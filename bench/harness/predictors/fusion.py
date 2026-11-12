"""fusion — two predictors, one answer: an equal-weight mean, or a confidence gate.

The arm this exists for is the 3B plus a cheap lexical member. The numbers, with their
source: `bench/results/fusion-2026-11-10/summary.json` holds the one pre-registered run.
Equal-weight mean on the 415 routing rows: **351/415 = 84.6%** against the 3B's 325/415 =
78.3%, fixed 37 / broke 11, exact McNemar p = 0.00022, positive in 5 of 5 fold seeds — which
**fails** its own pre-registered condition that no label may net worse than -3 (`web_search`
netts -5). The gate form at t=0.38 is weaker and not significant: 330/415, p = 0.267. The
members are not redundant: they are both wrong on 42 of 415 rows, φ = 0.274.

Three things a reader should not have to guess:

* **The base arm is a replay.** ``fusion`` reads the other member's per-row distribution from
  a recorded artifact (``base_path`` + ``base_arm``), not from a live server. That is what
  makes the arm free, offline and reproducible; it also means the fusion inherits that
  artifact's option order, which is why the predictor declares ``replays_a_file`` and a run
  refuses to score it under ``--order``.
* **``weights`` is the weight on the base arm** and defaults to 0.5, the null. It exists so a
  weighted mean is *expressible*; a weight picked by sweeping it against the rows being
  scored is a knob, not a result, and the honest form of that experiment tunes on a fit split
  and reports the tuned value with its selection cost.
* **The gate threshold is a claim about the base arm's calibration**, not a free parameter.
  ``gate=base_conf|t=0.38`` says "the base is unsure below 0.38". If the base emits no
  probabilities the gate cannot be evaluated and the spec is refused rather than guessed.

An unfitted fusion (one whose member has ``fit``) raises from ``predict`` with the reason, as
every fitted predictor here does: grouped CV builds a fresh one per fold.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..protocol import Prediction, PredictorError, build as build_predictor, register
from ..rows import Row

MODES = ("mean", "gate")
MASKS = ("auto", "off")
FUSION_KEYS = ("member", "mode", "base", "base_arm", "base_path", "t", "weights",
               "mask", "mask_folds", "mask_min_rows", "seed")
MEMBER_PIPE = "~"
DEFAULT_BASE = "recorded|arm=floor_a"
DEFAULT_BASE_PATH = "bench/results/p3-transfer/rows.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _argmax(known: list[str], values: dict[str, float]) -> str:
    """Highest value; ties break by option order, then name."""
    return min(known, key=lambda option: (-values[option], known.index(option), option))


def _resolve_mode(raw: str) -> str:
    text = (raw or "").strip().lower()
    if not text:
        return "mean"
    hits = [m for m in MODES if m.startswith(text)]
    if len(hits) == 1:
        return hits[0]
    near = f"a unique prefix matches {', '.join(hits)}" if hits else "no prefix matches"
    raise PredictorError(f"fusion: unknown mode {raw!r}; known: {', '.join(MODES)} ({near})")


def load_base(path: str | Path, arm: str) -> dict[str, dict]:
    """Per-row distributions for one arm of a recorded artifact, keyed by row id."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise PredictorError(f"fusion: base artifact {p} does not exist")
    out: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("arm") != arm:
            continue
        probs = row.get("probabilities")
        if not probs:
            raise PredictorError(
                f"fusion: arm {arm!r} of {p.name} has no probabilities on row {row.get('id')}; "
                f"a fusion needs both members to emit a distribution")
        if row["id"] in out:
            raise PredictorError(
                f"fusion: duplicate row id {row['id']!r} for arm {arm!r} in {p.name} — a "
                f"later line would silently replace an earlier one, so the base is ambiguous")
        out[row["id"]] = {"choice": row.get("choice"), "probabilities": probs}
    if not out:
        raise PredictorError(f"fusion: no rows for arm {arm!r} in {p}")
    return out


class FusionPredictor:
    replays_a_file = True

    def __init__(self, member: str = "", mode: str = "mean", base: str = DEFAULT_BASE,
                 base_arm: str = "floor_a", base_path: str = DEFAULT_BASE_PATH,
                 t: str = "0.38", weights: str = "0.5", mask: str = "off",
                 mask_folds: str = "5", mask_min_rows: str = "10", seed: str = "0") -> None:
        if not member.strip():
            raise PredictorError(
                "fusion: member= is empty; expected member=SPEC, where SPEC is any predictor "
                "spec and its own '|' separators are written as '~' "
                "(member=sklearn_text~vectorizer=tfidf_word~clf=logreg)")
        self.member_spec = member.strip().replace(MEMBER_PIPE, "|")
        self.mode = _resolve_mode(mode)
        self.base_spec = (base.strip() or DEFAULT_BASE).replace(MEMBER_PIPE, "|")
        self.base_path = base_path
        try:
            self.base_weight = float(weights)
            self.threshold = float(t)
        except ValueError as exc:
            raise PredictorError(f"fusion: weights={weights!r} t={t!r} must be numeric: {exc}") \
                from None
        if not 0.0 <= self.base_weight <= 1.0:
            raise PredictorError(f"fusion: weights={self.base_weight} must be in [0, 1]")
        if not 0.0 <= self.threshold <= 1.0:
            raise PredictorError(f"fusion: t={self.threshold} must be in [0, 1]")
        self.base_arm = base_arm or self._arm_from_spec()
        if mask not in MASKS:
            raise PredictorError(f"fusion: unknown mask {mask!r}; known: {', '.join(MASKS)}")
        self.mask = mask
        try:
            self.mask_folds = int(mask_folds)
            self.mask_min_rows = int(mask_min_rows)
            self.seed = int(seed)
        except ValueError as exc:
            raise PredictorError(
                f"fusion: mask_folds={mask_folds!r} mask_min_rows={mask_min_rows!r} "
                f"seed={seed!r} must be integers: {exc}") from None
        self.mask_labels: tuple[str, ...] | None = None
        self.mask_tally: dict[str, list[int]] = {}
        self.member = build_predictor(self.member_spec)
        self._base: dict[str, dict] | None = None
        self.name = f"fusion:{self.mode}@{self.base_weight:g}:{self.member.name}"

    def _arm_from_spec(self) -> str:
        kw = dict(part.split("=", 1) for part in self.base_spec.split("|")[1:] if "=" in part)
        return kw.get("arm", "floor_a")

    def base_rows(self) -> dict[str, dict]:
        if self._base is None:
            self._base = load_base(self.base_path, self.base_arm)
        return self._base

    def fit(self, rows: list[Row]) -> None:
        rows = list(rows)
        if not hasattr(self.member, "fit"):
            return
        if self.mask == "auto":
            self.mask_labels = self.learn_mask(rows)
        self.member.fit(rows)

    def learn_mask(self, rows: list[Row]) -> tuple[str, ...]:
        """Which labels this member may vote for, from its own out-of-fold confusion.

        The rule, declared before any run that uses it: the member may vote for label ``L`` only
        when, on the rows where it votes ``L``, it is right **more often than the base arm** is.
        That is the question a fusion actually asks — *should this member be allowed to override
        the base here* — and it is a question per-label precision cannot answer, because a member
        can be accurate on a label while voting it on rows that belong to another one.

        The tally is built from an inner cross-validation over exactly the rows handed to
        ``fit``, scored against the base arm's recorded choices for those same ids. Nothing in it
        touches the outer fold's held-out rows.
        """
        from .. import metrics  # noqa: PLC0415 - metrics imports score, which imports this package

        truth = {r.id: r.label for r in rows}
        inner = metrics.grouped_cv(rows, (lambda: build_predictor(self.member_spec)),
                                   folds=self.mask_folds, seed=self.seed, stratify=True)
        base = self.base_rows()
        tally: dict[str, list[int]] = {}
        for rec in inner["oof"]:
            if not rec.choice or rec.id not in base or rec.id not in truth:
                continue
            counts = tally.setdefault(rec.choice, [0, 0, 0])
            counts[0] += 1
            counts[1] += 1 if rec.correct else 0
            counts[2] += 1 if base[rec.id]["choice"] == truth[rec.id] else 0
        self.mask_tally = tally
        return tuple(sorted(label for label, (n, mine, theirs) in tally.items()
                            if n >= self.mask_min_rows and mine > theirs))

    def _masked(self, options: tuple[str, ...]) -> tuple[str, ...]:
        """The member's allowed vote, restricted to its demonstrated competence."""
        if self.mask_labels is None:
            return tuple(options)
        allowed = tuple(o for o in options if o in self.mask_labels)
        return allowed

    def describe(self) -> dict:
        return {"member": self.member_spec, "mode": self.mode,
                "base": self.base_spec, "base_weight": self.base_weight,
                "mask": self.mask, "mask_labels": list(self.mask_labels or ()),
                "mask_tally": {k: {"n": v[0], "member_right": v[1], "base_right": v[2]}
                               for k, v in sorted(self.mask_tally.items())}}

    def _member_prediction(self, row: Row, options: tuple[str, ...]) -> Prediction:
        return self.member.predict(row.text, options)

    def predict_for(self, row: Row, options: tuple[str, ...] | None = None) -> Prediction:
        t0 = time.perf_counter()
        offered = tuple(options or row.options)
        base = self.base_rows().get(row.id)
        if base is None:
            raise PredictorError(
                f"fusion: the base arm has no row {row.id!r}; it was recorded on a different "
                f"row set, so this row cannot be fused")
        member = self._member_prediction(row, offered)

        known = [o for o in offered if o in base["probabilities"]]
        if not known:
            raise PredictorError(
                f"fusion: the base distribution covers none of the {len(offered)} offered "
                f"options on row {row.id!r}")

        if self.mode == "gate":
            base_dist = {o: float(base["probabilities"][o]) for o in known}
            base_conf = max(base_dist.values())
            use_member = base_conf < self.threshold
            allowed = self._masked(offered)
            if use_member and member.choice not in allowed:
                choice = _argmax(known, base_dist)
                probabilities, source = base_dist, "base_masked"
            elif use_member:
                choice, probabilities, source = member.choice, member.probabilities, "member"
            else:
                choice = _argmax(known, base_dist)
                probabilities, source = base_dist, "base"
        else:
            if not member.probabilities:
                raise PredictorError(
                    f"fusion(mean): member {self.member.name} emits no probabilities, so an "
                    f"equal-weight mean is undefined. Give the member a probabilistic "
                    f"classifier (clf=logreg) or calibrate it (calibrate=sigmoid), or use "
                    f"mode=gate, which needs only the member's choice")
            allowed = self._masked(offered)
            if member.choice not in allowed:
                # "may not vote for L" is not "votes against L". A pruned distribution in a mean
                # blend is an active vote against everything pruned — measured: masking the
                # member's web_search vote cost web_search 8 net rows instead of saving it. A
                # member that cannot name its top choice has no opinion here, so the base's own
                # distribution stands alone.
                base_dist = {o: float(base["probabilities"][o]) for o in known}
                return Prediction(choice=_argmax(known, base_dist), probabilities=base_dist,
                                  latency_ms=(time.perf_counter() - t0) * 1000.0,
                                  meta={"fusion": "base_masked"})
            member_dist = {o: float(member.probabilities[o]) for o in known
                           if o in member.probabilities}
            if not member_dist:
                # the mask removed every label the member was willing to name: it has no vote
                # here, so the base's own answer stands rather than a uniform guess
                base_dist = {o: float(base["probabilities"][o]) for o in known}
                return Prediction(choice=_argmax(known, base_dist), probabilities=base_dist,
                                  latency_ms=(time.perf_counter() - t0) * 1000.0,
                                  meta={"fusion": "base_masked"})
            base_dist = {o: float(base["probabilities"][o]) for o in known}
            blended = {o: self.base_weight * base_dist[o]
                          + (1.0 - self.base_weight) * member_dist.get(o, 0.0) for o in known}
            total = sum(blended.values())
            if not total > 0.0:
                raise PredictorError(f"fusion(mean): blended distribution is empty on {row.id!r}")
            probabilities = {o: v / total for o, v in blended.items()}
            choice, source = _argmax(known, probabilities), "mean"

        return Prediction(choice=choice, probabilities=probabilities,
                          latency_ms=(time.perf_counter() - t0) * 1000.0,
                          meta={"fusion": source})

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        raise PredictorError(
            "fusion: this predictor is bound to a recorded row set and needs row identity; "
            "score it through the harness (which calls predict_for) rather than directly")


@register("fusion")
def build(member: str = "", mode: str = "mean", base: str = DEFAULT_BASE,
          base_arm: str = "", base_path: str = DEFAULT_BASE_PATH, t: str = "0.38",
          weights: str = "0.5", mask: str = "off", mask_folds: str = "5",
          mask_min_rows: str = "10", seed: str = "0", **_kw) -> FusionPredictor:
    if _kw:
        raise PredictorError(
            f"fusion: unknown key(s) {sorted(_kw)}; known: {', '.join(FUSION_KEYS)}. A member or "
            f"base spec nests by writing its own pipes as ~, so a stray key here means the "
            f"nesting was lost")
    return FusionPredictor(member=member, mode=mode, base=base, base_arm=base_arm,
                           base_path=base_path, t=t, weights=weights, mask=mask,
                           mask_folds=mask_folds, mask_min_rows=mask_min_rows, seed=seed)
