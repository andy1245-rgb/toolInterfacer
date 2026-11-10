"""ensemble — other predictors composed, plus the number that says whether that bought anything.

Three modes:

* ``vote``  — each member names an option, the most-voted offered option wins. Stdlib only, no
  fitting, no probabilities (``probabilities=None``: a share of votes is not a probability, and
  putting it in the calibration column would be a fake number).
* ``mean``  — each member's probability distribution is restricted to the offered options,
  renormalised, and averaged per option over the members that cover it. Stdlib only, no fitting.
  A member that cannot emit probabilities is skipped and counted, and if no member can, the mean
  raises rather than falling back to a vote nobody asked for.
* ``stacked`` — a logistic meta-learner over the members' per-option scores.

**The grammar is not the top-level one.** A member spec contains ``|``, so members are separated
by ``;;`` and the member list continues across ``|`` fragments until a fragment whose key is one
of the ensemble's own (``members``, ``mode``, ``folds``) appears. ``~`` is accepted as a
member-internal ``|``, and it is the form to use whenever two members set the same parameter::

    ensemble|members=a:sklearn_text~clf=logreg;;b:torch_text~checkpoint=models/head.pt|mode=mean
    ensemble|members=a:sklearn_text|clf=linsvc|calibrate=sigmoid;;b:fake|accuracy=0.7|mode=vote

Why the second form is a convenience and not the safe one: ``protocol.build`` runs
``split_spec``, which is a flat ``key=value`` split into a **dict**, so a repeated key name is
overwritten *before* this factory sees the spec. ``members=a:X|k=1;;b:Y|k=2`` therefore arrives
here as ``members=a:X`` plus ``k=2``: the first member's value is gone and no parser downstream
can recover it. With ``~`` the whole member list travels inside one fragment and nothing can
collide, which is why it is the documented form for members that share parameter names.

A key that is neither a member continuation nor an ensemble key is an error naming the three
ensemble keys — silently ignoring it would score a model nobody specified.

**Ties break deterministically**: highest score, then the caller's option order, then the option
name. A tie decided by dict iteration order is a bug that shows up as a different accuracy on a
different Python.

**Stacked is out-of-fold or it is nothing.** ``fit`` assigns folds *by ``row.group``* (round
robin over sorted groups), refits a fresh copy of each member on the remaining folds through the
registered factories, and takes the meta-learner's features only from rows a member did not see.
It then refits the deployed members on all rows, which is ordinary stacking. If the dataset has
fewer than two groups, or a fold leaves a single training label, ``fit`` raises instead of
fitting the meta-learner on the rows it will score — an in-fold stack reports an accuracy it did
not earn, and that is exactly the failure this harness exists to catch.

**The error correlation is a deliverable.** An ensemble of predictors that fail on the same rows
buys nothing, so ``pairwise_error_agreement(masks)`` is a pure function anyone can call with two
lists of booleans, ``member_error_correlation(records_by_member)`` turns ``protocol.score``
output into the same table aligned by row id, and ``EnsemblePredictor.error_correlation()``
reports it for the calls this instance has actually served (``predict_for(row)`` supplies the
label that turns a choice into a wrong/right mask; the label is read *after* the prediction and
never reaches ``_combine``, so it cannot change a choice).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import combinations

from ..protocol import Prediction, PredictorError, build as build_predictor, register
from ..rows import Row

INSTALL = "python3 -m pip install scikit-learn"

MODES = ("vote", "mean", "stacked")
ENSEMBLE_KEYS = ("members", "mode", "folds")
MEMBER_SEPARATOR = ";;"
MEMBER_PIPE = "~"


def _missing(exc: ImportError, where: str) -> PredictorError:
    return PredictorError(
        f"ensemble(stacked): {where} needs scikit-learn, which is not importable ({exc}). "
        f"Install it with `{INSTALL}`. Modes vote and mean need nothing outside the standard "
        f"library; only the stacked meta-learner needs scikit-learn.")


def _require_sklearn(where: str) -> None:
    try:
        import sklearn  # noqa: F401,PLC0415 - optional dependency, imported on demand
    except ImportError as exc:
        raise _missing(exc, where) from None


def _argmax(known: list[str], values: dict[str, float]) -> str:
    """Highest value; ties break by option order, then name."""
    return min(known, key=lambda option: (-values[option], known.index(option), option))


def _resolve_mode(raw: str) -> str:
    """Mode names accept a unique prefix, so a hand-typed spec still parses."""
    text = (raw or "").strip().lower()
    if not text:
        return "vote"
    hits = [m for m in MODES if m.startswith(text)]
    if len(hits) == 1:
        return hits[0]
    near = f"a unique prefix matches {', '.join(hits)}" if hits else "no prefix matches"
    raise PredictorError(
        f"ensemble: unknown mode {raw!r}; known: {', '.join(MODES)} ({near})")


def _split_members(spec: str) -> list[tuple[str, str]]:
    if not spec.strip():
        raise PredictorError(
            "ensemble: members= is empty; expected members=a:SPEC;;b:SPEC, where each SPEC is a "
            "predictor spec and ';;' separates members (a member spec contains '|')")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in spec.split(MEMBER_SEPARATOR):
        text = chunk.strip()
        if not text:
            continue
        name, sep, member_spec = text.partition(":")
        member_spec = member_spec.replace(MEMBER_PIPE, "|")
        name, member_spec = name.strip(), member_spec.strip()
        if not sep or not member_spec:
            raise PredictorError(
                f"ensemble: member {text!r} has no spec; expected name:SPEC, e.g. "
                f"a:sklearn_text|clf=logreg")
        if name in seen:
            raise PredictorError(
                f"ensemble: member name {name!r} is used twice — member names are the keys of "
                f"the error-correlation table, so they have to be unique")
        seen.add(name)
        out.append((name, member_spec))
    if not out:
        raise PredictorError(f"ensemble: members={spec!r} names no members")
    return out


@dataclass
class _Call:
    """One served row: what each member said, what this ensemble said, and the truth if known."""
    label: str | None
    options: tuple[str, ...]
    choices: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    choice: str = ""


# ---------------------------------------------------------------- error correlation

def pairwise_error_agreement(masks: dict[str, list[bool]]) -> dict:
    """Pairwise agreement of error masks — ``True`` means the member got that row wrong.

    Returns a dict ready to print or serialise::

        {"n": 415, "members": ["a", "b"],
         "pairs": {"a|b": {"n": 415, "agreement": 0.83, "correlation": 0.41,
                           "both_wrong": 12, "a_only": 4, "b_only": 40, "both_right": 359}},
         "mean_agreement": 0.83}

    ``agreement`` is the fraction of rows on which the two masks are equal; ``correlation`` is
    the phi coefficient over the same 2x2 table (``None`` when a member is never wrong or always
    wrong, where a correlation is undefined rather than zero). ``both_wrong`` is the number that
    decides whether the ensemble is worth having: members that fail on the same rows agree at
    1.0 and buy only the rows where both are right anyway.
    """
    names = sorted(masks)
    if not names:
        raise PredictorError("pairwise_error_agreement: no member masks")
    lengths = {name: len(masks[name]) for name in names}
    if len(set(lengths.values())) > 1:
        raise PredictorError(
            f"pairwise_error_agreement: masks have different lengths {lengths} — masks over "
            f"different row sets are not comparable")
    n = lengths[names[0]]
    result: dict = {"n": n, "members": names, "pairs": {}, "mean_agreement": None}
    if n == 0 or len(names) < 2:
        return result
    agreements: list[float] = []
    for left, right in combinations(names, 2):
        a, b = masks[left], masks[right]
        both_wrong = sum(1 for x, y in zip(a, b) if x and y)
        a_only = sum(1 for x, y in zip(a, b) if x and not y)
        b_only = sum(1 for x, y in zip(a, b) if y and not x)
        both_right = n - both_wrong - a_only - b_only
        agreement = (both_wrong + both_right) / n
        denom = math.sqrt(float(both_wrong + a_only) * (b_only + both_right)
                          * (both_wrong + b_only) * (a_only + both_right))
        correlation = ((both_wrong * both_right) - (a_only * b_only)) / denom if denom else None
        agreements.append(agreement)
        result["pairs"][f"{left}|{right}"] = {
            "n": n, "agreement": agreement, "correlation": correlation,
            "both_wrong": both_wrong, "a_only": a_only, "b_only": b_only,
            "both_right": both_right,
        }
    result["mean_agreement"] = sum(agreements) / len(agreements) if agreements else None
    return result


def member_error_correlation(records_by_member: dict[str, list]) -> dict:
    """The same table, built from ``protocol.score`` output: ``{member: [Record, ...]}``.

    Records are aligned by ``row.id`` (every member of an ensemble is scored on the same rows, so
    a missing id means the two runs are not comparable and this raises). Bare booleans or
    ``Prediction``-shaped objects without ids are accepted positionally, ``True`` meaning wrong.
    """
    if not records_by_member:
        raise PredictorError("member_error_correlation: no members")
    masks: dict[str, list[bool]] = {}
    order: list[str] | None = None
    base = ""
    for name, records in records_by_member.items():
        items = list(records)
        if items and hasattr(items[0], "correct"):
            ids = [getattr(item, "id", None) for item in items]
            if all(isinstance(i, str) for i in ids):
                if order is None:
                    order, base = ids, name
                    if len(set(ids)) != len(ids):
                        raise PredictorError(
                            f"member_error_correlation: {name!r} repeats a row id, so the masks "
                            f"cannot be aligned by id")
                else:
                    position = {i: k for k, i in enumerate(ids)}
                    missing = [i for i in order if i not in position]
                    if missing or len(ids) != len(order):
                        raise PredictorError(
                            f"member_error_correlation: {name!r} covers a different row set than "
                            f"{base!r} (missing {missing[:5]}, {len(ids)} vs "
                            f"{len(order)} rows) — masks over different rows are not "
                            f"comparable")
                    items = [items[position[i]] for i in order]
            masks[name] = [not bool(item.correct) for item in items]
        else:
            masks[name] = [bool(item) for item in items]
    return pairwise_error_agreement(masks)


# ---------------------------------------------------------------- the predictor

class EnsemblePredictor:
    """Members composed by ``vote``, ``mean`` or a ``stacked`` meta-learner."""

    def __init__(self, members: list[tuple[str, str]], mode: str = "vote", folds: int = 5):
        if not members:
            raise PredictorError("ensemble: no members")
        self.member_specs: dict[str, str] = {name: spec for name, spec in members}
        self.member_names: tuple[str, ...] = tuple(name for name, _ in members)
        self.members: dict[str, object] = {name: build_predictor(spec)
                                           for name, spec in members}
        self.mode = mode
        self.folds = folds
        self.meta = None
        self.meta_classes: tuple[str, ...] = ()
        self.folds_used = 0
        self.fit_skipped: dict[str, str] = {}
        self.member_failures: dict[str, int] = {}
        self._calls: list[_Call] = []
        self.name = f"ensemble:{mode}({','.join(self.member_names)})"

    def __repr__(self) -> str:
        return f"EnsemblePredictor({self.name!r}, members={list(self.member_names)})"

    # ---------------------------------------------------------------- fit

    def fit(self, rows: list[Row]) -> None:
        if not rows:
            raise PredictorError("ensemble: fit needs at least one row")
        if self.mode == "stacked":
            self._fit_stacked(rows)
            return
        # vote and mean have no parameters of their own; a member that can fit still needs the
        # rows, which is how an ensemble of sklearn_text members is fitted through one call
        for name, member in self.members.items():
            if callable(getattr(member, "fit", None)):
                self._fit_member(name, member, rows)

    def _fit_member(self, name: str, member, rows: list[Row], fold: int | None = None) -> None:
        try:
            member.fit(rows)
        except NotImplementedError as exc:
            # a member that declares its fit unsupported keeps its loaded weights; the reason is
            # kept so a run can print why one member is untrained instead of guessing
            self.fit_skipped[name] = str(exc)
        except Exception as exc:
            where = f" at fold {fold}" if fold is not None else ""
            raise PredictorError(
                f"ensemble: member {name!r} failed to fit on {len(rows)} rows{where}: "
                f"{type(exc).__name__}: {exc}") from exc

    def _fit_stacked(self, rows: list[Row]) -> None:
        _require_sklearn("fitting a stacked ensemble")
        labels = sorted({r.label for r in rows})
        if len(labels) < 2:
            raise PredictorError(
                f"ensemble(stacked): fit needs at least two labels and got {labels}")
        assignment, folds = self._fold_assignment(rows)
        self.meta_classes = tuple(labels)
        features: list[list[float]] = []
        targets: list[str] = []
        for fold in range(folds):
            train = [r for r, f in zip(rows, assignment) if f != fold]
            test = [r for r, f in zip(rows, assignment) if f == fold]
            if not test:
                continue
            train_labels = sorted({r.label for r in train})
            if len(train_labels) < 2:
                raise PredictorError(
                    f"ensemble(stacked): fold {fold} leaves a single training label "
                    f"{train_labels} over {len(train)} rows — a grouped out-of-fold split "
                    f"cannot be formed honestly on this dataset, so this refuses instead of "
                    f"fitting the meta-learner on the rows it will score")
            fold_members = {name: build_predictor(spec)
                            for name, spec in self.member_specs.items()}
            for name, member in fold_members.items():
                if callable(getattr(member, "fit", None)):
                    self._fit_member(name, member, train, fold=fold)
            for row in test:
                predictions = {}
                for name, member in fold_members.items():
                    try:
                        predictions[name] = member.predict(row.text, row.options)
                    except Exception as exc:  # noqa: BLE001 - a fold member that cannot answer
                        self.member_failures[f"{name}@fold{fold}"] = (
                            self.member_failures.get(f"{name}@fold{fold}", 0) + 1)
                        self.fit_skipped.setdefault(
                            f"{name}@fold{fold}", f"{type(exc).__name__}: {exc}")
                features.append(self._feature_vector(predictions))
                targets.append(row.label)
        if not features:
            raise PredictorError(
                f"ensemble(stacked): no out-of-fold rows were produced over {len(rows)} rows in "
                f"{folds} folds")
        self.folds_used = folds
        self.meta = self._fit_meta(features, targets)
        # the deployed members are refitted on every row; only the meta-learner's features had
        # to be out-of-fold, and they were
        for name, member in self.members.items():
            if callable(getattr(member, "fit", None)):
                self._fit_member(name, member, rows)

    def _fold_assignment(self, rows: list[Row]) -> tuple[list[int], int]:
        groups = sorted({r.group for r in rows})
        folds = min(self.folds, len(groups))
        if folds < 2:
            raise PredictorError(
                f"ensemble(stacked): out-of-fold needs at least two groups and this dataset has "
                f"{len(groups)} ({groups[:5]}) — a meta-learner fitted on the rows it scores "
                f"reports an accuracy it did not earn, so this refuses instead")
        fold_of = {group: index % folds for index, group in enumerate(groups)}
        return [fold_of[r.group] for r in rows], folds

    def _fit_meta(self, features: list[list[float]], targets: list[str]):
        try:
            from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on the box
            raise _missing(exc, "the meta-learner") from None
        model = LogisticRegression(max_iter=2000)
        try:
            model.fit(features, targets)
        except Exception as exc:
            raise PredictorError(
                f"ensemble(stacked): the meta-learner failed to fit on {len(features)} "
                f"out-of-fold rows x {len(features[0])} features over {len(set(targets))} "
                f"labels: {type(exc).__name__}: {exc}") from exc
        return model

    def _feature_vector(self, predictions: dict[str, Prediction]) -> list[float]:
        vector: list[float] = []
        for name in self.member_names:
            prediction = predictions.get(name)
            for label in self.meta_classes:
                if prediction is None:
                    vector.append(0.0)
                elif prediction.probabilities:
                    vector.append(float(prediction.probabilities.get(label, 0.0)))
                else:
                    vector.append(1.0 if prediction.choice == label else 0.0)
        return vector

    # ---------------------------------------------------------------- predict

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        return self._predict(text, options, None)

    def predict_for(self, row: Row) -> Prediction:
        """Same prediction, plus the label kept for the error mask — read after, never before."""
        return self._predict(row.text, row.options, row.label)

    def _predict(self, text: str, options: tuple[str, ...], label: str | None) -> Prediction:
        t0 = time.perf_counter()
        offered = tuple(options)
        if not offered:
            raise PredictorError("ensemble: predict got an empty option set")
        predictions, errors = self._ask(text, offered)
        choice, probabilities = "", None
        try:
            choice, probabilities = self._combine(offered, predictions, errors)
        finally:
            # logged even when the combination raises, so a member that fails on the same rows
            # every time still shows up in error_correlation() as failing on them
            self._calls.append(_Call(label=label, options=offered,
                                     choices={n: p.choice for n, p in predictions.items()},
                                     errors=errors, choice=choice))
        return Prediction(choice=choice, probabilities=probabilities,
                          latency_ms=(time.perf_counter() - t0) * 1000.0)

    def _ask(self, text: str, offered: tuple[str, ...]
             ) -> tuple[dict[str, Prediction], dict[str, str]]:
        predictions: dict[str, Prediction] = {}
        errors: dict[str, str] = {}
        for name, member in self.members.items():
            try:
                predictions[name] = member.predict(text, offered)
            except Exception as exc:  # noqa: BLE001 - kept with its message, not swallowed
                errors[name] = f"{type(exc).__name__}: {exc}"
                self.member_failures[name] = self.member_failures.get(name, 0) + 1
        return predictions, errors

    def _combine(self, offered: tuple[str, ...], predictions: dict[str, Prediction],
                 errors: dict[str, str]) -> tuple[str, dict[str, float] | None]:
        if not predictions:
            raise PredictorError(
                f"ensemble({self.mode}): every member raised on this row: "
                + "; ".join(f"{name}: {message}" for name, message in errors.items()))
        if self.mode == "vote":
            return self._vote(offered, predictions)
        if self.mode == "mean":
            return self._mean(offered, predictions, errors)
        return self._stacked(offered, predictions)

    def _vote(self, offered: tuple[str, ...],
              predictions: dict[str, Prediction]) -> tuple[str, None]:
        tally = {option: 0 for option in offered}
        off_grid: dict[str, str] = {}
        for name, prediction in predictions.items():
            if prediction.choice in tally:
                tally[prediction.choice] += 1
            else:
                off_grid[name] = prediction.choice
        if not any(tally.values()):
            raise PredictorError(
                f"ensemble(vote): no member returned an option from the offered set "
                f"(no votes for {list(offered)}; members answered {off_grid})")
        scores = {option: float(count) for option, count in tally.items()}
        return _argmax(list(offered), scores), None

    def _mean(self, offered: tuple[str, ...], predictions: dict[str, Prediction],
              errors: dict[str, str]) -> tuple[str, dict[str, float]]:
        totals = {option: 0.0 for option in offered}
        cover = {option: 0 for option in offered}
        without: list[str] = []
        for name, prediction in predictions.items():
            if not prediction.probabilities:
                without.append(name)
                continue
            known = [o for o in offered if o in prediction.probabilities]
            if not known:
                raise PredictorError(
                    f"ensemble(mean): member {name!r} has no probability on any offered option "
                    f"(its distribution covers {sorted(prediction.probabilities)[:6]})")
            total = sum(prediction.probabilities[o] for o in known)
            if not total > 0.0:
                raise PredictorError(
                    f"ensemble(mean): member {name!r} put zero probability mass on the offered "
                    f"options {known}")
            # averaged per option over the members that cover it, so a member trained on a
            # different label space does not silently down-weight the options it never saw
            for option in known:
                totals[option] += prediction.probabilities[option] / total
                cover[option] += 1
        if not any(cover.values()):
            raise PredictorError(
                f"ensemble(mean): no member produced probabilities (members without: {without}; "
                f"members that raised: {sorted(errors)}) — mean needs a probability column, so "
                f"calibrate a linsvc or use mode=vote")
        averaged = {option: totals[option] / cover[option]
                    for option in offered if cover[option]}
        norm = sum(averaged.values())
        if not norm > 0.0:
            raise PredictorError(
                f"ensemble(mean): the members' averaged distribution has no mass on any offered "
                f"option {list(offered)}")
        probabilities = {option: value / norm for option, value in averaged.items()}
        return _argmax(list(probabilities), probabilities), probabilities

    def _stacked(self, offered: tuple[str, ...],
                 predictions: dict[str, Prediction]) -> tuple[str, dict[str, float]]:
        if self.meta is None:
            raise PredictorError(
                f"ensemble(stacked): the meta-learner is not fitted — call fit(rows) first; "
                f"stacked is the one mode that needs fitting")
        vector = self._feature_vector(predictions)
        try:
            weights = self.meta.predict_proba([vector])[0]
        except Exception as exc:
            raise PredictorError(
                f"ensemble(stacked): the meta-learner could not score this row: "
                f"{type(exc).__name__}: {exc}") from exc
        raw = {str(label): float(p) for label, p in zip(self.meta.classes_, weights)}
        known = [option for option in offered if option in raw]
        if not known:
            raise PredictorError(
                f"ensemble(stacked): none of the offered options is a meta-learner class "
                f"({list(raw)[:6]}) — an option set the stack never saw cannot be scored")
        total = sum(raw[option] for option in known)
        if not total > 0.0:
            raise PredictorError(
                f"ensemble(stacked): the meta-learner put no mass on any offered option {known}")
        probabilities = {option: raw[option] / total for option in known}
        return _argmax(known, probabilities), probabilities

    # ---------------------------------------------------------------- error correlation

    def error_correlation(self) -> dict:
        """Pairwise error agreement over the calls this instance has actually served.

        A member counts as wrong on a row when it raised or when its choice differed from the
        row's label. Rows scored through ``predict`` carry no label, so they cannot contribute a
        mask — they are counted in ``unlabelled`` instead of being treated as right, because a
        silent zero would read as perfect agreement.
        """
        masks = {name: [] for name in self.member_names}
        labelled = 0
        for call in self._calls:
            if call.label is None:
                continue
            labelled += 1
            for name in self.member_names:
                masks[name].append(name in call.errors
                                   or call.choices.get(name) != call.label)
        result = pairwise_error_agreement(masks) if labelled else {
            "n": 0, "members": sorted(self.member_names), "pairs": {}, "mean_agreement": None}
        result["calls"] = len(self._calls)
        result["labelled"] = labelled
        result["unlabelled"] = len(self._calls) - labelled
        result["member_wrong"] = {name: sum(mask) for name, mask in masks.items()}
        result["member_failed"] = dict(self.member_failures)
        return result

    def reset_error_log(self) -> None:
        self._calls.clear()
        self.member_failures.clear()


@register("ensemble")
def build(**fragments: str) -> EnsemblePredictor:
    member_spec = ""
    own: dict[str, str] = {}
    joining = False
    for key, value in fragments.items():
        if key == "members":
            if member_spec:
                raise PredictorError(
                    "ensemble: members= appears twice; one ensemble has one member list "
                    "(separate members with ';;')")
            member_spec = value
            joining = True
            continue
        if joining and key not in ENSEMBLE_KEYS:
            member_spec += "|" + key + "=" + value
            continue
        joining = False
        own[key] = value
    unknown = sorted(key for key in own if key not in ENSEMBLE_KEYS)
    if unknown:
        raise PredictorError(
            f"ensemble: unknown key(s) {unknown}; the ensemble's own keys are "
            f"{', '.join(ENSEMBLE_KEYS)}. A member spec continues across '|' until one of those "
            f"appears, so a member key written after mode=/folds= has to move before them: "
            f"ensemble|members=a:SPEC|k=v;;b:SPEC|mode=vote")
    mode = _resolve_mode(own.get("mode", "vote"))
    try:
        folds = int(own.get("folds", "5"))
    except ValueError as exc:
        raise PredictorError(
            f"ensemble: folds={own.get('folds')!r} must be an int: {exc}") from exc
    if folds < 2:
        raise PredictorError(f"ensemble: folds={folds} must be at least 2 for an out-of-fold fit")
    if mode == "stacked":
        _require_sklearn("building mode=stacked")
    return EnsemblePredictor(_split_members(member_spec), mode=mode, folds=folds)
