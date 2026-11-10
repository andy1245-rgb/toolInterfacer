"""recorded — replay an arm that is already on disk.

This is what makes the historical records the regression suite: every arm in
`bench/results/p3-transfer/rows.jsonl` becomes a predictor, so the harness can be asked to
reproduce a published number with no server, no model and no network.

Replay needs the row identity, not just its text, so this predictor also implements
``predict_for(row)``; ``protocol.score`` prefers that hook when it exists. A text-only lookup
is kept as a fallback and refuses duplicates rather than guessing.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..protocol import PredictorError, Prediction, register

ROOT = Path(__file__).resolve().parents[3]


class RecordedPredictor:
    """One arm of a ``rows.jsonl``, replayed."""

    replays_a_file = True

    def __init__(self, path: str, arm: str, order_sha: str = ""):
        src = Path(path) if path else ROOT / "bench/results/p3-transfer/rows.jsonl"
        if not src.is_absolute():
            src = (ROOT / src) if (ROOT / src).exists() else src
        if not src.exists():
            raise PredictorError(f"recorded: {src} is missing")
        self.path = src
        self.arm = arm
        self.expected_order = order_sha
        self.by_id: dict[str, dict] = {}
        self.by_text: dict[str, dict] = {}
        dupes = 0
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("arm") != arm:
                continue
            self.by_id[rec["id"]] = rec
            text = rec.get("utterance") or rec.get("text") or ""
            if text in self.by_text:
                dupes += 1
                self.by_text[text] = None  # ambiguous: refuse rather than pick one
            else:
                self.by_text[text] = rec
        if not self.by_id:
            arms = sorted({json.loads(l)["arm"] for l in
                           src.read_text(encoding="utf-8").splitlines() if l.strip()})
            raise PredictorError(f"recorded: arm {arm!r} not in {src} (arms: {arms})")
        self.dupe_texts = dupes
        self.name = f"recorded:{arm}"

    def _prediction(self, rec: dict) -> Prediction:
        probs = rec.get("probabilities")
        if rec.get("ms") is not None:
            latency = float(rec["ms"])
        else:
            latency = float(rec.get("latency_ms") or 0.0)
        return Prediction(choice=rec.get("choice") or rec.get("pred") or "",
                          probabilities=probs, latency_ms=latency)

    def predict_for(self, row) -> Prediction:
        self._check_order(getattr(row, "options", None))
        rec = self.by_id.get(row.id)
        if rec is None:
            raise PredictorError(f"recorded: {self.arm} has no row {row.id!r}")
        return self._prediction(rec)

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        self._check_order(options)
        rec = self.by_text.get(text)
        if rec is None:
            raise PredictorError(f"recorded: {self.arm} has no unique row with this text")
        return self._prediction(rec)

    def _check_order(self, options) -> None:
        if not self.expected_order or options is None:
            return
        from ..protocol import option_order_sha
        got = option_order_sha(tuple(options))
        if got != self.expected_order:
            raise PredictorError(
                f"recorded: {self.arm} was recorded under order {self.expected_order}, "
                f"asked under {got} — replaying a fixed order under a permuted one is not a "
                f"measurement of anything")


@register("recorded")
def build(path: str = "", arm: str = "floor_a", order_sha: str = "", **_kw) -> RecordedPredictor:
    return RecordedPredictor(path=path or "", arm=arm, order_sha=order_sha)
