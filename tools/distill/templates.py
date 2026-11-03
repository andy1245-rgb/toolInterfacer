"""templates — expand the authored frames into rows, and check every band claim.

Three jobs, in order:

1. **fill** a frame's slots, rotating values per frame so two frames for the same
   op never produce the same sentence;
2. **check the band** the frame claims, with the same vocabulary ``bands.py`` uses
   for mined text — a t2 frame that leaks "grep" is a template bug, not a hard
   row, so it is dropped and counted rather than trained on;
3. **build the row**: candidate block (gold + named rival + fillers, sorted by
   name), the packed state, the split family.

Volume is fixed per band and per op so the head learns every op equally (01 §5
"balance by tool for the head"), and t3 carries the largest share because the
boundary pairs are where routing actually fails (01 §4: hand-authored confusable
rows are worth ~10x a generic teacher row).
"""
from __future__ import annotations

import hashlib
import itertools
import re
from typing import Any, Iterator

from . import bands, catalog, corpus, frames

# The one instruction string every row carries. The request itself lives in
# state["intent"]; keeping the instruction generic means the model learns to read
# the state rather than to match an instruction it will never see twice.
ROUTE_INSTRUCTION = ("Pick the single candidate op that carries out the user's request. "
                     "Answer with one candidate key, copied verbatim, and a confidence "
                     "from 0.0 to 1.0.")

T1_PER_OP = 18
T2_PER_OP = 14
T3_PER_DIRECTION = 8
VARIANTS_PER_FRAME = 2

_SLOT = re.compile(r"\{(\w+)\}")
_CANDIDATE_SIZES = (5, 5, 4, 6, 6, 5, 8)


def _salt(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def fill(text: str, salt: int) -> Iterator[str]:
    """Each variant picks the next value per slot; `salt` keeps frames out of step."""
    names = _SLOT.findall(text)
    if not names:
        yield text
        return
    limit = max(len(frames.SLOTS[n]) for n in names)
    for i in range(limit):
        def sub(m: re.Match[str], _i: int = i) -> str:
            values = frames.SLOTS[m.group(1)]
            return values[(salt + _i) % len(values)]
        yield _SLOT.sub(sub, text)


def _band_violation(band: str, intended: str, rival: str | None, text: str) -> str:
    """Why `text` does not earn the band the frame claims ("" when it does).

    t3 is checked for the label side only. The rival side of a boundary row is a
    *semantic* claim ("the other op is a defensible reading", per the frozen
    set's notes) and most of these frames state it by negation — "the names, not
    what's in them" — which no vocabulary match can see. What the row can prove
    mechanically is that the rival is on the ballot, and `corpus.assert_row`
    enforces exactly that. Whether the rival was cued lexically is counted and
    reported instead of asserted.
    """
    hits = bands.cue_hits(text)
    if band == "t1":
        if not hits:
            return "t1 frame carries no op-signature vocabulary"
        ranked = sorted(hits, key=lambda op: (-len(hits[op]), op))
        if ranked[0] != intended:
            return f"t1 frame cues {ranked[0]!r}, not the intended {intended!r}"
        if len(ranked) > 1:
            return f"t1 frame also cues {ranked[1:]!r} — that is a boundary row"
        return ""
    if band == "t2":
        return f"t2 frame leaks {sorted(hits)!r}" if hits else ""
    if not hits.get(intended):
        return f"t3 frame does not cue its label {intended!r}"
    return ""


def _state(op: str, utterance: str) -> dict[str, Any]:
    """The state the decision model reads, bounded the way the service bounds it."""
    from ti.context.packer import guard_state

    ctx: dict[str, Any] = {}
    digest: dict[str, Any] = {}
    if op.startswith("filesys_"):
        ctx["cwd"] = "~/repo"
    if op in ("decide_pick", "decide_extract"):
        digest = {"summary": "the previous step returned a list of items; they are on screen"}
    state = {"intent": utterance, "params_context": ctx, "digest": digest}
    return guard_state(state)


def _candidates(op: str, family: str, rival: str | None) -> dict[str, str]:
    size = _CANDIDATE_SIZES[_salt(family) % len(_CANDIDATE_SIZES)]
    must = (rival,) if rival else ()
    return catalog.candidate_set(op, seed=family, size=max(size, len(must) + 1),
                                 must_include=must)


def build_row(*, band: str, source: str, op: str, rival: str | None, family: str,
              text: str, index: int) -> dict[str, Any]:
    """One row from a declared (band, op, rival, family, utterance).

    ``op`` seeds the candidate block: for a template row it is the frame's gold,
    for a mined row it is the cue-derived anchor. The *label* is the teacher's
    answer either way and arrives in ``label.py`` — this function never writes one.
    """
    return corpus.build(
        band=band, source=source, family=family, utterance=text,
        instructions=ROUTE_INSTRUCTION, state=_state(op, text),
        candidates=_candidates(op, family, rival), intended=op,
        tier_target=rival, index=index)


def _row(*, band: str, op: str, rival: str | None, frame_index: int, text: str,
         index: int) -> dict[str, Any]:
    family = (f"{band}:{op}->{rival}:{frame_index:02d}" if rival
              else f"{band}:{op}:{frame_index:02d}")
    return build_row(band=band, source="template", op=op, rival=rival,
                     family=family, text=text, index=index)


class _Sink:
    """Collects rows while enforcing dedup + the frozen-set refusal, with counts."""

    def __init__(self, gate: corpus.Gate) -> None:
        self.rows: list[dict[str, Any]] = []
        self.gate = gate

    @property
    def dropped(self) -> dict[str, int]:
        return self.gate.dropped

    @property
    def rejected(self) -> list[dict[str, str]]:
        return self.gate.rejected

    def offer(self, *, band: str, op: str, rival: str | None, frame_index: int,
              text: str) -> bool:
        violation = _band_violation(band, op, rival, text)
        if violation:
            self.gate.drop("band", text, violation)
            return False
        reason = self.gate.offer(text)
        if reason:
            return False
        self.rows.append(_row(band=band, op=op, rival=rival, frame_index=frame_index,
                              text=text, index=len(self.rows) + 1))
        return True


def generate(gate: corpus.Gate | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every template row, plus the report on what was dropped and why."""
    sink = _Sink(gate if gate is not None else corpus.Gate())

    for op in sorted(frames.T1):
        made, fi = 0, 0
        for text in frames.T1[op]:
            fi += 1
            for variant in itertools.islice(fill(text, _salt("t1", op, str(fi))),
                                            VARIANTS_PER_FRAME):
                if made >= T1_PER_OP:
                    break
                if sink.offer(band="t1", op=op, rival=None, frame_index=fi, text=variant):
                    made += 1

    for op in sorted(frames.T2):
        made, fi = 0, 0
        for text in frames.T2[op]:
            fi += 1
            for variant in itertools.islice(fill(text, _salt("t2", op, str(fi))),
                                            VARIANTS_PER_FRAME):
                if made >= T2_PER_OP:
                    break
                if sink.offer(band="t2", op=op, rival=None, frame_index=fi, text=variant):
                    made += 1

    for label, rival, texts in frames.T3:
        made, fi = 0, 0
        for text in texts:
            fi += 1
            for variant in itertools.islice(fill(text, _salt("t3", label, rival, str(fi))),
                                            VARIANTS_PER_FRAME):
                if made >= T3_PER_DIRECTION:
                    break
                if sink.offer(band="t3", op=label, rival=rival, frame_index=fi,
                              text=variant):
                    made += 1

    per_band: dict[str, int] = {}
    per_op: dict[str, int] = {}
    for row in sink.rows:
        per_band[row["band"]] = per_band.get(row["band"], 0) + 1
        per_op[row["intended"]] = per_op.get(row["intended"], 0) + 1
    report = {
        "rows": len(sink.rows),
        "per_band": per_band,
        "per_op": per_op,
        "dropped": sink.dropped,
        "dropped_examples": sink.rejected,
        "targets": {"t1_per_op": T1_PER_OP, "t2_per_op": T2_PER_OP,
                    "t3_per_direction": T3_PER_DIRECTION,
                    "t3_pairs": len(frames.T3)},
    }
    return sink.rows, report
