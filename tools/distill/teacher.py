"""teacher — the model whose decisions become the corpus's targets.

Two implementations behind one method, ``choose(state, instructions, criteria)``:

``HostedTeacher``  the service's own ``openrouter`` backend (``ti.decide.backends.
                   openrouter``), subclassed to keep the wire format, the retry
                   policy, the provider pin and the key handling exactly as the
                   product uses them. Two things are added for labelling and
                   nothing else is changed: the request body asks OpenRouter to
                   include the charge (``usage: {include: true}``), and the reply
                   envelope is captured so a row can record its raw response,
                   usage and the upstream that actually served it.

``FixtureTeacher`` a deterministic stand-in with no network and no key, used by
                   ``--dry-run`` so CI can exercise the whole path for free. It reads
                   the op-signature vocabulary to pick a candidate; that is a
                   property of a fixture, never a claim about decision quality.

The teacher is *not* an oracle. ``label.py`` records what it said, ``audit.py``
puts a stratified sample in front of a human, and ``stats.py`` reports how far the
teacher's confidence has actually been checked — because 01 §5 puts teacher label
error at 5–15% before verification, and a corpus that treats the teacher as truth
just trains the student on that error.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from . import bands, similarity

# Imported through the service package so the labelling path *is* the product path.
from ti.decide.backends.openrouter import (OpenRouterBackend, OpenRouterError,
                                           _message_text)
from ti.decide.primitives import ChoiceAnswer


class TeacherError(RuntimeError):
    """A call that produced no usable decision: transport, HTTP, or a bad schema.

    Distinct from a low-confidence answer: this one is a row that must not exist.
    """


@dataclass
class TeacherAnswer:
    """One teacher decision and everything needed to audit it later."""

    choice: str
    confidence: float
    raw: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    model_served: str = ""
    provider: str = ""
    reply_id: str = ""
    decode: dict[str, Any] = field(default_factory=dict)

    def target(self) -> dict[str, Any]:
        """The `target` block a row carries. `probabilities` is the point asserted.

        A chat completion carries no distribution over the candidate set, so this is
        one number for one key, not a spread invented to look like a softmax — the
        same reading ``OpenRouterBackend.choice`` documents.
        """
        return {"tool": self.choice, "conf": round(float(self.confidence), 6),
                "probabilities": {self.choice: round(float(self.confidence), 6)}}

    def teacher(self) -> dict[str, Any]:
        """Everything needed to re-read the label later, decode config included.

        03 §6 lists decode settings as a confound that must travel with a number:
        temperature, max_tokens, response format and reasoning effort move accuracy,
        so a row that does not record them cannot be compared with one that does.
        ``max_tokens`` is the shipped 64, and a reply truncated by it is refused
        rather than repaired (see `rejects.jsonl`).
        """
        return {"model": self.model, "model_served": self.model_served,
                "provider": self.provider, "raw": self.raw,
                "usage": self.usage, "reply_id": self.reply_id,
                "decode": self.decode, "schema_valid": True}


class FixtureTeacher:
    """Deterministic, offline, key-free: the whole pipeline runs without spending.

    Picks with the band vocabulary first (so a dry run exercises the real prompt
    shape and produces plausible per-band distributions), then by lexical overlap
    with the candidate descriptions, then by name order. Confidence is derived from
    a hash of the utterance, so two runs of the same corpus agree exactly.
    """

    name = "fixture"

    def choose(self, state: Any, instructions: str,
               criteria: dict[str, str]) -> TeacherAnswer:
        intent = state.get("intent") if isinstance(state, dict) else str(state)
        intent = str(intent or instructions)
        ranked = sorted(bands.cue_hits(intent),
                        key=lambda op: (-len(bands.cue_hits(intent)[op]), op))
        choice = next((op for op in ranked if op in criteria), "")
        if not choice:
            words = {w for w in similarity.normalize(intent).split() if len(w) > 3}
            for name in sorted(criteria):
                if any(w in name.lower() or w in criteria[name].lower() for w in words):
                    choice = name
                    break
        if not choice:
            choice = sorted(criteria)[0] if criteria else ""
        digest = hashlib.sha256(f"{intent}|{choice}".encode()).digest()
        conf = round(0.55 + (digest[0] / 255.0) * 0.40, 3)
        return TeacherAnswer(choice=choice, confidence=conf,
                             raw=json.dumps({"choice": choice, "confidence": conf}),
                             model="fixture", model_served="fixture",
                             provider="fixture", decode={"backend": "fixture"})


class HostedTeacher(OpenRouterBackend):
    """The service's openrouter backend, plus the two things labelling needs.

    Inherited unchanged: the config-driven model, the provider pin, ``temperature 0``,
    ``json_object``, the 429/5xx retry with Retry-After, the reply parsing (fenced,
    noisy or clean), the degradation to an empty choice, and the rule that the key
    lives in the environment and travels only in the ``Authorization`` header.
    """

    name = "openrouter+distill"

    def __init__(self) -> None:
        super().__init__()
        self.last_reply: dict[str, Any] = {}

    def _payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = super()._payload(messages)
        # OpenRouter returns the billed amount in the reply's usage object when the
        # request asks for it. Without this the run can only *estimate* its cost, and
        # the budget gate would be reasoning about a guess.
        body["usage"] = {"include": True}
        return body

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        out = super()._post(payload)
        self.last_reply = out
        return out

    def choose(self, state: Any, instructions: str,
               criteria: dict[str, str]) -> TeacherAnswer:
        try:
            answer: ChoiceAnswer = self.choice(state, instructions, criteria)
        except OpenRouterError as exc:
            raise TeacherError(str(exc)) from None
        reply = self.last_reply if isinstance(self.last_reply, dict) else {}
        usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
        if not answer.choice or answer.choice not in criteria:
            raise TeacherError(
                "schema-invalid reply: no candidate chosen "
                f"(raw: {_message_text(reply)[:200]!r})")
        if not 0.0 <= float(answer.confidence) <= 1.0:  # the backend clamps; assert it
            raise TeacherError(f"schema-invalid confidence {answer.confidence!r}")
        return TeacherAnswer(
            choice=answer.choice, confidence=float(answer.confidence),
            raw=_message_text(reply), usage=dict(usage), model=str(self.model),
            model_served=str(reply.get("model") or ""),
            provider=str(self.last_provider or ""),
            reply_id=str(reply.get("id") or ""),
            decode={"temperature": 0, "max_tokens": self.max_tokens,
                    "json_object": self.json_object, "reasoning": self.reasoning,
                    "provider_pin": self.provider})


def make_teacher(dry_run: bool) -> Any:
    """The fixture when asked for offline, the hosted teacher otherwise."""
    return FixtureTeacher() if dry_run else HostedTeacher()


def is_transient(exc: BaseException) -> bool:
    """Whether a failed row is worth retrying later (a 429 is not a row)."""
    text = str(exc).lower()
    return "429" in text or "rate limit" in text
