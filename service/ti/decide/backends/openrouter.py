"""openrouter — a hosted decision model behind an explicit provider pin.

One POST per decision to ``{base_url}/chat/completions``, temperature 0, a
JSON-only reply. The wire format is OpenAI's, so this is stdlib urllib like every
other backend here: no SDK, no new dependency.

The provider pin is the point of the file. ``decide.openrouter.provider`` is
copied into the request body verbatim — the shipped config pins
``{"order": ["Relace"], "allow_fallbacks": true}`` — so which upstream serves a
decision is a config edit (``order``, ``only``, ``ignore``, ``sort``,
``data_collection``), never a code change. OpenRouter owns that schema; nothing
here validates it beyond it being a JSON object.

Because ``allow_fallbacks`` can route around the pin, the reply's ``provider``
field — which upstream actually answered — is captured on ``last_provider`` and
logged at DEBUG. A silent fallback otherwise changes both the price and the
quantization behind an answer, and the numbers a run was calibrated on would no
longer be the numbers it got.

Degradation, and it is deliberate: a reply that does not parse, or one whose
``choice`` is not one of the candidate keys, returns ``ChoiceAnswer("", {}, 0.0)``.
The slot gate blocks on an empty choice, so the run visibly asks back or falls
back to a best guess. Inventing a choice would be worse — it would look like a
valid decision and reach the executor. Transport and HTTP failures are the
opposite case (nothing was answered at all): 429 and 5xx are retried a bounded
number of times, then raised, so the runner records a structured error status
rather than a decision.

``noul`` reads an unparseable reply as 0.0: that is the "no evidence" value d1
returns when the field is absent, and it fails verification rather than passing
it. The key comes from ``OPENROUTER_API_KEY`` only, read per call, and travels in
the ``Authorization`` header only — it is never a config value, an attribute, or
part of a message this module builds.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from ...conf import load
from ..primitives import ChoiceAnswer, DecideBackend, register

log = logging.getLogger(__name__)

ENV_KEY = "OPENROUTER_API_KEY"
ENDPOINT = "/chat/completions"

# Retry policy. 429 and 5xx are the only retryable classes; anything else that
# stops an answer from arriving is raised on the spot (see _post).
MAX_ATTEMPTS = 3
BACKOFF_S = 0.5
RETRY_AFTER_CAP_S = 30.0

SYSTEM = ("You are a decision function. You read an instruction and a state, then "
          "answer with exactly one JSON object and nothing else.")


class OpenRouterError(RuntimeError):
    """A call that produced no decision: missing key, transport, or HTTP failure."""


@register("openrouter")
class OpenRouterBackend(DecideBackend):
    """Hosted `choice`/`noul` over OpenRouter, with the provider pinned by config.

    ``last_provider`` is the upstream OpenRouter reports for the most recent call
    ("" when the reply carried no provider field). It lives on the instance rather
    than in ChoiceAnswer because that dataclass is the primitive every backend
    shares and the other backends have no such field to fill.
    """

    name = "openrouter"

    def __init__(self) -> None:
        cfg = load()["decide"]["openrouter"]
        self.base_url = str(cfg["base_url"])
        self.model = str(cfg["model"])
        provider = cfg.get("provider")
        if provider is None:
            provider = {}
        if not isinstance(provider, dict):
            raise ValueError(
                "openrouter: decide.openrouter.provider must be a JSON object, got "
                f"{type(provider).__name__} — OpenRouter owns the schema, this backend "
                "copies it into the request body unchanged")
        self.provider = provider
        self.timeout_s = float(cfg.get("timeout_s", 30))
        self.max_tokens = int(cfg.get("max_tokens", 64))
        self.json_object = bool(cfg.get("json_object", True))
        self.max_attempts = MAX_ATTEMPTS
        self.backoff_s = BACKOFF_S
        self.retry_after_cap_s = RETRY_AFTER_CAP_S
        self.last_provider = ""

    def __repr__(self) -> str:
        """Config the backend runs on. The key is not an attribute, so it cannot appear."""
        return (f"OpenRouterBackend(model={self.model!r}, base_url={self.base_url!r}, "
                f"provider={self.provider!r})")

    def choice(self, state: Any, instructions: str,
               criteria: dict[str, str]) -> ChoiceAnswer:
        out = self._post(self._payload(_choice_messages(
            instructions, criteria, _state_text(state))))
        self._record_provider(out)
        obj = _first_json_object(_unfence(_message_text(out)))
        choice = obj.get("choice") if obj else None
        if not isinstance(choice, str) or choice not in criteria:
            # Absent, unparseable, or a key the caller never offered: all three are
            # "no decision" and all three must look that way to the slot gate.
            log.debug("openrouter: no usable choice in the reply (%r, %d candidates)",
                      choice, len(criteria))
            return ChoiceAnswer("", {}, 0.0)
        confidence = _clamp01(_number(obj.get("confidence"), 0.0))
        # A chat completion carries no distribution over the candidate set, so
        # `probabilities` reports the single point the model asserted rather than a
        # spread invented here.
        return ChoiceAnswer(choice, {choice: confidence}, confidence)

    def noul(self, state: Any, question: str) -> float:
        out = self._post(self._payload(_noul_messages(question, _state_text(state))))
        self._record_provider(out)
        obj = _first_json_object(_unfence(_message_text(out)))
        if obj is None:
            log.debug("openrouter: no JSON object in the noul reply; reading it as 0.0")
            return 0.0
        return _clamp01(_number(obj.get("noul"), 0.0))

    # -- wire ------------------------------------------------------------------

    def _payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,           # a decision wall must answer the same twice
            "max_tokens": self.max_tokens,
            "provider": self.provider,  # the pin, verbatim: config owns the schema
        }
        if self.json_object:
            body["response_format"] = {"type": "json_object"}
        return body

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one payload; {} when the reply body was not a JSON object."""
        key = _api_key()  # read per call: the key is environment state, not config
        url = self.base_url.rstrip("/") + ENDPOINT
        body = json.dumps(payload).encode()
        for attempt in range(1, self.max_attempts + 1):
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
            except urllib.error.HTTPError as err:
                detail = _redact(_error_body(err), key)
                if err.code != 429 and not 500 <= err.code < 600:
                    raise OpenRouterError(
                        f"openrouter: HTTP {err.code} from {url}: {detail}") from None
                if attempt >= self.max_attempts:
                    raise OpenRouterError(
                        f"openrouter: HTTP {err.code} after {attempt} attempts: "
                        f"{detail}") from None
                self._wait(attempt, err.headers.get("Retry-After") if err.headers else None)
                continue
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                # Nothing came back: a second attempt would spend the run's clock
                # again on a call the transport already abandoned. Raise, so the
                # runner records a structured error status instead of a decision.
                reason = _redact(str(getattr(err, "reason", err)), key)
                raise OpenRouterError(f"openrouter: {url} failed: {reason}") from None
            try:
                out = json.loads(raw)
            except ValueError:
                log.debug("openrouter: reply body is not JSON (%d bytes)", len(raw))
                return {}
            return out if isinstance(out, dict) else {}
        raise OpenRouterError(f"openrouter: gave up after {self.max_attempts} attempts")

    def _wait(self, attempt: int, retry_after: str | None) -> None:
        """Exponential backoff between retries, with Retry-After taking precedence.

        The header is capped so a mistaken (or hostile) value cannot hold a run
        past the budget the run itself owns, and an HTTP-date form falls back to
        the doubling.
        """
        delay = self.backoff_s * (2 ** (attempt - 1))
        if retry_after:
            try:
                delay = min(float(retry_after), self.retry_after_cap_s)
            except ValueError:
                pass
        if delay > 0:
            time.sleep(delay)

    def _record_provider(self, out: dict[str, Any]) -> None:
        """Which upstream answered the call; with allow_fallbacks the pin can be bypassed."""
        provider = out.get("provider")
        self.last_provider = provider if isinstance(provider, str) else ""
        if self.last_provider:
            log.debug("openrouter: %s served the request (pin: provider=%s)",
                      self.last_provider, self.provider)
        else:
            log.debug("openrouter: reply carried no provider field; upstream unknown")


def _api_key() -> str:
    """The key, read at call time so a test can set it after import."""
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        raise OpenRouterError(
            f"openrouter: {ENV_KEY} is not set — export it; it is read from the "
            "environment only and never stored in config")
    return key


def _redact(text: str, key: str) -> str:
    """Keep the key out of every message this module builds."""
    return text.replace(key, "[redacted]") if key else text


def _state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, default=str)


def _choice_messages(instructions: str, criteria: dict[str, str],
                     state_text: str) -> list[dict[str, str]]:
    candidates = "\n".join(f'- "{name}": {desc or "(no description)"}'
                           for name, desc in criteria.items())
    keys = ", ".join(f'"{name}"' for name in criteria)
    user = (f"[Instruction]\n{instructions}\n\n"
            f"[State]\n{state_text}\n\n"
            f"[Candidates]\n{candidates}\n\n"
            "[Reply]\nReply with exactly one JSON object and nothing else: no prose, "
            "no markdown, no code fence.\n"
            '{"choice": "<one candidate key, copied verbatim>", '
            '"confidence": <number from 0.0 to 1.0>}\n'
            f'"choice" must be exactly one of: {keys}.')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _noul_messages(question: str, state_text: str) -> list[dict[str, str]]:
    user = (f"[Question]\n{question}\n\n"
            f"[State]\n{state_text}\n\n"
            "[Reply]\nReply with exactly one JSON object and nothing else: no prose, "
            "no markdown, no code fence.\n"
            '{"noul": <probability from 0.0 to 1.0>}\n'
            "1.0 means yes / verified, 0.0 means no / not verified.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _message_text(out: dict[str, Any]) -> str:
    """The first choice's message content; some providers send it as content parts."""
    choices = out.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part["text"] for part in content
                       if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


_FENCE = re.compile(r"```[A-Za-z0-9_+-]*\s*(.*?)```", re.DOTALL)


def _unfence(text: str) -> str:
    """A fenced reply's body; the text unchanged when there is no fence."""
    match = _FENCE.search(text)
    return match.group(1) if match else text


def _first_json_object(text: str) -> dict[str, Any] | None:
    """The first balanced JSON object in `text`, ignoring prose around it."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    """0..1, with NaN read as 0.0 — a non-answer, never a pass."""
    if value != value:
        return 0.0
    return min(1.0, max(0.0, value))


def _error_body(err: urllib.error.HTTPError) -> str:
    """A short slice of the provider's error text for the raised message."""
    try:
        return err.read()[:300].decode("utf-8", "replace")
    except (OSError, ValueError, AttributeError):
        return ""
