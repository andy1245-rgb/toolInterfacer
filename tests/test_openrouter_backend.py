"""openrouter backend tests — every case against a stub server on 127.0.0.1.

What is pinned here is the feature: the request body carries the configured model,
``temperature: 0`` and the ``provider`` object exactly as config spells it, so
switching the pin (or ``sort``/``only``/``ignore``) is a config edit rather than a
code change. The rest is the reply contract — a clean, fenced or noisy JSON reply
parses; a reply that names no candidate degrades to an empty answer instead of
inventing one; 429/5xx are retried a bounded number of times before raising.

No case touches the real API. The backend is pointed at an ephemeral port served
by http.server, and OPENROUTER_API_KEY is a fake value patched into the
environment; the model on 127.0.0.1:8080 and the service on :1113 are never used.
"""
import json
import os
import pathlib
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.conf import load  # noqa: E402
from ti.decide.backends.openrouter import (ENV_KEY, OpenRouterBackend,  # noqa: E402
                                           OpenRouterError)
from ti.decide.primitives import ChoiceAnswer, make_backend  # noqa: E402

TEST_KEY = "sk-or-v1-0000000000000000000000000000000000000000000000000000000000000000"
PIN = {"only": ["Relace"], "allow_fallbacks": True}
MODEL = "z-ai/glm-5.3-flash"
INSTRUCTION = "which file should the read step open next?"
STATE = {"intent_context": "read the file we discussed"}
CRITERIA = {
    "README.md": "the project overview",
    "docs/architecture.md": "the module map",
    "service/ti/decide/resolve.py": "how a backend is chosen",
    "service/ti/conf/defaults.json": "every tuning knob",
    "tests/test_openrouter_backend.py": "these tests",
    "bench/replay.py": "the offline runner",
    "spec/1.0-spec.md": "the contract",
    "git status --porcelain": "the working tree state",
}
ANSWER = {"choice": "docs/architecture.md", "confidence": 0.82}


def _chat(content: Any, *, provider: str | None = "Relace") -> dict[str, Any]:
    """A 200 chat completion whose assistant message carries `content`."""
    body: dict[str, Any] = {
        "id": "gen-test", "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
    }
    if provider is not None:
        body["provider"] = provider
    return {"status": 200, "body": body}


def _error(status: int, text: str = "", headers: dict[str, str] | None = None,
           delay: float = 0.0) -> dict[str, Any]:
    return {"status": status, "headers": headers or {}, "body": text.encode(),
            "delay": delay}


class _Handler(BaseHTTPRequestHandler):
    """One canned reply per POST, and a record of what the backend actually sent."""

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        stub = self.server.stub
        stub.requests.append({"path": self.path, "body": json.loads(raw or b"{}"),
                              "auth": self.headers.get("Authorization"),
                              "content_type": self.headers.get("Content-Type")})
        status, headers, body, delay = stub.take()
        if delay:
            time.sleep(delay)
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except OSError:
            pass  # the client timed out and closed first: expected in the timeout case

    def log_message(self, *args: Any) -> None:
        pass


class _Stub:
    """A local HTTP server replaying canned replies; the last one repeats."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.stub = self
        self._httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        # poll_interval drives how long shutdown() waits for the loop to notice; the
        # default 0.5 s would dominate the suite's wall clock across this many stubs.
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self) -> "_Stub":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    def take(self) -> tuple[int, dict[str, str], Any, float]:
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return (int(reply.get("status", 200)), reply.get("headers") or {},
                reply.get("body", b""), float(reply.get("delay", 0.0)))


class OpenRouterCase(unittest.TestCase):
    """Points decide.openrouter at a stub server and a fake key; restores after."""

    def setUp(self) -> None:
        self.decide = load()["decide"]
        self._saved = self.decide["openrouter"]
        self.addCleanup(self._restore)
        # The key variable is config-driven (decide.openrouter.key_env), so the suite
        # patches whichever name the shipped config points at rather than assuming one.
        env_name = str(self._saved.get("key_env") or ENV_KEY)
        patcher = mock.patch.dict(os.environ, {ENV_KEY: TEST_KEY, env_name: TEST_KEY})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _restore(self) -> None:
        self.decide["openrouter"] = self._saved

    def backend(self, stub: _Stub, **overrides: Any) -> OpenRouterBackend:
        self.decide["openrouter"] = dict(self._saved, base_url=stub.url, **overrides)
        backend = OpenRouterBackend()
        backend.backoff_s = 0.0  # retries must not spend wall-clock in the suite
        return backend


class RequestShapeTest(OpenRouterCase):
    """The pin and the determinism knobs travel with every decision."""

    def test_request_body_carries_the_model_temperature_and_the_pin(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            answer = self.backend(stub).choice(STATE, INSTRUCTION, CRITERIA)
        sent = stub.requests[0]
        self.assertEqual(sent["path"], "/chat/completions")
        self.assertEqual(sent["body"]["model"], MODEL)
        self.assertEqual(sent["body"]["temperature"], 0)
        self.assertEqual(sent["body"]["provider"], PIN)
        self.assertEqual(sent["body"]["max_tokens"], 64)
        self.assertEqual(sent["body"]["response_format"], {"type": "json_object"})
        self.assertEqual(answer, ChoiceAnswer("docs/architecture.md",
                                              {"docs/architecture.md": 0.82}, 0.82))

    def test_provider_pin_is_config_not_code(self):
        pin = {"sort": "price", "data_collection": "deny", "allow_fallbacks": False}
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            self.backend(stub, provider=pin).choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(stub.requests[0]["body"]["provider"], pin)

    def test_provider_must_be_a_json_object(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            with self.assertRaises(ValueError):
                self.backend(stub, provider=["Relace"])

    def test_prompt_presents_every_candidate_the_instruction_and_the_json_demand(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            self.backend(stub).choice(STATE, INSTRUCTION, CRITERIA)
        messages = stub.requests[0]["body"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        text = " ".join(m["content"] for m in messages)
        self.assertIn(INSTRUCTION, text)
        self.assertIn(STATE["intent_context"], text)
        for key, description in CRITERIA.items():
            self.assertIn(key, text)
            self.assertIn(description, text)
        self.assertIn('"choice"', text)
        self.assertIn('"confidence"', text)

    def test_authorization_header_comes_from_the_patched_env_var(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            self.backend(stub).choice(STATE, INSTRUCTION, CRITERIA)
        sent = stub.requests[0]
        self.assertEqual(sent["auth"], f"Bearer {TEST_KEY}")
        self.assertEqual(sent["content_type"], "application/json")
        self.assertNotIn(TEST_KEY, json.dumps(sent["body"]))

    def test_json_object_off_omits_response_format_and_still_parses(self):
        fenced = "```json\n" + json.dumps(ANSWER) + "\n```"
        with _Stub(_chat(fenced)) as stub:
            answer = self.backend(stub, json_object=False).choice(
                STATE, INSTRUCTION, CRITERIA)
        self.assertNotIn("response_format", stub.requests[0]["body"])
        self.assertEqual(answer.choice, "docs/architecture.md")


class ReplyParsingTest(OpenRouterCase):
    """Provider replies differ in packaging; the answer must not."""

    def _answer(self, content: Any) -> ChoiceAnswer:
        with _Stub(_chat(content)) as stub:
            return self.backend(stub).choice(STATE, INSTRUCTION, CRITERIA)

    def test_clean_json_reply_parses(self):
        self.assertEqual(self._answer(json.dumps(ANSWER)),
                         ChoiceAnswer("docs/architecture.md", {"docs/architecture.md": 0.82},
                                      0.82))

    def test_fenced_reply_parses(self):
        content = "```json\n" + json.dumps(ANSWER) + "\n```"
        self.assertEqual(self._answer(content).choice, "docs/architecture.md")

    def test_noisy_reply_parses(self):
        content = ("I weighed {README.md, bench/replay.py} for a moment and then answered:\n"
                   '{"choice": "docs/architecture.md", "confidence": "0.82"}\n'
                   "Say the word if you want another pass.")
        answer = self._answer(content)
        self.assertEqual(answer, ChoiceAnswer("docs/architecture.md",
                                              {"docs/architecture.md": 0.82}, 0.82))

    def test_invalid_choice_degrades_to_an_empty_answer(self):
        answer = self._answer('{"choice": "src/main.py", "confidence": 0.99}')
        self.assertEqual(answer, ChoiceAnswer("", {}, 0.0))

    def test_non_string_choice_degrades_to_an_empty_answer(self):
        self.assertEqual(self._answer('{"choice": 3, "confidence": 0.9}'),
                         ChoiceAnswer("", {}, 0.0))

    def test_reply_without_json_degrades_to_an_empty_answer(self):
        self.assertEqual(self._answer("I would rather not decide."),
                         ChoiceAnswer("", {}, 0.0))

    def test_reply_with_no_choices_degrades_to_an_empty_answer(self):
        with _Stub({"status": 200, "body": {"id": "gen-test", "choices": []}}) as stub:
            answer = self.backend(stub).choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(answer, ChoiceAnswer("", {}, 0.0))

    def test_non_json_body_degrades_to_an_empty_answer(self):
        with _Stub({"status": 200, "body": b"<html>502 bad gateway</html>"}) as stub:
            answer = self.backend(stub).choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(answer, ChoiceAnswer("", {}, 0.0))

    def test_confidence_missing_keeps_the_choice(self):
        self.assertEqual(self._answer('{"choice": "README.md"}'),
                         ChoiceAnswer("README.md", {"README.md": 0.0}, 0.0))

    def test_confidence_is_clamped(self):
        for raw, expected in ((1.7, 1.0), (-0.3, 0.0), ("0.6", 0.6), (None, 0.0)):
            with self.subTest(confidence=raw):
                content = json.dumps({"choice": "README.md", "confidence": raw})
                answer = self._answer(content)
                self.assertEqual(answer.choice, "README.md")
                self.assertEqual(answer.confidence, expected)


class RetryPolicyTest(OpenRouterCase):
    """429 and 5xx are retried a bounded number of times; nothing else is."""

    def test_429_then_200_is_retried(self):
        replies = [_error(429, "slow down", {"Retry-After": "0"}),
                   _chat(json.dumps(ANSWER))]
        with _Stub(*replies) as stub:
            answer = self.backend(stub).choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(len(stub.requests), 2)
        self.assertEqual(answer.choice, "docs/architecture.md")

    def test_retry_after_header_is_honoured(self):
        replies = [_error(429, "slow down", {"Retry-After": "2"}),
                   _chat(json.dumps(ANSWER))]
        with _Stub(*replies) as stub:
            backend = self.backend(stub)
            with mock.patch("ti.decide.backends.openrouter.time.sleep") as sleep:
                backend.choice(STATE, INSTRUCTION, CRITERIA)
        sleep.assert_called_once_with(2.0)
        self.assertEqual(len(stub.requests), 2)

    def test_persistent_500_raises_after_bounded_retries(self):
        with _Stub(_error(500, "upstream exploded")) as stub:
            backend = self.backend(stub)
            with self.assertRaises(OpenRouterError) as caught:
                backend.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(len(stub.requests), backend.max_attempts)
        self.assertIn("500", str(caught.exception))

    def test_client_error_is_not_retried(self):
        with _Stub(_error(401, "no auth")) as stub:
            with self.assertRaises(OpenRouterError) as caught:
                self.backend(stub).choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(len(stub.requests), 1)
        self.assertIn("401", str(caught.exception))

    def test_timeout_raises(self):
        with _Stub(_chat(json.dumps(ANSWER)) | {"delay": 0.6}) as stub:
            backend = self.backend(stub, timeout_s=0.15)
            with self.assertRaises(OpenRouterError):
                backend.choice(STATE, INSTRUCTION, CRITERIA)


class NoulTest(OpenRouterCase):
    """noul answers a probability on the same endpoint, and clamps it."""

    def _noul(self, content: Any) -> tuple[float, dict[str, Any]]:
        with _Stub(_chat(content)) as stub:
            value = self.backend(stub).noul(STATE, "is this page relevant?")
        return value, stub.requests[0]["body"]

    def test_noul_parses_and_clamps(self):
        for raw, expected in ((0.42, 0.42), (1.5, 1.0), (-0.2, 0.0), ("0.7", 0.7)):
            with self.subTest(noul=raw):
                value, _ = self._noul(json.dumps({"noul": raw}))
                self.assertEqual(value, expected)

    def test_noul_uses_the_same_pin_and_determinism(self):
        _, body = self._noul(json.dumps({"noul": 0.42}))
        self.assertEqual(body["provider"], PIN)
        self.assertEqual(body["temperature"], 0)
        text = " ".join(m["content"] for m in body["messages"])
        self.assertIn("is this page relevant?", text)
        self.assertIn('"noul"', text)

    def test_noul_reply_without_json_reads_as_zero(self):
        value, _ = self._noul("no idea, sorry")
        self.assertEqual(value, 0.0)


class CredentialsTest(OpenRouterCase):
    """The key is environment state: never config, never an attribute, never a message."""

    def test_missing_key_raises_before_any_request(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            backend = self.backend(stub)
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(OpenRouterError) as caught:
                    backend.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertIn(ENV_KEY, str(caught.exception))
        self.assertEqual(stub.requests, [])

    def test_key_never_appears_in_repr_str_or_an_exception(self):
        with _Stub(_error(500, f"upstream echoed: Bearer {TEST_KEY}")) as stub:
            backend = self.backend(stub)
            with self.assertRaises(OpenRouterError) as caught:
                backend.choice(STATE, INSTRUCTION, CRITERIA)
        message = str(caught.exception)
        self.assertNotIn(TEST_KEY, message)
        self.assertNotIn(TEST_KEY, repr(caught.exception))
        self.assertIn("[redacted]", message)
        self.assertNotIn(TEST_KEY, repr(backend))
        self.assertNotIn(TEST_KEY, str(backend))

    def test_key_never_appears_in_the_answer_object(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            answer = self.backend(stub).choice(STATE, INSTRUCTION, CRITERIA)
        self.assertNotIn(TEST_KEY, repr(answer))
        self.assertNotIn(TEST_KEY, str(answer))


class ProviderObservabilityTest(OpenRouterCase):
    """allow_fallbacks can route around the pin; the reply says who answered."""

    def test_served_provider_is_captured_and_logged(self):
        with _Stub(_chat(json.dumps(ANSWER), provider="Relace")) as stub:
            backend = self.backend(stub)
            with self.assertLogs("ti.decide.backends.openrouter", level="DEBUG") as logs:
                backend.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(backend.last_provider, "Relace")
        self.assertTrue(any("Relace" in line for line in logs.output))

    def test_a_fallback_to_another_upstream_is_visible(self):
        with _Stub(_chat(json.dumps(ANSWER), provider="SomeOtherUpstream")) as stub:
            backend = self.backend(stub)
            backend.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(backend.last_provider, "SomeOtherUpstream")

    def test_reply_without_a_provider_field_leaves_it_empty(self):
        with _Stub(_chat(json.dumps(ANSWER), provider=None)) as stub:
            backend = self.backend(stub)
            backend.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(backend.last_provider, "")


class RegistrationTest(OpenRouterCase):
    """Selectable by name, and not the shipped default route."""

    def test_backend_is_registered_and_reads_its_config_block(self):
        backend = make_backend("openrouter")
        self.assertEqual(backend.name, "openrouter")
        self.assertEqual(backend.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(backend.model, MODEL)
        self.assertEqual(backend.provider, PIN)

    def test_default_routing_still_names_d1_systemone(self):
        self.assertEqual(self.decide["backends_in_use"]["choice"], "d1_systemone")
        self.assertEqual(self.decide["backends_in_use"]["noul"], "fixture")


if __name__ == "__main__":
    unittest.main()


class PerRequestScopeTest(OpenRouterCase):
    """These settings live in one request body — they must not reach anything else.

    The operator shares one OpenRouter account with other tools, so the pin, the
    reasoning effort and the response format have to be per-call fields and the key
    has to come from a variable this backend alone names. Anything global here would
    silently reshape unrelated traffic.
    """

    def test_reasoning_and_user_travel_in_the_body(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            self.backend(stub, reasoning={"effort": "minimal"},
                         user="toolinterfacer").choice(STATE, INSTRUCTION, CRITERIA)
        body = stub.requests[0]["body"]
        self.assertEqual(body["reasoning"], {"effort": "minimal"})
        self.assertEqual(body["user"], "toolinterfacer")

    def test_reasoning_must_be_a_json_object(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            with self.assertRaises(ValueError):
                self.backend(stub, reasoning="minimal")

    def test_absent_reasoning_and_user_send_no_such_field(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            self.backend(stub, reasoning=None, user=None).choice(
                STATE, INSTRUCTION, CRITERIA)
        body = stub.requests[0]["body"]
        self.assertNotIn("reasoning", body)
        self.assertNotIn("user", body)

    def test_key_env_selects_its_own_variable(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            backend = self.backend(stub, key_env="TI_OPENROUTER_API_KEY")
            with mock.patch.dict(os.environ, {"TI_OPENROUTER_API_KEY": TEST_KEY},
                                 clear=True):
                backend.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(stub.requests[0]["auth"], f"Bearer {TEST_KEY}")

    def test_a_missing_key_names_the_configured_variable(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            backend = self.backend(stub, key_env="TI_OPENROUTER_API_KEY")
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(OpenRouterError) as caught:
                    backend.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertIn("TI_OPENROUTER_API_KEY", str(caught.exception))

    def test_constructing_a_backend_mutates_no_environment(self):
        before = dict(os.environ)
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            self.backend(stub)
        self.assertEqual(dict(os.environ), before)

    def test_two_backends_do_not_share_their_pin(self):
        with _Stub(_chat(json.dumps(ANSWER))) as stub:
            self.backend(stub).choice(STATE, INSTRUCTION, CRITERIA)
            self.backend(stub, provider={"only": ["GMICloud"]}).choice(
                STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(stub.requests[0]["body"]["provider"], dict(PIN))
        self.assertEqual(stub.requests[1]["body"]["provider"],
                         {"only": ["GMICloud"]})
