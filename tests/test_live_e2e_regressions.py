"""Regressions for the breaks the live probe found in the shipped path.

Each case here pins one defect that `bench/live_e2e.py` surfaced against a real
service and a real d1-3B, recorded with its own name:

  * the L1 slot-fill read a picked relative name against the *service's* cwd
    instead of the caller's, so every session whose cwd was not the service's got
    `[Errno 2] No such file or directory: 'notes.md'`;
  * the L2 research plan named two ops that are not registered, so every
    "research …" intent died at step one with `unknown op '__decide_extract__'`;
  * the research extract step never saw the intent, so its queries were built
    from its own instruction text;
  * a gate block over HTTP degraded to a guessed result instead of returning
    `need_input`, and `params.answer` was never consumed, so the ask-back loop
    could not close;
  * the question offered the rejected pick as its only option (and a URL
    candidate's option id was a 400-character dict repr);
  * an empty candidate list blamed the decide backend ("pair: no stage produced
    a choice") instead of naming the real cause.

Model-free throughout: a counting/fixed backend is pinned in the registry and the
decision cache is cleared, the way test_loop and test_conf_degraded do it.
"""
import json
import pathlib
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

import ti.decide.primitives as prim  # noqa: E402
from ti.conf import load  # noqa: E402
from ti.core.artifacts import ArtifactStore  # noqa: E402
from ti.core.gates import GateBlocked  # noqa: E402
from ti.core.runner import Step, _handle_block  # noqa: E402
from ti.decide.cache import clear as clear_decisions  # noqa: E402
from ti.decide.primitives import ChoiceAnswer, DecideBackend  # noqa: E402
from ti.ops.op_api import get as get_op, import_all  # noqa: E402
from ti.ops.decisions import run_extract  # noqa: E402
from ti.service_entry import use_tool  # noqa: E402

REPORTS = ("report-draft.md", "report-final.md", "report-old.md",
           "report-v1.md", "report-v2.md")


class _Pick(DecideBackend):
    """Picks `choice` among the candidates at `conf` — the live model's stand-in."""

    name = "live_e2e_pick"

    def __init__(self, choice: str, conf: float) -> None:
        self._choice = choice
        self._conf = conf

    def choice(self, state, instructions, criteria):
        choice = self._choice if self._choice in criteria else next(iter(criteria), self._choice)
        probs = dict.fromkeys(criteria, self._conf) or {choice: self._conf}
        return ChoiceAnswer(choice, probs, self._conf)

    def noul(self, state, question):
        return 0.9


class PinTestCase(unittest.TestCase):
    """Pins decide.backends_in_use.choice at a registered fake; clears the cache."""

    def pin(self, backend: DecideBackend) -> None:
        import_all()
        prim._REGISTRY[backend.name] = lambda: backend
        backends = load()["decide"]["backends_in_use"]
        previous = backends["choice"]
        backends["choice"] = backend.name
        clear_decisions()

        def restore() -> None:
            backends["choice"] = previous
            prim._REGISTRY.pop(backend.name, None)
            clear_decisions()

        self.addCleanup(restore)

    def report_dir(self) -> pathlib.Path:
        path = pathlib.Path(tempfile.mkdtemp(prefix="ti-regress-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(path, ignore_errors=True))
        for name in REPORTS:
            (path / name).write_text(f"# {name}\n\nbody of {name}\n", encoding="utf-8")
        return path


class L1ReadFrameTest(PinTestCase):
    def test_a_picked_relative_path_is_read_in_the_callers_frame(self):
        # The service's own cwd is this repo; the digest cwd is a temp directory.
        # The picked name is relative to the *digest* cwd, so reading it against
        # the process cwd raised FileNotFoundError on every live run.
        self.pin(_Pick("notes.md", 0.95))
        with tempfile.TemporaryDirectory() as tmp:
            notes = pathlib.Path(tmp, "notes.md")
            notes.write_text("live-e2e-notes-body\n", encoding="utf-8")
            pathlib.Path(tmp, "budget.csv").write_text("item,qty\nbolts,12\n", encoding="utf-8")
            out = use_tool({"intent": "read the notes file", "digest": {"cwd": tmp}})
        self.assertEqual(out["status"], "done", out)
        self.assertEqual(out["result"], "live-e2e-notes-body\n")
        self.assertEqual([h["op"] for h in out["hops"]],
                         ["filesys_glob", "decide_pick", "filesys_read"])

    def test_an_absolute_path_is_left_alone(self):
        self.pin(_Pick("notes.md", 0.95))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp, "abs.md")
            path.write_text("absolute\n", encoding="utf-8")
            kind, data, _conf = get_op("filesys_read").run(
                {"path": str(path), "cwd": "/nonexistent-should-be-ignored"}, None)
        self.assertEqual((kind, data), ("file_content", "absolute\n"))


class ResearchPlanTest(PinTestCase):
    def test_every_step_of_the_research_plan_is_a_registered_op(self):
        out = use_tool({"intent": "research the current stable python version",
                        "mode": "plan"})
        self.assertEqual(out["status"], "plan", out)
        ops = [step["op"] for step in out["steps"]]
        # spec §7's graph: extract queries, search, pick, fetch, assemble a digest.
        self.assertEqual(ops, ["decide_extract", "web_search", "decide_pick",
                               "web_fetch", "digest_assemble"])
        for op in ops:
            get_op(op)  # KeyError here is the defect: an unregistered name in a template

    def test_the_extract_step_is_told_the_intent(self):
        # The research template passes instructions only; the runner merges the
        # intent in as `intent_context`, which the op must declare and use.
        kind, values, conf = run_extract(
            {"instructions": "Extract 3-6 search queries for the intent.",
             "intent_context": "kafka consumer lag diagnosis"}, None)
        self.assertEqual(kind, "value_list")
        self.assertTrue(values, "no queries at all — the intent never reached the extractor")
        self.assertIn("kafka", " ".join(values).lower())
        # `rule_extract` measures no distribution, so it reports no confidence —
        # see docs/confidence-sources.md. Absence is the correct answer here, and
        # the intent actually reaching the extractor is what this test is about.
        self.assertIsNone(conf)


class AskBackChannelTest(PinTestCase):
    def _body(self, cwd, **params):
        body = {"intent": "read the report file", "digest": {"cwd": str(cwd)}}
        if params:
            body["params"] = params
        return body

    def test_a_gate_block_over_http_returns_need_input_with_the_candidates(self):
        self.pin(_Pick("report-final.md", 0.26))
        cwd = self.report_dir()
        out = use_tool(self._body(cwd), ask_via_response=True)
        self.assertEqual(out["status"], "need_input", out)
        question = out["question"]
        self.assertEqual(question["kind"], "choice")
        self.assertEqual(sorted(o["id"] for o in question["options"]), sorted(REPORTS))
        self.assertIn("askback", [h["op"] for h in out["hops"]])
        # The best guess still rides along: a degraded result has information value.
        self.assertTrue(out["degraded"])
        self.assertEqual(out["result"], "# report-final.md\n\nbody of report-final.md\n")

    def test_without_a_channel_the_run_degrades_as_before(self):
        # Callers that pass neither on_ask nor ask_via_response keep P1 behaviour.
        self.pin(_Pick("report-final.md", 0.26))
        out = use_tool(self._body(self.report_dir()))
        self.assertEqual(out["status"], "done", out)
        self.assertTrue(out["degraded"])
        self.assertNotIn("question", out)

    def test_the_answer_resolves_the_gate_the_question_named(self):
        self.pin(_Pick("report-final.md", 0.26))
        cwd = self.report_dir()
        out = use_tool(self._body(cwd, answer="report-v1.md"), ask_via_response=True)
        self.assertEqual(out["status"], "done", out)
        self.assertNotIn("question", out)
        self.assertFalse(out["degraded"], out)
        self.assertEqual(out["result"], "# report-v1.md\n\nbody of report-v1.md\n")
        resolved = [h for h in out["hops"] if h["op"] == "askback.resolved"]
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["resolves"], "picked_file")
        self.assertEqual(out["confidence"], resolved[0]["conf"])

    def test_the_options_are_the_enumeration_not_the_rejected_pick(self):
        # A blocked pick used to offer picked_items (one item: its own answer) as
        # the whole option list, which no caller can answer.
        store = ArtifactStore()
        store.put("candidates", "candidates", "path_list", list(REPORTS), 1.0)
        store.put("picked_file", "picked_file", "picked_items", ["report-final.md"], 0.26)
        asked: list = []
        blocked = GateBlocked("content", "artifact 'picked_file' conf 0.26 below gate",
                              {"artifact": "picked_file", "kind": "conf_gte", "min": 0.7})
        retry = _handle_block(blocked, Step(id="content", op="filesys_read"),
                              store, [], [], lambda question: asked.append(question) or {"choice": REPORTS[0]})
        self.assertTrue(retry)
        self.assertEqual([o["id"] for o in asked[0].options], list(REPORTS))

    def test_a_url_candidate_offers_its_url_as_the_option_id(self):
        store = ArtifactStore()
        store.put("hits", "hits", "url_list", [
            {"title": "quantization guide", "url": "https://example.test/q", "snippet": "s"}])
        asked: list = []
        blocked = GateBlocked("pages", "artifact 'picked' conf 0.25 below gate",
                              {"artifact": "picked", "kind": "conf_gte", "min": 0.7})
        _handle_block(blocked, Step(id="pages", op="web_fetch"), store, [], [],
                      lambda question: asked.append(question) or {"choice": None})
        self.assertEqual(asked[0].options, [{"id": "https://example.test/q", "desc": ""}])


class EmptyCandidateTest(PinTestCase):
    def test_an_empty_candidate_list_names_its_cause(self):
        self.pin(_Pick("notes.md", 0.95))
        with tempfile.TemporaryDirectory() as tmp:
            out = use_tool({"intent": "read the notes file", "digest": {"cwd": tmp}})
        self.assertEqual(out["status"], "error", out)
        # The invariant is that the failure names the step that really failed. An
        # empty directory is not a decision the backend got wrong: asking it anyway
        # made d1 abstain and the pair surface "pair: no stage produced a choice",
        # which points at the model for what is an empty `filesys_glob`.
        self.assertIn("decide_pick", out["error"])
        self.assertNotIn("no stage produced a choice", out["error"])
        self.assertNotIn("pair:", out["error"])


class HttpRouteTest(PinTestCase):
    """/v1/useTool must be the transport that carries the ask-back (not a guess)."""

    def setUp(self):
        from ti.http.api import Handler
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(self._stop)

    def _stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as resp:
            return json.loads(resp.read())

    def _post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def test_health_names_the_decide_backends(self):
        health = self._get("/health")
        self.assertEqual(health["status"], "ok")
        self.assertIn("filesys_read", health["ops"])
        self.assertIn("choice", health["decide"])
        self.assertIn("extract", health["decide"])
        self.assertIsInstance(health["cascade"], bool)

    def test_the_route_carries_need_input_and_then_the_answer(self):
        self.pin(_Pick("report-final.md", 0.26))
        cwd = self.report_dir()
        first = self._post("/v1/useTool", {"intent": "read the report file",
                                           "digest": {"cwd": str(cwd)}})
        self.assertEqual(first["status"], "need_input", first)
        question = first["question"]
        self.assertEqual(question["kind"], "choice")
        reply = self._post("/v1/useTool", {"intent": "read the report file",
                                           "params": {"answer": question["options"][3]["id"]},
                                           "digest": {"cwd": str(cwd)}})
        self.assertEqual(reply["status"], "done", reply)
        self.assertIn(question["options"][3]["id"], reply["result"])

    def test_plan_route_does_not_run_anything(self):
        out = self._post("/v1/plan", {"intent": "read the report file",
                                      "digest": {"cwd": str(self.report_dir())}})
        self.assertEqual(out["status"], "plan")
        self.assertEqual(out["hops"], [])
        self.assertEqual([s["op"] for s in out["steps"]],
                         ["filesys_glob", "decide_pick", "filesys_read"])


if __name__ == "__main__":
    unittest.main()
