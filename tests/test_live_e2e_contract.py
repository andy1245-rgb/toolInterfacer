"""The live probe's contract assertions, exercised with no model and no socket.

``bench/live_e2e.py`` posts real calls to a real service; the parts that decide
PASS from FAIL are pure — spec §3 field checks, the hop-vs-registry check, and
each case's own expectation. This module drives them through a stub transport,
which is what lets CI hold the contract on a box with no llama-server: a body
whose hops lie about what ran, whose read has no bytes behind it, or whose
``need_input`` carries no answerable question must FAIL here.

The stub bodies are the shapes the shipped service returns for the probe's five
cases (docs/live-e2e-2026-11-03.md records the captured live ones); the files
they point at are the ones on disk, so "backed by a real file" stays a real check
offline.
"""
import importlib.util
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
sys.path.insert(0, str(BENCH))


def _load_probe():
    spec = importlib.util.spec_from_file_location("live_e2e_probe", BENCH / "live_e2e.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_e2e_probe"] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()

# What GET /health reports for the shipped op registry.
OPS = ["decide_extract", "decide_pick", "filesys_glob", "filesys_grep", "filesys_read",
       "reddit_digest", "web_fetch", "web_search"]


def hop(step, op, ms=1.0, conf=1.0, summary="artifact"):
    return {"step": step, "op": op, "ms": ms, "conf": conf, "artifact_summary": summary}


def read_body(path, hops, status="done", **extra):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    body = {"status": status, "result": text, "degraded": False, "assumptions": [],
            "confidence": 1.0, "hops": hops}
    body.update(extra)
    return body


class StubBodies(unittest.TestCase):
    """Probe context + a conformant body for every case, and a transport serving them."""

    def setUp(self):
        self.ctx = probe.Ctx()
        self.files = self.ctx.fixture_files()
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="ti-e2e-contract-"))
        self.ctx.ambiguous = self.tmp
        for name in probe.AMBIGUOUS_FILES:
            (self.tmp / name).write_text(f"# {name}\n\nbody of {name}\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- bodies ---------------------------------------------------------
    def l0_body(self):
        text = self.ctx.readme.read_text(encoding="utf-8")
        return read_body(self.ctx.readme, [hop("L0", "filesys_read",
                                               summary=f"file_content :: {text[:120]}")])

    def l1_body(self, name="notes.md"):
        text = dict(self.files)[name]
        return read_body(self.ctx.fixture / name, [
            hop("candidates", "filesys_glob", summary=f"path_list[{len(self.files)}]  e.g. {name}"),
            hop("picked_file", "decide_pick", ms=1340.0, conf=0.96,
                summary=f"picked_items[1]  e.g. {name}"),
            hop("content", "filesys_read", summary=f"file_content :: {text[:120]}"),
        ], confidence=0.96)

    def l2_body(self):
        return {
            "status": "need_input", "result": [{"url": "https://example.test/q", "content": "x"}],
            "degraded": True, "assumptions": ["pages: gate unresolved, using the best guess"],
            "confidence": 0.25,
            "hops": [
                hop("queries", "decide_extract", summary="value_list[4]  e.g. llama.cpp quantization"),
                hop("hits", "web_search", ms=54000.0,
                    summary="url_list[6]  e.g. {'title': 'guide', 'url': 'https://x'}"),
                hop("picked", "decide_pick", ms=20500.0, conf=0.25,
                    summary="picked_items[4]  e.g. ['https://arxiv.org/abs/1']"),
                hop("pages", "askback", conf=0.0, summary="artifact 'picked' conf 0.25 below gate"),
                hop("pages", "web_fetch", ms=120000.0, summary="pages[1]  e.g. {'url': 'https://x'}"),
            ],
            "question": {"kind": "choice", "question": "Step 'pages' blocked",
                         "options": [{"id": "https://example.test/q", "desc": ""}]},
        }

    def ambiguous_body(self):
        text = dict(self.ctx.ambiguous_files())["report-final.md"]
        return {
            "status": "need_input", "result": text, "degraded": True,
            "assumptions": ["content: gate unresolved, using the best guess"],
            "confidence": 0.26,
            "hops": [
                hop("candidates", "filesys_glob",
                    summary=f"path_list[{len(probe.AMBIGUOUS_FILES)}]  e.g. report-draft.md"),
                hop("picked_file", "decide_pick", ms=3650.0, conf=0.26,
                    summary="picked_items[1]  e.g. report-final.md"),
                hop("content", "askback", conf=0.0,
                    summary="artifact 'picked_file' conf 0.26 below gate"),
                hop("content", "filesys_read", summary=f"file_content :: {text[:120]}"),
            ],
            "question": {"kind": "choice",
                         "question": "Step 'content' blocked: artifact 'picked_file' conf 0.26 "
                                     "below gate. Answer to resolve:",
                         "options": [{"id": name, "desc": ""} for name in probe.AMBIGUOUS_FILES]},
        }

    def answer_body(self):
        text = dict(self.ctx.ambiguous_files())[probe.AMBIGUOUS_ANSWER]
        return read_body(self.ctx.ambiguous / probe.AMBIGUOUS_ANSWER, [
            hop("candidates", "filesys_glob", summary="path_list[5]  e.g. report-draft.md"),
            hop("picked_file", "decide_pick", conf=0.26,
                summary="picked_items[1]  e.g. report-final.md"),
            hop("content", "askback", conf=0.0,
                summary="artifact 'picked_file' conf 0.26 below gate"),
            hop("content", "askback.resolved", conf=0.9, summary=probe.AMBIGUOUS_ANSWER),
            hop("content", "filesys_read", summary=f"file_content :: {text[:120]}"),
        ], confidence=0.9)

    def transport(self):
        """Serve the conformant body a case's request should get from a real service."""
        def call(body, timeout):
            params = body.get("params") or {}
            intent = body.get("intent", "")
            if params.get("tool"):
                return self.l0_body()
            if intent == probe.L2_INTENT:
                return self.l2_body()
            if intent == probe.AMBIGUOUS_INTENT:
                return self.answer_body() if "answer" in params else self.ambiguous_body()
            if intent == "read the notes file":
                return self.l1_body()
            raise AssertionError(f"no stub body for {body}")
        return call


class StubbedRunTest(StubBodies):
    """Every case passes against a conformant stub — and only then."""

    def test_all_five_cases_pass_on_conformant_bodies(self):
        rows = probe.run_cases(probe.cases(False, 60.0), self.ctx, OPS, self.transport())
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertTrue(row.ok, f"{row.case.name}: {row.problems}")

    def test_a_read_with_no_file_behind_it_fails_the_case(self):
        body = dict(self.l1_body(), result="text that is on no disk\n")
        rows = probe.run_cases([probe.cases(False, 60.0)[1]], self.ctx, OPS,
                               lambda b, t: body)
        self.assertFalse(rows[0].ok)
        self.assertTrue(any("matches no file" in p for p in rows[0].problems), rows[0].problems)

    def test_a_pick_hop_that_does_not_name_the_read_file_fails(self):
        body = self.l1_body()
        body["hops"][1] = hop("picked_file", "decide_pick", conf=0.96,
                              summary="picked_items[1]  e.g. budget.csv")
        rows = probe.run_cases([probe.cases(False, 60.0)[1]], self.ctx, OPS,
                               lambda b, t: body)
        self.assertFalse(rows[0].ok)
        self.assertTrue(any("does not name the file" in p for p in rows[0].problems))

    def test_need_input_without_a_question_fails(self):
        body = dict(self.ambiguous_body())
        body.pop("question")
        rows = probe.run_cases([probe.cases(False, 60.0)[3]], self.ctx, OPS, lambda b, t: body)
        self.assertFalse(rows[0].ok)
        self.assertTrue(any("need_input" in p for p in rows[0].problems), rows[0].problems)

    def test_a_one_option_choice_question_fails(self):
        body = self.ambiguous_body()
        body["question"] = dict(body["question"], options=[{"id": "report-final.md", "desc": ""}])
        rows = probe.run_cases([probe.cases(False, 60.0)[3]], self.ctx, OPS, lambda b, t: body)
        self.assertFalse(rows[0].ok)
        self.assertTrue(any("option" in p for p in rows[0].problems), rows[0].problems)

    def test_the_supplied_answer_must_be_what_was_read(self):
        body = dict(self.answer_body(),
                    result=dict(self.ctx.ambiguous_files())["report-final.md"])
        rows = probe.run_cases([probe.cases(False, 60.0)[4]], self.ctx, OPS, lambda b, t: body)
        self.assertFalse(rows[0].ok)
        self.assertTrue(any("supplied answer was not used" in p for p in rows[0].problems),
                        rows[0].problems)

    def test_a_dead_transport_is_a_failed_case_not_a_crash(self):
        def boom(body, timeout):
            raise OSError("connection refused")

        rows = probe.run_cases(probe.cases(False, 60.0), self.ctx, OPS, boom)
        self.assertTrue(all(not row.ok for row in rows))
        self.assertIn("transport failed", rows[0].problems[0])

    def test_skip_l2_and_only_narrow_the_list(self):
        names = [c.name for c in probe.cases(True, 60.0)]
        self.assertNotIn("l2_compound_research", names)
        self.assertEqual(len(names), 4)
        only = probe.cases(False, 60.0, ["ambiguous_needs_input"])
        self.assertEqual([c.name for c in only], ["ambiguous_needs_input"])


class ContractFieldTest(unittest.TestCase):
    """check_contract, one field at a time (spec §3)."""

    def base(self):
        return {"status": "done", "result": "x", "degraded": False, "assumptions": [],
                "confidence": 0.5, "hops": [hop("s", "filesys_read")]}

    def problems(self, body, ops=OPS):
        return probe.check_contract(body, ops)

    def test_conformant_body_has_no_problems(self):
        self.assertEqual(self.problems(self.base()), [])

    def test_status_must_be_one_of_the_four(self):
        self.assertIn("status", " ".join(self.problems(dict(self.base(), status="started"))))

    def test_hops_are_required_and_typed(self):
        body = self.base()
        body.pop("hops")
        self.assertTrue(any("hops" in p for p in self.problems(body)))
        self.assertTrue(any("hops" in p for p in self.problems(dict(self.base(), hops="nope"))))

    def test_every_hop_carries_the_five_fields(self):
        body = dict(self.base(), hops=[{"step": "s", "op": "filesys_read", "ms": 1.0}])
        problems = self.problems(body)
        self.assertTrue(any("'conf'" in p for p in problems), problems)
        self.assertTrue(any("'artifact_summary'" in p for p in problems), problems)

    def test_ms_and_conf_must_be_numbers_in_range(self):
        joined = " ".join(self.problems(dict(self.base(),
                                             hops=[hop("s", "filesys_read", ms="12ms", conf=1.4)])))
        self.assertIn("ms", joined)
        self.assertIn("outside [0,1]", joined)

    def test_done_needs_a_result(self):
        self.assertTrue(any("no result" in p
                            for p in self.problems(dict(self.base(), result=None))))

    def test_error_status_needs_a_message_and_no_error_otherwise(self):
        self.assertTrue(any("no message" in p
                            for p in self.problems({"status": "error", "hops": []})))
        self.assertTrue(any("carries an error" in p
                            for p in self.problems(dict(self.base(), error="stale"))))

    def test_confidence_is_optional_and_bounded_when_present(self):
        # Optional: the service reports it only when a hop measured one, so an L0
        # passthrough and an error run both omit it, and demanding it scored real
        # bodies as failures (docs/review2-2026-11-03.md NB3).
        self.assertEqual(self.problems(dict(self.base(), confidence=None)), [])
        self.assertTrue(any("outside [0,1]" in p
                            for p in self.problems(dict(self.base(), confidence=1.5))))
        self.assertTrue(any("not a number" in p
                            for p in self.problems(dict(self.base(), confidence="high"))))

    def test_sources_is_a_list_when_present(self):
        self.assertEqual(self.problems(dict(self.base(), sources=["a"])), [])
        self.assertTrue(any("sources" in p
                            for p in self.problems(dict(self.base(), sources={"a": 1}))))

    def test_question_shape(self):
        joined = " ".join(self.problems(dict(self.base(), status="need_input",
                                             question={"kind": "quiz", "question": ""})))
        self.assertIn("question.kind", joined)
        self.assertIn("missing or empty", joined)

    def test_a_hop_op_must_be_registered_or_a_runner_record(self):
        self.assertTrue(any("neither a registered op" in p
                            for p in self.problems(dict(self.base(),
                                                        hops=[hop("s", "__decide_extract__")]))))
        self.assertEqual(self.problems(dict(self.base(),
                                            hops=[hop("s", "askback.resolved", conf=0.9)])), [])


class BackendEvidenceTest(unittest.TestCase):
    """The probe must refuse to claim the live path on a replay backend."""

    def test_live_choice_backend_is_accepted(self):
        self.assertEqual(probe.backend_problem({"choice": "pair"}, False), "")
        self.assertEqual(probe.backend_problem({"choice": "d1_systemone"}, False), "")

    def test_fixture_backend_is_refused_unless_allowed(self):
        for name in ("fixture", "low_pick"):
            self.assertIn("replay backend", probe.backend_problem({"choice": name}, False))
            self.assertEqual(probe.backend_problem({"choice": name}, True), "")

    def test_unknown_backend_is_refused(self):
        self.assertIn("cannot say what answered", probe.backend_problem({}, False))
        self.assertEqual(probe.backend_problem({}, True), "")


class RealBodyTest(unittest.TestCase):
    """The contract, checked against bodies the service actually produced.

    Every other case here is a hand-written stub, and that is exactly how the
    probe came to reject the service's own correct responses: two hand-merged
    branches disagreed about whether `confidence` and a hop's `conf` are
    optional, and 22 stub cases happily agreed with the wrong side
    (docs/review2-2026-11-03.md NB3). This case asks the real entry point for a
    body and runs the real checker over it, so the contract cannot drift away
    from the thing it describes again.
    """

    def setUp(self):
        sys.path.insert(0, str(ROOT / "service"))
        from ti.conf import load
        self._filesys = dict(load()["ops"]["filesys"])
        self.addCleanup(load()["ops"]["filesys"].update, self._filesys)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "notes.md").write_text("body\n", encoding="utf-8")
        (self.root / "budget.csv").write_text("a,b\n", encoding="utf-8")

    def _body(self, request):
        from ti.service_entry import use_tool
        return use_tool(request)

    def test_an_l0_body_passes_the_contract(self):
        # `digest.cwd` is the run root; L0 paths are confined to it.
        body = self._body({
            "intent": "read the notes file",
            "digest": {"cwd": str(self.root)},
            "params": {"tool": "filesys_read",
                       "args": {"path": str(self.root / "notes.md")}},
        })
        self.assertEqual(body["status"], "done", body)
        self.assertEqual(probe.check_contract(body, set(OPS)), [])

    def test_an_error_body_passes_the_contract(self):
        body = self._body({
            "intent": "x",
            "digest": {"cwd": str(self.root)},
            "params": {"tool": "filesys_read",
                       "args": {"path": str(self.root / "absent.txt")}},
        })
        self.assertEqual(body["status"], "error", body)
        self.assertEqual(probe.check_contract(body, set(OPS)), [])

    def test_a_plan_body_passes_the_contract(self):
        body = self._body({"intent": "research local inference latency",
                           "mode": "plan"})
        self.assertEqual(body["status"], "plan", body)
        self.assertEqual(probe.check_contract(body, set(OPS)), [])


if __name__ == "__main__":
    unittest.main()
