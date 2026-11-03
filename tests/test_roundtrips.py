"""Round-trip bench tests — the counting, the cost model, and the checks.

The bench's numbers are the whole point of it, so the parts that decide those
numbers are pinned here offline, with no service and no model:

* a round trip is one entry in a native policy, and one POST (plus the answer
  POST when the service asks back) — the counting is asserted against a
  recorder transport, not against the bench's own report;
* the cost model is asserted against hand arithmetic, including the breakeven
  per-task solve and its two degenerate branches;
* every task's correctness check is exercised in both directions against a
  throwaway fixture tree, so a check that always passes (or always fails) is a
  test failure here rather than a row in the results file.
"""
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "service"))

import roundtrip_tasks as tasklist  # noqa: E402
import roundtrips  # noqa: E402


def make_tree(base: pathlib.Path) -> tasklist.Ctx:
    """A miniature of the repo the tasks read: README, walk.py, tests, fixtures."""
    (base / "README.md").write_text("# readme\nbody\n", encoding="utf-8")
    walk = base / "service/ti/ops/filesys/walk.py"
    walk.parent.mkdir(parents=True, exist_ok=True)
    walk.write_text('"""walk — bounded os.walk."""\n\n# max_depth keeps the walk finite\n'
                    'MAX = 12\n' + "x = 1\n" * 40, encoding="utf-8")
    (base / "tests/data").mkdir(parents=True)
    (base / "tests/test_a.py").write_text("a\n", encoding="utf-8")
    (base / "tests/data/deep.py").write_text("deep\n", encoding="utf-8")
    notes, reports = tasklist.fixtures(base / "fx")
    return tasklist.Ctx(root=base, notes=notes, reports=reports).build()


class ChecksTest(unittest.TestCase):
    """Every check must reject a wrong artifact and accept the right one."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = make_tree(pathlib.Path(self.tmp.name))
        self.tasks = {t.id: t for t in tasklist.tasks(self.ctx)}

    def tearDown(self):
        self.tmp.cleanup()

    def _check(self, task_id, good, bad):
        task = self.tasks[task_id]
        ok, _detail = task.check(good, self.ctx)
        self.assertTrue(ok, f"{task_id}: a correct artifact was scored wrong")
        ok, detail = task.check(bad, self.ctx)
        self.assertFalse(ok, f"{task_id}: a wrong artifact was scored correct")
        self.assertTrue(detail, f"{task_id}: a failed check must say what it wanted")

    def test_read_checks(self):
        self._check("l0_read", self.ctx.readme, self.ctx.readme[:-3])
        self._check("l1_read", self.ctx.notes_text, "pick me later\n")
        self._check("l1_ask", self.ctx.report_text, "report draft\n")
        self._check("compound_walk", self.ctx.walk_text, "no such token")

    def test_listing_check_is_exact(self):
        good = self.ctx.tests_py
        self._check("l1_list", good, good[:-1])
        self._check("l1_list", good, good + ["tests/test_extra.py"])
        # and it is the recursive set: the stack's glob matches across separators
        self.assertIn("tests/data/deep.py", self.ctx.tests_py)

    def test_grep_check_wants_the_defining_file(self):
        self._check("l0_grep", [{"path": "disk_lookup.py", "excerpt": "def build"}],
                    [{"path": "research.py", "excerpt": "def build"}])

    def test_symbol_check_takes_hits_or_text(self):
        task = self.tasks["l1_symbol"]
        self.assertTrue(task.check([{"path": "impl.py", "excerpt": "def run"}], self.ctx)[0])
        self.assertTrue(task.check("see service/ti/ops/filesys/impl.py", self.ctx)[0])
        self.assertFalse(task.check("[build-system]\nrequires = []\n", self.ctx)[0])

    def test_research_check_accepts_both_arm_shapes(self):
        task = self.tasks["l2_research"]
        page = [{"url": "https://example.org/x", "content": "c" * 400}]
        self.assertTrue(task.check(page, self.ctx)[0])
        self.assertFalse(task.check([{"url": "https://example.org/x", "content": "short"}],
                                    self.ctx)[0])
        digest = {"sources": [{"url": "https://example.org/x", "claim": "c" * 80}]}
        self.assertTrue(task.check(digest, self.ctx)[0])
        empty = {"sources": [{"url": "", "claim": ""}]}
        self.assertFalse(task.check(empty, self.ctx)[0])

    def test_reddit_check(self):
        self._check("l2_reddit", {"threads": [{"id": "1"}]}, {"threads": []})


class TaskListTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = make_tree(pathlib.Path(self.tmp.name))
        self.tasks = tasklist.tasks(self.ctx)

    def tearDown(self):
        self.tmp.cleanup()

    def test_at_least_eight_mixed_shapes(self):
        self.assertGreaterEqual(len(self.tasks), 8)
        shapes = {t.shape.split()[0] for t in self.tasks}
        for level in ("L0", "L1", "L2"):
            self.assertTrue(any(s.startswith(level) for s in shapes), f"no {level} task")

    def test_ids_unique_and_every_task_has_a_policy(self):
        ids = [t.id for t in self.tasks]
        self.assertEqual(len(ids), len(set(ids)))
        for task in self.tasks:
            self.assertTrue(task.native, f"{task.id} has no native policy")
            self.assertTrue(callable(task.check), f"{task.id} has no check")

    def test_native_steps_name_ops_or_the_human(self):
        # A typo'd op name would run as an error in arm A and quietly inflate or
        # deflate its round-trip count; the registry is the source of truth.
        from ti.ops.op_api import all_ops, import_all
        import_all()
        known = set(all_ops()) | {tasklist.WAIT_FOR_HUMAN}
        for task in self.tasks:
            for name, _spec in task.native:
                self.assertIn(name, known, f"{task.id} names unknown tool {name!r}")

    def test_answer_is_one_of_the_fixture_candidates(self):
        task = {t.id: t for t in self.tasks}["l1_ask"]
        self.assertIn(task.answer, tasklist.REPORT_FILES)

    def test_digest_is_stable_and_distinguishing(self):
        self.assertEqual(tasklist.digest_of("abc"), tasklist.digest_of("abc"))
        self.assertNotEqual(tasklist.digest_of("abc"), tasklist.digest_of("abd"))
        self.assertEqual(len(tasklist.digest_of({"a": 1})), 12)


class CountingTest(unittest.TestCase):
    """Round trips are counted from the two arms' own structure, not from prose."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = make_tree(pathlib.Path(self.tmp.name))
        self.tasks = {t.id: t for t in tasklist.tasks(self.ctx)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_native_counts_every_policy_step_including_the_human(self):
        run = roundtrips.run_native(self.tasks["l1_ask"], self.ctx.reports)
        self.assertEqual(run.round_trips, len(self.tasks["l1_ask"].native))
        self.assertEqual(run.round_trips, 3)
        self.assertEqual(run.result, self.ctx.report_text)
        self.assertEqual([s["tool"] for s in run.steps],
                         ["filesys_glob", "ask_user", "filesys_read"])

    def test_native_executes_real_ops_and_times_them(self):
        run = roundtrips.run_native(self.tasks["l0_read"], self.ctx.root)
        self.assertEqual(run.error, "")
        self.assertEqual(run.result, self.ctx.readme)
        self.assertGreaterEqual(run.exec_ms, 0.0)

    def test_native_records_a_tool_error_without_crashing(self):
        task = tasklist.Task(id="bad", shape="L0 read", intent="read nothing",
                             params={}, check=lambda _r, _c: (False, "n/a"),
                             native=(("filesys_read", {"path": "no/such/file"}),))
        run = roundtrips.run_native(task, self.ctx.root)
        self.assertEqual(run.round_trips, 1)
        self.assertIn("filesys_read", run.error)

    def test_service_counts_one_post_when_it_answers_directly(self):
        transport = Recorder([{"status": "done", "result": "x", "hops": [{"op": "a"}]}])
        run = roundtrips.run_service(self.tasks["l0_read"], self.ctx.root, transport)
        self.assertEqual(run.round_trips, 1)
        self.assertEqual(len(transport.bodies), 1)
        self.assertEqual(run.status, "done")

    def test_service_counts_the_answer_call_as_a_second_round_trip(self):
        first = {"status": "need_input",
                 "question": {"kind": "choice",
                              "options": [{"id": "report-v1.md", "desc": ""}]},
                 "hops": [{"op": "filesys_glob", "ms": 1.0}]}
        second = {"status": "done", "result": "ok",
                  "hops": [{"op": "askback.resolved", "ms": 2.0}]}
        transport = Recorder([first, second])
        run = roundtrips.run_service(self.tasks["l1_ask"], self.ctx.reports, transport)
        self.assertEqual(run.round_trips, 2)
        self.assertEqual(transport.bodies[1]["params"]["answer"], "report-v1.md")
        self.assertEqual(len(run.hops), 2)
        self.assertEqual(run.status, "done")

    def test_service_does_not_loop_on_repeated_questions(self):
        ask = {"status": "need_input",
               "question": {"kind": "choice", "options": [{"id": "a", "desc": ""}]},
               "hops": []}
        transport = Recorder([ask, ask, ask])
        run = roundtrips.run_service(self.tasks["l1_ask"], self.ctx.reports, transport)
        self.assertEqual(run.round_trips, 2)
        self.assertEqual(len(transport.bodies), 2)

    def test_decide_time_is_read_from_decide_hops_only(self):
        body = {"status": "done", "result": "x",
                "hops": [{"op": "filesys_glob", "ms": 5.0},
                         {"op": "decide_pick", "ms": 900.0},
                         {"op": "filesys_read", "ms": 2.0}]}
        run = roundtrips.run_service(self.tasks["l1_read"], self.ctx.notes, Recorder([body]))
        self.assertEqual(run.decide_ms, 900.0)
        self.assertEqual(len(run.decide_hops), 1)

    def test_ask_user_executes_no_tool_and_costs_no_exec_time(self):
        run = roundtrips.run_native(self.tasks["l1_ask"], self.ctx.reports)
        step = [s for s in run.steps if s["tool"] == "ask_user"][0]
        self.assertEqual(step["ms"], 0.0)


class CostModelTest(unittest.TestCase):
    def test_wall_is_tool_work_plus_one_pass_per_round_trip(self):
        self.assertEqual(roundtrips.wall_ms(100.0, 3, 2000.0), 6100.0)
        self.assertEqual(roundtrips.wall_ms(0.0, 0, 2000.0), 0.0)

    def test_breakeven_solves_the_measured_difference(self):
        native = roundtrips.NativeRun(round_trips=3, exec_ms=50.0)
        service = roundtrips.ServiceRun(round_trips=1, wall_ms=2050.0)
        # (2050 - 50) / (3 - 1) = 1000 ms: at a 1 s pass the two arms tie.
        self.assertEqual(roundtrips.breakeven_pass_ms(native, service), 1000.0)
        # below the breakeven arm A is cheaper, above it the service is: that is
        # the direction of the crossover, and getting it backwards would invert
        # every conclusion the run draws.
        self.assertLess(roundtrips.wall_ms(50.0, 3, 999.0),
                        roundtrips.wall_ms(2050.0, 1, 999.0))
        self.assertGreater(roundtrips.wall_ms(50.0, 3, 1001.0),
                           roundtrips.wall_ms(2050.0, 1, 1001.0))

    def test_breakeven_is_none_when_no_round_trip_is_saved(self):
        native = roundtrips.NativeRun(round_trips=1, exec_ms=1.0)
        service = roundtrips.ServiceRun(round_trips=2, wall_ms=10.0)
        self.assertIsNone(roundtrips.breakeven_pass_ms(native, service))

    def test_breakeven_never_goes_negative(self):
        native = roundtrips.NativeRun(round_trips=2, exec_ms=100.0)
        service = roundtrips.ServiceRun(round_trips=1, wall_ms=5.0)   # B cheaper outright
        self.assertEqual(roundtrips.breakeven_pass_ms(native, service), 0.0)


class AggregateTest(unittest.TestCase):
    def _row(self, a_rt, b_rt, a_exec, b_wall, hops, decide_ms=0.0):
        native = roundtrips.NativeRun(round_trips=a_rt, exec_ms=a_exec)
        service = roundtrips.ServiceRun(round_trips=b_rt, wall_ms=b_wall,
                                        hops=[{"op": "x", "ms": 0.0}] * hops)
        service.hops = ([{"op": "decide_pick", "ms": decide_ms}] if decide_ms else []) + \
                       [{"op": "x", "ms": 0.0}] * hops
        return roundtrips.Row(task=tasklist.Task(id="t", shape="s", intent="i",
                                                 params={}, native=(), check=lambda *_: (True, "")),
                              native=native, service=service,
                              a_correct=True, a_detail="", b_correct=True, b_detail="")

    def test_totals_double_count_nothing(self):
        rows = [self._row(2, 1, 10.0, 110.0, 3), self._row(1, 1, 5.0, 20.0, 1)]
        total = roundtrips.aggregate(rows, 1000.0)
        self.assertEqual(total["round_trips"]["a"], 3)
        self.assertEqual(total["round_trips"]["b"], 2)
        self.assertEqual(total["round_trips"]["saved"], 1)
        self.assertEqual(total["wall_ms"]["a"], 10.0 + 5.0 + 3 * 1000.0)
        self.assertEqual(total["wall_ms"]["b"], 110.0 + 20.0 + 2 * 1000.0)
        self.assertEqual(total["service_overhead_ms"], 115.0)     # (110+20) - (10+5)
        self.assertEqual(total["hops_per_round_trip"], 2.0)

    def test_projection_reprices_only_the_decide_time(self):
        rows = [self._row(2, 1, 10.0, 1110.0, 1, decide_ms=1000.0)]
        total = roundtrips.aggregate(rows, 1000.0, project_decide_ms=250.0)
        # 1110 - 1000 + 1 * 250 = 360 ms of service, plus one pass: 1360 ms.
        self.assertEqual(total["projected"]["service_ms"], 360.0)
        self.assertEqual(total["projected"]["wall_ms"], 1360.0)
        self.assertEqual(total["projected"]["saved_ms"], 650.0)
        self.assertEqual(total["decide_hops_on_b"], 1)

    def test_no_projection_unless_asked_for(self):
        rows = [self._row(2, 1, 10.0, 1110.0, 1, decide_ms=1000.0)]
        self.assertNotIn("projected", roundtrips.aggregate(rows, 1000.0))

    def test_a_run_that_saves_nothing_reports_zero_pct(self):
        rows = [self._row(1, 1, 1.0, 2.0, 1)]
        total = roundtrips.aggregate(rows, 2000.0)
        self.assertEqual(total["round_trips"]["saved"], 0)
        self.assertLess(total["wall_ms"]["saved"], 0)             # B pays the service
        self.assertEqual(total["round_trips"]["saved_pct"], 0.0)


class Recorder:
    """A transport that answers from a script and remembers every body."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.bodies = []

    def post(self, body):
        self.bodies.append(body)
        payload = self.payloads[min(len(self.bodies) - 1, len(self.payloads) - 1)]
        return payload, 1.0


if __name__ == "__main__":
    unittest.main()
