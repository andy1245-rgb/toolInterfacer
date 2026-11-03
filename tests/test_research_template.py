"""The research template — the plan it builds, and the run it produces.

Two things are pinned here, and they are pinned for different reasons.

**Plan completeness** is a class, not a case: every step of every registered
template must name an op the registry serves, satisfy that op's required params
from what the template itself supplies, and read only artifacts an earlier step
produced. `research` shipped `op="__decide_extract__"` — a name nothing
registers, born of an internal-namespace convention that was never implemented —
so every "research …" intent died on step 0 for the whole life of the template,
and the suite never noticed because no test ever built a plan. A test that only
fixed those two strings would let the next template repeat it; this one cannot.

**The run** is driven the way production drives it — `use_tool`, through the
fixture decide backend and `tests/data/fake_cli.py` behind every `ops.*` script,
so nothing here touches the network or a live skill CLI.
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.conf import load, reload  # noqa: E402
from ti.decide.cache import clear as clear_decisions  # noqa: E402
from ti.ops.op_api import all_ops, get as get_op, import_all  # noqa: E402
from ti.planner.templates import TEMPLATES, apply_template  # noqa: E402
from ti.service_entry import use_tool  # noqa: E402

FAKE = ROOT / "tests" / "data" / "fake_cli.py"
_ENV_KEYS = ("TI_CONF", "TI_CONF_OVERLAY", "FAKE_CLI_OUT")

#: Ops spec §7 names by name. A template may only be built from these.
SPEC_OPS = {"decide_extract", "decide_pick", "web_search", "web_fetch",
            "reddit_digest", "digest_assemble"}

PLANS = [
    ("research", "research the state of local tool routing", {"cwd": "."}),
    ("research", "research what r/MechanicalKeyboards says about hall effect", {"cwd": "."}),
    ("reddit_digest", "what does reddit say about mechanical keyboards", {"cwd": "."}),
    ("disk_lookup", "read the file we discussed", {"cwd": "."}),
    ("disk_lookup", "list the python files in bench", {"cwd": "."}),
]


class PlanCompletenessTest(unittest.TestCase):
    """No template may publish a step the registry cannot execute."""

    def setUp(self):
        import_all()

    def test_the_registry_serves_every_op_the_spec_names(self):
        self.assertEqual(SPEC_OPS - set(all_ops()), set(), all_ops())

    def test_every_step_off_every_template_is_executable(self):
        for name, intent, digest in PLANS:
            with self.subTest(template=name, intent=intent):
                steps, _final_key, _notes = apply_template(name, intent, digest)
                self.assertTrue(steps, "a plan with no steps has nothing to run")
                known = all_ops()
                for step in steps:
                    self.assertIn(step.op, known,
                                  f"{name}.{step.id} names op {step.op!r}, which no "
                                  f"module registers; known: {known}")
                    supplied = set(step.params) | set(step.inject) | set(step.over)
                    self.assertEqual(set(get_op(step.op).required) - supplied, set(),
                                     f"{name}.{step.id} ({step.op}) cannot run: it leaves "
                                     f"a required param for the caller to guess")

    def test_step_ids_are_unique_and_the_final_key_is_one_of_them(self):
        for name, intent, digest in PLANS:
            with self.subTest(template=name, intent=intent):
                steps, final_key, _notes = apply_template(name, intent, digest)
                ids = [step.id for step in steps]
                self.assertEqual(len(ids), len(set(ids)), ids)
                self.assertIn(final_key, ids,
                              f"{name}: terminal artifact {final_key!r} is not written by "
                              f"any step, so a successful run would return None")

    def test_every_step_reads_only_artifacts_an_earlier_step_produced(self):
        for name, intent, digest in PLANS:
            with self.subTest(template=name, intent=intent):
                steps, _final_key, _notes = apply_template(name, intent, digest)
                produced: set[str] = set()
                for step in steps:
                    for param, artifact in {**step.inject, **step.over}.items():
                        self.assertIn(artifact, produced,
                                      f"{name}.{step.id}: param {param!r} reads {artifact!r}, "
                                      f"which no earlier step writes (have {sorted(produced)})")
                    produced.add(step.id)


class FixtureCase(unittest.TestCase):
    """Fixture decide backend, fake CLI behind every op, temp artifact dir."""

    INTENT = "research how local tool routing models are evaluated"

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ti-research-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
        self.addCleanup(self._restore_env)
        os.environ.pop("TI_CONF", None)
        os.environ["FAKE_CLI_OUT"] = self.tmp
        self.configure()

    def _restore_env(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_decisions()
        reload()

    def configure(self, **sections):
        """Point every op at the fake CLI and every decision at the fixture backend."""
        overlay = {
            "decide": {"backends_in_use": {"choice": "fixture", "noul": "fixture",
                                           "extract": "rule_extract"},
                       "cache": {"enabled": False}},
            "ops": {"web_search": {"script": str(FAKE), "backend": "free-search",
                                   "brave": "off"},
                    "web_fetch": {"script": str(FAKE), "max_chars": 500},
                    "reddit": {"cwd": str(pathlib.Path(self.tmp) / "no-such-project"),
                               "cwd_candidates": [], "cli": ["fake"]},
                    "cli_timeout_s": 20},
            "hops_log": {"enabled": False},
        }
        for section, values in sections.items():
            overlay[section] = _merge(overlay.get(section, {}), values)
        path = pathlib.Path(self.tmp) / "overlay.json"
        path.write_text(json.dumps(overlay), encoding="utf-8")
        os.environ["TI_CONF_OVERLAY"] = str(path)
        reload()
        clear_decisions()

    def run_intent(self, intent=None, **body):
        return use_tool({"intent": intent or self.INTENT, "digest": {"cwd": str(self.tmp)},
                         **body})

    def steps_of(self, out):
        return [hop["step"] for hop in out["hops"]]


def _merge(base, over):
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


class ResearchRunTest(FixtureCase):
    """P4's acceptance criterion: a research intent completes, sourced."""

    def test_a_research_intent_does_not_error(self):
        out = self.run_intent()
        self.assertEqual(out["status"], "done", out.get("error"))
        self.assertIsNone(out.get("error"))
        self.assertTrue(out["hops"], "a two-decision pipeline that ran zero hops did not run")

    def test_every_hop_names_a_registered_op(self):
        out = self.run_intent()
        known = set(all_ops()) | {"askback", "askback.resolved"}
        for hop in out["hops"]:
            self.assertIn(hop["op"], known, hop)

    def test_the_run_walks_the_spec_pipeline_in_order(self):
        steps = self.steps_of(self.run_intent())
        head = steps[:4]
        self.assertEqual(head, ["queries", "hits", "picked", "pages#0"])
        self.assertEqual(steps[-1], "digest")
        self.assertTrue(all(s.startswith("pages#") for s in steps[3:-1]), steps)

    def test_the_terminal_artifact_is_a_digest_not_a_page_list(self):
        digest = self.run_intent()["result"]
        self.assertEqual(digest["intent"], self.INTENT)
        self.assertEqual(set(digest) >= {"intent", "queries", "sources", "skipped",
                                         "threads", "coverage", "confidence"}, True,
                         sorted(digest))
        self.assertTrue(digest["queries"], "the digest must say what it searched")
        self.assertTrue(digest["sources"], "no source survived a fake CLI that never fails")
        for source in digest["sources"]:
            self.assertTrue(source["url"].startswith("http"), source)
            self.assertTrue(source["claim"], source)
            self.assertTrue(source["title"], "attribution comes from the search hits")

    def test_digest_confidence_is_page_coverage(self):
        digest = self.run_intent()["result"]
        coverage = digest["coverage"]
        self.assertEqual(coverage["sources"], len(digest["sources"]))
        self.assertAlmostEqual(digest["confidence"],
                               min(1.0, coverage["sources"] / coverage["target"]), places=3)

    def test_plan_mode_publishes_the_same_steps(self):
        out = use_tool({"intent": self.INTENT, "mode": "plan", "digest": {"cwd": "."}})
        self.assertEqual(out["status"], "plan")
        self.assertEqual([step["op"] for step in out["steps"]],
                         ["decide_extract", "web_search", "decide_pick", "web_fetch",
                          "digest_assemble"])
        loop = out["steps"][3]
        self.assertEqual(loop["gate_post"][0]["verifier"], "is_relevant")

    def test_the_template_hardcodes_no_degree(self):
        """Every number the plan carries came out of conf, not out of the file."""
        conf = load()
        steps, _key, _notes = apply_template("research", self.INTENT, {"cwd": "."})
        by_id = {step.id: step for step in steps}
        self.assertEqual(by_id["queries"].params["count"], conf["research"]["query_count"][-1])
        self.assertEqual(by_id["hits"].params["backend"], conf["research"]["backends"][0])
        self.assertEqual(by_id["picked"].params["top_k"], conf["research"]["pick_top_k"])
        self.assertEqual(by_id["pages"].stop_after, conf["research"]["good_pages_target"])
        self.assertEqual(by_id["pages"].gate.post[0]["min"], conf["policy"]["gate_noul"])
        for cond in by_id["pages"].gate.pre:
            self.assertEqual(cond["min"], conf["policy"]["gate_slot"])

    def test_the_fan_out_is_authorized_by_coverage_and_by_the_pick(self):
        steps, _key, _notes = apply_template("research", self.INTENT, {"cwd": "."})
        gate = {step.id: step for step in steps}["pages"].gate.pre
        self.assertEqual([cond["artifact"] for cond in gate], ["hits", "picked"],
                         "web_search's conf is coverage; it must gate the fan-out")

    def test_a_gated_fetch_step_degrades_instead_of_aborting(self):
        self.configure(policy={"gate_slot": 0.99})
        out = self.run_intent()
        self.assertEqual(out["status"], "done", out.get("error"))
        self.assertTrue(out["degraded"])
        self.assertTrue(any("gate unresolved" in note for note in out["assumptions"]),
                        out["assumptions"])

    def test_a_pick_from_no_candidates_fails_at_the_pick(self):
        """Not one step later, on a URL no search ever returned."""
        from ti.core.artifacts import ArtifactStore
        store = ArtifactStore()
        store.put("hits", "hits", "url_list", [])
        with self.assertRaises(ValueError) as ctx:
            get_op("decide_pick").run({"source_artifact": "hits",
                                       "instructions": "pick one"}, store)
        self.assertIn("nothing to pick from", str(ctx.exception))


class GoodPagesStopTest(FixtureCase):
    """§7.5's stop condition: fetch until the target is met, then stop paying."""

    def test_the_loop_stops_at_the_good_page_target(self):
        self.configure(research={"pick_top_k": 6, "good_pages_target": 2})
        out = self.run_intent()
        pages = [step for step in self.steps_of(out) if step.startswith("pages#")]
        self.assertEqual(pages, ["pages#0", "pages#1"])
        self.assertEqual(len(out["result"]["sources"]), 2)

    def test_a_page_below_the_noul_gate_is_dropped_not_kept(self):
        self.configure(decide={"fixture": {"default_noul": 0.4}},
                       research={"good_pages_target": 2})
        out = self.run_intent()
        self.assertEqual(out["status"], "done", out.get("error"))
        verdicts = [hop.get("verified") for hop in out["hops"]
                    if hop["step"].startswith("pages#")]
        self.assertTrue(verdicts)
        self.assertTrue(all(v is False for v in verdicts), verdicts)
        self.assertEqual(out["result"]["sources"], [])
        self.assertTrue(out["degraded"], "an empty digest is a degraded run, not a good one")
        self.assertTrue(any("dropped" in note for note in out["assumptions"]), out["assumptions"])

    def test_each_page_is_verified_on_its_own(self):
        self.configure(decide={"fixture": {"default_noul": 0.9}})
        out = self.run_intent()
        seen = [hop["artifact_summary"] for hop in out["hops"]
                if hop["step"].startswith("pages#")]
        self.assertGreater(len(seen), 1)
        for summary in seen:
            self.assertTrue(summary.startswith("pages[1]"),
                            "a per-page verdict must see one page, not the batch")

    def test_a_shortfall_is_stated_not_silent(self):
        self.configure(research={"good_pages_target": 8})
        out = self.run_intent()
        self.assertEqual(out["status"], "done", out.get("error"))
        notes = [note for note in out["assumptions"] if note.startswith("pages:")]
        self.assertTrue(notes, out["assumptions"])
        self.assertIn("target 8", notes[0])

    def test_query_count_comes_from_the_configured_band(self):
        self.configure(research={"query_count": [3, 3]})
        out = self.run_intent()
        self.assertLessEqual(len(out["result"]["queries"]), 3)


class CrossCheckTest(FixtureCase):
    """§7.6: the reddit step is optional and must skip cleanly when unconfigured."""

    COMMUNITY = "research what r/MechanicalKeyboards says about hall effect switches"

    def steps(self, intent):
        steps, _key, notes = apply_template("research", intent, {"cwd": "."})
        return [step.id for step in steps], notes

    def test_auto_mode_leaves_a_plain_research_ask_alone(self):
        ids, notes = self.steps(self.INTENT)
        self.assertNotIn("cross_check", ids)
        self.assertEqual(notes, [], "an ask that never wanted a cross-check was not denied one")

    def test_an_unconfigured_cross_check_is_skipped_by_name(self):
        self.configure(research={"reddit_cross_check": "on"})
        ids, notes = self.steps(self.INTENT)
        self.assertNotIn("cross_check", ids)
        self.assertTrue(any("cross-check skipped" in note for note in notes), notes)

    def test_the_run_survives_an_unavailable_cross_check(self):
        self.configure(research={"reddit_cross_check": "on"})
        out = self.run_intent(self.COMMUNITY)
        self.assertEqual(out["status"], "done", out.get("error"))
        self.assertIsNone(out["result"]["threads"])
        self.assertTrue(any("cross-check skipped" in note for note in out["assumptions"]), out)

    def test_a_configured_cross_check_lands_in_the_digest(self):
        project = pathlib.Path(self.tmp) / "reddit-project"
        (project / ".venv" / "bin").mkdir(parents=True)
        entry = project / ".venv" / "bin" / "python"
        shutil.copy(FAKE, entry)
        entry.chmod(0o755)
        self.configure(ops={"reddit": {"cwd": str(project), "cwd_candidates": [],
                                      "cli": [".venv/bin/python"]}})
        out = self.run_intent(self.COMMUNITY)
        self.assertEqual(out["status"], "done", out.get("error"))
        self.assertIn("cross_check", self.steps_of(out))
        threads = out["result"]["threads"]
        self.assertIsNotNone(threads, out["assumptions"])
        self.assertEqual(threads["subreddit"], "MechanicalKeyboards")
        self.assertTrue(threads["posts"])
        self.assertTrue(all("unknown_field" not in post for post in threads["posts"]))

    def test_a_project_that_vanishes_mid_run_degrades_instead_of_failing(self):
        project = pathlib.Path(self.tmp) / "half-project"
        project.mkdir()
        self.configure(ops={"reddit": {"cwd": str(project), "cwd_candidates": [],
                                      "cli": [".venv/bin/python"]}})
        out = self.run_intent(self.COMMUNITY)
        self.assertEqual(out["status"], "done", out.get("error"))
        self.assertTrue(out["degraded"])
        self.assertTrue(any("optional step" in note for note in out["assumptions"]),
                        out["assumptions"])
        self.assertIsNone(out["result"]["threads"])


class AssembleOpTest(unittest.TestCase):
    """`digest_assemble`'s contract, at the op boundary."""

    def setUp(self):
        import_all()
        self.op = get_op("digest_assemble")

    def run_op(self, **params):
        return self.op.run({"intent": "why do people use local models", **params}, store=None)

    def test_a_page_without_text_becomes_a_skip_not_a_source(self):
        _kind, data, _conf = self.run_op(pages=[
            {"url": "https://a.test/x", "content": "text", "method": "curl"},
            {"url": "https://b.test/x", "content": "", "method": None, "error": "boom"},
        ])
        self.assertEqual([s["url"] for s in data["sources"]], ["https://a.test/x"])
        self.assertEqual(data["skipped"], [{"url": "https://b.test/x", "reason": "boom"}])

    def test_a_degraded_extraction_is_graded_half(self):
        _kind, data, _conf = self.run_op(pages=[
            {"url": "https://a.test/x", "content": "shell", "method": "python-stdlib",
             "degraded": True}])
        self.assertEqual(data["sources"][0]["conf"], 0.5)

    def test_the_claim_is_cut_on_a_word_boundary(self):
        self.configure_claim_chars(20)
        text = "alpha beta gamma delta epsilon zeta eta theta"
        _kind, data, _conf = self.run_op(pages=[
            {"url": "https://a.test/x", "content": text, "method": "curl"}])
        claim = data["sources"][0]["claim"]
        self.assertTrue(claim.endswith("…"), claim)
        self.assertLessEqual(len(claim), 21)
        self.assertEqual(claim[:-1], claim[:-1].rstrip(), "a cut claim must not end in a space")
        self.assertTrue(text.startswith(claim[:-1]), claim)

    def test_a_short_page_is_quoted_whole(self):
        _kind, data, _conf = self.run_op(pages=[
            {"url": "https://a.test/x", "content": "  short   page ", "method": "curl"}])
        self.assertEqual(data["sources"][0]["claim"], "short page")

    def test_confidence_is_coverage_against_the_configured_target(self):
        self.configure_target(4)
        _kind, data, conf = self.run_op(pages=[
            {"url": f"https://a.test/{i}", "content": "x", "method": "curl"}
            for i in range(2)])
        self.assertEqual(data["coverage"], {"sources": 2, "target": 4})
        self.assertAlmostEqual(data["confidence"], 0.5, places=3)
        self.assertAlmostEqual(conf, 0.5, places=3)

    def test_an_empty_digest_reports_no_confidence(self):
        _kind, data, conf = self.run_op(pages=[])
        self.assertEqual((data["sources"], data["confidence"], conf), ([], 0.0, 0.0))

    def test_a_missing_artifact_is_an_empty_digest_not_a_crash(self):
        _kind, data, _conf = self.run_op(pages="(missing)", threads="(missing)",
                                         hits="(missing)", queries="(missing)")
        self.assertEqual((data["sources"], data["threads"], data["queries"]), ([], None, []))

    def test_the_intent_is_required(self):
        with self.assertRaises(ValueError):
            self.op.run({"intent": "  "}, store=None)

    def configure_target(self, target):
        """Point the op's coverage target at `target` for one case."""
        self._conf("good_pages_target", target)

    def configure_claim_chars(self, chars):
        """Narrow the claim window so the cut is exercised on a short fixture."""
        digest = load()["research"].setdefault("digest", {})
        self.addCleanup(digest.__setitem__, "claim_chars", digest.get("claim_chars"))
        digest["claim_chars"] = chars

    def _conf(self, key, value):
        research = load()["research"]
        saved = research[key]
        research[key] = value
        self.addCleanup(research.__setitem__, key, saved)


if __name__ == "__main__":
    unittest.main()
