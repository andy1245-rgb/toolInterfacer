"""distill pipeline tests — the whole path exercised offline, for free.

Three things are pinned here, and each one is a way this pipeline has failed in
someone else's hands:

* **the row contract** — every generated row carries the band, the source, the split
  and the candidate block the eval needs to slice on, and the labelled rows carry a
  target that is one of those candidates with a confidence in 0..1;
* **the leakage refusal** — a row that restates one of the frozen 150 utterances is
  rejected at generation *and* again at label time, so a hand-edited candidate file
  cannot smuggle eval content into train/val;
* **the run mechanics** — a 429 is not a row, a schema-invalid reply is not a row,
  the checkpoint is resumable without duplicating ids, and the budget gate stops a
  run whose projection exceeds the ceiling.

No case touches the network: the dry-run path uses the fixture teacher, and the
hosted-teacher cases run against a stub server on an ephemeral 127.0.0.1 port with a
fake key in the environment. Nothing here reads the real key or the real API.
"""
import atexit
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "service"))

from distill import (audit, bands, catalog, corpus, cost, label, mine,  # noqa: E402
                     similarity, stats, teacher, templates, utterances)
from ti.conf import load  # noqa: E402

FAKE_KEY = "sk-or-v1-" + "0" * 64


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        stub: "_Stub" = self.server.stub  # type: ignore[attr-defined]
        stub.bodies.append(json.loads(raw or b"{}"))
        status, body = stub.take()
        payload = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except OSError:  # the client hung up first
            pass

    def log_message(self, *args: Any) -> None:
        pass


class _Stub:
    """Canned replies for a POST endpoint; the last reply repeats."""

    def __init__(self, *replies: tuple[int, Any]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.stub = self  # type: ignore[attr-defined]
        self._httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self) -> "_Stub":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    def take(self) -> tuple[int, Any]:
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _reply(content: Any, usage: dict[str, Any] | None = None) -> tuple[int, Any]:
    body: dict[str, Any] = {
        "id": "gen-test", "model": "z-ai/glm-5.3-flash", "provider": "Relace",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
    }
    if usage is not None:
        body["usage"] = usage
    return 200, body


_SHARED: dict[str, Any] = {}


def shared_corpus() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate the candidate corpus once for the whole module.

    Generation is deterministic and costs a second, so every case that only *reads*
    a corpus shares one; tests that mutate rows get copies.
    """
    if not _SHARED:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="distill-test-"))
        atexit.register(shutil.rmtree, tmp, True)
        manifest = utterances.generate(tmp, use_mining=False)
        _SHARED["rows"] = corpus.read_jsonl(tmp / "candidates.jsonl")
        _SHARED["manifest"] = manifest
    return [dict(row) for row in _SHARED["rows"]], dict(_SHARED["manifest"])


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)

    def generate(self) -> dict[str, Any]:
        rows, manifest = shared_corpus()
        corpus.write_jsonl(self.dir / "candidates.jsonl", rows)
        return manifest

    def candidates(self) -> list[dict[str, Any]]:
        return corpus.read_jsonl(self.dir / "candidates.jsonl")


class CatalogTest(unittest.TestCase):
    def test_catalog_is_derived_from_the_service_registry(self) -> None:
        cat = catalog.catalog()
        self.assertEqual(sorted(cat), sorted(catalog.ops()))
        self.assertEqual(len(cat), 8)
        for name, description in cat.items():
            self.assertTrue(description and len(description) < 120, name)

    def test_description_drift_fails_loudly(self) -> None:
        broken = dict(catalog.DESCRIPTIONS)
        broken.pop("web_fetch")
        with mock.patch.object(catalog, "DESCRIPTIONS", broken):
            with self.assertRaises(AssertionError):
                catalog.catalog()

    def test_candidate_set_offers_the_rival_and_is_name_sorted(self) -> None:
        block = catalog.candidate_set("filesys_glob", seed="x", size=5,
                                      must_include=("filesys_grep",))
        self.assertIn("filesys_glob", block)
        self.assertIn("filesys_grep", block)
        self.assertEqual(list(block), sorted(block))
        self.assertLessEqual(len(block), 5)

    def test_every_rival_is_a_real_op(self) -> None:
        ops = set(catalog.catalog())
        for a, b in catalog.CONFUSABLE:
            self.assertIn(a, ops)
            self.assertIn(b, ops)


class BandTest(unittest.TestCase):
    def test_t1_t2_t3_classification(self) -> None:
        self.assertEqual(bands.classify("grep the tree for handle_stream").band, "t1")
        self.assertEqual(bands.classify("what does the config actually say?").band, "t2")
        boundary = bands.classify("which files mention the timeout, not which files exist")
        self.assertEqual(boundary.band, "t3")
        self.assertTrue(boundary.rival)

    def test_gap_cues_match_intervening_words(self) -> None:
        self.assertIn("filesys_grep", bands.cue_hits("which files in the repo mention X"))
        self.assertIn("filesys_grep", bands.cue_hits("which files talk about the retry"))


class GenerateTest(_Tmp):
    def test_rows_carry_the_band_source_split_and_candidate_block(self) -> None:
        manifest = self.generate()
        rows = self.candidates()
        self.assertEqual(manifest["rows"], len(rows))
        self.assertGreater(len(rows), 300)
        for row in rows:
            self.assertIn(row["band"], corpus.BANDS)
            self.assertIn(row["source"], corpus.SOURCES)
            self.assertIn(row["split"], corpus.SPLITS)
            self.assertIsNone(row["target"], "generation must never write a label")
            self.assertIsNone(row["teacher"])
            self.assertGreaterEqual(len(row["candidates"]), 2)
            if row["band"] == "t3":
                self.assertIn(row["tier_target"], row["candidates"])
                self.assertNotEqual(row["tier_target"], row["intended"])
            else:
                self.assertIsNone(row["tier_target"])

    def test_every_band_and_every_op_has_rows(self) -> None:
        self.generate()
        rows = self.candidates()
        per_band = {b: sum(1 for r in rows if r["band"] == b) for b in corpus.BANDS}
        for band, count in per_band.items():
            self.assertGreater(count, 40, f"band {band} thin: {count}")
        per_op = {op: sum(1 for r in rows if r["intended"] == op)
                  for op in catalog.catalog()}
        for op, count in per_op.items():
            self.assertGreater(count, 20, f"op {op} thin: {count}")

    def test_splits_are_family_level_not_row_level(self) -> None:
        self.generate()
        rows = self.candidates()
        by_family: dict[str, set[str]] = {}
        for row in rows:
            by_family.setdefault(row["family"], set()).add(row["split"])
        for family, splits in by_family.items():
            self.assertEqual(len(splits), 1, f"family {family} straddles {splits}")
        val = {r["split"] for r in rows}
        self.assertEqual(val, {"train", "val"})

    def test_no_generated_row_restates_the_frozen_eval_set(self) -> None:
        manifest = self.generate()
        self.assertTrue(manifest["leakage"]["ok"])
        self.assertEqual(manifest["leakage"]["collisions"], [])
        self.assertEqual(manifest["leakage"]["frozen_rows"], 150)
        self.assertLess(manifest["leakage"]["max_ratio"], similarity.LEAK_RATIO)

    def test_similarity_method_is_emitted(self) -> None:
        manifest = self.generate()
        self.assertIn("jaccard", manifest["similarity"])
        self.assertIn(str(similarity.LEAK_RATIO), manifest["similarity"])

    def test_dry_run_writes_no_personal_paths(self) -> None:
        utterances.generate(self.dir, use_mining=False)
        blob = (self.dir / "manifest.json").read_text(encoding="utf-8")
        self.assertNotIn(str(pathlib.Path.home()), blob)
        self.assertNotRegex(blob, "/" + "home" + r"/[A-Za-z0-9_.-]+/")


class LeakageTest(_Tmp):
    def _frozen_row(self) -> dict[str, Any]:
        frozen = json.loads(corpus.FROZEN_PATH.read_text(encoding="utf-8"))
        return frozen[0]

    def test_a_frozen_utterance_is_refused_by_the_guard(self) -> None:
        frozen_row = self._frozen_row()
        with self.assertRaises(corpus.LeakageError):
            corpus.assert_not_frozen(frozen_row["utterance"])

    def test_a_paraphrase_of_a_frozen_utterance_is_refused(self) -> None:
        frozen_row = self._frozen_row()
        paraphrase = frozen_row["utterance"].rstrip(".") + " please"
        with self.assertRaises(corpus.LeakageError):
            corpus.assert_not_frozen(paraphrase)

    def test_the_generator_refuses_a_colliding_candidate(self) -> None:
        frozen_row = self._frozen_row()
        gate = corpus.Gate()
        self.assertEqual(gate.offer("grep the tree for a symbol nobody wrote"), "")
        self.assertIn("frozen", gate.offer(frozen_row["utterance"]))

    def test_labelling_refuses_a_candidates_file_that_smuggles_eval_content(self) -> None:
        self.generate()
        rows = self.candidates()
        rows[3]["utterance"] = self._frozen_row()["utterance"]
        corpus.write_jsonl(self.dir / "smuggled.jsonl", rows)
        with self.assertRaises(corpus.LeakageError):
            label.label_all(rows=rows, out_path=self.dir / "smuggled_labels.jsonl",
                            dry_run=True, gate=0, budget=100.0, min_interval=0.0,
                            limit=5, resume=False)

    def test_eval_split_names_are_refused(self) -> None:
        self.generate()
        row = dict(self.candidates()[0])
        row["split"] = "eval"
        with self.assertRaises(corpus.RowError):
            corpus.assert_row(row)


class LabelTest(_Tmp):
    def test_labelled_rows_are_schema_valid_and_sliceable(self) -> None:
        self.generate()
        run = label.label_all(rows=self.candidates(),
                              out_path=self.dir / "labels.jsonl", dry_run=True,
                              gate=5, budget=100.0, min_interval=0.0, limit=40,
                              resume=False)
        self.assertEqual(run["labelled"], 40)
        rows = corpus.read_jsonl(self.dir / "labels.jsonl")
        self.assertEqual(len(rows), 40)
        for row in rows:
            corpus.assert_labelled(row)
            self.assertIn(row["target"]["tool"], row["candidates"])
            self.assertTrue(0.0 <= row["target"]["conf"] <= 1.0)
            self.assertEqual(row["label_method"], cost.FIXTURE)
            self.assertEqual(row["teacher"]["schema_valid"], True)
            self.assertTrue(row["ts"].endswith("Z"))
            self.assertIn(row["band"], corpus.BANDS)
            self.assertIn(row["source"], corpus.SOURCES)

    def test_resume_labels_nothing_twice(self) -> None:
        self.generate()
        candidates = self.candidates()
        first = label.label_all(rows=candidates, out_path=self.dir / "labels.jsonl",
                               dry_run=True, gate=0, budget=100.0, min_interval=0.0,
                               limit=25, resume=False)
        self.assertEqual(first["labelled"], 25)
        second = label.label_all(rows=candidates, out_path=self.dir / "labels.jsonl",
                                dry_run=True, gate=0, budget=100.0, min_interval=0.0,
                                limit=25, resume=True)
        self.assertEqual(second["labelled"], 25)
        self.assertEqual(second["resumed_rows"], 25)
        rows = corpus.read_jsonl(self.dir / "labels.jsonl")
        self.assertEqual(len(rows), 50)
        self.assertEqual(len({r["id"] for r in rows}), 50, "an id was labelled twice")
        fresh = label.label_all(rows=candidates, out_path=self.dir / "labels.jsonl",
                               dry_run=True, gate=0, budget=100.0, min_interval=0.0,
                               limit=10, resume=True)
        self.assertEqual(fresh["labelled"], 10)
        self.assertEqual(len(corpus.read_jsonl(self.dir / "labels.jsonl")), 60)

    def test_resume_refuses_a_different_candidate_set(self) -> None:
        self.generate()
        candidates = self.candidates()
        label.label_all(rows=candidates, out_path=self.dir / "labels.jsonl",
                        dry_run=True, gate=0, budget=100.0, min_interval=0.0,
                        limit=5, resume=False)
        mutated = [dict(r) for r in candidates]
        mutated[0]["utterance"] = mutated[0]["utterance"] + " changed"
        with self.assertRaises(SystemExit):
            label.label_all(rows=mutated, out_path=self.dir / "labels.jsonl",
                            dry_run=True, gate=0, budget=100.0, min_interval=0.0,
                            limit=5, resume=True)

    def test_refusing_to_overwrite_an_existing_file_without_resume(self) -> None:
        self.generate()
        candidates = self.candidates()
        label.label_all(rows=candidates, out_path=self.dir / "labels.jsonl",
                        dry_run=True, gate=0, budget=100.0, min_interval=0.0,
                        limit=5, resume=False)
        with self.assertRaises(SystemExit):
            label.label_all(rows=candidates, out_path=self.dir / "labels.jsonl",
                            dry_run=True, gate=0, budget=100.0, min_interval=0.0,
                            limit=5, resume=False)

    def test_the_gate_stops_a_run_whose_projection_exceeds_the_budget(self) -> None:
        """A measured per-row cost above the ceiling stops the run after the gate."""

        class _Costly(teacher.FixtureTeacher):
            def choose(self, state: Any, instructions: str,
                       criteria: dict[str, str]) -> teacher.TeacherAnswer:
                answer = super().choose(state, instructions, criteria)
                answer.usage = {"cost": 0.01, "prompt_tokens": 700,
                                "completion_tokens": 20}
                return answer

        self.generate()
        with mock.patch.object(teacher, "make_teacher", return_value=_Costly()):
            run = label.label_all(rows=self.candidates(),
                                  out_path=self.dir / "labels.jsonl", dry_run=False,
                                  gate=20, budget=2.0, min_interval=0.0, resume=False)
        self.assertTrue(run["stopped"])
        self.assertEqual(run["labelled"], 20)
        self.assertEqual(len(corpus.read_jsonl(self.dir / "labels.jsonl")), 20)
        self.assertGreater(run["gate"]["projected_total_usd"], 2.0)
        self.assertEqual(run["gate"]["method"], cost.MEASURED)
        # The checkpoint is intact: resuming continues the same file.
        with mock.patch.object(teacher, "make_teacher", return_value=_Costly()):
            more = label.label_all(rows=self.candidates(),
                                   out_path=self.dir / "labels.jsonl", dry_run=False,
                                   gate=0, budget=100.0, min_interval=0.0, limit=5,
                                   resume=True)
        self.assertEqual(more["labelled"], 5)
        self.assertEqual(len(corpus.read_jsonl(self.dir / "labels.jsonl")), 25)

    def test_a_second_writer_is_refused_and_a_stale_lock_is_taken_over(self) -> None:
        self.generate()
        out = self.dir / "labels.jsonl"
        argv = ["--dry-run", "--in", str(self.dir / "candidates.jsonl"), "--out", str(out),
                "--gate", "0", "--limit", "4"]
        lock = out.with_suffix(".lock")
        lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        with self.assertRaises(SystemExit):
            label.main(argv)
        lock.write_text(json.dumps({"pid": 99_999_999}), encoding="utf-8")  # nobody home
        self.assertEqual(label.main(argv), 0)
        self.assertEqual(len(corpus.read_jsonl(out)), 4)
        self.assertFalse(lock.exists(), "the lock is released when the run ends")

    def test_the_run_stops_at_the_wall_clock_cap_and_resumes(self) -> None:
        self.generate()
        out = self.dir / "labels.jsonl"
        run = label.label_all(rows=self.candidates(), out_path=out, dry_run=True,
                              gate=0, budget=100.0, min_interval=0.0, resume=False,
                              max_seconds=0.0)
        self.assertTrue(run["timed_out"])
        self.assertEqual(run["labelled"], 0)
        run = label.label_all(rows=self.candidates(), out_path=out, dry_run=True,
                              gate=0, budget=100.0, min_interval=0.0, limit=6,
                              resume=True, max_seconds=None)
        self.assertFalse(run["timed_out"])
        self.assertEqual(run["labelled"], 6)

    def test_the_gate_proceeds_when_inside_the_budget(self) -> None:
        self.generate()
        run = label.label_all(rows=self.candidates(), out_path=self.dir / "labels.jsonl",
                              dry_run=True, gate=20, budget=100.0, min_interval=0.0,
                              limit=60, resume=False)
        self.assertFalse(run["stopped"])
        self.assertEqual(run["labelled"], 60)
        self.assertEqual(run["gate"]["labelled"], 20)
        self.assertGreater(run["gate"]["remaining"], 0)
        bands_labeled = {r["band"] for r in corpus.read_jsonl(self.dir / "labels.jsonl")}
        self.assertEqual(bands_labeled, set(corpus.BANDS), "a truncated run stays band-balanced")


class HostedTeacherTest(unittest.TestCase):
    """The hosted path against a stub: usage captured, junk refused, 429 not a row."""

    def setUp(self) -> None:
        self.conf = load()["decide"]
        self._saved = self.conf["openrouter"]
        self.addCleanup(self._restore)
        patcher = mock.patch.dict(os.environ, {"TI_OPENROUTER_API_KEY": FAKE_KEY})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _restore(self) -> None:
        self.conf["openrouter"] = self._saved

    def hosted(self, stub: _Stub) -> teacher.HostedTeacher:
        self.conf["openrouter"] = dict(self._saved, base_url=stub.url)
        impl = teacher.HostedTeacher()
        impl.backoff_s = 0.0
        return impl

    @staticmethod
    def _row() -> dict[str, Any]:
        return templates.build_row(
            band="t1", source="template", op="web_fetch", rival=None,
            family="t1:web_fetch:01", text="fetch the article at that address",
            index=1)

    def test_the_request_asks_for_the_charge_and_records_the_reply(self) -> None:
        usage = {"prompt_tokens": 700, "completion_tokens": 18, "cost": 0.00021}
        with _Stub(_reply('{"choice": "web_fetch", "confidence": 0.91}', usage)) as stub:
            answer = self.hosted(stub).choose(
                self._row()["state"], templates.ROUTE_INSTRUCTION,
                self._row()["candidates"])
        self.assertEqual(stub.bodies[0]["usage"], {"include": True})
        self.assertEqual(answer.choice, "web_fetch")
        self.assertEqual(answer.usage["cost"], 0.00021)
        self.assertEqual(answer.provider, "Relace")
        self.assertIn("web_fetch", answer.raw)
        usd, method = cost.row_cost(answer.usage, 2000, 40)
        self.assertEqual(method, cost.MEASURED)
        self.assertAlmostEqual(usd, 0.00021)
        self.assertNotIn(FAKE_KEY, json.dumps(answer.teacher()))

    def test_a_choice_outside_the_candidates_is_a_refusal_not_a_row(self) -> None:
        with _Stub(_reply('{"choice": "web_fetch_all_the_things", "confidence": 0.9}')) as stub:
            with self.assertRaises(teacher.TeacherError):
                self.hosted(stub).choose(self._row()["state"],
                                         templates.ROUTE_INSTRUCTION,
                                         self._row()["candidates"])

    def test_a_prose_reply_is_a_refusal_not_a_row(self) -> None:
        with _Stub(_reply("I would fetch that page for you.")) as stub:
            with self.assertRaises(teacher.TeacherError):
                self.hosted(stub).choose(self._row()["state"],
                                         templates.ROUTE_INSTRUCTION,
                                         self._row()["candidates"])

    def test_a_429_is_retried_and_never_becomes_a_row(self) -> None:
        with _Stub((429, b'{"error": "rate limited"}')) as stub:
            with self.assertRaises(teacher.TeacherError) as caught:
                self.hosted(stub).choose(self._row()["state"],
                                         templates.ROUTE_INSTRUCTION,
                                         self._row()["candidates"])
        self.assertTrue(teacher.is_transient(caught.exception))
        self.assertEqual(len(stub.bodies), 3, "the backend's own retries should run")

    def test_a_transient_failure_is_deferred_and_then_retried(self) -> None:
        self.generate_dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.generate_dir,
                                                            ignore_errors=True))
        utterances.generate(self.generate_dir, use_mining=False)
        rows = corpus.read_jsonl(self.generate_dir / "candidates.jsonl")[:3]
        replies = [(429, b'{"error":"rate limited"}'),
                   (429, b'{"error":"rate limited"}'),
                   (429, b'{"error":"rate limited"}')]
        target = rows[0]["candidates"]
        choice = sorted(target)[0]
        with _Stub(*replies, _reply(json.dumps({"choice": choice, "confidence": 0.6}),
                                    {"prompt_tokens": 700, "completion_tokens": 18,
                                     "cost": 0.00019})) as stub:
            impl = self.hosted(stub)
            with mock.patch.object(teacher, "make_teacher", return_value=impl):
                run = label.label_all(rows=rows,
                                      out_path=self.generate_dir / "labels.jsonl",
                                      dry_run=False, gate=0, budget=100.0,
                                      min_interval=0.0, limit=1, resume=False)
        # Row 1 exhausted its retries and was deferred; the deferred pass then got it.
        self.assertEqual(run["labelled"], 1)
        self.assertEqual(run["deferred"], [])
        written = corpus.read_jsonl(self.generate_dir / "labels.jsonl")
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["target"]["tool"], choice)
        self.assertEqual(written[0]["label_method"], cost.MEASURED)


class AuditStatsTest(_Tmp):
    def _label(self, limit: int = 60) -> pathlib.Path:
        self.generate()
        label.label_all(rows=self.candidates(), out_path=self.dir / "labels.jsonl",
                        dry_run=True, gate=0, budget=100.0, min_interval=0.0,
                        limit=limit, resume=False)
        return self.dir / "labels.jsonl"

    def test_audit_samples_every_band_and_writes_an_annotatable_csv(self) -> None:
        rows = corpus.read_jsonl(self._label(120))
        picked = audit.sample(rows, per_band=3)
        self.assertEqual(len(picked), 9)
        self.assertEqual({r["band"] for r in picked}, set(corpus.BANDS))
        path = self.dir / "audit.csv"
        audit.write_csv(picked, path)
        import csv as _csv

        with path.open(encoding="utf-8") as fh:
            reader = list(_csv.DictReader(fh))
        self.assertEqual(len(reader), 9)
        for column in ("utterance", "teacher_choice", "teacher_conf", "verdict"):
            self.assertIn(column, reader[0])
        self.assertEqual({r["verdict"] for r in reader}, {""}, "human columns start blank")

    def test_audit_is_deterministic(self) -> None:
        rows = corpus.read_jsonl(self._label(120))
        first = [r["id"] for r in audit.sample(rows, per_band=2, seed="s")]
        second = [r["id"] for r in audit.sample(rows, per_band=2, seed="s")]
        self.assertEqual(first, second)

    def test_stats_says_calibration_is_unknown_without_an_annotation(self) -> None:
        rows = corpus.read_jsonl(self._label(60))
        summary = stats.summarise(rows)
        self.assertTrue(summary["leakage"]["ok"])
        self.assertEqual(set(summary["by_band"]), set(corpus.BANDS))
        cal = stats.calibration(self.dir / "missing.csv")
        self.assertFalse(cal["known"])
        self.assertIn("UNKNOWN", cal["reason"])
        self.assertIsNone(cal.get("ece"))

    def test_stats_reports_ece_once_the_csv_is_annotated(self) -> None:
        rows = corpus.read_jsonl(self._label(120))
        picked = audit.sample(rows, per_band=4)
        path = self.dir / "audit.csv"
        audit.write_csv(picked, path)
        import csv as _csv

        with path.open(encoding="utf-8") as fh:
            table = list(_csv.DictReader(fh))
        for row in table:
            row["verdict"] = "right" if float(row["teacher_conf"]) > 0.7 else "wrong"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=audit.COLUMNS)
            writer.writeheader()
            writer.writerows(table)
        cal = stats.calibration(path)
        self.assertTrue(cal["known"])
        self.assertEqual(cal["annotated"], len(table))
        self.assertGreaterEqual(cal["ece"], 0.0)
        self.assertIn("smoke test", cal["caveat"])


class MineFilterTest(unittest.TestCase):
    def test_pasted_output_is_not_a_request(self) -> None:
        cases = {
            "## GitHub Thread Replies": "pasted-or-code",
            "READ: Complete feedback without reacting": "pasted-or-code",
            "This changes the pick rationale.": "sentence-fragment",
            "or choose a different namespace name": "sentence-fragment",
            "Resumed session": "words",
            "the neighbouring test in that edit and restored it": "no-request-lead",
            "4e857c2 chore: scaffold the fetch skill": "digits",
        }
        for text, expected in cases.items():
            self.assertEqual(mine.accept(text), expected, text)
        # A sentence the filters cannot classify is still refused, never guessed at.
        for text in ("and maybe one for the reddit thing too?",
                     "get a read-a-file plan.",
                     "the box changes — direction is structural"):
            self.assertNotEqual(mine.accept(text), "", text)

    def test_mined_rows_are_quarantined_unless_asked_for(self) -> None:
        """The miner ships, its rows do not reach the corpus without an explicit flag."""
        mined = [mine.Mined(text="can you fetch the page at that url", band="t1",
                            op="web_fetch", rival=None, seed="fetch the page",
                            session_id="s1", seq=4)]
        with tempfile.TemporaryDirectory() as tmp, \
                tempfile.TemporaryDirectory() as tmp2:
            out, out2 = pathlib.Path(tmp), pathlib.Path(tmp2)
            with mock.patch.object(mine, "mine", return_value=(mined, mine.Report())):
                manifest = utterances.generate(out, use_mining=True)
                merged = utterances.generate(out2, use_mining=True, include_mined=True)
            self.assertEqual(manifest["counts"]["source"], {"template": manifest["rows"]})
            self.assertEqual(len(corpus.read_jsonl(out / "candidates_session.jsonl")), 1)
            self.assertFalse(manifest["mining"]["merged"])
            self.assertEqual(merged["counts"]["source"],
                             {"template": merged["rows"] - 1, "session": 1})
            self.assertTrue(merged["mining"]["merged"])

    def test_a_request_shaped_unit_with_an_op_cue_passes(self) -> None:
        self.assertEqual(mine.accept("can you fetch the page at that url"), "")
        self.assertEqual(mine.accept("what files mention the retry budget?"), "")

    def test_mining_is_off_when_the_cli_is_absent(self) -> None:
        with mock.patch.object(mine, "available", return_value=False):
            rows, report = mine.mine(limit=1)
        self.assertEqual(rows, [])
        self.assertIn("not found", report.skipped)


if __name__ == "__main__":
    unittest.main()
