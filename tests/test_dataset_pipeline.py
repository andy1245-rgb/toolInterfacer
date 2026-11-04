"""The dataset pipeline: schema rejects bad rows, the split is deterministic and
leakage-free, export round-trips, and empty input does not crash.

Every fixture here is built in a temp dir. The real logs under ``service/logs``
are runtime output that may be absent or mid-append, so a test that read them
would report on the machine rather than on the code.
"""
import ast
import itertools
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.dataset import criteria as C  # noqa: E402
from tools.dataset import cli, export as E, schema, split as S, stats as ST  # noqa: E402
from tools.dataset.ingest import ingest  # noqa: E402
from tools.dataset.paths import LOGS_ENV, OUT_ENV  # noqa: E402


_COUNTER = itertools.count()


def choice_record(**over):
    """A schema-valid routing record; override one field to make it invalid."""
    rec = {
        "id": f"id-{next(_COUNTER)}",
        "kind": "choice",
        "source": "utterances_v1",
        "group": "u:" + str(over.get("state", "state")),
        "label_source": "gold",
        "criteria_id": C.ROUTING_OPS,
        "state": "list the files",
        "criteria": dict(C.OP_DESCRIPTIONS),
        "label": "filesys_glob",
    }
    rec.update(over)
    return rec


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def hop(ts, op, intent=None):
    rec = {"ts": ts, "op": op, "params": {}, "conf": 1.0, "kind": "k", "ms": 1.0,
           "artifact_summary": "[]", "soft": {}}
    if intent is not None:
        rec["soft"] = {"intent_context": intent}
    return rec


class SchemaTest(unittest.TestCase):
    def test_valid_records_pass(self):
        self.assertEqual(schema.check_record(choice_record()), [])
        self.assertEqual(schema.check_record({
            "id": "n1", "kind": "noul", "source": "s", "group": "g",
            "label_source": "gold", "state": "state", "question": "is it read?",
            "p": 0.25}), [])
        self.assertEqual(schema.check_record({
            "id": "x1", "kind": "extract", "source": "s", "group": "g",
            "label_source": "gold", "state": {"intent": "i"},
            "schema": {"query": "string"}, "value": {"query": "example query"}}), [])

    def test_bad_rows_are_rejected_with_a_reason(self):
        cases = {
            "missing label": choice_record(label=None),
            "label not in criteria": choice_record(label="web_teleport"),
            "label wrong type": choice_record(label=7),
            "empty state string": choice_record(state=""),
            "whitespace state": choice_record(state="   "),
            "empty state object": choice_record(state={}),
            "empty criteria": choice_record(criteria={}),
            "criteria not a dict": choice_record(criteria=["filesys_glob"]),
            "criteria mismatch": choice_record(label="filesys_glob",
                                               criteria={"filesys_glob": "other text"}),
            "unknown criteria_id": choice_record(criteria_id="routing.ops_v9"),
            "unknown kind": choice_record(kind="decide"),
            "unknown label_source": choice_record(label_source="guess"),
            "empty group": choice_record(group=""),
            "missing required key": {k: v for k, v in choice_record().items()
                                     if k != "label_source"},
        }
        for name, rec in cases.items():
            with self.subTest(name):
                self.assertNotEqual(schema.check_record(rec), [], f"{name} was accepted")

    def test_noul_probability_bounds(self):
        base = {"id": "n", "kind": "noul", "source": "s", "group": "g",
                "label_source": "gold", "state": "s", "question": "q"}
        for p in (-0.01, 1.01, "0.5", True, None):
            with self.subTest(p=p):
                self.assertTrue(schema.check_record({**base, "p": p}))
        for p in (0.0, 0.5, 1.0):
            with self.subTest(p=p):
                self.assertEqual(schema.check_record({**base, "p": p}), [])

    def test_duplicate_ids_reported_with_both_positions(self):
        recs = [choice_record(id="dup", state="a"), choice_record(id="dup", state="b")]
        errs = schema.check_corpus(recs)
        self.assertEqual(len(errs), 1)
        self.assertIn("duplicate id", errs[0])
        self.assertIn("record[1]", errs[0])
        self.assertIn("record[0]", errs[0])

    def test_corpus_reports_every_problem_not_just_the_first(self):
        recs = [choice_record(state=""), choice_record(label="nope")]
        self.assertEqual(len(schema.check_corpus(recs)), 2)


class IngestTest(unittest.TestCase):
    def test_hops_rule_keeps_only_consistent_routable_intents(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = pathlib.Path(tmp) / "logs"
            write_jsonl(logs / "hops.jsonl", [
                hop(1.0, "filesys_glob", "list the files"),
                hop(1.1, "filesys_glob", "list the files"),
                hop(2.0, "filesys_glob", "read that file"),
                hop(2.1, "filesys_read", "read that file"),
                hop(3.0, "filesys_glob"),
                hop(4.0, "askback", "which one do you mean"),
            ])
            records, report = ingest(logs=logs, data=pathlib.Path(tmp) / "data")

        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["state"], "list the files")
        self.assertEqual(rec["label"], "filesys_glob")
        self.assertEqual(rec["label_source"], "observed")
        self.assertEqual(rec["meta"]["observations"], 2)
        self.assertEqual(report["skipped"]["no_intent_context (pure execution trace)"], 1)
        self.assertEqual(
            report["skipped"]["contradictory intent (executed as more than one op)"], 2)
        self.assertEqual(report["skipped"]["op 'askback' not a routable option"], 1)

    def test_live_session_uses_the_first_hop_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = pathlib.Path(tmp) / "logs"
            write_jsonl(logs / "live.jsonl", [
                {"ts": 10, "intent": "list the files", "status": "done",
                 "hops": [{"op": "filesys_glob"}, {"op": "decide_pick"}]},
                {"ts": 11, "intent": "timed out", "status": "error", "hops": []},
            ])
            records, report = ingest(logs=logs, data=pathlib.Path(tmp) / "data")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["label"], "filesys_glob")
        self.assertEqual(records[0]["group"], "s:10")
        self.assertEqual(report["skipped"]["live: no hops (errored before routing)"], 1)

    def test_frozen_label_outside_the_criteria_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = pathlib.Path(tmp) / "data"
            data.mkdir()
            (data / "utterances.json").write_text(json.dumps([
                {"utterance": "list the files", "label": "filesys_glob"},
                {"utterance": "teleport", "label": "quantum_op"},
            ]))
            records, report = ingest(logs=pathlib.Path(tmp) / "logs", data=data)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["label_source"], "gold")
        self.assertIsNone(records[0].get("band"))
        self.assertEqual(
            report["skipped"]["utterances_v1: label 'quantum_op' not an option of "
                             "routing.ops_v1"], 1)

    def test_frozen_bands_are_carried(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = pathlib.Path(tmp) / "data"
            data.mkdir()
            (data / "utterances_v2.json").write_text(json.dumps([
                {"utterance": "list the files", "label": "filesys_glob", "band": "t1"},
                {"utterance": "which one matters", "label": "decide_pick", "band": "t3",
                 "tier_target": "filesys_glob"},
            ]))
            records, _ = ingest(logs=pathlib.Path(tmp) / "logs", data=data)

        self.assertEqual([r["band"] for r in records], ["t1", "t3"])
        self.assertEqual(records[1]["meta"]["tier_target"], "filesys_glob")

    def test_records_are_unique_and_schema_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = pathlib.Path(tmp) / "logs"
            write_jsonl(logs / "hops.jsonl",
                        [hop(1.0, "filesys_glob", "list the files")] * 20)
            records, _ = ingest(logs=logs, data=pathlib.Path(tmp) / "data")

        self.assertEqual(len(records), 1, "20 identical hops are one decision, not 20")
        self.assertEqual(schema.check_corpus(records), [])


class SplitTest(unittest.TestCase):
    def _corpus(self, n=80):
        records = []
        labels = sorted(C.OP_DESCRIPTIONS)
        for i in range(n):
            label = labels[i % len(labels)]
            state = f"utterance number {i} about {label}"
            records.append(choice_record(id=f"r{i}", state=state,
                                         group=f"u:{state}", label=label))
        return records

    def test_same_seed_gives_the_same_split(self):
        records = self._corpus()
        first = S.assign_groups(records, seed=13)
        second = S.assign_groups(records, seed=13)
        self.assertEqual(first, second)

    def test_different_seed_moves_rows(self):
        records = self._corpus()
        self.assertNotEqual(S.assign_groups(records, seed=13),
                            S.assign_groups(records, seed=14))

    def test_input_order_does_not_change_the_split(self):
        records = self._corpus()
        self.assertEqual(S.assign_groups(records, seed=13),
                         S.assign_groups(list(reversed(records)), seed=13))

    def test_no_group_straddles_a_split(self):
        splits = S.split_records(self._corpus(200), seed=13)
        self.assertEqual(S.check_leakage(splits), [])

    def test_rows_from_one_utterance_stay_together(self):
        state = "list the files"
        records = [
            choice_record(id="a", state=state, group=f"u:{state}", label="filesys_glob"),
            choice_record(id="b", state=state, group=f"u:{state}", label="singular",
                          source="shape_expectations", criteria_id=C.SHAPE),
        ]
        splits = S.split_records(records, seed=13)
        self.assertEqual(S.check_leakage(splits), [])
        homes = [name for name, rows in splits.items() if rows]
        self.assertEqual(len(homes), 1)
        self.assertEqual(len(splits[homes[0]]), 2)

    def test_ratios_hold_and_every_label_survives_in_every_split(self):
        splits = S.split_records(self._corpus(400), seed=13)
        total = sum(len(rows) for rows in splits.values())
        self.assertGreater(len(splits["train"]),
                           len(splits["val"]) + len(splits["test"]))
        for name, rows in splits.items():
            with self.subTest(name):
                self.assertGreater(len(rows) / total, 0.05)
        for label in C.OP_DESCRIPTIONS:
            for name, rows in splits.items():
                with self.subTest(label=label, split=name):
                    self.assertIn(label, {r["label"] for r in rows})

    def test_leakage_check_catches_a_wilful_split(self):
        records = self._corpus(10)
        splits = {"train": records[:5], "val": records[4:6], "test": []}
        problems = S.check_leakage(splits)
        self.assertEqual(len(problems), 1)
        self.assertIn("straddles", problems[0])


class ExportTest(unittest.TestCase):
    def _export(self, out, records, **kwargs):
        splits = S.split_records(records, seed=13)
        written = E.write_export(splits, out, **kwargs)
        return splits, written

    def test_round_trip_rebuilds_every_record(self):
        records = [choice_record(id=f"r{i}", state=f"utterance {i}",
                                 group=f"u:utterance {i}",
                                 label=sorted(C.OP_DESCRIPTIONS)[i % 8])
                   for i in range(24)]
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            _, written = self._export(out, records)
            sft = E.read_sft(out / E.SFT_NAME)
            manifest = E.read_manifest(out / E.MANIFEST_NAME)
            legend = E.load_legend(out / E.LABELS_NAME)

        self.assertEqual(len(sft), len(manifest))
        self.assertEqual(len(sft), written["sft"])
        for line, meta in zip(sft, manifest):
            with self.subTest(meta["id"]):
                self.assertEqual(list(line), ["messages"], "chat line carries only messages")
                system, user, assistant = line["messages"]
                self.assertEqual([m["role"] for m in line["messages"]],
                                 ["system", "user", "assistant"])
                self.assertEqual(system["content"], C.system_prompt(C.ROUTING_OPS))
                self.assertEqual(user["content"], meta["state"])
                self.assertEqual(legend["letters"][assistant["content"]], meta["label"])
                self.assertIn(meta["label"], legend["descriptions"])

    def test_letter_is_a_single_letter_and_maps_back(self):
        for label in C.OP_DESCRIPTIONS:
            with self.subTest(label):
                letter = C.letter_for(C.ROUTING_OPS, label)
                self.assertRegex(letter, r"^[A-Z]$")
                self.assertEqual(C.label_for(C.ROUTING_OPS, letter), label)

    def test_observed_records_are_excluded_from_sft_by_default(self):
        records = [choice_record(id="obs", state="observed row",
                                 group="u:observed row", label_source="observed")]
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            _, written = self._export(out, records)
            default_rows = E.read_sft(out / E.SFT_NAME)
            _, written_all = self._export(out, records, include_observed=True)
            all_rows = E.read_sft(out / E.SFT_NAME)

        self.assertEqual(default_rows, [])
        self.assertEqual(written["skipped"], 1)
        self.assertEqual(len(all_rows), 1)
        self.assertEqual(written_all["sft"], 1)

    def test_eval_file_is_the_shape_the_bench_harness_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            self._export(out, [choice_record(id=f"r{i}", state=f"utterance {i}",
                                             group=f"u:utterance {i}") for i in range(30)])
            payload = (out / E.EVAL_TEMPLATE.format(split="test")).read_text()

        rows = json.loads(payload)                      # live_probe: json.loads(path)
        self.assertIsInstance(rows, list, "the harness iterates a JSON array")
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(row["utterance"]):
                self.assertIn(row["label"], C.OP_DESCRIPTIONS)
                self.assertTrue(row["utterance"])       # live_probe: row["utterance"]
                self.assertEqual(row["label"], C.label_for(
                    C.ROUTING_OPS, C.letter_for(C.ROUTING_OPS, row["label"])))

    def test_op_descriptions_still_match_the_bench_harness(self):
        tree = ast.parse((ROOT / "bench" / "ti_bench" / "prompts.py").read_text(encoding="utf-8"))
        found = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "OP_DESCRIPTIONS"
                    for t in node.targets):
                found = ast.literal_eval(node.value)
        self.assertIsNotNone(found, "bench/ti_bench/prompts.py no longer defines OP_DESCRIPTIONS")
        self.assertEqual(found, C.OP_DESCRIPTIONS,
                         "the dataset's option text has drifted from what the harness sends")

    def test_shape_records_do_not_reach_the_routing_export(self):
        records = [
            choice_record(id="route", state="list the files", group="u:list the files"),
            choice_record(id="shape", state="is it one file", group="u:is it one file",
                          criteria_id=C.SHAPE, criteria=dict(C.SHAPE_DESCRIPTIONS),
                          label="singular"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            _, written = self._export(out, records)
            manifest = E.read_manifest(out / E.MANIFEST_NAME)

        self.assertEqual({m["criteria_id"] for m in manifest}, {C.ROUTING_OPS})
        self.assertEqual(written["skipped"], 1)


class EmptyInputTest(unittest.TestCase):
    def _run(self, argv, logs):
        with mock.patch.dict(os.environ, {LOGS_ENV: str(logs)}):
            return cli.main(argv)

    def test_missing_logs_do_not_crash_and_report_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "out"
            logs = pathlib.Path(tmp) / "logs"          # never created
            data = pathlib.Path(tmp) / "data"
            data.mkdir()
            self.assertEqual(self._run(["--out", str(out), "ingest",
                                        "--data", str(data)], logs), 0)
            corpus = [json.loads(line) for line in
                      (out / "corpus.jsonl").read_text().splitlines() if line.strip()]
            report = json.loads((out / "ingest_report.json").read_text())
        self.assertEqual(corpus, [])
        self.assertFalse(report["sources"]["hops"]["present"])
        self.assertEqual(report["total"], 0)

    def test_verbs_on_an_empty_ingest_exit_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "out"
            logs = pathlib.Path(tmp) / "logs"
            data = pathlib.Path(tmp) / "data"
            data.mkdir()
            self._run(["--out", str(out), "ingest", "--data", str(data)], logs)
            self.assertEqual(self._run(["--out", str(out), "validate"], logs), 2)
            self.assertEqual(self._run(["--out", str(out), "split"], logs), 2)
            self.assertEqual(self._run(["--out", str(out), "stats"], logs), 2)
            self.assertEqual(self._run(["--out", str(out), "export"], logs), 2)

    def test_no_verb_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                cli.main(["--out", tmp])
        self.assertEqual(caught.exception.code, 2)

    def test_one_record_survives_the_whole_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = pathlib.Path(tmp) / "logs"
            write_jsonl(logs / "hops.jsonl", [hop(1.0, "filesys_glob", "list the files")])
            data = pathlib.Path(tmp) / "data"
            data.mkdir()
            out = pathlib.Path(tmp) / "out"
            self.assertEqual(self._run(["--out", str(out), "ingest",
                                        "--data", str(data)], logs), 0)
            self.assertEqual(self._run(["--out", str(out), "validate"], logs), 0)
            self.assertEqual(self._run(["--out", str(out), "split"], logs), 0)
            self.assertEqual(self._run(["--out", str(out), "export"], logs), 0)
            self.assertEqual(self._run(["--out", str(out), "stats"], logs), 0)
            splits = [json.loads(line) for line in
                      (out / "splits.jsonl").read_text().splitlines() if line.strip()]
            self.assertEqual([r["split"] for r in splits], ["train"])

    def test_ingest_tolerates_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = pathlib.Path(tmp) / "logs"
            logs.mkdir(parents=True)
            (logs / "hops.jsonl").write_text(
                "not json\n\n" + json.dumps(hop(1.0, "filesys_glob", "list the files")) + "\n")
            records, _ = ingest(logs=logs, data=pathlib.Path(tmp) / "data")
        self.assertEqual(len(records), 1)


class StatsTest(unittest.TestCase):
    def test_counts_label_distribution_and_band_coverage(self):
        records = [
            choice_record(id="a", state="s1", group="u:s1", label="filesys_glob",
                          band="t1"),
            choice_record(id="b", state="s2", group="u:s2", label="web_search",
                          band="t2", source="utterances_v2"),
            choice_record(id="c", state="s3", group="u:s3", label="decide_pick",
                          source="utterances_v1"),
        ]
        splits = S.split_records(records, seed=13)
        summary = ST.collect(splits)
        self.assertEqual(summary["totals"]["n"], 3)
        self.assertEqual(summary["totals"]["by_label"]["filesys_glob"], 1)
        self.assertEqual(summary["totals"]["bands"], {"t1": 1, "t2": 1, "t3": 0})
        text = ST.render(summary)
        self.assertIn("band coverage", text)
        self.assertIn("label distribution", text)
        self.assertIn("t3=0", text)

    def test_missing_band_is_called_out(self):
        records = [choice_record(id="a", state="s1", group="u:s1", band="t1")]
        summary = ST.collect(S.split_records(records, seed=13))
        self.assertTrue(any("no t2, t3 rows" in gap for gap in summary["band_gaps"]))


if __name__ == "__main__":
    unittest.main()
