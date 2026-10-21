"""rule_extract — the `extract` primitive, model-free and deterministic.

Pins the two real uses (research queries, the literal paths/urls a request
names), the degraded paths that a caller gates on, and the wiring that made
every L2 research run die at its first step: `decide.backends_in_use` had no
"extract" key, so resolve("extract") raised KeyError inside `__decide_extract__`.

The session state is deliberately the awkward one — context arrives as a dict,
a list, or a raw string depending on what the previous step stored, and {} when
the step has no source at all.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.conf import reload  # noqa: E402
from ti.decide.primitives import make_backend  # noqa: E402
from ti.decide.resolve import resolve  # noqa: E402

RESEARCH = "Extract 3-6 search queries for the intent."


def backend():
    return make_backend("rule_extract")


def extract(context, instructions, count=4):
    return backend().extract({"context": context, "instructions": instructions},
                             {"type": "value_list", "count": count})


class EnvCase(unittest.TestCase):
    """Every case runs against the shipped defaults, not a leftover overlay."""

    def setUp(self):
        import os
        self._saved = {k: os.environ.pop(k) for k in ("TI_CONF", "TI_CONF_OVERLAY")
                       if k in os.environ}
        reload()
        self.addCleanup(self._restore)

    def _restore(self):
        import os
        os.environ.update({k: v for k, v in self._saved.items() if v is not None})
        reload()


class QueryTest(EnvCase):
    def test_research_instructions_produce_a_dense_query_band(self):
        out = extract({"intent": "how do I quantize a llama model for cpu inference"},
                      RESEARCH)
        values = out["values"]
        self.assertTrue(3 <= len(values) <= 6, values)
        self.assertEqual(len(values), len(set(values)), "queries must be distinct")
        for query in values:
            self.assertTrue(query.strip(), "a blank query is a wasted web_search round")
            self.assertLessEqual(len(query.split()), 3)

    def test_stopwords_never_rank_as_query_terms(self):
        out = extract({"intent": "how do I make the most of it for the better part"}, RESEARCH)
        terms = {w for query in out["values"] for w in query.split()}
        self.assertFalse(terms & {"how", "do", "the", "most", "of", "it", "for", "a", "i"},
                         terms)

    def test_terms_carry_the_intent_not_the_instruction(self):
        # "search"/"queries" live in the instructions; they are the ask, not the topic
        out = extract({"intent": "kafka consumer lag diagnosis"}, RESEARCH)
        terms = {w for query in out["values"] for w in query.split()}
        self.assertIn("kafka", terms)
        self.assertNotIn("search", terms)
        self.assertNotIn("queries", terms)

    def test_query_confidence_is_moderate(self):
        out = extract({"intent": "kafka consumer lag diagnosis"}, RESEARCH)
        self.assertTrue(0.5 <= out["conf"] < 0.9, out["conf"])

    def test_a_thin_payload_still_returns_what_it_has(self):
        out = extract({"intent": "quantization"}, RESEARCH)
        self.assertEqual(out["values"], ["quantization"])
        self.assertLessEqual(out["conf"], 0.5)
        self.assertGreater(out["conf"], 0.1)


class UrlTest(EnvCase):
    def test_urls_are_extracted_from_nested_context_in_text_order(self):
        context = {"hits": [{"url": "https://example.com/a"},
                            {"snippet": "see (https://docs.python.org/3/library/asyncio.html)."}]}
        out = extract(context, "list the urls")
        self.assertEqual(out["values"], ["https://example.com/a",
                                         "https://docs.python.org/3/library/asyncio.html"])
        self.assertGreaterEqual(out["conf"], 0.9)

    def test_a_url_in_the_ask_is_a_literal_too(self):
        out = extract({}, "fetch https://example.com/docs")
        self.assertEqual(out["values"], ["https://example.com/docs"])

    def test_duplicate_urls_collapse(self):
        out = extract("https://a.example/x and https://a.example/x again", "list the links")
        self.assertEqual(out["values"], ["https://a.example/x"])


class PathTest(EnvCase):
    def test_paths_are_extracted_when_the_ask_names_files(self):
        out = extract("see ./service/ti/ops/matcher.py and /tmp/x/file.ext",
                      "extract the file paths mentioned in the last message")
        self.assertEqual(out["values"], ["./service/ti/ops/matcher.py",
                                         "/tmp/x/file.ext"])
        self.assertGreaterEqual(out["conf"], 0.9)

    def test_a_bare_filename_counts_only_when_the_ask_names_files(self):
        self.assertEqual(extract({"files": ["matcher.py"]},
                                 "write out the file names").get("values"), ["matcher.py"])

    def test_urls_are_not_paths(self):
        out = extract("https://example.com/a/b/c", "extract the file paths mentioned")
        self.assertNotIn("https://example.com/a/b/c", out["values"])


class MixedTest(EnvCase):
    CONTEXT = ("draft: see https://a.example/b for the spec, the local copy is "
               "./docs/spec.md and the band is 3-6")

    def test_a_path_ask_returns_paths_and_not_urls(self):
        out = extract(self.CONTEXT, "extract the file paths mentioned")
        self.assertEqual(out["values"], ["./docs/spec.md"])

    def test_an_url_ask_returns_urls_and_not_paths(self):
        out = extract(self.CONTEXT, "list the urls to search")
        self.assertEqual(out["values"], ["https://a.example/b"])

    def test_no_ask_at_all_follows_the_payload_structure(self):
        out = extract(self.CONTEXT, "")
        self.assertEqual(out["values"], ["https://a.example/b"])
        self.assertGreaterEqual(out["conf"], 0.9)

    def test_prose_with_no_literals_falls_back_to_distinctive_tokens(self):
        out = extract("the migration notes need a rewrite before friday",
                      "pull out what matters")
        self.assertIn("migration", out["values"])
        self.assertLessEqual(out["conf"], 0.5)

    def test_quoted_spans_beat_bare_tokens_in_the_fallback(self):
        out = extract('the note says "run the tests first" otherwise ignore it',
                      "pull out what matters")
        self.assertEqual(out["values"][0], "run the tests first")
        self.assertGreaterEqual(out["conf"], 0.4)


class EmptyAndDegradedTest(EnvCase):
    def test_empty_context_is_an_honest_empty_answer(self):
        out = extract({}, RESEARCH)
        self.assertEqual(out["values"], [])
        self.assertLessEqual(out["conf"], 0.1)

    def test_no_source_artifact_does_not_crash(self):
        # run_extract passes {} when the step carries no source artifact
        for context in ({}, [], "", None):
            with self.subTest(context=context):
                out = extract(context, RESEARCH)
                self.assertIsInstance(out["values"], list)

    def test_a_literal_ask_that_finds_nothing_degrades_instead_of_lying(self):
        out = extract("nothing useful in here at all", "list the urls")
        self.assertTrue(out["values"], "never return nothing silently")
        self.assertLessEqual(out["conf"], 0.3)


class CountTest(EnvCase):
    CONTEXT = {"intent": "compare kafka and rabbitmq throughput benchmarks for python clients"}

    def test_count_is_clamped_to_the_3_6_query_band(self):
        for given, expected_max in ((1, 3), (2, 3), (3, 3), (4, 4), (6, 6), (12, 6)):
            with self.subTest(count=given):
                values = extract(self.CONTEXT, RESEARCH, count=given)["values"]
                self.assertLessEqual(len(values), expected_max)
                self.assertGreaterEqual(len(values), min(3, expected_max))

    def test_a_missing_or_unusable_count_uses_the_band_floor(self):
        for schema_count in (None, "x"):
            with self.subTest(count=schema_count):
                schema = {"type": "value_list"}
                if schema_count is not None:
                    schema["count"] = schema_count
                out = backend().extract({"context": self.CONTEXT, "instructions": RESEARCH}, schema)
                self.assertEqual(len(out["values"]), 3)


class DeterminismTest(EnvCase):
    def test_same_input_twice_is_the_same_output(self):
        context = {"intent": "compare kafka and rabbitmq throughput benchmarks for python clients"}
        first = extract(context, RESEARCH, count=5)
        second = extract(context, RESEARCH, count=5)
        self.assertEqual(first, second)

    def test_two_backend_instances_agree(self):
        context = {"note": "see https://a.example/b and ./docs/spec.md"}
        self.assertEqual(make_backend("rule_extract").extract(
            {"context": context, "instructions": "list the urls"}, {"type": "value_list", "count": 3}),
            make_backend("rule_extract").extract(
            {"context": context, "instructions": "list the urls"}, {"type": "value_list", "count": 3}))


class WiringTest(EnvCase):
    def test_conf_names_rule_extract_for_the_extract_kind(self):
        from ti.decide.resolve import load_backends_in_use
        self.assertEqual(load_backends_in_use().get("extract"), "rule_extract")

    def test_resolve_extract_returns_the_backend(self):
        self.assertEqual(resolve("extract").name, "rule_extract")

    def test_the_other_kinds_are_untouched(self):
        from ti.decide.resolve import load_backends_in_use
        mapping = load_backends_in_use()
        self.assertEqual(mapping.get("choice"), "fixture")
        self.assertEqual(mapping.get("noul"), "fixture")

    def test_it_refuses_the_primitives_it_does_not_serve(self):
        be = backend()
        with self.assertRaises(NotImplementedError):
            be.choice({}, "route", {"a": "b"})
        with self.assertRaises(NotImplementedError):
            be.noul({}, "is it relevant?")


class OpSmokeTest(EnvCase):
    """`decide_extract` end to end through the op registry — no network, no model."""

    def setUp(self):
        super().setUp()
        from ti.core.artifacts import ArtifactStore
        from ti.ops.op_api import get, import_all
        import_all()
        self.store = ArtifactStore()
        self.op = get("decide_extract")

    def test_op_returns_a_value_list_the_runner_can_attach(self):
        self.store.put("ctx", "step", "digest",
                       {"intent": "how do I quantize a llama model for cpu inference"})
        kind, data, conf = self.op.run(
            {"instructions": RESEARCH, "source_artifact": "ctx", "count": 4}, self.store)
        self.assertEqual(kind, "value_list")
        self.assertTrue(3 <= len(data) <= 6, data)
        self.assertTrue(0.0 <= conf <= 1.0)

    def test_op_without_a_source_artifact_still_runs(self):
        kind, data, conf = self.op.run({"instructions": RESEARCH, "count": 3}, self.store)
        self.assertEqual((kind, data, conf), ("value_list", [], 0.1))

    def test_op_extracts_paths_for_a_filesys_handoff(self):
        self.store.put("note", "step", "digest", {"note": "the file is ./service/ti/conf/defaults.json"})
        kind, data, conf = self.op.run(
            {"instructions": "extract the file paths mentioned", "source_artifact": "note"},
            self.store)
        self.assertEqual(kind, "value_list")
        self.assertEqual(data, ["./service/ti/conf/defaults.json"])
        self.assertGreaterEqual(conf, 0.9)


if __name__ == "__main__":
    unittest.main()
