"""classical backend: the `choice` contract, the abstain class, determinism.

The fixture corpus is synthetic and tiny on purpose — the real registry_382
corpus costs ~2 s to fit and these cases must stay cheap. The reproduction of
the recorded 93.19% is measured, not asserted here: `python3
bench/classical_baseline.py` writes `bench/results/classical_382.json` with the
model fingerprint beside the accuracy.
"""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.conf import load  # noqa: E402
from ti.decide.backends.classical import (ClassicalBackend, load_corpus,  # noqa: E402
                                          resolve_path)
from ti.decide.linear_text import fit, tokenize  # noqa: E402
from ti.decide.primitives import make_backend  # noqa: E402

CORPUS = [
    ("set a timer for ten minutes", "timer_create"),
    ("start a timer for the pasta", "timer_create"),
    ("cancel the timer", "timer_cancel"),
    ("stop the running timer", "timer_cancel"),
    ("turn on the kitchen lights", "light_set"),
    ("dim the bedroom lamps", "light_set"),
    ("what is the temperature in here", "thermostat_read"),
    ("tell me how warm the office is", "thermostat_read"),
    ("add milk to the shopping list", "list_add"),
    ("put bread on the grocery list", "list_add"),
    ("what is the capital of Portugal", "__none__"),
    ("tell me a joke", "__none__"),
]
CANDIDATES = {"timer_create": "start a timer", "timer_cancel": "stop a timer",
              "light_set": "change the lights", "thermostat_read": "read the thermostat",
              "list_add": "add to a list"}
CONFIG = {"corpus_path": "(test fixture)", "loss": "squared_hinge", "c": 1.0,
          "epochs": 20, "eta": 0.3, "ngram_max": 2, "min_df": 1,
          "sublinear_tf": True, "seed": 42, "abstain_label": "__none__",
          "temperature": 1.0}


def backend(config: dict | None = None, corpus: list[tuple[str, str]] | None = CORPUS
            ) -> ClassicalBackend:
    return ClassicalBackend(config=dict(CONFIG, **(config or {})),
                            corpus=list(corpus) if corpus is not None else None)


class ChoiceContractTest(unittest.TestCase):
    def test_probabilities_are_a_distribution_over_the_offer_set(self):
        ans = backend().choice("set a timer for 2 minutes", "route", CANDIDATES)
        self.assertAlmostEqual(sum(ans.probabilities.values()), 1.0, places=9)
        for label in CANDIDATES:
            self.assertIn(label, ans.probabilities)
        self.assertGreaterEqual(min(ans.probabilities.values()), 0.0)

    def test_confidence_is_the_peak_probability(self):
        ans = backend().choice("dim the lamps", "route", CANDIDATES)
        self.assertAlmostEqual(ans.confidence, max(ans.probabilities.values()), places=12)

    def test_routes_a_clear_utterance(self):
        ans = backend().choice("start a timer for nine minutes", "route", CANDIDATES)
        self.assertEqual(ans.choice, "timer_create")

    def test_empty_candidate_list_does_not_crash(self):
        ans = backend().choice("anything at all", "route", {})
        self.assertIsNone(ans.choice)
        self.assertEqual(ans.probabilities, {})
        self.assertEqual(ans.confidence, 0.0)

    def test_candidates_the_model_never_saw_are_not_invented(self):
        ans = backend().choice("summarise this thread", "route",
                               {"reddit_digest": "digest a subreddit"})
        self.assertIsNone(ans.choice)  # nothing on offer is scoreable
        self.assertEqual(ans.probabilities, {"reddit_digest": 1.0})

    def test_unknown_label_alongside_known_ones_keeps_negligible_mass(self):
        ans = backend().choice("turn the lights up", "route",
                               dict(CANDIDATES, reddit_digest="digest a subreddit"))
        self.assertAlmostEqual(sum(ans.probabilities.values()), 1.0, places=9)
        self.assertLess(ans.probabilities["reddit_digest"], 0.01)

    def test_serves_choice_only(self):
        with self.assertRaises(NotImplementedError):
            backend().noul("state", "is this relevant?")
        with self.assertRaises(NotImplementedError):
            backend().score("state", "how good?", ["a", "b"])


class AbstainClassTest(unittest.TestCase):
    def test_off_scaffold_utterance_abstains(self):
        ans = backend().choice("what is the capital of Portugal", "route", CANDIDATES)
        self.assertIsNone(ans.choice, "an off-scaffold question must not pick a tool")
        self.assertIn("__none__", ans.probabilities)
        self.assertEqual(max(ans.probabilities, key=ans.probabilities.get), "__none__")

    def test_abstain_competes_even_though_the_caller_never_offers_it(self):
        ans = backend().choice("tell me a joke", "route", {"light_set": "change the lights"})
        self.assertIn("__none__", ans.probabilities)
        self.assertNotIn("__none__", CANDIDATES)

    def test_a_corpus_without_an_abstain_class_puts_nothing_extra_on_the_ballot(self):
        corpus = [c for c in CORPUS if c[1] != "__none__"]
        ans = backend(corpus=corpus).choice("dim the lamps", "route", CANDIDATES)
        self.assertNotIn("__none__", ans.probabilities)
        self.assertEqual(set(ans.probabilities), set(CANDIDATES))


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_fingerprint_and_same_weights(self):
        texts = [t for t, _ in CORPUS]
        labels = [label for _, label in CORPUS]
        first = fit(texts, labels, epochs=20, eta=0.3, seed=7)
        second = fit(texts, labels, epochs=20, eta=0.3, seed=7)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.model.weights, second.model.weights)

    def test_a_different_seed_is_a_different_model(self):
        texts = [t for t, _ in CORPUS]
        labels = [label for _, label in CORPUS]
        self.assertNotEqual(fit(texts, labels, epochs=20, eta=0.3, seed=7).fingerprint,
                            fit(texts, labels, epochs=20, eta=0.3, seed=8).fingerprint)

    def test_repeated_choice_calls_agree(self):
        b = backend()
        first = b.choice("cancel the alarm", "route", CANDIDATES)
        second = b.choice("cancel the alarm", "route", CANDIDATES)
        self.assertEqual(first.choice, second.choice)
        self.assertEqual(first.probabilities, second.probabilities)


class CorpusAndConfigTest(unittest.TestCase):
    def test_shipped_config_resolves_a_relative_corpus_path(self):
        cfg = load()["decide"]["classical"]
        self.assertFalse(pathlib.Path(cfg["corpus_path"]).is_absolute())
        path = resolve_path(cfg["corpus_path"])
        self.assertTrue(path.exists(), f"{path} missing — the backend cannot fit")

    def test_registry_shaped_corpus_is_flattened_to_rows(self):
        rows = load_corpus(resolve_path(load()["decide"]["classical"]["corpus_path"]))
        self.assertEqual(len(rows), 200)
        self.assertEqual(len({label for _, label in rows}), 18)
        self.assertIn("__none__", {label for _, label in rows})

    def test_row_shaped_corpus_is_accepted(self):
        path = ROOT / "tests/data/utterances.json"
        rows = load_corpus(path)
        self.assertEqual(len(rows), 50)
        self.assertEqual(len({label for _, label in rows}), 8)

    def test_backend_registers_by_name(self):
        resolved = make_backend("classical")
        self.assertEqual(resolved.name, "classical")

    def test_tokenizer_matches_the_sklearn_default(self):
        self.assertEqual(tokenize("Turn ON the kitchen Lights!"),
                         ["turn", "on", "the", "kitchen", "lights"])
        self.assertEqual(tokenize("a b c"), [])  # single characters are dropped


class VectoriserTest(unittest.TestCase):
    def test_rows_are_l2_normalised_and_idf_is_smoothed(self):
        from ti.decide.linear_text import TfidfVectorizer
        vec = TfidfVectorizer().fit(["set a timer", "cancel the timer"])
        for _idxs, vals in vec.transform(["set a timer"]):
            self.assertAlmostEqual(sum(v * v for v in vals) ** 0.5, 1.0, places=12)

    def test_unseen_text_vectorises_to_nothing_rather_than_failing(self):
        model = backend().model()
        idxs, vals = model.vectorizer.vectorize("zzz qqq")
        self.assertEqual((idxs, vals), ([], []))


class ImportedDataTest(unittest.TestCase):
    def test_the_382_rows_keep_the_imported_shape(self):
        """Not `verbatim`: the `projects_query` slot vocabulary was substituted
        2026-11-04 (four private project names). Shape is what a reader of the
        score depends on, and `routing_382_PROVENANCE.md` records the digests on
        both sides of that edit."""
        rows = json.loads((ROOT / "tests/data/routing_382.json").read_text())
        self.assertEqual(len(rows), 382)
        self.assertEqual(len({r["tool"] for r in rows}), 18)
        self.assertEqual(sum(r["tool"] == "__none__" for r in rows), 25)
        self.assertEqual(list(rows[0]), ["text", "tool", "family"])


if __name__ == "__main__":
    unittest.main()
