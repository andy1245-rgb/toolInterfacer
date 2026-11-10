"""sklearn_text — a scikit-learn text pipeline as an arm.

This is the seam that makes "score a classifier" a spec string instead of a new script:
``sklearn_text|vectorizer=tfidf_word_char|clf=linsvc|calibrate=sigmoid``. Everything is
in-process, so ``latency_ms`` is a real number and not a log line from a server.

Four things a reader should not have to guess:

* **The word+char union is not optional.** ``tfidf_word_char`` is a ``FeatureUnion`` of a word
  ``TfidfVectorizer`` and a ``char_wb`` one. On rows of three to eight tokens most of the signal
  is morphology and punctuation, which the word analyser throws away; the character channel is
  the first cheap win on short utterances, so it ships on by default rather than as a later idea.
* **``linsvc`` has no ``predict_proba``.** Without calibration a ``LinearSVC`` arm returns
  ``probabilities=None`` — an honest gap that excludes the row from ECE/Brier/NLL — and the
  choice comes from ``decision_function`` restricted to the offered options. With
  ``calibrate=sigmoid|isotonic`` the classifier is wrapped in ``CalibratedClassifierCV`` and
  real probabilities are produced. ``logreg`` and ``sgd`` emit them either way; calibrating
  those wraps the calibration around an already-probabilistic estimator, which is allowed and
  is sometimes what a curve you want to compare needs.
* **Restriction to the offered options is the seam.** The fitted classes are the training
  labels, which may be a superset of the options a given call offers (a 17-intent model scored
  on an 8-way subset, a model trained on the union of two projects' labels). ``predict``
  therefore keeps only the offered options that the model knows, renormalises the distribution
  over them, and argmaxes within them. Options the model never saw are dropped from the
  distribution and cannot be chosen — they are not given a fake zero mass in a dict that
  claims to cover the option set. If *none* of the offered options is in the fitted label
  space, that is an error, not a guess.
* **scikit-learn is imported inside the factory and inside ``fit``, never at module import.**
  The harness core has to import on a box with no scikit-learn, so a missing dependency is an
  actionable ``PredictorError`` naming the install command at the moment someone asks for this
  predictor, instead of a traceback from importing ``bench.harness.predictors``.

An unfitted predictor raises ``PredictorError`` from ``predict`` with the reason: grouped CV
builds a *fresh* one per fold through the registered factory, so an unfitted instance means the
fit step was skipped, not that the model is bad.
"""
from __future__ import annotations

import time

from ..protocol import PredictorError, Prediction, register
from ..rows import Row

INSTALL = "python3 -m pip install scikit-learn"

VECTORIZERS = ("tfidf_word", "tfidf_word_char", "count_word")
CLASSIFIERS = ("linsvc", "logreg", "sgd")
CALIBRATIONS = ("", "sigmoid", "isotonic")
PROBABILISTIC_CLASSIFIERS = ("logreg", "sgd")


def _missing(exc: ImportError, where: str) -> PredictorError:
    return PredictorError(
        f"sklearn_text: {where} needs scikit-learn, which is not importable ({exc}). "
        f"Install it with `{INSTALL}`. The harness core does not need scikit-learn — "
        f"protocol, rows, metrics, and the constant/fake/recorded/ensemble predictors all "
        f"run without it.")


def _require(where: str) -> None:
    try:
        import sklearn  # noqa: F401,PLC0415 - optional dependency, imported on demand
    except ImportError as exc:
        raise _missing(exc, where) from None


def _vectorizer(name: str, ngram_max: int):
    try:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
    except ImportError as exc:  # pragma: no cover - depends on the box
        raise _missing(exc, "the vectoriser") from None
    if name == "tfidf_word":
        return TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True)
    if name == "count_word":
        return CountVectorizer(ngram_range=(1, ngram_max))
    if name == "tfidf_word_char":
        return FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                     sublinear_tf=True, min_df=2)),
        ])
    raise PredictorError(f"sklearn_text: unknown vectorizer {name!r}; known: {VECTORIZERS}")


def _classifier(name: str, C: float, seed: int):
    try:
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.svm import LinearSVC
    except ImportError as exc:  # pragma: no cover - depends on the box
        raise _missing(exc, "the classifier") from None
    if name == "linsvc":
        return LinearSVC(C=C, random_state=seed)
    if name == "logreg":
        return LogisticRegression(C=C, max_iter=2000, random_state=seed)
    if name == "sgd":
        return SGDClassifier(loss="log_loss", C=C, max_iter=2000, random_state=seed)
    raise PredictorError(f"sklearn_text: unknown clf {name!r}; known: {CLASSIFIERS}")


def _argmax(known: list[str], values: dict[str, float]) -> str:
    """Highest value; ties break by the caller's option order, never by dict iteration."""
    return min(known, key=lambda option: (-values[option], known.index(option), option))


class SklearnTextPredictor:
    """A ``vectoriser -> classifier`` pipeline, fitted on ``row.text`` -> ``row.label``."""

    def __init__(self, vectorizer: str = "tfidf_word", clf: str = "linsvc",
                 calibrate: str = "", C: float = 1.0, ngram_max: int = 2, seed: int = 0):
        self.vectorizer = vectorizer
        self.clf = clf
        self.calibrate = calibrate
        self.C = C
        self.ngram_max = ngram_max
        self.seed = seed
        self.pipeline = None
        self.classes_: tuple[str, ...] = ()
        self.fit_rows = 0
        tail = f":cal={calibrate}" if calibrate else ""
        self.name = f"sklearn_text:{vectorizer}:{clf}{tail}"

    @property
    def can_emit_probabilities(self) -> bool:
        return bool(self.calibrate) or self.clf in PROBABILISTIC_CLASSIFIERS

    def _build_pipeline(self):
        try:
            from sklearn.pipeline import Pipeline
        except ImportError as exc:  # pragma: no cover - depends on the box
            raise _missing(exc, "the pipeline") from None
        estimator = _classifier(self.clf, self.C, self.seed)
        if self.calibrate:
            try:
                from sklearn.calibration import CalibratedClassifierCV
            except ImportError as exc:  # pragma: no cover - depends on the box
                raise _missing(exc, "calibrate=") from None
            estimator = CalibratedClassifierCV(estimator, method=self.calibrate, cv=3)
        return Pipeline([("vec", _vectorizer(self.vectorizer, self.ngram_max)),
                         ("clf", estimator)])

    def fit(self, rows: list[Row]) -> None:
        if not rows:
            raise PredictorError("sklearn_text: fit needs at least one row")
        labels = [r.label for r in rows]
        distinct = sorted(set(labels))
        if len(distinct) < 2:
            raise PredictorError(
                f"sklearn_text: fit needs at least two labels and got {distinct} — a one-class "
                f"fit produces a model whose whole option distribution is degenerate")
        pipeline = self._build_pipeline()
        try:
            pipeline.fit([r.text for r in rows], labels)
        except Exception as exc:
            raise PredictorError(
                f"sklearn_text: fit failed for vectorizer={self.vectorizer} clf={self.clf} "
                f"calibrate={self.calibrate or 'none'} over {len(rows)} rows / {len(distinct)} "
                f"labels: {type(exc).__name__}: {exc}") from exc
        self.pipeline = pipeline
        self.classes_ = tuple(str(c) for c in pipeline.classes_)
        self.fit_rows = len(rows)

    def _guarded_classes(self) -> tuple[str, ...]:
        if self.pipeline is None or not self.classes_:
            raise PredictorError(
                f"sklearn_text({self.name}): predict before fit — call fit(rows) first. Grouped "
                f"CV builds a fresh unfitted predictor per fold through the registered factory, "
                f"so an unfitted instance here means the fit step was skipped.")
        return self.classes_

    def _scores(self, text: str) -> dict[str, float]:
        """``decision_function`` mapped onto class names, including the binary shape."""
        scores = self.pipeline.decision_function([text])[0]
        if hasattr(scores, "__len__"):
            return {c: float(v) for c, v in zip(self.classes_, scores)}
        # a two-class LinearSVC returns one margin for classes_[1]; the negative is classes_[0]
        return {self.classes_[0]: -float(scores), self.classes_[1]: float(scores)}

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        t0 = time.perf_counter()
        classes = self._guarded_classes()
        offered = tuple(options)
        if not offered:
            raise PredictorError("sklearn_text: predict got an empty option set")
        known = [o for o in offered if o in classes]
        if not known:
            raise PredictorError(
                f"sklearn_text({self.name}): none of the {len(offered)} offered options is in "
                f"the fitted label space {list(classes)} — an option set the model was never "
                f"trained on cannot be scored")
        if self.can_emit_probabilities:
            raw = {c: float(p) for c, p in
                   zip(classes, self.pipeline.predict_proba([text])[0])}
            total = sum(raw[o] for o in known)
            if not total > 0.0:
                raise PredictorError(
                    f"sklearn_text({self.name}): the model put no probability mass on any "
                    f"offered option {known} (fitted over {len(classes)} classes), so the "
                    f"restricted distribution is undefined rather than uniform")
            probabilities = {o: raw[o] / total for o in known}
            choice = _argmax(known, probabilities)
        else:
            raw = self._scores(text)
            choice = _argmax(known, raw)
            probabilities = None
        return Prediction(choice=choice, probabilities=probabilities,
                          latency_ms=(time.perf_counter() - t0) * 1000.0)


@register("sklearn_text")
def build(vectorizer: str = "tfidf_word", clf: str = "linsvc", calibrate: str = "",
          C: str = "1.0", ngram_max: str = "2", seed: str = "0", **_kw) -> SklearnTextPredictor:
    if vectorizer not in VECTORIZERS:
        raise PredictorError(
            f"sklearn_text: unknown vectorizer {vectorizer!r}; known: {', '.join(VECTORIZERS)}")
    if clf not in CLASSIFIERS:
        raise PredictorError(
            f"sklearn_text: unknown clf {clf!r}; known: {', '.join(CLASSIFIERS)}")
    if calibrate not in CALIBRATIONS:
        raise PredictorError(
            f"sklearn_text: unknown calibrate {calibrate!r}; known: "
            f"{', '.join(repr(c) for c in CALIBRATIONS)}")
    try:
        c_value = float(C)
        ngrams = int(ngram_max)
        seed_value = int(seed)
    except ValueError as exc:
        raise PredictorError(
            f"sklearn_text: C={C!r} ngram_max={ngram_max!r} seed={seed!r} must be numeric: "
            f"{exc}") from exc
    if ngrams < 1:
        raise PredictorError(f"sklearn_text: ngram_max={ngrams} must be at least 1")
    _require("building a spec")
    return SklearnTextPredictor(vectorizer=vectorizer, clf=clf, calibrate=calibrate,
                                C=c_value, ngram_max=ngrams, seed=seed_value)
