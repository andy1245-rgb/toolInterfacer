"""torch_text — inference over a fine-tuned classification head.

The arm this makes possible: an encoder someone already fine-tuned in its owning project is
scored by the same harness, on the same rows, with the same metrics, as a scikit-learn pipeline
or a llama-server. Inference only — training stays where the corpus and the GPU are.

Checkpoint contract (a ``torch.save``d dict, written so it loads with ``weights_only=True``;
only tensors, strings, ints and dicts, so nothing has to be unpickled into code)::

    torch.save({
        "state_dict": {
            "head.weight": torch.Tensor[n_labels, hidden],   # required
            "head.bias":   torch.Tensor[n_labels],           # required
            "encoder.<key>": ...,                            # optional, <key> is exactly the
        },                                                   # AutoModel state_dict key
        "labels": ["op_a", "op_b", ...],   # head output order: logit i belongs to labels[i]
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",  # optional, overrides the spec
        "pooling": "mean",                # "mean" or "cls"
        "max_length": 128,                # optional
    }, path)

``checkpoint`` may be that ``.pt`` file or a directory containing ``model.pt``/``head.pt``.
Loading always uses ``map_location="cpu"`` and ``weights_only=True``, so a checkpoint saved on
the training box's GPU runs here. ``encoder.*`` tensors are applied with ``strict=False`` and the
counts kept on ``encoder_state``; a state dict whose encoder keys *all* fail to match is an
error, because a silently un-updated encoder is a silently different model.

``predict`` encodes (tokenizer + encoder, ``torch.no_grad()``), pools, runs the linear head,
softmaxes, and then — exactly as ``sklearn_text`` does — keeps only the offered options the
head knows, renormalises over them, and argmaxes within them. The head's label list may be a
superset of a call's options; options the head never saw are dropped rather than given a fake
zero mass.

``fit`` is a hook, not a training loop. Pass ``trainer=module:function`` in the spec and the
harness calls ``trainer(predictor, rows)`` (imported lazily by dotted path, never ``eval``'d).
With no trainer, ``fit`` raises ``NotImplementedError`` that says training stays in the owning
project and points at ``docs/harness.md``. A trainer may rewrite the checkpoint; the predictor
drops its loaded state after the call so the next ``predict`` reloads.

torch and transformers are imported inside the factory and inside the load path, never at module
import: the harness core has to import on a box with neither, and a missing dependency should be
an actionable ``PredictorError`` naming the install command at the moment it is asked for.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..protocol import PredictorError, Prediction, register
from ..rows import Row

INSTALL = "python3 -m pip install torch transformers"
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
POOLINGS = ("mean", "cls")
CHECKPOINT_NAMES = ("model.pt", "head.pt")


def _missing(exc: ImportError, where: str) -> PredictorError:
    return PredictorError(
        f"torch_text: {where} needs torch/transformers, which are not importable ({exc}). "
        f"Install them with `{INSTALL}`. The harness core does not need either — protocol, "
        f"rows, metrics, and the constant/fake/recorded/ensemble predictors run without them.")


def _require(where: str) -> None:
    try:
        import torch  # noqa: F401,PLC0415 - optional dependency, imported on demand
    except ImportError as exc:
        raise _missing(exc, where) from None


def _argmax(known: list[str], values: dict[str, float]) -> str:
    """Highest value; ties break by the caller's option order, never by dict iteration."""
    return min(known, key=lambda option: (-values[option], known.index(option), option))


def _checkpoint_file(checkpoint: str) -> Path:
    path = Path(checkpoint) if checkpoint else Path("")
    if not str(checkpoint).strip():
        raise PredictorError(
            "torch_text: checkpoint= is required — a .pt file, or a directory holding one of "
            f"{', '.join(CHECKPOINT_NAMES)}")
    if path.is_dir():
        for name in CHECKPOINT_NAMES:
            if (path / name).is_file():
                return path / name
        raise PredictorError(
            f"torch_text: {path} is a directory with no {' or '.join(CHECKPOINT_NAMES)} in it")
    if not path.is_file():
        raise PredictorError(f"torch_text: checkpoint {path} does not exist")
    return path


class TorchTextPredictor:
    """A frozen (or partly updated) encoder plus a linear head, scored in-process on CPU."""

    def __init__(self, checkpoint: str, encoder: str = DEFAULT_ENCODER, device: str = "cpu",
                 max_length: int = 128, trainer: str = ""):
        self.checkpoint = _checkpoint_file(checkpoint)
        self.encoder = encoder or DEFAULT_ENCODER
        self.device = device or "cpu"
        self.max_length = int(max_length)
        self.trainer_spec = trainer
        self.trainer_fn = None
        self.labels: tuple[str, ...] = ()
        self.pooling = "mean"
        self.encoder_state: dict[str, int] = {}
        self._torch = None
        self.tokenizer = None
        self.encoder_model = None
        self.head = None
        self.name = f"torch_text:{self.checkpoint.stem}"

    def __repr__(self) -> str:  # keeps a pytest failure readable
        return f"TorchTextPredictor({self.name!r}, encoder={self.encoder!r})"

    # ---------------------------------------------------------------- loading

    def _load(self) -> None:
        try:
            import torch  # noqa: PLC0415
            from torch import nn  # noqa: PLC0415
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise _missing(exc, "loading a checkpoint") from None
        try:
            blob = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True)
        except TypeError:
            # torch older than the weights_only guard: same contract, only the guard is absent
            try:
                blob = torch.load(str(self.checkpoint), map_location="cpu")
            except Exception as exc:
                raise PredictorError(
                    f"torch_text: cannot load {self.checkpoint}: {type(exc).__name__}: {exc}"
                ) from exc
        except Exception as exc:
            raise PredictorError(
                f"torch_text: cannot load {self.checkpoint}: {type(exc).__name__}: {exc}. The "
                f"contract is a dict with state_dict/labels/encoder/pooling, saveable with "
                f"weights_only=True (tensors, str, int, dict only)") from exc
        state_dict, labels, pooling, encoder_name, max_length = self._validate(blob)
        try:
            tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            encoder_model = AutoModel.from_pretrained(encoder_name)
        except Exception as exc:
            raise PredictorError(
                f"torch_text: cannot build encoder {encoder_name!r}: {type(exc).__name__}: "
                f"{exc}") from exc
        head_keys = {k[len("head."):]: v for k, v in state_dict.items()
                     if k.startswith("head.")}
        hidden = int(getattr(encoder_model.config, "hidden_size", 0) or 0)
        if not hidden:
            raise PredictorError(
                f"torch_text: encoder {encoder_name!r} reports no hidden_size, so the head's "
                f"input width cannot be checked against it")
        head = nn.Linear(hidden, len(labels))
        try:
            head.load_state_dict(head_keys)
        except RuntimeError as exc:
            raise PredictorError(
                f"torch_text: head in {self.checkpoint} does not fit a Linear({hidden}, "
                f"{len(labels)}): {exc}") from exc
        encoder_state = self._load_encoder_state(encoder_model, state_dict)
        try:
            encoder_model.eval()
            head.eval()
            encoder_model.to(self.device)
            head.to(self.device)
        except Exception as exc:
            raise PredictorError(
                f"torch_text: cannot place the model on device {self.device!r}: "
                f"{type(exc).__name__}: {exc}") from exc
        self._torch = torch
        self.tokenizer = tokenizer
        self.encoder_model = encoder_model
        self.head = head
        self.labels = labels
        self.pooling = pooling
        self.max_length = max_length
        self.encoder_state = encoder_state

    def _validate(self, blob) -> tuple[dict, tuple[str, ...], str, str, int]:
        where = self.checkpoint
        if not isinstance(blob, dict):
            raise PredictorError(
                f"torch_text: {where} does not hold a dict; the contract is "
                f"{{'state_dict': ..., 'labels': [...], 'encoder': ..., 'pooling': ...}}")
        state_dict = blob.get("state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise PredictorError(f"torch_text: {where} has no non-empty state_dict")
        stray = [k for k in state_dict
                 if not isinstance(k, str) or not k.startswith(("head.", "encoder."))]
        if stray:
            raise PredictorError(
                f"torch_text: {where} has state_dict keys outside head.* / encoder.*: "
                f"{sorted(stray)[:5]} — a checkpoint the head cannot claim is not one this "
                f"predictor can score")
        if not any(k.startswith("head.") for k in state_dict):
            raise PredictorError(
                f"torch_text: {where} has no head.* tensors; the required keys are "
                f"head.weight and head.bias")
        raw_labels = blob.get("labels")
        if not isinstance(raw_labels, (list, tuple)) or not raw_labels:
            raise PredictorError(
                f"torch_text: {where} has no labels list, so logit i cannot be named")
        if not all(isinstance(lab, str) and lab for lab in raw_labels):
            raise PredictorError(f"torch_text: {where} labels must be non-empty strings")
        labels = tuple(raw_labels)
        if len(set(labels)) != len(labels):
            raise PredictorError(
                f"torch_text: {where} labels repeat ({len(labels)} labels, "
                f"{len(set(labels))} distinct)")
        pooling = blob.get("pooling") or "mean"
        if pooling not in POOLINGS:
            raise PredictorError(
                f"torch_text: {where} pooling={pooling!r}; known: {', '.join(POOLINGS)}")
        encoder_name = blob.get("encoder") or self.encoder
        if not isinstance(encoder_name, str) or not encoder_name:
            raise PredictorError(f"torch_text: {where} encoder must be a non-empty string")
        try:
            max_length = int(blob.get("max_length") or self.max_length)
        except (TypeError, ValueError) as exc:
            raise PredictorError(f"torch_text: {where} max_length is not an int: {exc}") from exc
        return state_dict, labels, pooling, encoder_name, max_length

    def _load_encoder_state(self, encoder_model, state_dict: dict) -> dict[str, int]:
        encoder_keys = {k[len("encoder."):]: v for k, v in state_dict.items()
                        if k.startswith("encoder.")}
        if not encoder_keys:
            return {"provided": 0, "applied": 0, "unexpected": 0, "missing": 0}
        missing, unexpected = encoder_model.load_state_dict(encoder_keys, strict=False)
        if unexpected and len(unexpected) >= len(encoder_keys):
            raise PredictorError(
                f"torch_text: none of the {len(encoder_keys)} encoder.* tensors in "
                f"{self.checkpoint} matches this encoder's state_dict (first unexpected: "
                f"{list(unexpected)[:3]}) — a checkpoint whose weights are silently ignored is "
                f"a different model than the one that was trained")
        return {"provided": len(encoder_keys), "applied": len(encoder_keys) - len(unexpected),
                "unexpected": len(unexpected), "missing": len(missing)}

    def _ensure_loaded(self) -> None:
        if self.head is None:
            self._load()

    def _reset(self) -> None:
        self._torch = None
        self.tokenizer = None
        self.encoder_model = None
        self.head = None

    # ---------------------------------------------------------------- inference

    def _probabilities(self, text: str) -> dict[str, float]:
        torch = self._torch
        encoded = self.tokenizer([text], truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = self.encoder_model(**encoded).last_hidden_state
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            logits = self.head(pooled)
            weights = torch.softmax(logits.float(), dim=-1)[0].tolist()
        return {label: float(p) for label, p in zip(self.labels, weights)}

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        t0 = time.perf_counter()
        offered = tuple(options)
        if not offered:
            raise PredictorError("torch_text: predict got an empty option set")
        self._ensure_loaded()
        known = [o for o in offered if o in self.labels]
        if not known:
            raise PredictorError(
                f"torch_text({self.name}): none of the {len(offered)} offered options is a head "
                f"label (head has {len(self.labels)}: {list(self.labels)[:6]}...) — an option "
                f"set the head never saw cannot be scored")
        raw = self._probabilities(text)
        total = sum(raw[o] for o in known)
        if not total > 0.0:
            raise PredictorError(
                f"torch_text({self.name}): the head put no probability mass on any offered "
                f"option {known}, so the restricted distribution is undefined rather than "
                f"uniform")
        probabilities = {o: raw[o] / total for o in known}
        return Prediction(choice=_argmax(known, probabilities), probabilities=probabilities,
                          latency_ms=(time.perf_counter() - t0) * 1000.0)

    # ---------------------------------------------------------------- training hook

    def _trainer(self):
        if not self.trainer_spec:
            return None
        if self.trainer_fn is not None:
            return self.trainer_fn
        module_name, _, func_name = self.trainer_spec.partition(":")
        if not module_name or not func_name:
            raise PredictorError(
                f"torch_text: trainer={self.trainer_spec!r} is not a dotted path; expected "
                f"module:function, e.g. trainer=mypkg.train:fit_head")
        try:
            import importlib  # noqa: PLC0415
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise PredictorError(
                f"torch_text: cannot import trainer module {module_name!r}: {exc}") from None
        func = getattr(module, func_name, None)
        if not callable(func):
            raise PredictorError(
                f"torch_text: {module_name}:{func_name} is missing or not callable")
        self.trainer_fn = func
        return func

    def fit(self, rows: list[Row]) -> None:
        trainer = self._trainer()
        if trainer is None:
            raise NotImplementedError(
                f"torch_text({self.name}): no trainer. Training stays in the owning project — "
                f"the corpus, the schedule and the checkpoint contract live there, not in a "
                f"harness arm. Pass trainer=module:function in the spec and the harness will "
                f"call trainer(predictor, rows); see docs/harness.md")
        if not rows:
            raise PredictorError("torch_text: fit was handed no rows")
        trainer(self, rows)
        # a trainer is free to write a new checkpoint; drop the loaded state so the next
        # predict reads what it produced instead of the weights it started from
        self._reset()


@register("torch_text")
def build(checkpoint: str = "", encoder: str = DEFAULT_ENCODER, device: str = "cpu",
          max_length: str = "128", trainer: str = "", **_kw) -> TorchTextPredictor:
    try:
        max_len = int(max_length)
    except ValueError as exc:
        raise PredictorError(
            f"torch_text: max_length={max_length!r} must be an int: {exc}") from exc
    if max_len < 1:
        raise PredictorError(f"torch_text: max_length={max_len} must be at least 1")
    predictor = TorchTextPredictor(checkpoint=checkpoint, encoder=encoder, device=device,
                                   max_length=max_len, trainer=trainer)
    _require("building a spec")
    return predictor
