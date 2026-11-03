"""arms — the arm spec (`bench/arms.json`) read as data.

An arm is one scorer's worth of configuration: a registered backend name, what
that backend needs to run, and the identity the run has to prove. Nothing in this
module names a backend, so adding one is an entry in the JSON and no code change.

Arm fields:

  ``backend``       registered decide backend, built through ``make_backend``
  ``requires``      ``[]`` offline/in-process · ``"llama-server"`` a live
                    llama.cpp at ``endpoint`` · ``"hosted"`` a hosted API whose
                    key must be in the environment
  ``expect_model``  gguf basename the live server must be serving (/props)
  ``expect_sha`` / ``expect_size``  optional exact identity checks
  ``endpoint``      dotted conf path holding the server URL, so ``--base-url``
                    repoints the arm without editing the JSON
  ``override``      dotted conf paths applied before the backend is built; this
                    is how a variant (tau 0.86, a different stage2) becomes a
                    named arm instead of a CLI flag nobody recorded

There is deliberately no "cache" or "pre-gate" field. Arms are built with
``make_backend`` directly, so ``FastPathBackend`` cannot answer a row from the
decision cache or the ``heuristic_micro`` gate and put a 0 ms sample into a
latency distribution — the contamination that made
docs/decision-pair-vs-hosted.md build its arms by hand.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ti.conf import load as load_conf
from ti.decide import primitives
from ti.decide.primitives import DecideBackend, make_backend

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "arms.json"

# The closed vocabulary of "what an arm needs". A typo'd requirement would
# silently skip the guard, so it is validated at load.
NEEDS_ENDPOINT = "llama-server"
NEEDS_HOSTED = "hosted"
REQUIREMENTS = (NEEDS_ENDPOINT, NEEDS_HOSTED)

FIELDS = ("name", "backend", "requires", "expect_model", "expect_sha", "expect_size",
          "endpoint", "override")


class ArmSpecError(ValueError):
    """The arm spec on disk does not describe a runnable arm."""


@dataclass
class Arm:
    name: str
    backend: str
    requires: tuple[str, ...] = ()
    expect_model: str = ""
    expect_sha: str = ""
    expect_size: int | None = None
    endpoint: str = ""
    override: dict[str, Any] = field(default_factory=dict)

    @property
    def offline(self) -> bool:
        """True when the arm runs in-process: no endpoint, no key, no network."""
        return not self.requires

    @property
    def needs_llama(self) -> bool:
        return NEEDS_ENDPOINT in self.requires

    @property
    def needs_hosted(self) -> bool:
        return NEEDS_HOSTED in self.requires

    @property
    def real_call(self) -> bool:
        """Whether a row's answer should come from a real call, not in-process logic."""
        return bool(self.requires)

    @property
    def where(self) -> str:
        """'offline', or the endpoints the arm needs, for `list` and the report."""
        return "+".join(self.requires) if self.requires else "offline"

    def as_dict(self) -> dict:
        return {"name": self.name, "backend": self.backend, "requires": list(self.requires),
                "where": self.where, "offline": self.offline, "real_call": self.real_call,
                "expect_model": self.expect_model, "expect_sha": self.expect_sha,
                "expect_size": self.expect_size, "endpoint": self.endpoint,
                "override": dict(self.override)}


def _arm_from(name: str, spec: dict) -> Arm:
    if not isinstance(spec, dict):
        raise ArmSpecError(f"arm {name!r}: expected an object, got {type(spec).__name__}")
    unknown = set(spec) - set(FIELDS) - {"comment"}
    if unknown:
        raise ArmSpecError(f"arm {name!r}: unknown field(s) {', '.join(sorted(unknown))}")
    backend = spec.get("backend")
    if not backend:
        raise ArmSpecError(f"arm {name!r}: 'backend' is required")
    requires = spec.get("requires") or []
    if isinstance(requires, str):
        requires = [requires]
    bad = [r for r in requires if r not in REQUIREMENTS]
    if bad:
        raise ArmSpecError(f"arm {name!r}: unknown requirement(s) {', '.join(map(str, bad))}; "
                           f"known: {', '.join(REQUIREMENTS)}")
    if NEEDS_ENDPOINT in requires and not spec.get("endpoint"):
        raise ArmSpecError(f"arm {name!r}: needs a llama-server but names no 'endpoint' "
                           "(the dotted conf path holding its URL)")
    if spec.get("endpoint") and NEEDS_ENDPOINT not in requires:
        raise ArmSpecError(f"arm {name!r}: names an endpoint but does not require "
                           f"{NEEDS_ENDPOINT!r}")
    return Arm(name=name, backend=str(backend), requires=tuple(requires),
               expect_model=str(spec.get("expect_model") or ""),
               expect_sha=str(spec.get("expect_sha") or ""),
               expect_size=spec.get("expect_size"),
               endpoint=str(spec.get("endpoint") or ""),
               override=dict(spec.get("override") or {}))


def load_arms(path: str | Path | None = None) -> dict[str, Arm]:
    """Every arm in the spec file, in file order."""
    spec_path = Path(path) if path else DEFAULT_SPEC
    raw = json.loads(spec_path.read_text())
    specs = raw.get("arms")
    if not isinstance(specs, dict) or not specs:
        raise ArmSpecError(f"{spec_path}: no 'arms' object")
    return {name: _arm_from(name, spec) for name, spec in specs.items()}


def get_arm(name: str, path: str | Path | None = None) -> Arm:
    spec = load_arms(path)
    if name not in spec:
        raise SystemExit(f"unknown arm {name!r} (known: {', '.join(spec)})")
    return spec[name]


def registered_backends() -> list[str]:
    """Every backend the registry holds, whether or not an arm names it."""
    primitives._ensure_backends_loaded()
    return sorted(primitives._REGISTRY)


def _walk(cfg: dict, dotted: str, create: bool) -> tuple[dict, str] | None:
    node: Any = cfg
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            if not create:
                return None
            raise ArmSpecError(f"conf path {dotted!r}: {key!r} is not a section")
    return (node, leaf) if isinstance(node, dict) else None

def conf_value(dotted: str, cfg: dict | None = None) -> Any:
    """Read a dotted conf path ('decide.d1_systemone.base_url'); None when absent."""
    found = _walk(cfg if cfg is not None else load_conf(), dotted, create=False)
    return found[0].get(found[1]) if found else None


def set_conf(dotted: str, value: Any, cfg: dict | None = None) -> None:
    """Write a dotted conf path on the loaded config (the arm spec's ``override``)."""
    found = _walk(cfg if cfg is not None else load_conf(), dotted, create=False)
    if found is None:
        raise ArmSpecError(f"{dotted!r} has no parent section in the config")
    found[0][found[1]] = value


def apply(arm: Arm, base_url: str = "", cfg: dict | None = None) -> dict:
    """Apply the arm's overrides (and ``--base-url``) to the loaded config.

    ``ti.conf.load()`` hands out the shared cache, so this mutates the config the
    backend constructors will read — the same mechanism
    ``bench/live_probe.py --base-url`` uses. One process measures one arm.
    """
    cfg = cfg if cfg is not None else load_conf()
    for dotted, value in arm.override.items():
        set_conf(dotted, value, cfg)
    if base_url:
        if not arm.endpoint:
            raise ArmSpecError(f"arm {arm.name!r} has no endpoint to repoint; "
                               "--base-url does not apply")
        set_conf(arm.endpoint, base_url, cfg)
    return cfg


def build(arm: Arm) -> DecideBackend:
    """Construct the arm's backend. Outside ``resolve()`` on purpose: no cache, no gate."""
    return make_backend(arm.backend)
