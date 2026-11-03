"""guard — prove the endpoint is what the arm says, before any probe traffic.

`/health` answers ``ok`` for *any* llama-server, so a lane that gates only on
health can probe a stale server squatting the port. That is exactly how four
result files titled "Qwen3.5-4B" came to hold d1-3B-Q4 numbers
(docs/model-sweep-consolidated.md, CORRECTION). This module is the harness-side
gate in front of that failure, reusing ``bench/model_guard.py``'s ``/props``
assertion rather than re-implementing it.

Three rules, in the order they run:

1. A ``hosted`` arm asserts its API key is in the environment. A missing key does
   not fail loudly downstream — ``pair`` keeps answering from stage1 (policy rule
   5) and ``openrouter`` returns no choice — so the run would quietly measure a
   different arm under the same name.
2. A ``llama-server`` arm asserts ``/props`` against the expected gguf basename
   (plus sha/size when given) and aborts on mismatch, before the backend is even
   constructed.
3. An arm that cannot say which model it expects is refused unless the caller
   passes ``allow_unasserted``. That asymmetry is deliberate: the phantom-lane
   bug was caused by an unasserted lane, so reproducing it now takes a flag
   somebody has to type.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any, TextIO

from model_guard import NO_EXPECT_WARNING, ModelAssertionError, assert_model_match, served_model
from ti.conf import load as load_conf

from .arms import Arm, conf_value

TIMEOUT_S = 10


class GuardError(Exception):
    """The arm's endpoint is not (yet) the thing this run intends to measure."""


def _get_json(url: str) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
            return json.loads(resp.read())
    except (OSError, ValueError) as e:  # unreachable, HTTP error, or a body that is not JSON
        raise GuardError(f"cannot read {url}: {e}") from e


def api_key_env() -> str:
    """The environment variable the hosted backend reads its key from."""
    return str((load_conf().get("decide", {}).get("openrouter") or {})
               .get("key_env") or "OPENROUTER_API_KEY")


def preflight(arm: Arm, expect_model: str = "", allow_unasserted: bool = False,
              stream: TextIO | None = None) -> dict:
    """Assert the arm is runnable, or raise GuardError. Returns the report block."""
    stream = stream if stream is not None else sys.stderr
    expected = expect_model or arm.expect_model
    block: dict[str, Any] = {"required": list(arm.requires), "checked": False,
                             "expect_model": expected, "served_basename": "",
                             "source": "", "ftype": "", "build_info": "",
                             "api_key_env": "", "api_key_present": None}

    if arm.needs_hosted:
        env = api_key_env()
        block["api_key_env"] = env
        block["api_key_present"] = bool(os.environ.get(env))
        if not block["api_key_present"]:
            raise GuardError(
                f"arm {arm.name!r} needs a hosted key but {env} is unset — refusing to "
                f"measure a degraded arm under the same name")

    if not arm.needs_llama:
        return block

    url = conf_value(arm.endpoint)
    if not url:
        raise GuardError(f"arm {arm.name!r}: {arm.endpoint} is empty in the config")
    health = _get_json(url.rstrip("/") + "/health")
    if str(health.get("status")) != "ok":
        raise GuardError(f"arm {arm.name!r}: {url}/health is not ok: {health}")
    source = url.rstrip("/") + "/props"
    props = _get_json(source)
    block["source"] = source

    if not expected:
        if not allow_unasserted:
            raise GuardError(
                f"arm {arm.name!r} declares no expect_model, so the served model cannot be "
                "asserted (see docs/model-sweep-consolidated.md, CORRECTION). Set "
                "expect_model in the arm spec or pass --expect-model; to probe an "
                "unasserted server deliberately, pass --allow-unasserted-model")
        print(NO_EXPECT_WARNING, file=stream)
        served = served_model(props)
        block.update({"served_basename": served["basename"], "ftype": served["ftype"],
                      "build_info": served["build_info"], "served_path": served["path"]})
        return block

    try:
        served = assert_model_match(props, expected, arm.expect_sha, arm.expect_size, source)
    except ModelAssertionError as e:
        raise GuardError(str(e)) from e
    block.update({"checked": True, "served_basename": served["basename"],
                  "ftype": served["ftype"], "build_info": served["build_info"],
                  "served_path": served["path"]})
    return block
