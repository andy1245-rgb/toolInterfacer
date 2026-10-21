#!/usr/bin/env python3
"""pyhost_qwen35.py — qwen35-family in-process serving driver for dell321.

Purpose
-------
One-stop driver for serving Qwen3.5-family GGUFs in-process on dell321 via the
pyhost recipe (LoadLibrary llama-server-impl.dll, python.exe hardlinked into
the llama.cpp bin dir), with per-run config through a local JSON sidecar so no
PowerShell quoting is needed over ssh.

NEW FILE (serving-fix run, 2026-10-17). Does not modify
.../jevResearch/lab/** or any of the research-space scripts.

Paths
-----
dell-side copy of this driver: toolInterfacerLab/qdiag.py (same logic; reads
qdiag_config.json next to itself). See service.example in the docs:
  bench/ results/dell/fix-*.json  - per-row lane artifacts
  docs/serving-fix-qwen35.md      - root-cause analysis + validation numbers

Status (2026-10-17): the driver works for serving (health/RSS/log capture
incl. load_tensors/offload lines via fd redirect at -lv 5) but the
qwen35-arch CPU-offload issue documented in the analysis doc is NOT yet
resolved: any consumer context in this llama.cpp build runs qwen35 prefill at
CPU-class speed in-process (llama-bench does GPU at 585-919 t/s on the same
GGUF). d1 (lfm2-d1) lanes remain GPU-fast through this driver.

Usage (from a machine with ssh access to dell321):
  # 1. write dell-side qdiag_config.json (mode/args/env/log)
  # 2. launch:
  ssh dell321 "C:\\...\\d1bench\\e3\\python.exe C:\\...\\toolInterfacerLab\\qdiag.py"
  # 3. /health on dell:8080, probe over the 18080 tunnel, then taskkill PIDs.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

E3PY = os.environ.get("DELL_E3_PY", "")
LAB = os.environ.get("DELL_LAB", "")
QDIAG = LAB + r"\qdiag.py"
MODELS = LAB + r"\models"

MODES = {
    "qwen35": ["-m", MODELS + r"\Qwen3.5-4B-Q4_K_M.gguf",
               "--host", "127.0.0.1", "--port", "8080", "-t", "8", "-ngl", "99"],
    "intern": [r"\Intern-Decision-4B-Q4_K_M.gguf"],
    "kev": [r"\Kev-4B-Q4_K_M.gguf"],
}


def write_config(mode: str, extra: list[str], env: dict[str, str] | None = None) -> str:
    args = list(MODES.get(mode, ["-m"]) )
    cfg = {"mode": mode,
           "args": args + extra,
           "env": env or {},
           "log": LAB + r"\logs\qdiag.log"}
    path = os.path.join(os.environ.get("DELL_MOUNT", ""), "Documents/PersonalCodingProjects/personal/toolInterfacerLab/qdiag_config.json")
    open(path, "w").write(json.dumps(cfg, indent=1))
    return path


def serve(ssh_host: str = "dell321", wait_s: int = 90) -> bool:
    p = subprocess.Popen(["ssh", ssh_host, f"{E3PY} {QDIAG}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.time() + wait_s
    while time.time() < end:
        try:
            with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=3) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(2)
    print("serve failed", file=sys.stderr)
    return False


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "qwen35"
    extra = sys.argv[2:]
    write_config(mode, extra)
    print(f"config written for {mode}; start via: ssh dell321 \"{E3PY} {QDIAG}\"")
