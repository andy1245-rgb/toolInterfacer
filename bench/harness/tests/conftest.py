"""Make the harness importable however pytest was invoked.

`python3 -m pytest` puts the cwd on sys.path; a bare `pytest` does not. The harness is a
package under `bench/`, so the repo root has to be importable for `bench.harness.*` and the
repo's own `bench.ti_bench.*` to resolve.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
