"""Registry load-order regression: importing one backend must not hide the rest.

primitives._ensure_backends_loaded() used to early-return whenever _REGISTRY was
non-empty. Backends self-register on import, so a caller that imported one
backend module first (a bench script pinning --backend, a test importing
logit_read for its entropy helper) left every other backend unregistered and
make_backend() raised KeyError for them. Import order is not a contract.

Each case runs in a fresh interpreter: this process has already imported every
backend, which is exactly the state that hides the bug.
"""
import subprocess
import sys
import unittest
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVICE = ROOT / "service"

BACKENDS = ["d1_systemone", "fixture", "heuristic_micro", "logit_read", "rule_extract"]


def _fresh_import(first: str | None) -> subprocess.CompletedProcess:
    prelude = f"from ti.decide.backends import {first}\n" if first else ""
    code = (f"import sys\nsys.path.insert(0, {str(SERVICE)!r})\n"
            f"{prelude}"
            "from ti.decide.primitives import make_backend\n"
            f"for name in {BACKENDS!r}:\n"
            "    make_backend(name)\n"
            "print('ok')\n")
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120, cwd=str(ROOT))


class RegistryLoadOrderTest(unittest.TestCase):
    def test_single_backend_import_first_still_registers_the_others(self):
        for first in BACKENDS:
            with self.subTest(first_import=first):
                proc = _fresh_import(first)
                self.assertEqual(proc.returncode, 0,
                                 f"importing {first} first broke the registry:\n{proc.stderr}")
                self.assertEqual(proc.stdout.strip(), "ok")

    def test_cold_registry_still_resolves_every_backend(self):
        proc = _fresh_import(None)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "ok")

    def test_loader_is_guarded_by_its_own_flag(self):
        import ti.decide.primitives as prim
        self.assertTrue(hasattr(prim, "_LOADED"))
        prim._ensure_backends_loaded()
        before = dict(prim._REGISTRY)
        prim._ensure_backends_loaded()
        self.assertEqual(before, dict(prim._REGISTRY))
        self.assertTrue(set(BACKENDS) <= set(prim._REGISTRY))


if __name__ == "__main__":
    unittest.main()
