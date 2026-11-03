"""The demo's lifecycle: a service that comes up, is killable, and is never orphaned.

`bench/demo.py` runs a real service in its own session, on a port it picked. Three
things about that are worth a test rather than a comment, because all three were
broken or nearly so:

- **Signals.** SIGINT unwinds through main()'s `finally`, but only once main() has
  been entered past the spawn; SIGTERM has no Python handler at all by default. A
  `timeout` or a `kill` therefore used to leave the child running in its own
  session, holding its port, with the terminal's Ctrl-C unable to reach it.
- **Two at once.** The demo picks a free port, so two runs must not land on the
  same one — and if one ever does, it must say so instead of polling the port and
  talking to whoever answers.
- **Temp files.** Both the fixture CLI's artifact and the service's overlay live in
  a directory that goes away on every exit path.

These spawn real processes, so they live here rather than beside the pure rendering
tests; the alternative is a demo nobody has proven can be stopped.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))


def _load_demo():
    spec = importlib.util.spec_from_file_location("ti_demo_lifecycle",
                                                  ROOT / "bench" / "demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ti_demo_lifecycle"] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


def _children(pid: int) -> list[int]:
    done = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
    return [int(x) for x in done.stdout.split()]


def _alive(pid: int) -> bool:
    """Is this pid a live process (not gone, not a zombie waiting to be reaped)?"""
    done = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                          capture_output=True, text=True)
    state = done.stdout.strip()
    return bool(state) and not state.startswith("Z")


def _demo_tmp_dirs(root: pathlib.Path) -> set[str]:
    """The demo's own temp dirs, under the private TMPDIR the caller gave it."""
    return {name for name in os.listdir(root)
            if name.startswith(("ti-demo-", "ti-demo-files-"))}


class ServiceLifecycleTest(unittest.TestCase):
    def test_a_service_that_dies_at_startup_is_reported_not_polled(self):
        # The old code waited the full 25 s in silence and then blamed the URL.
        log = pathlib.Path(tempfile.mkdtemp(prefix="ti-demo-log-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(log, ignore_errors=True))
        tail = log / "service.log"
        tail.write_text("ti: cannot bind 127.0.0.1:1 — Address already in use\n",
                        encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
        proc.wait(timeout=30)
        started = time.monotonic()
        with self.assertRaises(SystemExit) as caught:
            demo._wait_health("http://127.0.0.1:9/health", 25.0, proc=proc, log=tail)
        message = str(caught.exception)
        self.assertIn("exited during startup", message)
        self.assertIn("Address already in use", message)
        self.assertLess(time.monotonic() - started, 5.0, "it should not wait for the timeout")

    def test_an_interrupt_while_waiting_for_health_does_not_orphan_the_service(self):
        # The window the old code leaked in: the child exists, no atexit hook is
        # registered yet, and the interrupt unwinds straight past main()'s try.
        service = demo.Service(live=False)
        original = demo._wait_health

        def interrupted(*_args, **_kwargs):
            raise KeyboardInterrupt

        demo._wait_health = interrupted
        try:
            with self.assertRaises(KeyboardInterrupt):
                service.start()
        finally:
            demo._wait_health = original
        self.assertIsNone(service.proc, "start() must tear its own child down")
        self.assertFalse(service.tmp.exists(), "and its temp dir with it")

    def test_two_services_start_at_once_on_different_ports(self):
        first = demo.Service(live=False)
        second = demo.Service(live=False)
        self.assertNotEqual(first.port, second.port)
        for service in (first, second):
            self.addCleanup(service.stop)
            service.start()
        self.assertEqual(first.health()["status"], "ok")
        self.assertEqual(second.health()["status"], "ok")
        self.assertNotEqual(first.base, second.base)

    def test_a_taken_port_is_named_rather_than_answered_by_a_stranger(self):
        # Bind the port first, then hand it to the demo: the child cannot bind and
        # must die loudly, because the alternative is polling /health and reaching
        # a listener that is not ours.
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            port = taken.getsockname()[1]
            service = demo.Service(live=False)
            self.addCleanup(service.stop)
            service.port = port
            service.base = f"http://127.0.0.1:{port}"
            with self.assertRaises(SystemExit) as caught:
                service.start()
            self.assertIn("exited during startup", str(caught.exception))
            self.assertIn("cannot bind", str(caught.exception))
        self.assertIsNone(service.proc)

    def test_stop_is_idempotent_and_safe_before_start(self):
        service = demo.Service(live=False)
        service.stop()
        service.stop()
        self.assertIsNone(service.proc)
        self.assertFalse(service.tmp.exists())


class SignalTest(unittest.TestCase):
    """The demo, killed mid-run, takes its service with it."""

    def _private_tmp(self) -> pathlib.Path:
        """A TMPDIR of this test's own, so only what this test started is observed.

        A scan of the shared `/tmp` for `ti-demo-*` also sees a second suite run,
        or a `bench/demo.py` someone left going, and calls their directories this
        run's leak — a check that is only correct on a quiet machine. The demo's
        `mkdtemp` honours `TMPDIR`, so the child is pointed at this directory and
        the assertion below reads exactly what the child wrote there.
        """
        holder = tempfile.TemporaryDirectory(prefix="ti-demo-harness-")
        self.addCleanup(holder.cleanup)
        return pathlib.Path(holder.name)

    def _run_until_service_is_up(self, tmp: pathlib.Path) -> tuple[subprocess.Popen, list[int]]:
        proc = subprocess.Popen([sys.executable, str(ROOT / "bench" / "demo.py"),
                                 "--pace", "5"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True, start_new_session=True,
                                env={**os.environ, "TMPDIR": str(tmp)})
        self.addCleanup(self._kill, proc)
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            children = _children(proc.pid)
            if children:
                time.sleep(0.4)          # let it reach the scenes, past registration
                return proc, _children(proc.pid) or children
            if proc.poll() is not None:
                self.fail(f"the demo exited before starting a service: {proc.stderr.read()}")
            time.sleep(0.05)
        self.fail("the demo never started a service")

    def _kill(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)

    def _assert_signal_takes_the_service_down(self, signum: int, expected: int) -> None:
        tmp = self._private_tmp()
        proc, children = self._run_until_service_is_up(tmp)
        # Non-vacuous: the demo did put a temp dir in there, so "it removed it"
        # is a statement about something that existed.
        self.assertNotEqual(_demo_tmp_dirs(tmp), set(),
                            "the demo left nothing in its TMPDIR to leak")
        os.kill(proc.pid, signum)
        self.assertEqual(proc.wait(timeout=40), expected)
        time.sleep(0.5)
        survivors = [pid for pid in children if _alive(pid)]
        for pid in survivors:               # never leave one behind, even on failure
            os.kill(pid, signal.SIGKILL)
        self.assertEqual(survivors, [], f"the service outlived SIG{signum}")
        self.assertEqual(_demo_tmp_dirs(tmp), set(), "temp dirs were left behind")

    def test_sigint_takes_the_service_down(self):
        self._assert_signal_takes_the_service_down(signal.SIGINT, 130)

    def test_sigterm_takes_the_service_down(self):
        # What `timeout` and a plain `kill` send, and what had no handler at all.
        self._assert_signal_takes_the_service_down(signal.SIGTERM, 128 + signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
