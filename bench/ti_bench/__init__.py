"""ti_bench — the pieces behind `bench/run.py`, the one-command scoring harness.

Layout, one concern per module:

  stats.py     percentile / Wilson interval / McNemar exact / calibration
  prompts.py   the frozen probe instruction + the 8 op criteria (shared with
               bench/live_probe.py, so both benches ask the same question)
  sets.py      the frozen utterance sets (v1, v2) and their stable row ids
  arms.py      the arm spec (bench/arms.json) read as data, plus conf application
  guard.py     the served-model / API-key assertion, run before any probe traffic
  scoring.py   one arm over one set -> the report dict
  report.py    the no-clobber write and the report-shape check
  compare.py   two reports -> McNemar exact + latency delta
  selftest.py  the whole path offline, for CI
"""
