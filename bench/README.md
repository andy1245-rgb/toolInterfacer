# bench/

- `replay.py` (P1) — runs a plan end-to-end through the fixture-player backend
  with zero network; prints hops, confidences, wall time.
- `live_probe.py` (P2) — routing probe over held-out pi-tool utterances
  against a live d1 server; writes per-row timing + conf to stdout/JSONL.
- `baseline.py` (P4) — scripted LLM-driver baseline for the research template
  comparison.
