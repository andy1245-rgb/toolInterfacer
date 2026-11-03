"""Decision-router distillation pipeline (stdlib only).

Generate candidate routing utterances, label them with the hosted teacher, and
audit/measure what came back — the feedstock for a fine-tuned routing model.

Entry points, each runnable as `python3 tools/distill/<name>.py --help`:
  utterances.py  candidates.jsonl from the session corpus + template coverage
  label.py       teacher labels over those candidates (resumable, budgeted)
  audit.py       the human-check sample, printed and as a CSV to annotate
  stats.py       per-band counts, label/confidence distributions, calibration

Shared modules: catalog (the op set, derived), bands (t1/t2/t3), similarity
(normalisation + near-dup), corpus (row schema, leakage guard), mine (pi-sessions),
frames + templates (surface variety), teacher (fixture / hosted), cost.
"""
