#!/bin/bash
# serve_local.sh — run the decision model (d1-3B) locally on this box.
#
# CPU-only build here, so this is not fast (~1.7 s per routing decision), but it
# is enough to exercise useTool end to end. Keep it running: the extension talks
# to the Python service, and the service talks to this server on :8080.
#
#   bash bench/serve_local.sh          # foreground
#   bash bench/serve_local.sh &        # background
#
# Overridable: TI_MODEL, TI_MMPROJ (optional vision projector), TI_MODEL_PORT,
# TI_MODEL_CTX, TI_THREADS, TI_SERVER.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SERVER="${TI_SERVER:-$ROOT/extern/llama.cpp/build/bin/llama-server}"
MODEL="${TI_MODEL:-$ROOT/models/d1-3B-Q4_K_M.gguf}"
PORT="${TI_MODEL_PORT:-8080}"
CTX="${TI_MODEL_CTX:-4096}"
THREADS="${TI_THREADS:-8}"

[ -x "$SERVER" ] || { echo "!! no llama-server at $SERVER (build extern/llama.cpp first)"; exit 1; }
[ -f "$MODEL" ] || { echo "!! no model at $MODEL"; exit 1; }

ARGS=(-m "$MODEL" --host 127.0.0.1 --port "$PORT" -c "$CTX" -t "$THREADS" --parallel 1)
# Routing is text-only; loading the vision projector costs ~850 MB and init time.
if [ -n "${TI_MMPROJ:-}" ] && [ -f "${TI_MMPROJ}" ]; then ARGS+=(--mmproj "$TI_MMPROJ"); fi

echo "== llama-server :$PORT  model=$(basename "$MODEL")  ctx=$CTX  threads=$THREADS"
exec "$SERVER" "${ARGS[@]}"
