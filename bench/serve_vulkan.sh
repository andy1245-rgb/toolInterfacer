#!/bin/bash
# serve_vulkan.sh — run the decision model (d1-3B) on the iGPU via ggml's Vulkan backend.
#
# Same server, same model, same port conventions as bench/serve_local.sh, but the
# binary comes from an out-of-tree build with -DGGML_VULKAN=ON and every layer is
# offloaded (no CPU split). Measured against the CPU build on this box in
# docs/vulkan-igpu-2026-11-03.md; the short version is the CPU lane is still the
# default — this script exists so the experiment is repeatable.
#
#   bash bench/serve_vulkan.sh          # foreground
#   bash bench/serve_vulkan.sh &        # background
#
# Needs a Vulkan build of llama.cpp: the build recipe (glslc/SPIRV-Headers shims
# included, this box's stock headers are too old for ggml) is in the doc above.
#
# Overridable: TI_VK_BUILD (build dir), TI_SERVER, TI_MODEL, TI_MMPROJ,
# TI_MODEL_PORT, TI_MODEL_CTX, TI_THREADS, TI_DEVICE, TI_NGL, TI_EXTRA_ARGS.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VK_BUILD="${TI_VK_BUILD:-$ROOT/build-vk}"
SERVER="${TI_SERVER:-$VK_BUILD/bin/llama-server}"
MODEL="${TI_MODEL:-$ROOT/models/d1-3B-Q4_K_M.gguf}"
# Not 8080: that is the CPU lane's port, and both lanes are served at once when
# the two arms are interleaved.
PORT="${TI_MODEL_PORT:-8081}"
CTX="${TI_MODEL_CTX:-4096}"
THREADS="${TI_THREADS:-8}"
DEVICE="${TI_DEVICE:-Vulkan0}"
NGL="${TI_NGL:-99}"

[ -x "$SERVER" ] || { echo "!! no llama-server at $SERVER (build llama.cpp with -DGGML_VULKAN=ON first)"; exit 1; }
[ -f "$MODEL" ] || { echo "!! no model at $MODEL"; exit 1; }

ARGS=(-m "$MODEL" --host 127.0.0.1 --port "$PORT" -c "$CTX" -t "$THREADS"
      --parallel 1 -ngl "$NGL" --device "$DEVICE")
# Routing is text-only; loading the vision projector costs ~850 MB and init time.
if [ -n "${TI_MMPROJ:-}" ] && [ -f "${TI_MMPROJ}" ]; then ARGS+=(--mmproj "$TI_MMPROJ"); fi
# shellcheck disable=SC2206  # word splitting is the point: TI_EXTRA_ARGS is an argv fragment
if [ -n "${TI_EXTRA_ARGS:-}" ]; then ARGS+=(${TI_EXTRA_ARGS}); fi

echo "== llama-server :$PORT  model=$(basename "$MODEL")  ctx=$CTX  threads=$THREADS  device=$DEVICE  ngl=$NGL"
exec "$SERVER" "${ARGS[@]}"
