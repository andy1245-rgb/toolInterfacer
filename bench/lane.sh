#!/bin/bash
# lane.sh <lane> <models-dir> [backend] [out-name] — one bench lane on dell321.
#
# Serving is routed through bench/serve_lane.py (deployed to the dell box), which
# builds the llama-server argv in-process so a chat-template-kwargs JSON value
# never crosses a PowerShell command line. An unquoted {"enable_thinking": false}
# is a PowerShell 5.1 parse error: it killed the Qwen lane's serve silently and a
# stale server then answered the probes (see docs/model-sweep-consolidated.md,
# CORRECTION). This lane therefore (a) asserts the served model in /props before
# any probe traffic and (b) tears down without self-matching its own pkill.
#
#   lane.sh qwen 'C:/Users/andre/.../toolInterfacerLab/models' logit_read qwen35-4B
set -u

LANE="${1:?usage: lane.sh <lane> <models-dir> [backend] [out-name]}"
MODELS_DIR="${2:?models-dir required}"
BK="${3:-logit_read}"
NAME="${4:-$LANE}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

e3py='C:/Users/andre/Documents/PersonalCodingProjects/personal/Research/jevResearch/lab/d1bench/e3/python.exe'
DEPLOY='C:/Users/andre/Documents/PersonalCodingProjects/personal/toolInterfacerLab/v2serve'
SLOG="C:/Users/andre/Documents/PersonalCodingProjects/personal/toolInterfacerLab/logs/v2-${NAME}-serve.log"
ROUTE=bench/results/dell
EXPECT=$(python3 -c "import json,sys; print(json.load(open('bench/lanes.json'))['lanes'][sys.argv[1]]['model'])" "$LANE")

ps() { ssh -o ConnectTimeout=10 dell321 "powershell -NoProfile -Command \"$1\"" 2>/dev/null; }
health_once() { ssh -o ConnectTimeout=10 dell321 "curl.exe -s http://127.0.0.1:8080/health" 2>/dev/null; }
props_remote() { ssh -o ConnectTimeout=10 dell321 "curl.exe -s http://127.0.0.1:8080/props" 2>/dev/null; }
pythons() { ssh -o ConnectTimeout=10 dell321 "tasklist | findstr /I \"python\"" 2>/dev/null; }
port80() { ssh -o ConnectTimeout=10 dell321 "netstat -ano | findstr LISTENING | findstr :8080" 2>/dev/null | head -1; }

assert_props() { # $1 = stage label; $2 = props json (empty -> fetch from dell)
  local p="$2"
  [ -n "$p" ] || p=$(props_remote)
  if printf '%s' "$p" | python3 bench/model_guard.py --expect-model "$EXPECT" --props-file -; then
    return 0
  fi
  echo "!! /props assertion FAILED at '$1' — served model is not $EXPECT"; return 1
}

echo "== lane $NAME (lane=$LANE, expected model: $EXPECT, backend: $BK)"
if [ -n "$(port80)" ]; then echo "!! dell 8080 already busy — ABORT"; exit 42; fi
PYS_BEFORE=$(pythons | awk '{print $2}' | sort)

echo "== deploy argv-safe launcher -> $DEPLOY"
ps "New-Item -ItemType Directory -Force -Path '$DEPLOY' | Out-Null"
scp -q bench/serve_lane.py bench/lanes.json "dell321:$DEPLOY/" || { echo "!! scp FAILED"; exit 1; }

echo "== serve via serve_lane.py (argv built in-process, no shell quoting)"
nohup ssh -o ConnectTimeout=10 dell321 \
  "$e3py $DEPLOY/serve_lane.py $LANE --models-dir '$MODELS_DIR'" > "/tmp/lane_serve_${NAME}.log" 2>&1 &
SERVE_SSH=$!

H=""
for i in $(seq 1 40); do
  H=$(health_once)
  case "$H" in *ok*) echo "health ok (poll $i)"; break;; *) sleep 3;; esac
done
case "$H" in
  *ok*) : ;;
  *) echo "SERVE FAILED"; tail -20 "/tmp/lane_serve_${NAME}.log"
     ssh -o ConnectTimeout=10 dell321 "type '$SLOG'" 2>/dev/null | tail -20
     exit 1;;
esac

# model-identity guard: /health is answered by ANY llama-server, so this is the
# only thing that proves the port carries the model this lane intends to measure
assert_props "after health" "" || exit 43

sleep 15
assert_props "after settle (port-steal check)" "" || exit 43

ssh -f -N -L 18080:127.0.0.1:8080 -o ConnectTimeout=10 dell321
sleep 1
ss -tln | grep 18080 >/dev/null || { echo "!! tunnel FAILED"; exit 1; }
echo "tunnel up"
assert_props "through tunnel" "$(curl -s --max-time 10 http://127.0.0.1:18080/props)" || exit 43

for ds in v1 v2; do
  case "$ds" in
    v1) ut="tests/data/utterances.json"; out="$ROUTE/${NAME}-v1";;
    v2) ut="tests/data/utterances_v2.json"; out="$ROUTE/v2-${NAME}";;
  esac
  for run in "" "-run2"; do
    assert_props "before ${ds}${run}" "$(curl -s --max-time 10 http://127.0.0.1:18080/props)" || exit 43
    echo "=== probe $ds $run -> ${out}${run}.json"
    python3 bench/live_probe.py --backend "$BK" --base-url http://127.0.0.1:18080 \
      --host "dell-${NAME}" --threads 8 --expect-model "$EXPECT" \
      --utterances "$ut" --out "${out}${run}.json" \
      > "/tmp/probe_${NAME}_${ds}${run}.log" 2>&1 || { echo "!! probe $ds $run FAILED"; tail -5 "/tmp/probe_${NAME}_${ds}${run}.log"; exit 1; }
    grep -m1 "^model assertion:" "/tmp/probe_${NAME}_${ds}${run}.log"
    grep -m1 "^n=" "/tmp/probe_${NAME}_${ds}${run}.log"
  done
done

echo "== teardown"
if pgrep -f "[s]sh -f -N -L 18080" >/dev/null; then pkill -f "[s]sh -f -N -L 18080"; fi
kill "$SERVE_SSH" 2>/dev/null
sleep 2
for p in $(pythons | awk '{print $2}' | sort | comm -13 <(echo "$PYS_BEFORE") -); do
  ssh -o ConnectTimeout=10 dell321 "taskkill /F /PID $p" >/dev/null 2>&1
done
sleep 2
[ -z "$(port80)" ] && echo "dell 8080 free" || echo "!! dell 8080 still busy"
ss -tln | grep 18080 >/dev/null && echo "!! local tunnel still up" || echo "local tunnel free"
echo "== lane $NAME done"
