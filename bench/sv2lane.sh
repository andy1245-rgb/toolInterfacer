#!/bin/bash
# sv2lane.sh <tibench-key> <logname> <backend> <outname>
# v2 sweep lane on dell321 — adapted from bench/lane.sh (known-good ssh choreography).
# Serves via pyhost (e3\python.exe tibench.py pyfg KEY), tunnels 18080, 2 probe runs
# over tests/data/utterances_v2.json, teardown, verification. One restart allowed.
set -u
KEY="$1"; LOGNAME="$2"; BK="$3"; OUTNAME="$4"
cd /home/andrew/toolInterfacer

e3py='C:\Users\andre\Documents\PersonalCodingProjects\personal\Research\jevResearch\lab\d1bench\e3\python.exe'
LIB='C:\Users\andre\Documents\PersonalCodingProjects\personal\Research\jevResearch\lab\d1bench\tibench.py'
UT=tests/data/utterances_v2.json
ROUTE=bench/results/dell

L_TS=$(date +%s)
L_NEED_MB=3500
RESTARTED=0

ram_mb() { ssh -o ConnectTimeout=10 dell321 "powershell -NoProfile -Command \"[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB,0)\"" 2>/dev/null | tail -1; }
port80() { ssh -o ConnectTimeout=10 dell321 "netstat -ano | findstr LISTENING | findstr :8080" 2>/dev/null | head -1; }
pythons() { ssh -o ConnectTimeout=10 dell321 "tasklist | findstr /I \"python\"" 2>/dev/null; }
health_once() { ssh -o ConnectTimeout=10 dell321 "curl.exe -s http://127.0.0.1:8080/health" 2>/dev/null; }

echo "== [lane $OUTNAME] pre-checks"
L_RAM=$(ram_mb); echo "dell free RAM: ${L_RAM} MB (need $L_NEED_MB)"
L80=$(port80); if [ -n "$L80" ]; then echo "!! dell 8080 busy: $L80 — ABORT"; exit 42; fi
LP=$(ss -tln | grep -E "18080|8080"); if [ -n "$LP" ]; then echo "!! local busy: $LP — ABORT"; exit 42; fi
if [ "$L_RAM" -lt "$L_NEED_MB" ]; then echo "!! RAM below $L_NEED_MB — ABORT"; exit 42; fi
PYS_BEFORE=$(pythons | awk '{print $2}' | sort)
echo "python PIDs before (do not touch): ${PYS_BEFORE:-none}"

serve() {
  echo "== serve pyfg $KEY"
  L_SERVE_TS=$(date +%s)
  nohup ssh -o ConnectTimeout=10 dell321 "$e3py $LIB pyfg $KEY" > "/tmp/v2serve_${LOGNAME}.log" 2>&1 &
  SERVE_SSH=$!
  H=""
  for i in $(seq 1 60); do
    H=$(health_once)
    case "$H" in *ok*) echo "health ok (poll $i, $(date +%e%T), waited $(( $(date +%s) - L_SERVE_TS ))s)"; return 0;; *) sleep 3;; esac
  done
  echo "SERVE FAILED (waited 180s)"; tail -25 "/tmp/v2serve_${LOGNAME}.log"; return 1
}

serve
[ $? -ne 0 ] && exit 1
L_HEALTH_S=$(( $(date +%s) - L_SERVE_TS ))
# barrier in case another session races the port: verify OUR model is being served
P=$(ssh -o ConnectTimeout=10 dell321 "curl.exe -s http://127.0.0.1:8080/props" 2>/dev/null)
case "$P" in *"$OUTNAME"*) : ;; *) echo "!! /props mismatch (not our model): ${P:0:200} — ABORT"; exit 43;; esac
sleep 30
P2=$(ssh -o ConnectTimeout=10 dell321 "curl.exe -s http://127.0.0.1:8080/props" 2>/dev/null)
case "$P2" in *"$OUTNAME"*) : ;; *) echo "!! /props changed after settle (port stolen): ${P2:0:200} — ABORT"; exit 43;; esac

RSS=$(ssh -o ConnectTimeout=10 dell321 "powershell -NoProfile -Command \"[math]::Round((Get-Process python).WorkingSet64/1MB,0)\"" 2>/dev/null | tail -1)
echo "serve RSS: ${RSS} MB"

ssh -f -N -L 18080:127.0.0.1:8080 -o ConnectTimeout=10 dell321
sleep 1
ss -tln | grep 18080 >/dev/null || { echo "!! tunnel FAILED"; exit 1; }
echo "tunnel up"

run_probe() { # run 1 or 2
  N=$1
  PT=$(curl -s --max-time 10 http://127.0.0.1:18080/props)
  case "$PT" in *"$OUTNAME"*) : ;; *) echo "!! tunnel /props mismatch before run $N: ${PT:0:160} — ABORT"; exit 43;; esac
  out="$ROUTE/v2-${OUTNAME}${N:+-run2}.json"
  t0=$(date +%s)
  python3 bench/live_probe.py --backend "$BK" --base-url http://127.0.0.1:18080 \
    --host dell321 --threads 8 --utterances "$UT" --out "$out" \
    > "/tmp/v2probe_${OUTNAME}-${N}.log" 2>&1
  rc=$?
  L_WALL=$(( $(date +%s) - t0 ))
  if [ $rc -ne 0 ]; then return 1; fi
  sleep 2
  E=$(health_once)
  case "$E" in *ok*) return 0;; *) return 2;; esac    # 2 = server died during run
}

STABLE=true
for N in "" 2; do
  run_probe "$N"; S=$?
  if [ $S -eq 1 ]; then
    echo "!! probe run $N failed (python error)"
    E=$(health_once)
    case "$E" in
      *ok*) STABLE=false; exit 1;;
      *)
        if [ "$RESTARTED" -eq 0 ]; then
          echo "== server vanished; restarting once"
          RESTARTED=1; STABLE=false
          kill $SERVE_SSH 2>/dev/null
          serve || exit 1
          run_probe "$N"; S=$?
          [ $S -ne 0 ] && { echo "!! still failing after restart"; exit 1; }
        else
          STABLE=false; exit 1
        fi;;
    esac
  elif [ $S -eq 2 ]; then
    if [ "$RESTARTED" -eq 0 ]; then
      echo "== server crashed during run $N; restarting once"
      RESTARTED=1; STABLE=false
      kill $SERVE_SSH 2>/dev/null
      serve || exit 1
      run_probe "$N"; S=$?
      [ $S -ne 0 ] && { echo "!! still failing after restart"; exit 1; }
    else
      STABLE=false; exit 1
    fi
  fi
  grep -m1 "^n=" "/tmp/v2probe_${OUTNAME}-${N}.log" | head -1
done
E=$(health_once); case "$E" in *ok*) : ;; *) STABLE=false;; esac

echo "== teardown"
TEARS=$(pgrep -f "ssh -f -N -L 18080" || true)
[ -n "$TEARS" ] && pkill -f "ssh -f -N -L 18080"
kill $SERVE_SSH 2>/dev/null
sleep 3
for p in $(pythons | awk '{print $2}' | sort | comm -13 <(echo "$PYS_BEFORE") -); do
  ssh -o ConnectTimeout=10 dell321 "taskkill /F /PID $p" >/dev/null 2>&1
done
sleep 2
L80=$(port80)
[ -z "$L80" ] && echo "dell 8080 free" || echo "!! dell 8080 still busy: $L80"
ss -tln | grep 18080 >/dev/null && echo "!! local tunnel still up" || echo "local tunnel free"
LPY=$(pythons | awk '{print $2}' | sort | comm -13 <(echo "$PYS_BEFORE") -)
[ -z "$LPY" ] && echo "dell: no new pythons alive" || echo "!! pythons still alive: $LPY"

L_TOTAL=$(( $(date +%s) - L_TS ))
echo "{\"key\":\"$KEY\",\"restarted\":$RESTARTED,\"serving_stable\":$STABLE,\"health_wait_s\":$L_HEALTH_S,\"rss_mb\":${RSS:-0},\"lane_wall_s\":$L_TOTAL}" > "/tmp/sv2_${OUTNAME}.status"
cat "/tmp/sv2_${OUTNAME}.status"
echo "== lane $OUTNAME done ($L_TOTAL s)"
