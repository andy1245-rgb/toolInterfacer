#!/bin/bash
# lane.sh <model-file-on-dell-name> <serve-extra-args...>  — known-good loop for one model lane
MODEL="$1"; shift
EXTRA="$*"
LAB='Documents\PersonalCodingProjects\personal\toolInterfacerLab'
models='C:\Users\andre\Documents\PersonalCodingProjects\personal\toolInterfacerLab\models'
e3py='C:\Users\andre\Documents\PersonalCodingProjects\personal\Research\jevResearch\lab\d1bench\e3\python.exe'
logname=$(echo "$MODEL" | sed 's/\.[^.]*$//;s/-Q4_K_M//')
log="C:\Users\andre\Documents\PersonalCodingProjects\personal\toolInterfacerLab\logs\\${logname}-serve.log"

echo "=== serve $MODEL (extra: $EXTRA)"
nohup ssh -o ConnectTimeout=10 dell321 "$e3py $LAB\\serve_pyhost.py -m $models\\$MODEL --host 127.0.0.1 --port 8080 -c 8192 -t 8 -ngl 99 $EXTRA > $log 2>&1" > /tmp/lane_serve.log 2>&1 &
SERVE_SSH=$!
sleep 22
# health via direct dell curl
for i in $(seq 1 10); do
  h=$(ssh -o ConnectTimeout=10 dell321 "curl.exe -s http://127.0.0.1:8080/health" 2>/dev/null)
  case "$h" in *ok*) echo "health ok (after ${i}p)"; break;; *) sleep 3;; esac
done
case "$h" in *ok*) : ;; *) echo "SERVE FAILED"; tail -20 "$log" 2>/dev/null; ssh -o ConnectTimeout=10 dell321 "type $log" 2>/dev/null | head -20; exit 1;; esac
ssh -f -N -L 18080:127.0.0.1:8080 -o ConnectTimeout=10 dell321
sleep 1
echo "=== RSS snapshot"
ssh -o ConnectTimeout=10 dell321 "Get-Process python | Select-Object -ExpandProperty WorkingSet64" 2>/dev/null | head -2
cd /home/andrew/toolInterfacer
ROUTE=bench/results/dell
for ds in v1 v2; do
  case $ds in
    v1) ut="tests/data/utterances.json"; ds="v1";;
    v2) ut="tests/data/utterances_v2.json"; ds="v2";;
  esac
  for run in "" "-run2"; do
    out="$ROUTE/q4-${logname}-${ds}${run}.json"
    echo "=== probe $ds $run -> $out"
    python3 bench/live_probe.py --backend logit_read --base-url http://127.0.0.1:18080 --host dell-q4 --threads 8 --utterances "$ut" --out "$out" > "/tmp/probe_${logname}_${ds}${run}.log" 2>&1
    tail -3 "/tmp/probe_${logname}_${ds}${run}.log" | head -1
  done
done
echo "=== teardown"
TEARS=$(pgrep -f "ssh -f -N -L 18080" || true)
[ -n "$TEARS" ] && pkill -f "ssh -f -N -L 18080"
kill $SERVE_SSH 2>/dev/null
sleep 2
for p in $(ssh -o ConnectTimeout=10 dell321 "tasklist | findstr python" | awk '{print $2}'); do ssh dell321 "taskkill /F /PID $p" >/dev/null 2>&1; done
sleep 2
ssh -o ConnectTimeout=10 dell321 "netstat -ano | findstr LISTENING | findstr :8080" && echo "!! port still busy" || echo "dell8080 free"
ss -ltn | grep 18080 && echo "!! tunnel still up" || echo "local tunnel free"
echo "=== lane ${logname} done"
