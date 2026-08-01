#!/bin/bash
# Health monitor for m3 training runs on rtx ($1 = run name, default m3_v5).
# Polls the remote console log; exits 1 on error markers, exits 0 once
# step >= 500 (printing recent step lines incl. the unweighted durlin
# value for weight calibration).
RUN="${1:-m3_v5}"
LOG="E:\\sufei-training\\logs\\${RUN}_console.log"
for i in $(seq 1 30); do
  sleep 120
  log=$(ssh rtx "cmd /c type $LOG" 2>/dev/null)
  if echo "$log" | grep -qE "Traceback|RuntimeError|CUDA out of memory|OMP: Error"; then
    echo "=== ERROR DETECTED (check $i) ==="
    echo "$log" | tail -40
    exit 1
  fi
  laststep=$(echo "$log" | grep -oE "step +[0-9]+/" | grep -oE "[0-9]+" | tail -1)
  echo "check $i: last step=${laststep:-none}"
  if [ "${laststep:-0}" -ge 500 ]; then
    echo "=== HEALTHY: step >= 500 ==="
    echo "$log" | grep -E "step +[0-9]+/" | tail -10
    exit 0
  fi
done
echo "=== TIMEOUT: step < 500 after 30 checks (60 min) ==="
echo "$log" | tail -40
exit 1
