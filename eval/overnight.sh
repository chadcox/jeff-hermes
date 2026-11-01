#!/bin/bash
# Overnight A/B: every model x mode x task, repeated until MAX_REPS or END (no new run starts after END).
# Mode order rotates each rep; model order alternates. The Hermes gateway service is stopped for the whole batch;
# each `hermes chat` starts its own gateway, which is killed on every switch so the next one reads the new config.
# On exit (done, END reached, or killed) the original config is restored, the service started again, and the report written.
#   eval/overnight.sh <outdir>          e.g. eval/results/2026-11-02-27b
set -u
EVAL=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "${1:?outdir}")
MODELS=(${MODELS:-Qwen3.8-27B-UD-Q4_K_M-262k-MTP Qwen3.8-27B-UD-IQ3_S-5080-MTP})
TASKS=(${TASKS:-t1 t2 t3 t4 t5 t6 t7 t8})
MAX_REPS=${MAX_REPS:-5}
END=$(date -d "${END:-09:00}" +%s)
[ "$END" -le "$(date +%s)" ] && END=$((END + 86400))  # default: the next 09:00
CFG=~/.hermes/config.yaml
HERMES=~/.hermes/hermes-agent/venv/bin/hermes
ORDERS=("router@0.6 on off" "on off router@0.6" "off router@0.6 on" "router@0.6 off on" "on router@0.6 off")

mkdir -p "$OUT"
cp -p "$CFG" "$OUT/config.yaml.orig"
exec >> "$OUT/batch.log" 2>&1

finish() {
  cp -p "$OUT/config.yaml.orig" "$CFG"
  $HERMES gateway stop --all -y > /dev/null 2>&1
  systemctl --user start hermes-gateway.service
  echo "HERMES_STARTED $(systemctl --user is-active hermes-gateway.service)"
  echo "BATCH_DONE $(date -Is)"
  python3 "$EVAL/report.py" "$OUT" > "$OUT/report.md" 2>&1
}
trap finish EXIT
trap 'exit 1' INT TERM

loaded_since() {  # did a gateway load the plugin at or after this "YYYY-mm-dd HH:MM:SS"?
  cat ~/.hermes/logs/*.log* 2>/dev/null | grep -a 'jeff_router: shadow mode on' | cut -c1-19 | awk -v t="$1" '$0 >= t {f=1} END {exit !f}'
}

restart_into() {  # mode model: kill the running gateway, then a probe query starts a fresh one on the new config
  python3 "$EVAL/run.py" set "$1" "$2" || exit 1
  $HERMES gateway stop --all -y > /dev/null 2>&1
  local t; t=$(date '+%F %T')
  (cd "$(mktemp -d)" && timeout 600 $HERMES chat -Q -q "Reply with OK." > /dev/null 2>&1)
  if loaded_since "$t"; then echo "SWITCHED $1 $2 $(date -Is)"; else echo "SWITCH_FAILED $1 $2 $(date -Is)"; exit 1; fi
}

warm() {  # load the model on the configured provider so the first run doesn't pay for the swap
  local base key t0=$SECONDS
  read -r base key < <(~/.hermes/hermes-agent/venv/bin/python - <<'PY'
import os, yaml
from pathlib import Path
cfg = yaml.safe_load((Path.home() / ".hermes/config.yaml").read_text())
model = cfg.get("model") or {}
prov = (cfg.get("providers") or {}).get(model.get("provider"), {})
env = prov.get("key_env", "")
key = os.environ.get(env, "")
for line in (Path.home() / ".hermes/.env").read_text().splitlines() if env and not key else []:
    if line.startswith(env + "="):
        key = line.split("=", 1)[1].strip().strip('"')
print(model.get("base_url") or prov.get("base_url", ""), key or "-")
PY
)
  [ -z "$base" ] && { echo "WARM skipped for $1: no base_url in config"; return; }
  curl -s -m 600 -H "Authorization: Bearer $key" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
    "${base%/}/chat/completions" > /dev/null
  echo "WARM $1 $((SECONDS - t0))s"
}

systemctl --user stop hermes-gateway.service
$HERMES gateway stop --all -y > /dev/null 2>&1
echo "HERMES_STOPPED $(systemctl --user is-active hermes-gateway.service)"
echo "START $(date -Is) models=${MODELS[*]} tasks=${TASKS[*]} max_reps=$MAX_REPS end=$(date -d @$END -Is)"
for rep in $(seq "$MAX_REPS"); do
  order=("${MODELS[@]}"); [ $((rep % 2)) -eq 0 ] && order=($(printf '%s\n' "${MODELS[@]}" | tac))
  for model in "${order[@]}"; do
    warm "$model"
    for mode in ${ORDERS[$(((rep - 1) % 5))]}; do
      restart_into "$mode" "$model"
      for task in "${TASKS[@]}"; do
        [ "$(date +%s)" -ge "$END" ] && { echo "END reached before rep $rep $model $mode $task"; exit 0; }
        python3 "$EVAL/run.py" run "$mode" "$task" "r$rep" "$OUT"
      done
    done
  done
done
