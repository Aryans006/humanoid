#!/usr/bin/env bash
# Unattended finish, run on THIS machine: wait for HaWoR on the GPU machine to finish, pull the
# results, then run the local pipeline (hands into W, overlay, export) on every demo.
# Progress and outcome go to data/processed/<session>/AUTO_STATUS.txt.
#
#   setsid nohup scripts/remote/auto_finish.sh user@host <session> >/dev/null 2>&1 &
#
# Env: PROCESS_LOG (remote log to wait on, default logs/process.log), SIM=1 to also run a physics
# sim replay of every demo (metrics only, no video) after its export.
set -o pipefail
HOST=$1; SESSION=$2
REMOTE_DIR=${REMOTE_DIR:-drive2/ego2libero}
PROCESS_LOG=${PROCESS_LOG:-logs/process.log}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/data/processed/$SESSION
mkdir -p "$OUT"
STATUS=$OUT/AUTO_STATUS.txt
log() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
log "waiting for HaWoR on $HOST ($REMOTE_DIR/$PROCESS_LOG)"
until ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "grep -q EXIT_CODE $REMOTE_DIR/$PROCESS_LOG" 2>/dev/null; do
  sleep 120
done
rsync -az -e ssh "$HOST:$REMOTE_DIR/$PROCESS_LOG" "$OUT/hawor_process.log" 2>/dev/null
log "HaWoR finished on the GPU machine:"
sed 's/\x1b\[[0-9;]*m//g' "$OUT/hawor_process.log" | grep -E "^== summary|FAILED|EXIT_CODE" >> "$STATUS"
log "pulling results"
REMOTE_DIR=$REMOTE_DIR "$ROOT/scripts/remote/pull.sh" "$HOST" "$SESSION" >> "$OUT/auto_finish.log" 2>&1 || log "pull failed (see auto_finish.log)"
for d in "$ROOT"/data/interim/"$SESSION"/demo_*/; do
  clip=$(basename "$d")
  if [ ! -f "$d/hawor.npz" ]; then log "$clip: no hawor.npz (HaWoR failed for it; see $d/hawor.log)"; continue; fi
  log "$clip: hands into W, overlay, export"
  if "$ROOT/.venv/bin/python" "$ROOT/scripts/run_demo.py" "$SESSION" "$clip" >> "$OUT/auto_finish.log" 2>&1; then
    sed -n "/== $SESSION\/$clip/,/^$/p" "$OUT/auto_finish.log" | grep -E "PASS|FAIL|n/a|hand_observed" | tail -12 >> "$STATUS"
    if [ "${SIM:-0}" = 1 ]; then
      if MUJOCO_GL=egl PYTHONPATH= "$ROOT/.venv-sim/bin/python" "$ROOT/scripts/sim_replay.py" "$SESSION" "$clip" \
           --mode physics --no-video >> "$OUT/auto_finish.log" 2>&1; then
        log "$clip: sim replay $(python3 -c "import json;r=json.load(open('$OUT/sim/${clip}_physics.json'));print('IN BOX' if r['success_cube_in_box'] else 'MISSED')")"
      else
        log "$clip: sim replay FAILED to run"
      fi
    fi
  else
    log "$clip: local pipeline FAILED (see auto_finish.log)"
  fi
done
log "done. Trajectories: $OUT/demo_*.npz (+ .json); overlays: data/interim/$SESSION/demo_*/overlay.mp4"
