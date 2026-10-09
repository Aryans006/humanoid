#!/usr/bin/env bash
# Unattended: smoke test -> wait for HaWoR/exports -> record sim episodes -> split -> train on GPU box.
# Log: data/processed/2026-10-07_dataset/OVERNIGHT.txt
S=${SESSION:-2026-10-07_dataset}; H=${GPU_HOST:?set GPU_HOST, e.g. user@10.0.0.2}; RD=${REMOTE_DIR:-drive2/ego2libero}
cd "$(dirname "$0")/.."; ROOT=$PWD; LOG=$ROOT/data/processed/$S/OVERNIGHT.txt
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
SIMPY() { MUJOCO_GL=egl PYTHONPATH= "$ROOT/.venv-sim/bin/python" "$@"; }
ssh_() { ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 "$H" "$@"; }

if [ "${SKIP_SMOKE:-0}" != 1 ]; then
log "1/5 smoke test (3 cameras, 1 episode, 4 steps)"
rm -rf data/sim_episodes/smoketest; mkdir -p data/sim_episodes/smoketest
SIMPY scripts/sim_record.py $S demo_C026 >> "$LOG.detail" 2>&1
mv data/sim_episodes/$S/demo_C026_r0.npz data/sim_episodes/smoketest/; rm -f data/sim_episodes/$S/index.json
echo '{"train":["demo_C026_r0"],"val":["demo_C026_r0"]}' > data/sim_episodes/smoketest/split.json
rsync -a scripts/ $H:$RD/scripts/ && rsync -a --mkpath data/sim_episodes/smoketest/ $H:$RD/data/sim_episodes/smoketest/
ssh_ "cd $RD && rm -rf logs/train_smoketest work/smolvla_cube2box_smoketest && BS_LIST=4 bash scripts/remote/train_smolvla.sh smoketest 4; cat logs/train_smoketest/status.txt; test -s work/smolvla_cube2box_smoketest/val_loss.csv && cat work/smolvla_cube2box_smoketest/val_loss.csv" >> "$LOG" 2>&1 \
  || { log "SMOKE TEST FAILED - see logs/train_smoketest on the GPU box. Stopping."; exit 1; }
log "smoke test passed"
fi

log "2/5 waiting for HaWoR + exports + sim replays (AUTO_STATUS)"
until grep -q "done. Trajectories" data/processed/$S/AUTO_STATUS.txt 2>/dev/null; do sleep 120; done

log "3/5 recording sim episodes for every accepted demo"
rm -rf data/sim_episodes/$S; mkdir -p data/sim_episodes/$S
ACC=$(python3 -c "
import json,glob,os
print(' '.join(os.path.basename(f)[:-5] for f in sorted(glob.glob('data/processed/$S/demo_C*.json')) if '_' not in os.path.basename(f)[5:-5] and json.load(open(f)).get('accepted')))")
log "accepted demos: $(echo $ACC | wc -w)"
SIMPY scripts/sim_record.py $S $ACC >> "$LOG.detail" 2>&1
log "$(tail -1 $LOG.detail)"

log "4/5 split"
.venv/bin/python scripts/final_split.py $S >> "$LOG" 2>&1 || { log "split FAILED"; exit 1; }

log "5/5 upload + start training"
rsync -a scripts/ $H:$RD/scripts/ && rsync -a --mkpath data/sim_episodes/$S/ $H:$RD/data/sim_episodes/$S/ || { log "upload FAILED"; exit 1; }
ssh_ "cd $RD && tmux new-session -d -s smolvla_train 'bash scripts/remote/train_smolvla.sh $S 40000'" && log "training started on GPU box: tmux smolvla_train; status $RD/logs/train_$S/status.txt; val loss $RD/work/smolvla_cube2box_$S/val_loss.csv"
