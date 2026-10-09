#!/usr/bin/env bash
# Convert episodes, find the largest batch that fits, train SmolVLA, track val loss per checkpoint.
#   bash scripts/remote/train_smolvla.sh <session> [steps]
set -o pipefail
SESSION=$1; STEPS=${2:-40000}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
export XDG_CACHE_HOME=$ROOT/.cache PIP_CACHE_DIR=$ROOT/.cache/pip HF_HOME=$ROOT/.cache/huggingface TMPDIR=$ROOT/.cache/tmp CONDA_REGISTER_ENVS=false
source third_party/miniforge3/etc/profile.d/conda.sh; conda activate "$ROOT/third_party/miniforge3/envs/lerobot"
L=logs/train_$SESSION; mkdir -p "$L"
echo "[$(date)] convert" >> $L/status.txt
python scripts/remote/to_lerobot.py data/sim_episodes/$SESSION data/sim_episodes/$SESSION/split.json data/lerobot > $L/convert.log 2>&1 || { echo "convert FAILED" >> $L/status.txt; exit 1; }
COMMON=(--policy.path=lerobot/smolvla_base --dataset.repo_id=local/cube2box_train --dataset.root=data/lerobot/cube2box_train
        --policy.device=cuda --wandb.enable=false --policy.push_to_hub=false --dataset.video_backend=pyav)
BS=""
for bs in ${BS_LIST:-64 48 32 24 16}; do
  rm -rf work/bs_probe
  if lerobot-train "${COMMON[@]}" --batch_size=$bs --steps=4 --save_checkpoint=false --output_dir=work/bs_probe > $L/probe_$bs.log 2>&1; then BS=$bs; break; fi
  grep -q "out of memory\|OutOfMemory" $L/probe_$bs.log || { echo "probe bs=$bs failed (not OOM), see probe_$bs.log" >> $L/status.txt; exit 1; }
done
[ -n "$BS" ] || { echo "no batch size fits" >> $L/status.txt; exit 1; }
echo "[$(date)] batch_size=$BS, training $STEPS steps" >> $L/status.txt
RUN=work/smolvla_cube2box_$SESSION; rm -rf $RUN
lerobot-train "${COMMON[@]}" --batch_size=$BS --steps=$STEPS --save_freq=2000 --output_dir=$RUN --job_name=cube2box > $L/train.log 2>&1 &
TP=$!
# Validation loss is only tracked when a holdout exists. It is NOT the selection metric: on the
# 2026-10-07 run it ranked checkpoints backwards (success correlated with val loss at +0.83).
# Pick checkpoints with scripts/eval_policy.py, which rolls the policy out in the task.
if [ -d data/lerobot/cube2box_val ]; then
  sleep 120
  python scripts/remote/val_watch.py $RUN data/lerobot/cube2box_val $L/train.log $STEPS > $L/val.log 2>&1 &
fi
wait $TP; echo "[$(date)] training exit $?" >> $L/status.txt
wait; echo "[$(date)] done. Select a checkpoint with scripts/eval_policy.py" >> $L/status.txt
