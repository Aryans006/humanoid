#!/usr/bin/env bash
# Set up a FRESH rented GPU box (RunPod / Vast / Lambda / any Ubuntu + NVIDIA box) and start
# training SmolVLA on the cube-to-box episodes. Idempotent: safe to re-run.
#
#   # on the rented box, after uploading ego2libero_train/ (see scripts/remote/push_cloud.sh)
#   bash cloud_setup.sh [STEPS]          # default 40000
#
# Everything lives under this directory; nothing outside it is touched.
#
# numpy MUST stay < 2. lerobot's rerun-sdk dep asks for >= 2, but rerun is never imported on the
# training path, and numpy 2 silently changes what MuJoCo renders (~14 levels darker on the same
# scene). That does not affect this training run -- the episodes are already recorded -- but the
# same env is used for evaluation, where it halved the measured success rate. Pin it everywhere.
set -uo pipefail
STEPS=${1:-40000}
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
export XDG_CACHE_HOME=$ROOT/.cache PIP_CACHE_DIR=$ROOT/.cache/pip HF_HOME=$ROOT/.cache/huggingface
export TMPDIR=$ROOT/.cache/tmp
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" logs

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/setup.txt; }

log "GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | tee -a logs/setup.txt

if [ ! -d .venv ]; then
  log "creating venv"
  python3 -m venv .venv
fi
. .venv/bin/activate
python -m pip install -q --upgrade pip

if ! python -c "import lerobot" 2>/dev/null; then
  log "installing torch + lerobot (this is the slow part, ~5 min)"
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install -q "lerobot[smolvla]==0.4.4"
  pip install -q "numpy<2"        # last, so nothing re-upgrades it
fi
python - <<'PY' | tee -a logs/setup.txt
import numpy, torch
assert numpy.__version__.startswith("1."), f"numpy must be <2, got {numpy.__version__}"
print(f"numpy {numpy.__version__}  torch {torch.__version__}  cuda {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu {p.name}  {p.total_memory/1e9:.0f} GB")
PY

[ -d data/lerobot/cube2box_train ] || { log "ERROR: data/lerobot/cube2box_train missing. Upload it first."; exit 1; }
N=$(python -c "import json;print(json.load(open('data/lerobot/cube2box_train/meta/info.json'))['total_episodes'])")
log "dataset: $N episodes"

# Largest batch that fits. The 5090 run used 64; a 24 GB card will land lower. Probing costs ~1 min
# and is far cheaper than discovering an OOM two hours in.
COMMON=(--policy.path=lerobot/smolvla_base
        --dataset.repo_id=local/cube2box_train
        --dataset.root="$ROOT/data/lerobot/cube2box_train"
        --policy.device=cuda --wandb.enable=false --policy.push_to_hub=false
        --dataset.video_backend=pyav)
BS=""
for bs in ${BS_LIST:-64 48 32 24 16 8}; do
  rm -rf work/bs_probe
  log "probing batch_size=$bs"
  if lerobot-train "${COMMON[@]}" --batch_size=$bs --steps=4 --save_checkpoint=false \
       --output_dir=work/bs_probe > logs/probe_$bs.log 2>&1; then BS=$bs; break; fi
  grep -qiE "out of memory|OutOfMemory" logs/probe_$bs.log || {
    log "probe bs=$bs failed for a non-OOM reason; see logs/probe_$bs.log"; tail -20 logs/probe_$bs.log; exit 1; }
done
[ -n "$BS" ] || { log "no batch size fits"; exit 1; }
log "batch_size=$BS, training $STEPS steps"

RUN=work/smolvla_cube2box
# save_freq 2000: checkpoints are what make a deadline survivable. Stop whenever and pick the best
# by sim rollout (scripts/eval_policy.py), NOT by validation loss -- on the previous run val loss
# ranked checkpoints backwards (success correlated with val loss at Spearman +0.83, p = 0.0005).
tmux new-session -d -s train \
  "cd $ROOT && . .venv/bin/activate && \
   XDG_CACHE_HOME=$XDG_CACHE_HOME HF_HOME=$HF_HOME TMPDIR=$TMPDIR \
   lerobot-train ${COMMON[*]} --batch_size=$BS --steps=$STEPS --save_freq=2000 \
     --output_dir=$RUN --job_name=cube2box > logs/train.log 2>&1; \
   echo done >> logs/setup.txt"
log "training started in tmux session 'train'"
log "  progress : tail -f $ROOT/logs/train.log"
log "  checkpts : $ROOT/$RUN/checkpoints/"
log "  pull     : rsync -az <this box>:$ROOT/$RUN/checkpoints/ ./data/checkpoints_cloud/"
