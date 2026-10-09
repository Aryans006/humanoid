#!/usr/bin/env bash
# One-time HaWoR install on the GPU machine (Linux + NVIDIA driver). Idempotent: each step
# leaves a marker in third_party/.hawor_setup/ and is skipped on rerun (rm a marker to redo).
# process.sh calls this automatically if setup has not finished.
#
#   scripts/remote/setup_hawor.sh
#
# Self-contained: conda, caches and temp files all live under the repo folder; nothing in ~ or
# /tmp is written (except conda's ~/.conda/environments.txt registry line).
#
# Env overrides: HAWOR_ENV (conda env name, default hawor), CONDA_ROOT (default <repo>/third_party/miniforge3),
#   CUDA_VER / TORCH_VER / TV_VER / TORCH_CU   (auto-picked from the GPU; see below)
#
# Why not CPU: HaWoR's SLAM stage (masked DROID-SLAM) only builds CUDA kernels.
set -eo pipefail   # no -u: conda activate/deactivate hooks reference unset variables
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HAWOR_DIR=$ROOT/third_party/HaWoR
HAWOR_COMMIT=66c7d4108d58a716deccd192cb7645170cdc7bd7   # pinned; driver was written against it
MARK=$ROOT/third_party/.hawor_setup
ENV_NAME=${HAWOR_ENV:-hawor}
CONDA_ROOT=${CONDA_ROOT:-$ROOT/third_party/miniforge3}
mkdir -p "$MARK"
# keep every cache and temp dir inside the repo folder
export XDG_CACHE_HOME=$ROOT/.cache PIP_CACHE_DIR=$ROOT/.cache/pip TORCH_HOME=$ROOT/.cache/torch \
       HF_HOME=$ROOT/.cache/huggingface TMPDIR=$ROOT/.cache/tmp CONDA_PKGS_DIRS=$CONDA_ROOT/pkgs
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
# The env is addressed by path inside the repo, never by name: a user ~/.condarc envs_dirs entry
# would otherwise put a named env into someone else's conda install. Don't register it either.
ENV_PREFIX=$CONDA_ROOT/envs/$ENV_NAME
export CONDA_REGISTER_ENVS=false CONDA_ENVS_DIRS=$CONDA_ROOT/envs
activate() { conda activate "$ENV_PREFIX"; }
log() { echo -e "\n\033[1m== $*\033[0m"; }
done_() { [ -f "$MARK/$1" ]; }
mark() { touch "$MARK/$1"; }

# ---------------------------------------------------------------- GPU + CUDA choice
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first"; exit 1; }
GPU=$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1)
CAP=$(echo "$GPU" | cut -d, -f2 | tr -d ' ')
log "GPU: $GPU"
if [ -z "${TORCH_VER:-}" ]; then
  if awk "BEGIN{exit !($CAP >= 10.0)}"; then
    # Blackwell (RTX 50xx etc.) needs CUDA 12.8+. Less tested with HaWoR's old deps.
    CUDA_VER=12.8.1; TORCH_VER=2.7.1; TV_VER=0.22.1; TORCH_CU=cu128
    echo "!! compute capability $CAP: using torch $TORCH_VER/$TORCH_CU (HaWoR was tested on 1.13/cu117)"
  else
    # cu118 covers Pascal..Hopper incl. RTX 40xx (sm_89), which cu117 cannot compile for.
    CUDA_VER=11.8.0; TORCH_VER=2.1.2; TV_VER=0.16.2; TORCH_CU=cu118
  fi
fi
PYG_TORCH=${TORCH_VER%.*}.0
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$CAP}

# ---------------------------------------------------------------- conda (Miniforge: no channel ToS prompts)
if ! [ -x "$CONDA_ROOT/bin/conda" ]; then
  log "installing Miniforge into $CONDA_ROOT"
  curl -fsSL -o "$TMPDIR/miniforge.sh" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash "$TMPDIR/miniforge.sh" -b -p "$CONDA_ROOT"   # -b: no edits to ~/.bashrc
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# ---------------------------------------------------------------- HaWoR source
if ! done_ clone; then
  log "cloning HaWoR @ ${HAWOR_COMMIT:0:7}"
  rm -rf "$HAWOR_DIR"
  git clone https://github.com/ThunderVVV/HaWoR.git "$HAWOR_DIR"
  git -C "$HAWOR_DIR" checkout -q "$HAWOR_COMMIT"
  git -C "$HAWOR_DIR" submodule update --init --recursive
  mark clone
fi

# ---------------------------------------------------------------- env + toolchain
if ! done_ env; then
  log "conda env $ENV_PREFIX (python 3.10, CUDA toolkit $CUDA_VER, gcc 11)"
  case "$ENV_PREFIX" in "$ROOT"/*) rm -rf "$ENV_PREFIX" ;; *) echo "refusing to touch $ENV_PREFIX"; exit 1 ;; esac
  conda create -y -p "$ENV_PREFIX" -c conda-forge --override-channels python=3.10 "gxx_linux-64=11" "gcc_linux-64=11" ffmpeg git
  activate
  conda install -y -p "$ENV_PREFIX" -c "nvidia/label/cuda-$CUDA_VER" cuda-toolkit
  mark env
fi
activate
export CUDA_HOME=$CONDA_PREFIX
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
cd "$HAWOR_DIR"

if ! done_ torch; then
  log "torch $TORCH_VER+$TORCH_CU"
  pip install "torch==$TORCH_VER" "torchvision==$TV_VER" --index-url "https://download.pytorch.org/whl/$TORCH_CU"
  python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print(torch.__version__, torch.cuda.get_device_name())"
  mark torch
fi

if ! done_ reqs; then
  log "HaWoR requirements"
  pip install -U pip "setuptools<70" wheel gdown pyyaml
  # torch-scatter needs a wheel matching torch; pytorch3d/chumpy must build against the
  # installed torch/numpy, hence --no-build-isolation.
  grep -v -e '^torch-scatter' requirements.txt > "$TMPDIR/hawor_reqs.txt"
  pip install "numpy==1.26.4"
  pip install --no-build-isolation -r "$TMPDIR/hawor_reqs.txt"
  pip install torch-scatter -f "https://data.pyg.org/whl/torch-$PYG_TORCH+$TORCH_CU.html"
  pip install pytorch-lightning==2.2.4 --no-deps
  pip install lightning-utilities torchmetrics==1.4.0
  mark reqs
fi

if ! done_ droid; then
  log "building masked DROID-SLAM (CUDA kernels; takes a while)"
  python "$ROOT/scripts/remote/patch_droid.py" "$CAP"   # upstream has no Blackwell/Ada arch flags
  (cd thirdparty/DROID-SLAM && python setup.py install)
  python -c "import torch, droid_backends, lietorch; print('DROID-SLAM ok')"
  mark droid
fi

# ---------------------------------------------------------------- weights
if ! done_ weights; then
  log "model weights"
  mkdir -p weights/external weights/hawor/checkpoints thirdparty/Metric3D/weights
  [ -s weights/external/droid.pth ] || gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O weights/external/droid.pth
  [ -s thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth ] || \
    gdown 1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr -O thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth
  HF=https://huggingface.co
  [ -s weights/external/detector.pt ] || wget -q --show-progress "$HF/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt" -P weights/external/
  [ -s weights/hawor/checkpoints/hawor.ckpt ] || wget -q --show-progress "$HF/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt" -P weights/hawor/checkpoints/
  [ -s weights/hawor/checkpoints/infiller.pt ] || wget -q --show-progress "$HF/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt" -P weights/hawor/checkpoints/
  [ -s weights/hawor/model_config.yaml ] || wget -q --show-progress "$HF/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml" -P weights/hawor/
  mark weights
fi

# ---------------------------------------------------------------- MANO (license: user downloads)
MANO_SRC=$ROOT/third_party/mano
if [ -f "$MANO_SRC/MANO_RIGHT.pkl" ] && [ -f "$MANO_SRC/MANO_LEFT.pkl" ]; then
  cp "$MANO_SRC/MANO_RIGHT.pkl" _DATA/data/mano/MANO_RIGHT.pkl
  cp "$MANO_SRC/MANO_LEFT.pkl" _DATA/data_left/mano_left/MANO_LEFT.pkl
else
  echo "!! MANO files missing in $MANO_SRC. Push them with scripts/remote/push.sh (MANO_DIR=...)."
  exit 2
fi

# ---------------------------------------------------------------- headless rendering check
if ! ldconfig -p 2>/dev/null | grep -q libEGL.so; then
  echo "!! libEGL not found; HaWoR's mask renderer needs it headless: sudo apt install libegl1 libgl1"
fi

# ---------------------------------------------------------------- smoke test on HaWoR's own example
if ! done_ smoke; then
  log "smoke test: HaWoR on example/video_0.mp4"
  python "$ROOT/ego2libero/hands/hawor_driver.py" --hawor-root "$HAWOR_DIR" \
    --video "$HAWOR_DIR/example/video_0.mp4" --focal 600 \
    --work "$ROOT/work/smoke" --out "$ROOT/work/smoke/hawor.npz"
  python - "$ROOT/work/smoke/hawor.npz" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1])
print({k: d[k].shape for k in d.files})
assert np.isfinite(d["right_joints"]).all() and np.isfinite(d["R_c2w"]).all()
print("smoke test OK")
PY
  mark smoke
fi
log "HaWoR setup complete (env: $ENV_PREFIX)"
