#!/usr/bin/env bash
# Run on the GPU machine after push.sh. Sets HaWoR up on first use, then runs it on every
# pushed clip (data/interim/<session>/<demo>/undistorted.mp4 -> hawor.npz + hawor.log).
# Clips that already have hawor.npz are skipped unless --force.
#
#   scripts/remote/process.sh [--force] [--only 'demo_*'] [session ...]   # default: all sessions
#   scripts/remote/process.sh --video clip.mp4 --focal 1400     # one-off clip, output next to it
#
# Focal length comes from K_u in configs/camera.yaml (written by S0 calibration).
# Then on your machine: scripts/remote/pull.sh user@host
set -o pipefail   # no -u: conda activate hooks reference unset variables
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HAWOR_DIR=$ROOT/third_party/HaWoR
ENV_NAME=${HAWOR_ENV:-hawor}
CONDA_ROOT=${CONDA_ROOT:-$ROOT/third_party/miniforge3}
export XDG_CACHE_HOME=$ROOT/.cache PIP_CACHE_DIR=$ROOT/.cache/pip TORCH_HOME=$ROOT/.cache/torch \
       HF_HOME=$ROOT/.cache/huggingface TMPDIR=$ROOT/.cache/tmp
mkdir -p "$TMPDIR"

FORCE=0; VIDEO=""; FOCAL=""; SESSIONS=(); ONLY="demo_*"
while [ $# -gt 0 ]; do
  case $1 in
    --force) FORCE=1 ;;
    --video) VIDEO=$2; shift ;;
    --focal) FOCAL=$2; shift ;;
    --only) ONLY=$2; shift ;;   # clip-name glob, default demo_* (static/known_pos have no hands)
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) SESSIONS+=("$1") ;;
  esac
  shift
done

if ! [ -f "$ROOT/third_party/.hawor_setup/smoke" ]; then
  "$ROOT/scripts/remote/setup_hawor.sh" || { echo "setup failed; fix the error above and rerun"; exit 1; }
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
export CONDA_REGISTER_ENVS=false
conda activate "$CONDA_ROOT/envs/$ENV_NAME"   # by path: never a same-named env elsewhere

run_one() {  # video out_dir focal
  local video=$1 dir=$2 focal=$3 ts=()
  [ -f "$dir/timestamps.npy" ] && ts=(--timestamps "$dir/timestamps.npy")
  python "$ROOT/ego2libero/hands/hawor_driver.py" --hawor-root "$HAWOR_DIR" \
    --video "$video" --focal "$focal" "${ts[@]}" \
    --work "$ROOT/work/${dir#"$ROOT/data/interim/"}" --out "$dir/hawor.npz" 2>&1 | tee "$dir/hawor.log"
  return "${PIPESTATUS[0]}"
}

if [ -n "$VIDEO" ]; then
  [ -n "$FOCAL" ] || { echo "--video needs --focal"; exit 1; }
  run_one "$(realpath "$VIDEO")" "$(dirname "$(realpath "$VIDEO")")" "$FOCAL"; exit $?
fi

FOCAL=$(python - "$ROOT/configs/camera.yaml" <<'PY'
import sys, yaml
K = (yaml.safe_load(open(sys.argv[1])) or {}).get("K_u")
if K is None:
    sys.exit("configs/camera.yaml has no K_u yet: run calibration (S0) locally and push again")
if abs(K[0][0] - K[1][1]) > 0.5:
    sys.exit(f"K_u fx {K[0][0]} != fy {K[1][1]}; HaWoR needs square pixels")
print(K[0][0])
PY
) || exit 1

[ ${#SESSIONS[@]} -gt 0 ] || mapfile -t SESSIONS < <(ls -1 "$ROOT/data/interim" 2>/dev/null)
ok=(); failed=(); skipped=()
for s in "${SESSIONS[@]}"; do
  for vid in "$ROOT"/data/interim/"$s"/$ONLY/undistorted.mp4; do
    [ -f "$vid" ] || continue
    dir=$(dirname "$vid"); name="$s/$(basename "$dir")"
    if [ -f "$dir/hawor.npz" ] && [ $FORCE -eq 0 ]; then skipped+=("$name"); continue; fi
    echo -e "\n\033[1m== $name\033[0m"
    if run_one "$vid" "$dir" "$FOCAL"; then ok+=("$name"); else failed+=("$name"); fi
  done
done
echo -e "\n== summary: ${#ok[@]} done, ${#skipped[@]} skipped (exist), ${#failed[@]} failed"
for f in "${failed[@]}"; do echo "   FAILED $f  (see data/interim/$f/hawor.log)"; done
[ ${#failed[@]} -eq 0 ]
