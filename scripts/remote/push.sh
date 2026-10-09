#!/usr/bin/env bash
# Copy code, configs, MANO files and the undistorted clips to the GPU machine.
#
#   scripts/remote/push.sh user@host [session ...]      # default: every session in data/interim
#
# Env: REMOTE_DIR (default ego2libero, relative to the remote home), SSH_OPTS (e.g. "-p 2222"),
#      MANO_DIR (local folder holding MANO_RIGHT.pkl and MANO_LEFT.pkl; default third_party/mano).
# Then on the remote:  cd ~/ego2libero && scripts/remote/process.sh
set -euo pipefail
[ $# -ge 1 ] || { sed -n '2,9p' "$0"; exit 1; }
HOST=$1; shift
REMOTE_DIR=${REMOTE_DIR:-ego2libero}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MANO_DIR=${MANO_DIR:-$ROOT/third_party/mano}
RSYNC=(rsync -az --info=progress2 -e "ssh ${SSH_OPTS:-}")

ssh ${SSH_OPTS:-} "$HOST" "mkdir -p '$REMOTE_DIR/data/interim' '$REMOTE_DIR/third_party/mano'"

echo "== code + configs"
"${RSYNC[@]}" --exclude .venv --exclude data --exclude tags --exclude work --exclude '__pycache__' --exclude .pytest_cache --exclude .cache \
  --exclude 'third_party/HaWoR' --exclude 'third_party/mano' --exclude '*.egg-info' \
  "$ROOT/" "$HOST:$REMOTE_DIR/"

if [ -f "$MANO_DIR/MANO_RIGHT.pkl" ] && [ -f "$MANO_DIR/MANO_LEFT.pkl" ]; then
  echo "== MANO"
  "${RSYNC[@]}" "$MANO_DIR/MANO_RIGHT.pkl" "$MANO_DIR/MANO_LEFT.pkl" "$HOST:$REMOTE_DIR/third_party/mano/"
else
  echo "!! MANO_RIGHT.pkl / MANO_LEFT.pkl not found in $MANO_DIR (register at mano.is.tue.mpg.de)."
  echo "   HaWoR cannot run without them; set MANO_DIR or copy them there and push again."
fi

if [ $# -eq 0 ]; then
  mapfile -t SESSIONS < <(find "$ROOT/data/interim" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort)
else
  SESSIONS=("$@")
fi
for s in "${SESSIONS[@]}"; do
  echo "== clips: $s"
  "${RSYNC[@]}" --prune-empty-dirs --include '*/' --include 'undistorted.mp4' --include 'timestamps.npy' \
    --exclude '*' "$ROOT/data/interim/$s/" "$HOST:$REMOTE_DIR/data/interim/$s/"
done
[ ${#SESSIONS[@]} -gt 0 ] || echo "(no sessions in data/interim yet; only code was pushed)"
echo "done. next: ssh ${SSH_OPTS:-}$HOST, then: cd $REMOTE_DIR && tmux new -s hawor scripts/remote/process.sh"
