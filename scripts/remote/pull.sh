#!/usr/bin/env bash
# Fetch HaWoR results (hawor.npz + hawor.log per demo) back from the GPU machine.
#
#   scripts/remote/pull.sh user@host [session ...]      # default: every remote session
set -euo pipefail
[ $# -ge 1 ] || { sed -n '2,4p' "$0"; exit 1; }
HOST=$1; shift
REMOTE_DIR=${REMOTE_DIR:-ego2libero}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SESSIONS=("$@")
if [ ${#SESSIONS[@]} -eq 0 ]; then
  mapfile -t SESSIONS < <(ssh ${SSH_OPTS:-} "$HOST" "ls -1 '$REMOTE_DIR/data/interim' 2>/dev/null")
fi
for s in "${SESSIONS[@]}"; do
  echo "== $s"
  mkdir -p "$ROOT/data/interim/$s"
  rsync -az --info=progress2 -e "ssh ${SSH_OPTS:-}" --prune-empty-dirs --include '*/' \
    --include 'hawor.npz' --include 'hawor.log' --exclude '*' \
    "$HOST:$REMOTE_DIR/data/interim/$s/" "$ROOT/data/interim/$s/"
done
