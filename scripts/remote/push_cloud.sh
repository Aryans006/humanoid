#!/usr/bin/env bash
# Stage (and optionally upload) everything a rented GPU box needs to train SmolVLA.
#
#   scripts/remote/push_cloud.sh                      # just stage into data/cloud_bundle/
#   scripts/remote/push_cloud.sh user@host [port]     # stage, then rsync it over
#
# The bundle is small: the dataset is video-encoded, and the base policy downloads on the remote
# from lerobot/smolvla_base rather than being shipped.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
OUT=data/cloud_bundle/ego2libero_train
rm -rf "$OUT"; mkdir -p "$OUT/data" "$OUT/scripts/remote"

[ -d data/lerobot/cube2box_train ] || { echo "build the dataset first: scripts/remote/to_lerobot.py"; exit 1; }
cp -r data/lerobot "$OUT/data/lerobot"
cp scripts/remote/cloud_setup.sh "$OUT/cloud_setup.sh"
cp scripts/remote/to_lerobot.py "$OUT/scripts/remote/"
chmod +x "$OUT/cloud_setup.sh"

echo "staged $OUT"
du -sh "$OUT"
find "$OUT" -maxdepth 2 -type d | sed "s|$OUT|  .|"

if [ $# -ge 1 ]; then
  HOST=$1; PORT=${2:-22}
  echo "uploading to $HOST:~ (port $PORT)"
  rsync -az --info=progress2 -e "ssh -p $PORT" "$OUT" "$HOST:~/"
  echo
  echo "now, on the box:   cd ~/ego2libero_train && bash cloud_setup.sh 40000"
fi
