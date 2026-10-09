"""Validation loss for every training checkpoint (lerobot has no offline val loss).
Writes <run>/val_loss.csv: step, train_loss (from the train log), val_loss (held-out episodes).
Usage: python val_watch.py <run_dir> <val_root> <train_log>   (exits once the final checkpoint is done)"""
import csv, re, sys, time
from pathlib import Path
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

run, val_root, log = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
final = int(sys.argv[4]) if len(sys.argv) > 4 else 40000
out = run / "val_loss.csv"
done = set()
if out.exists():
    done = {int(r["step"]) for r in csv.DictReader(open(out))}

def train_loss_at(step):
    best = None
    for line in log.read_text(errors="ignore").splitlines():
        m = re.search(r"step:(\d+(?:\.\d+)?)(K?)\b.*?\bloss:([0-9.]+)", line)
        if m:
            s = float(m.group(1)) * (1000 if m.group(2) else 1)
            if s <= step:
                best = float(m.group(3))
    return best

while True:
    ckpts = sorted((int(p.name), p) for p in (run / "checkpoints").glob("[0-9]*") if p.name.isdigit()) if (run / "checkpoints").exists() else []
    for step, p in ckpts:
        if step in done or not (p / "pretrained_model" / "model.safetensors").exists():
            continue
        try:
            pm = p / "pretrained_model"
            policy = SmolVLAPolicy.from_pretrained(pm).to("cuda").eval()
            pre, _ = make_pre_post_processors(policy.config, pretrained_path=str(pm))
            meta = LeRobotDatasetMetadata("local/cube2box_val", root=val_root)
            ds = LeRobotDataset("local/cube2box_val", root=val_root, delta_timestamps=resolve_delta_timestamps(policy.config, meta), video_backend="pyav")
            g = torch.Generator().manual_seed(0)
            dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, generator=g, num_workers=4)
            torch.manual_seed(0)
            tot, n = 0.0, 0
            with torch.no_grad():
                for i, batch in enumerate(dl):
                    if i >= 40:
                        break
                    loss, _ = policy.forward(pre(batch))
                    tot, n = tot + float(loss), n + 1
            del policy
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); time.sleep(120); break
        new = not out.exists()
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["step", "train_loss", "val_loss"])
            w.writerow([step, train_loss_at(step), round(tot / max(n, 1), 6)])
        done.add(step)
    if final in done:
        break
    time.sleep(60)
