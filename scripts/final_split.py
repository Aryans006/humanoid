"""Train/val split over the recorded sim episodes.

Eligible = all pipeline gates passed (export 'accepted') AND the recorded sim episode succeeded.

By default every eligible episode goes to training and nothing is held out. That is deliberate:
checkpoints are chosen by rolling the policy out in sim (scripts/eval_policy.py), not by
validation loss. On the 2026-10-07 run validation loss actively pointed the wrong way -- it was
computed on a single episode, and action-MSE against one demo's trace fell as the policy got
*worse* at the task (success correlated with val loss at Spearman +0.83, p = 0.0005).

  python scripts/final_split.py <session> [--per N]

``--per N`` restores a holdout: the N best-scoring eligible episodes per start position train and
the rest validate.
"""
import argparse, csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("session")
ap.add_argument("--per", type=int, default=None, help="per-position training cap; default: no holdout")
a = ap.parse_args()

proc, eps = ROOT / "data/processed" / a.session, ROOT / "data/sim_episodes" / a.session
idx = json.loads((eps / "index.json").read_text())["episodes"]
rows = list(csv.DictReader(open(proc / "selection.csv")))
train, val, notes, by_pos = [], [], [], {}
for p in sorted({r["position"] for r in rows if r["position"]}):
    ok = []
    for r in sorted((r for r in rows if r["position"] == p), key=lambda r: -float(r["score"])):
        j = proc / f"{r['clip']}.json"
        acc = j.exists() and json.loads(j.read_text()).get("accepted", False)
        suc = idx.get(f"{r['clip']}_r0", {}).get("success", False)
        if acc and suc:
            ok.append(f"{r['clip']}_r0")
        else:
            notes.append(f"{r['clip']} ({p}) excluded: accepted={acc} sim_success={suc}")
    cut = len(ok) if a.per is None else a.per
    train += ok[:cut]
    val += ok[cut:]
    by_pos[p] = {"train": len(ok[:cut]), "val": len(ok[cut:])}
    if a.per is not None and len(ok) < a.per:
        notes.append(f"{p}: only {len(ok)} eligible (<{a.per})")

(eps / "split.json").write_text(json.dumps(
    {"train": train, "val": val, "by_position": by_pos, "notes": notes}, indent=2))
print(f"train {len(train)} val {len(val)}  " + "  ".join(f"{p}:{v['train']}+{v['val']}" for p, v in by_pos.items()))
print("; ".join(n for n in notes if "only" in n) or "(no per-position cap)")
