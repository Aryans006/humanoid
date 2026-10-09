"""Sim episodes (.npz from sim_record.py) -> local LeRobot datasets (train / val) per split.json.
Run in the lerobot env:  python to_lerobot.py <episodes_dir> <split.json> <out_root>"""
import json, shutil, sys
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ep_dir, split, out = Path(sys.argv[1]).resolve(), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]).resolve()
IMG = {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
FEATURES = {
    # smolvla_base's camera slots: camera1 = agentview (front), camera2 = wrist, camera3 = side view
    "observation.images.camera1": IMG, "observation.images.camera2": IMG, "observation.images.camera3": IMG,
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["x", "y", "z", "rx", "ry", "rz", "g1", "g2"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
}
for name in ("train", "val"):
    if not split.get(name):
        print(f"{name}: no episodes, skipped")  # checkpoints are picked by sim rollout, not val loss
        continue
    root = out / f"cube2box_{name}"
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=f"local/cube2box_{name}", fps=20, features=FEATURES, root=root,
                               robot_type="panda", use_videos=True)
    for ep in split[name]:
        e = np.load(ep_dir / f"{ep}.npz")
        task = str(e["task"])
        for k in range(len(e["action"])):
            ds.add_frame({"observation.images.camera1": e["image"][k], "observation.images.camera2": e["wrist_image"][k],
                          "observation.images.camera3": e["side_image"][k],
                          "observation.state": e["state"][k], "action": e["action"][k], "task": task})
        ds.save_episode()
    ds.finalize()
    print(f"{name}: {len(split[name])} episodes -> {root}")
