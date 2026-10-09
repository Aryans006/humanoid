"""SmolVLA inference server, one checkpoint per process (plan stage: policy eval).

Run in .venv-eval (torch + lerobot). It exists as a separate process because lerobot 0.4.4
needs numpy >= 2 (via rerun-sdk) while robosuite 1.4.1 needs numpy < 2, so the policy and the
simulator cannot share an interpreter. ``scripts/eval_policy.py`` spawns this and talks to it
over stdin/stdout with a length-prefixed JSON header plus raw bytes -- no pickle, so the two
numpy major versions never have to agree on a wire format.

  .venv-eval/bin/python scripts/policy_server.py <pretrained_model_dir>

Protocol, one line of JSON then that many bytes of payload:
  -> {"cmd": "act",   "state": [8], "images": {"camera1": [256,256,3], ...}, "task": "..."}
  <- {"action": [7]}                          one action, the policy owns the chunk queue
  -> {"cmd": "chunk", "state": [8], "images": {...}, "task": "..."}
  <- {"action": [50, 7]}                      the whole chunk, for callers that drive it
  -> {"cmd": "reset"}   <- {"ok": true}       (clears the action-chunk queue between episodes)
  -> {"cmd": "quit"}

``chunk`` exists for residual RL: the caller replays the 50 actions itself and only comes back on
a boundary, so the cameras need rendering 50x less often. The forward costs ~1.0-1.4 s on CPU and
saturates every core, so batching it buys almost nothing -- about 45-50 env-steps/s is the ceiling.
"""

from __future__ import annotations

import json
import struct
import sys

import numpy as np
import torch


def _read(f) -> tuple[dict, bytes]:
    head = f.readline()
    if not head:
        return {"cmd": "quit"}, b""
    msg = json.loads(head)
    n = msg.get("nbytes", 0)
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("short read")
        buf += chunk
    return msg, buf


def _write(f, msg: dict, payload: bytes = b"") -> None:
    msg["nbytes"] = len(payload)
    f.write((json.dumps(msg) + "\n").encode())
    f.write(payload)
    f.flush()


def main() -> None:
    ckpt = sys.argv[1]
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(ckpt)
    policy.eval()
    # the checkpoint was saved with device "cuda" baked into its processor steps
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})

    inp, outp = sys.stdin.buffer, sys.stdout.buffer
    _write(outp, {"ready": True})
    while True:
        msg, buf = _read(inp)
        cmd = msg["cmd"]
        if cmd == "quit":
            return
        if cmd == "reset":
            policy.reset()
            _write(outp, {"ok": True})
            continue

        off = 0
        obs = {}
        for name, shape in msg["images"].items():
            n = int(np.prod(shape))
            img = np.frombuffer(buf, np.uint8, count=n, offset=off).reshape(shape)
            off += n
            # training fed LeRobotDataset uint8 HWC; it serves float32 CHW in [0, 1]
            obs[f"observation.images.{name}"] = torch.from_numpy(
                np.ascontiguousarray(img.transpose(2, 0, 1), np.float32) / 255.0).unsqueeze(0)
        state = np.frombuffer(buf, np.float32, count=msg["state"][0], offset=off)
        obs["observation.state"] = torch.from_numpy(state.copy()).unsqueeze(0)
        obs["task"] = msg["task"]

        with torch.no_grad():
            batch = pre(obs)
            if cmd == "chunk":
                a = np.asarray(post(policy.predict_action_chunk(batch)[0]), np.float32)
            else:
                a = np.asarray(post(policy.select_action(batch)), np.float32).reshape(-1)
        _write(outp, {"action": list(a.shape)}, np.ascontiguousarray(a).tobytes())


if __name__ == "__main__":
    main()
