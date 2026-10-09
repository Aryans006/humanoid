"""A frozen SmolVLA base policy, in-process or behind a pipe.

Two interchangeable classes, both exposing ``reset()`` and ``chunk(images, state, task)``:

* ``InProcessBasePolicy`` -- torch in this interpreter. Use on a GPU box or in Colab.
* ``BasePolicy`` -- a subprocess speaking over a pipe. The wire format is a JSON header plus raw
  bytes, so two interpreters never have to agree on a pickle.

**Pin numpy < 2 wherever the simulator runs.** lerobot's rerun-sdk dependency asks for numpy >= 2,
but rerun is never imported on the inference path and lerobot runs fine on 1.26. numpy 2 is not
merely untested with robosuite 1.4.1 -- it silently changes what MuJoCo renders. Measured on the
identical scene, with identical mujoco 2.3.7 / PyOpenGL 3.1.10 / glfw and byte-identical physics:

    camera              numpy 1.26.4    numpy 2.2.6
    agentview           mean 216.10     mean 210.15
    robot0_eye_in_hand  mean 208.19     mean 194.59
    sideview            mean 164.17     mean 150.14

Deterministic in each environment, different between them. SmolVLA is a vision policy, so the
darker images are off-distribution: the same checkpoint over the same 40 start poses scored 20/40
under numpy 1.26 and 7-9/40 under numpy 2. An entire residual-RL study was run against that
crippled baseline before this was caught.

``seed`` matters: SmolVLA resamples flow-matching noise on every call, so the same observation
gives actions differing by up to 0.17 -- comparable to a 0.2 residual. Seeding per chunk makes
baseline-vs-residual comparisons paired instead of needing ~4x the rollouts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class InProcessBasePolicy:
    """Same interface as ``BasePolicy``, but torch runs here. Use on GPU / in Colab."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", seed: int | None = None):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        ck = Path(checkpoint)
        if (ck / "pretrained_model").is_dir():
            ck = ck / "pretrained_model"
        self.checkpoint, self.device, self.seed = ck, device, seed
        self.policy = SmolVLAPolicy.from_pretrained(ck).to(device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=ck,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}})
        self._n = 0

    def reset(self) -> None:
        self.policy.reset()
        self._n = 0

    def chunk(self, images: dict, state, task: str) -> np.ndarray:
        torch = self.torch
        obs = {f"observation.images.{k}": torch.from_numpy(
            np.ascontiguousarray(v.transpose(2, 0, 1), np.float32) / 255.0).unsqueeze(0)
            for k, v in images.items()}
        obs["observation.state"] = torch.from_numpy(np.asarray(state, np.float32)).unsqueeze(0)
        obs["task"] = task
        if self.seed is not None:
            torch.manual_seed(self.seed + self._n)  # reproducible flow-matching noise
        self._n += 1
        with torch.no_grad():
            a = self.post(self.policy.predict_action_chunk(self.pre(obs))[0])
        return np.asarray(a.cpu() if hasattr(a, "cpu") else a, np.float32)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BasePolicy:
    """``chunk(images, state, task)`` returns the whole (n_action_steps, 7) action chunk."""

    def __init__(self, checkpoint: str | Path, python: str | Path | None = None,
                 server: str | Path | None = None):
        ck = Path(checkpoint)
        if (ck / "pretrained_model").is_dir():
            ck = ck / "pretrained_model"
        self.checkpoint = ck
        python = Path(python or ROOT / ".venv-eval/bin/python")
        server = Path(server or ROOT / "scripts/policy_server.py")
        self.p = subprocess.Popen([str(python), str(server), str(ck)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  env={**os.environ, "HF_HOME": str(ROOT / "data/.hf_cache")})
        self._recv()  # ready

    def _send(self, msg: dict, payload: bytes = b"") -> None:
        msg["nbytes"] = len(payload)
        self.p.stdin.write((json.dumps(msg) + "\n").encode())
        self.p.stdin.write(payload)
        self.p.stdin.flush()

    def _recv(self):
        # skip any stray non-JSON line: a library that prints to stdout would
        # otherwise corrupt the protocol stream
        while True:
            head = self.p.stdout.readline()
            if not head:
                raise RuntimeError(f"policy server for {self.checkpoint} died")
            try:
                json.loads(head)
                break
            except Exception:
                sys.stderr.write('[policy server] ' + head.decode(errors='replace'))
        msg = json.loads(head)
        n = msg.get("nbytes", 0)
        buf = b""
        while len(buf) < n:
            c = self.p.stdout.read(n - len(buf))
            if not c:
                raise RuntimeError("policy server died mid-message")
            buf += c
        return msg, buf

    def reset(self) -> None:
        self._send({"cmd": "reset"})
        self._recv()

    def _call(self, cmd: str, images: dict, state, task: str) -> np.ndarray:
        payload = b"".join(np.ascontiguousarray(v, np.uint8).tobytes() for v in images.values())
        payload += np.ascontiguousarray(state, np.float32).tobytes()
        self._send({"cmd": cmd, "images": {k: list(v.shape) for k, v in images.items()},
                    "state": [len(state)], "task": task}, payload)
        msg, buf = self._recv()
        return np.frombuffer(buf, np.float32).reshape(msg["action"]).copy()

    def chunk(self, images: dict, state, task: str) -> np.ndarray:
        return self._call("chunk", images, state, task)

    def act(self, images: dict, state, task: str) -> np.ndarray:
        return self._call("act", images, state, task)

    def close(self) -> None:
        try:
            self._send({"cmd": "quit"})
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
