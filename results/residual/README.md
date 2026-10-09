# Residual RL runs

**Runs v1-v4 are VOID.** They ran in `.venv-eval` on numpy 2.2.6, which silently changes what
MuJoCo renders (~14 levels darker; same mujoco/PyOpenGL, byte-identical physics). SmolVLA is a
vision policy, so the base it was asked to correct was crippled to ~20% instead of ~50%. Pin
numpy 1.26.4 wherever the simulator runs; see `ego2libero/sim/base_policy.py`.

| run | residual dims | what it found | valid? |
| --- | --- | --- | --- |
| `026000_cpu_loud_v1` | 7 | exploration std 1 = 0.56 cm random walk/step; lifted 2% vs base 45% | no |
| `026000_cpu_7dim_v2` | 7 | SAC learned a sustained +0.33 rotation-y tilt residual | no |
| `026000_pos_badreward_v3` | 3 | reward punished progress 8x; return rose as lifting collapsed | no |
| `026000_pot` | 3 | residual lost 15-5 paired -- but against the numpy-2 baseline | no |
| `026000_fixed` | 3 | first run in a correct environment | - |

The three design bugs found in v1-v3 were real and are fixed with tests
(`tests/test_sac.py`, `tests/test_residual.py`). Only the *measurements* are void.
