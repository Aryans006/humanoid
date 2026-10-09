"""Patch HaWoR's DROID-SLAM build for the GPU at hand (run inside the HaWoR repo).

Upstream hard-codes -gencode for sm_60..sm_86 only (no PTX), so on newer GPUs (e.g. RTX 5090,
sm_120) the extension builds but every kernel launch fails with "no kernel image is
available". Also replaces the deprecated Tensor::type() in AT_DISPATCH calls. Idempotent."""
import re
import sys
from pathlib import Path

cap = sys.argv[1].replace(".", "")  # "12.0" -> "120"
root = Path("thirdparty/DROID-SLAM")
setup = root / "setup.py"
s = setup.read_text()
flags = f"'-gencode=arch=compute_{cap},code=sm_{cap}', '-gencode=arch=compute_{cap},code=compute_{cap}',"
s2 = re.sub(r"(\s*'-gencode=arch=compute_\d+,code=(?:sm|compute)_\d+',?\s*)+", "\n                    " + flags + "\n", s)
if s2 != s:
    setup.write_text(s2)
    print(f"setup.py: gencode -> sm_{cap} (+PTX)")
# Dispatch macros: pass ScalarType (Tensor::scalar_type()) instead of the removed
# DeprecatedTypeProperties (Tensor::type()); lietorch's own macro then takes it directly.
n = 0
srcs = [p for d in ("src", "thirdparty/lietorch/lietorch/src") for ext in ("*.cu", "*.cpp") for p in root.glob(f"{d}/{ext}")]
for f in srcs:
    t = f.read_text()
    t2 = re.sub(r"((?:AT_DISPATCH_\w+|DISPATCH_GROUP_AND_FLOATING_TYPES)\(\s*(?:\w+,\s*)?[\w.\[\]]+)\.type\(\)",
                r"\1.scalar_type()", t)
    if t2 != t:
        f.write_text(t2)
        n += 1
print(f".type() -> .scalar_type() in {n} files")
hdr = root / "thirdparty/lietorch/lietorch/include/dispatch.h"
h = hdr.read_text()
h2 = h.replace("at::ScalarType _st = ::detail::scalar_type(the_type);", "at::ScalarType _st = the_type;")
if h2 != h:
    hdr.write_text(h2)
    print("dispatch.h: macro takes ScalarType")
