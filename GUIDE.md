# Guide for Aryan

What you need to do, in what order, and how to hand each piece back.
The design is in [plan.md](plan.md). This file is the hands-on part.

---

## Where things stand

- **Done (code):** M0, which covers geometry, printable sheets and the GPU scripts. M1 is also done: calibration, undistortion, tag detection, board pose, box pose, overlay video and the M1 gates, tested end to end on rendered video.
- **Done (you):** board printed and taped down, cube folded, known spots marked and photographed.
- **Waiting on:** the clips (calib, static, known_pos, demo_000–002) in `data/raw/2026-10-03_setup/`.

---

## Your to-do list

Do these roughly in order. Steps 1–5 take an afternoon. Step 7 can happen any time.

- [ ] **1. Print** the three sheets in [tags/](tags/). See [section 1](#1-printing).
- [ ] **2. Build** the board and the box. See [sections 2–3](#2-building-the-board).
- [ ] **3. Measure** and fill in [measurements.yaml](measurements.yaml). See [section 4](#4-measuring).
- [ ] **4. Take photos** of the board and box. See [section 5](#5-how-to-get-the-info-to-me).
- [ ] **5. Record** the calib, static, known-position and 3 pilot clips. See [section 6](#6-recording).
- [ ] **6. Copy** the clips to `data/raw/<session>/`, then tell me. See [section 5](#5-how-to-get-the-info-to-me).
- [ ] **7. Get a GPU machine and MANO files**, and send me the IP. See [section 7](#7-gpu-machine).

---

## 1. Printing

| Sheet | What it is |
| --- | --- |
| `tags/board_A4.pdf` | Table board: 12 tags, IDs 0–11 |
| `tags/box_net_A4.pdf` | Fold-up paper cube with tags 20–25 printed on it |
| `tags/charuco_A4.pdf` | Checkerboard target, only for camera calibration |

**Printer settings**

- Scale: **100% / "Actual size"**. Never "Fit to page".
- Paper: plain **matte** paper. Glossy paper reflects room lights and tags stop being detected.
- Quality: normal or best, black and white is fine.

**Check right after printing:** every sheet has a ruler labeled **100 mm**. Measure it.
If it isn't 100 ± 0.5 mm, the printer scaled the page. Fix the setting and reprint before going further.

---

## 2. Building the board

1. Glue the board sheet **flat** onto rigid cardboard or foam board. Use spray glue or a glue stick spread evenly, and avoid bubbles and waves.
2. Tape it to the table so it **cannot move** during a whole session.
3. Leave the area around it clear. The box moves on or near the board.

> If the board moves between demos, the goal region and known spots stop matching. If it gets bumped, re-measure those or start a new session.

---

## 3. Building the box

The box is a **paper cube folded from `tags/box_net_A4.pdf`**: 50 mm faces with 40 mm tags.
The net bakes in which tag goes on which face and which way up, so there are no choices to get wrong.

**Steps**

1. **Print** `box_net_A4.pdf` with the same settings as the other sheets. Thicker matte paper or card (160–250 g/m²) is better if your printer takes it.
2. **Measure the tags now**, while the sheet is flat (see [section 4](#4-measuring)).
3. **Cut** along the **solid** lines, including the grey glue tabs.
4. **Fold** every **dashed** line and every tab **away from you**, so the printed side stays outside.
5. **Fill it.** Put a block of foam, wood or tightly packed card inside before closing it, so it doesn't dent when you pinch it.
6. **Glue** the tabs on the inside. Press each face flat while the glue sets.

> **Why it has to be filled:** when you grasp the cube, the faces must not bend. A dented face bends its tag, which throws off the box pose. It also breaks the assumption that the box moves rigidly with your hand, which is what fills in frames where no tag is visible.

**Why the tags are smaller than the faces:** each tag needs a white border at least 1/8 of its own width on every side, or the detector can't find its outline. A 40 mm tag needs 5 mm, which is exactly what a 50 mm face leaves. (The earlier 39 mm tags on a 40 mm cube had only 0.5 mm.)

**For reference, the folded cube looks like this.** Stand it on tag 25 and face tag 23:

| Tag | Face | Its top edge points |
| --- | --- | --- |
| 22 | Back (far from you) | Up |
| 23 | Front (toward you) | Up |
| 20 | Right | Up |
| 21 | Left | Up |
| 24 | Top | Away from you (toward 22) |
| 25 | Bottom | Away from you (toward 22) |

`uv run python tools/print_tags.py --check` simulates folding the net in 3D and confirms that this matches the code's box model.

---

## 4. Measuring

Fill in [measurements.yaml](measurements.yaml). Everything is in **millimetres**. Calipers are best; otherwise use a steel ruler and read to 0.5 mm.

**Always measure the black square of a tag**: outer edge to outer edge of the black border, ignoring the white margin.

| What | How | Field in the sheet |
| --- | --- | --- |
| Scale bars | The 100 mm ruler on each sheet | `printer_check` |
| Board tag size | One tag's black square, e.g. tag 8 | `board.tag_edge_mm` |
| Board row span | Bottom row: left edge of tag 8 → right edge of tag 11 | `board.bottom_row_span_mm` |
| Board column span | Left column: bottom edge of tag 8 → top edge of tag 0 | `board.left_column_span_mm` |
| Box tags | Each tag's black square, measured flat before folding | `box_tags.*` |
| Cube size | Folded cube's outside edges: face 21 → 20, face 23 → 22, bottom → top | `box.x_mm`, `y_mm`, `z_mm` |
| Calibration board | Across all 7 squares on the long side, plus one small marker | `charuco.*` |

You don't measure the gaps between board tags directly. I work them out from the row and column spans, which is more accurate than measuring one gap.

### Table coordinates (for the known spots and the goal region)

The table frame starts at the **bottom-left corner of tag 8's black square**. The board PDF has an arrow labeled "W origin" pointing at it.

```
     y ↑
       │   tag 0   tag 1   tag 2   tag 3
       │   tag 4   tag 5   tag 6   tag 7
       │   tag 8   tag 9   tag 10  tag 11
       ●──────────────────────────────────→ x
     origin
```

- **x**: along the board's bottom edge, toward tag 11.
- **y**: toward tag 0.
- Everything on the board has positive x and y.

**Known spots:** stick two small tape crosses on the table (on or near the board), measure each one's (x, y) from the origin, and write them in `known_spots`. These let us check the pipeline's accuracy against a ruler.

---

## 5. How to get the info to me

Everything goes into this repo folder. Then send me a message saying what's ready, and I'll check it, write the configs, and run the pipeline.

### Measurements

Fill in [measurements.yaml](measurements.yaml), save it, and tell me **"measurements done"**.
Leave anything you couldn't measure as `null`. Don't edit the files in `configs/` yourself; I'll fill those in from the sheet and check the numbers against each other.

### Photos

Phone photos are fine. Put them in `data/raw/<session>/photos/`:

- The board taped to the table, whole board in frame
- The folded cube from 2–3 angles, so I can check it's square and the faces are flat
- The tape marks (known spots, goal region) next to the board

I can look at the images and catch assembly mistakes before they cost a recording session.

### Videos

**Copying from the phone:** use a USB cable, AirDrop, or the Files app's "original" export.
**Never** send them through WhatsApp, Telegram, Google Photos sharing or similar. Those re-encode the video and wreck both timing and quality.

**Folder layout.** Name the session `YYYY-MM-DD_<label>`, for example `2026-10-05_pilot`:

```
data/raw/2026-10-05_pilot/
  session.yaml        ← copy configs/session.template.yaml here and fill it in
  calib.mp4
  static.mp4
  known_pos.mp4
  box_refine.mp4      ← optional
  demo_000.mp4
  demo_001.mp4
  demo_002.mp4
  photos/
```

If the phone produces `.mov` files, that's fine; keep the original files and just rename them.

---

## 6. Recording

### Phone settings (identical for every clip, including calibration)

Use a manual camera app (**Blackmagic Camera** is free on iOS and Android).

- [ ] **1920 × 1080 at 60 fps**, same orientation (landscape) every time
- [ ] **Stabilization OFF**, including any "action", "enhanced" or "cinematic" mode
- [ ] **Zoom 0.5× in Blackmagic**, every clip. On the Pixel 10 that's the full main camera; Blackmagic's "1×" is a 2× digital crop. Never change it between the calib clip and the demos
- [ ] **Focus locked** at your working distance (about 50 cm, focused on the board)
- [ ] **Exposure, ISO and white balance locked**
- [ ] **Shutter and ISO.** Blur matters far more than grain, so raise ISO before slowing the shutter:
  - **Demos, static, known_pos:** shutter **1/250** (1/500 if you have the light), ISO **2000–3200**. A desk lamp pointed at the table lets you lower the ISO.
  - **Calib clip:** the board is held still, so **1/120** at ISO about 1400 is fine, as long as you pause at each pose.
  - Never slower than 1/120. At 60 fps the slowest possible is 1/60 anyway.

> Focus is part of the camera calibration. If you change focus after recording the calib clip, the calibration is wrong for everything that follows. Lock it once and record everything with that lock.

### Clips, in this order

| Clip | What to do | Length |
| --- | --- | --- |
| `calib` | Follow the **calib routine** below | ~60–90 s |
| `static` | Phone, board and box all completely still, with the board and box in view. Don't touch anything | 10 s |
| `known_pos` | Stand the box on tape spot A, hold still 3 s, move it to spot B, hold still 3 s | ~15 s |
| `box_refine` | *Optional.* Turn the box slowly in your hand so two faces are visible at a time | 20 s |
| `demo_000`… | One pick-and-place per clip (rules below) | ~5–15 s |

### Calib routine

The board lies flat on the table and you move the phone. Hold each pose still for about 1 s, because pauses give sharp frames.

1. **Distance:** hold the phone 40–60 cm away (your demo distance, focus locked there). The board should fill only **about a third** of the frame, not most of it.
2. **Corners:** move the phone so the board sits in each of the **four corners** of the frame, then the four edges, then the centre. Keep the phone roughly square-on.
3. **Tilts:** with the board in the centre, tilt the phone **30–45°** so you look at the board from the bottom, then the top, then the left, then the right. The board's far edge should look clearly smaller than its near edge.
4. **Mixed:** repeat a few tilted views with the board near the corners.

**The two mistakes the checker catches:** the board only ever flat-on to the camera (median tilt under about 10°), and the board never reaching the corners of the image. Either one gives a calibration that looks fine by its error number and is still wrong.

**Check it yourself before sending:**

```bash
uv run python -m ego2libero.calib.calibrate data/raw/<session>/calib.mp4 --dry-run
```

It should print no `PROBLEM:` lines.

### Rules for every demo

- **Start and end still:** 1–2 s with hand and box not moving, and at least 4 board tags plus 1 box tag visible.
- **Right hand only**, bare, short sleeves.
- **The camera looks straight down, so the cube's top tag is usually the only one it sees.** Keep your hand off the top face: pinch the cube on two opposite side faces, coming in from the side, and keep your fingers below the top edge. Every frame where the top tag is covered has to be filled in from the hand track.
- **Put the cube down with any face up,** but keep it inside the camera view and away from the image edges.
- Keep the box **on or over the board**. Move smoothly: no throws, no regrasps.
- **Vary** the box's starting position and rotation between demos.

**Pilots first.** Record just 3 demos, send them, and wait until they've been through the whole pipeline before recording the full set of 25+. Capture mistakes are the most expensive bugs in this project.

---

## 7. GPU machine

HaWoR, the hand-tracking model, **needs an NVIDIA GPU**. It can't run on CPU because part of it only exists as GPU code.

### What the machine needs

- Linux (Ubuntu 22.04 is the easiest) with the NVIDIA driver installed (`nvidia-smi` works)
- Roughly **12 GB+ of GPU memory** recommended (estimate, not yet measured) and **~30 GB free disk**
- `git`, `curl`, `rsync`, and `libegl1` installed: `sudo apt install git curl rsync libegl1 libgl1`
- SSH access from this machine without a password: run `ssh-copy-id user@GPU_IP` once

Everything else (conda, PyTorch, CUDA toolkit, HaWoR, model weights) installs automatically on the first run.

### MANO hand model (needs your own account)

1. Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) and download the MANO model zip.
2. From it, put `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` in `third_party/mano/` in this repo.

### Running it

Once you have the machine, send me `user@IP` and I'll drive it. Or run it yourself:

```bash
scripts/remote/push.sh user@GPU_IP            # 1. here: send code, MANO files, clips
ssh user@GPU_IP                               # 2. log in to the GPU machine
cd ~/ego2libero && tmux new -s hawor scripts/remote/process.sh
                                              #    first run: ~30–60 min of installs + a self-test
scripts/remote/pull.sh user@GPU_IP            # 3. back here: fetch the results
```

`tmux` keeps the job running if your SSH connection drops. Reattach with `tmux attach -t hawor`.

> The remote scripts haven't run yet, because there's no GPU here. Expect one or two fixes on the first real run, most likely in dependency installs.

---

## 8. Things worth knowing

Short version of what I found while building M0. None of it needs action from you beyond what's above.

**HaWoR can't run on CPU.**
Its camera-tracking step is built only as GPU code. That's why the remote-machine scripts exist.

**HaWoR secretly converts video to 30 fps.**
That would break the frame-to-timestamp mapping for our 60 fps clips. Our wrapper extracts every frame itself, so nothing is lost.

**HaWoR assumes an ideal camera.**
It assumes the optical center is exactly at the image center and the pixels are square. The undistortion step (M1) produces images that meet those assumptions, and a check enforces it.

**Box position accuracy has a physical floor.**
From a single frame at arm's length, the box's distance from the camera is uncertain by about 1 mm (2.3 mm worst case in simulation). Sideways error is far smaller. The plan's "2 mm" target holds for about 90% of frames. Averaging over the still periods at the start and end of each demo takes care of the rest.

**Tags are 55 mm, not 60 mm.**
They were shrunk so the board fits on one A4 sheet. That's fine for detection at arm's length.

**The table origin moved.**
It's at the board's bottom-left corner (tag 8), not tag 0's top-left as in the plan, so that every table coordinate you measure is positive.

**OpenCV version.**
The environment uses OpenCV 5.0. The plan said 4.8+; the board layout was checked against 5.0's actual behavior.

---

## 9. Decisions still open (yours)

These don't block M1. I've left placeholders in `configs/` rather than guessing.

- [ ] **Mount:** head strap (egocentric) or a fixed tripod at head height?
- [ ] **Task:** a single place-on-target, or more? Where is the goal region?
- [ ] **LIBERO scene**, and where the board origin sits in it, given the Panda's reach (`configs/libero_frame.yaml`)
- [ ] **Final demo count** (plan says 25+)

---

## Quick reference

```bash
uv sync                                   # set up the Python environment
uv run pytest                             # run the tests
uv run python tools/print_tags.py --check # regenerate + verify the PDFs
scripts/remote/push.sh user@IP            # send to GPU machine
scripts/remote/pull.sh user@IP            # fetch HaWoR results
```

| File | What it's for |
| --- | --- |
| [measurements.yaml](measurements.yaml) | You fill this in |
| [configs/session.template.yaml](configs/session.template.yaml) | Copy into each session folder |
| [FINDINGS.md](FINDINGS.md) | Technical notes and deviations from the plan |
| [plan.md](plan.md) | The full design |
