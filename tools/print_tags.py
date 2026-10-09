"""Render the printable targets as A4-landscape vector PDFs at exact physical size.

  board_A4.pdf    3x4 AprilTag 36h11 GridBoard, ids 0-11 (table frame W)
  box_net_A4.pdf  fold-up cube net, AprilTag 36h11 ids 20-25 (or --box-style separate:
                  box_tags_A4.pdf, six tags with cut outlines for a solid box)
  charuco_A4.pdf  ChArUco calibration target (DICT_5X5_100, own ids, no clash with tags)

Print at 100% / "Actual size" (never "fit to page"), on matte paper. Then measure the
printed sizes and put the MEASURED values in configs/*.yaml; each page has a 100 mm scale
bar to check the printer's scaling first.

  uv run python tools/print_tags.py            # writes to tags/
  uv run python tools/print_tags.py --check    # also re-detects every page from a render
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

A4_W, A4_H = 297.0, 210.0  # mm, landscape
MM_PER_IN = 25.4
APRIL = cv2.aruco.DICT_APRILTAG_36h11
CHARUCO_DICT = cv2.aruco.DICT_5X5_100

# Box faces: id -> (label, face normal, which box dims span the face as (width, height)).
# Must match configs/box.yaml: side tags upright (top edge -> +z), top/bottom tags' top -> +y.
BOX_FACES = [
    (20, "+x side", (1, 2)),
    (22, "+y side", (0, 2)),
    (21, "-x side", (1, 2)),
    (23, "-y side", (0, 2)),
    (24, "+z top", (0, 1)),
    (25, "-z bottom", (0, 1)),
]


# --------------------------------------------------------------------------- drawing


def new_page():
    fig = plt.figure(figsize=(A4_W / MM_PER_IN, A4_H / MM_PER_IN))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, A4_W)
    ax.set_ylim(A4_H, 0)  # y down, like the page and OpenCV board coordinates
    ax.axis("off")
    return fig, ax


def _rects_path(rects: list[tuple[float, float, float, float]]) -> MplPath:
    """One compound path of axis-aligned rects (x, y, w, h): renders as a union, no seams."""
    verts, codes = [], []
    for x, y, w, h in rects:
        verts += [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        codes += [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def marker_rects(dictionary, marker_id: int, x: float, y: float, size: float, half_turn: bool = False):
    """Black cells of a marker whose black square has top-left (x, y) and edge ``size`` mm.
    ``half_turn`` prints it rotated 180 degrees on the page."""
    n = dictionary.markerSize + 2
    bits = cv2.aruco.generateImageMarker(dictionary, marker_id, n, borderBits=1)
    if half_turn:
        bits = np.rot90(bits, 2)
    cell = size / n
    rects = []
    for r in range(n):
        c = 0
        while c < n:  # merge horizontal runs of black cells
            if bits[r, c] == 0:
                c0 = c
                while c < n and bits[r, c] == 0:
                    c += 1
                rects.append((x + c0 * cell, y + r * cell, (c - c0) * cell, cell))
            else:
                c += 1
    return rects


def fill(ax, rects):
    ax.add_patch(PathPatch(_rects_path(rects), facecolor="black", edgecolor="none", lw=0, antialiased=False))


def scale_bar(ax, x, y, length=100.0, vertical=False):
    """A 100 mm ruler with 10 mm ticks, for checking print scaling before measuring tags."""
    kw = dict(color="black", lw=0.6)
    if vertical:
        ax.plot([x, x], [y, y + length], **kw)
        for k in range(0, int(length) + 1, 10):
            ax.plot([x - (1.5 if k % 50 else 3), x], [y + k, y + k], **kw)
        ax.text(x + 1.2, y + length / 2, f"{length:.0f} mm", rotation=90, va="center", ha="left", fontsize=6)
    else:
        ax.plot([x, x + length], [y, y], **kw)
        for k in range(0, int(length) + 1, 10):
            ax.plot([x + k, x + k], [y, y - (1.5 if k % 50 else 3)], **kw)
        ax.text(x + length / 2, y + 1.2, f"{length:.0f} mm", va="top", ha="center", fontsize=6)


def note(ax, x, y, text, **kw):
    ax.text(x, y, text, fontsize=kw.pop("fontsize", 6.5), va=kw.pop("va", "top"), ha=kw.pop("ha", "left"),
            family="monospace", **kw)


# --------------------------------------------------------------------------- pages


def board_page(tag_mm: float, gap_mm: float, cols=4, rows=3, first_id=0):
    d = cv2.aruco.getPredefinedDictionary(APRIL)
    ids = np.arange(first_id, first_id + cols * rows)
    board = cv2.aruco.GridBoard((cols, rows), tag_mm, gap_mm, d, ids)
    w, h = cols * tag_mm + (cols - 1) * gap_mm, rows * tag_mm + (rows - 1) * gap_mm
    if w > A4_W - 20 or h > A4_H - 2 * (tag_mm / 8):
        raise SystemExit(f"board {w:.0f}x{h:.0f} mm does not fit A4 with margins; shrink --board-tag-mm/--board-gap-mm")
    ox, oy = (A4_W - w) / 2, (A4_H - h) / 2
    fig, ax = new_page()
    rects = []
    for corners, mid in zip(board.getObjPoints(), board.getIds().ravel()):
        x, y = np.asarray(corners)[0, :2]  # top-left corner of the marker, board coords (y down)
        rects += marker_rects(d, int(mid), ox + x, oy + y, tag_mm)
    fill(ax, rects)
    side = ox / 2
    scale_bar(ax, side, (A4_H - 100) / 2, vertical=True)
    ax.annotate("", xy=(ox - 0.8, oy + h + 0.8), xytext=(side, oy + h - 12),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black"))
    note(ax, side - 4, oy + h - 21, "W\norigin", fontsize=5)
    note(ax, A4_W - side, A4_H / 2,
         f"TABLE BOARD  tag36h11 ids {ids[0]}-{ids[-1]}  {rows}x{cols}\n"
         f"tag {tag_mm:g} mm  gap {gap_mm:g} mm (nominal)\n"
         "print 100% / actual size, matte\n"
         "measure -> configs/board.yaml\n"
         "W origin = bottom-left corner of tag 8\n"
         "x along bottom edge, y up page, z up",
         rotation=90, va="center", ha="center", fontsize=5.5)
    return fig, board, (ox, oy)


def box_tags_page(tag_mm: float, dims_mm: tuple[float, float, float], first_id=20):
    d = cv2.aruco.getPredefinedDictionary(APRIL)
    fig, ax = new_page()
    faces = [(i - 20 + first_id, lbl, span) for i, lbl, span in BOX_FACES]
    gap = 12.0
    rows = [faces[:4], faces[4:]]
    y = 14.0
    rects, placed = [], []
    for row in rows:
        sizes = [(dims_mm[a], dims_mm[b]) for _, _, (a, b) in row]
        x = (A4_W - (sum(s[0] for s in sizes) + gap * (len(row) - 1))) / 2
        row_h = max(s[1] for s in sizes)
        for (mid, lbl, _), (fw, fh) in zip(row, sizes):
            if tag_mm > min(fw, fh) - 2 * tag_mm / 8:
                raise SystemExit(f"{tag_mm} mm tag leaves < 1 bit of white border on a {fw}x{fh} mm face")
            cx, cy = x + fw / 2, y + fh / 2
            ax.add_patch(Rectangle((x, y), fw, fh, fill=False, ec="0.55", lw=0.4, ls=(0, (3, 2))))
            rects += marker_rects(d, mid, cx - tag_mm / 2, cy - tag_mm / 2, tag_mm)
            ax.annotate("", xy=(cx, y - 7), xytext=(cx, y - 1), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.3"))
            ax.text(cx + 1.5, y - 4, "top", fontsize=5, va="center", color="0.3")
            note(ax, x, y + fh + 1.5, f"ID {mid}  {lbl}\nface {fw:g}x{fh:g} mm", fontsize=5.5)
            placed.append(mid)
            x += fw + gap
        y += row_h + 18.0
    fill(ax, rects)
    scale_bar(ax, A4_W - 115, A4_H - 12)
    note(ax, 12, A4_H - 34,
         f"BOX TAGS  tag36h11  tag {tag_mm:g} mm (nominal)  box {dims_mm[0]:g}x{dims_mm[1]:g}x{dims_mm[2]:g} mm\n"
         "Pencil each 'top' arrow on the back, then cut on the dashed lines;\n"
         "each tag ends up centered on its face.\n"
         "Stand box on a small face. Sides 20, 22, 21, 23 counter-clockwise seen from\n"
         "above, upright ('top' arrow up). 24 on top and 25 underneath, 'top' arrow\n"
         "toward the face with 22. Matches configs/box.yaml. Measure tag + box -> box.yaml",
         fontsize=5.5)
    return fig, placed


def charuco_page(squares=(7, 5), square_mm=36.0, marker_mm=27.0):
    d = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    board = cv2.aruco.CharucoBoard(squares, square_mm, marker_mm, d)
    board.setLegacyPattern(False)
    nx, ny = squares
    w, h = nx * square_mm, ny * square_mm
    if w > A4_W - 20 or h > A4_H - 20:
        raise SystemExit(f"ChArUco {w:.0f}x{h:.0f} mm does not fit A4 with margins")
    ox, oy = (A4_W - w) / 2, (A4_H - h) / 2
    fig, ax = new_page()
    # Squares holding a marker are white; the rest of the checkerboard is black.
    has_marker = np.zeros((ny, nx), bool)
    rects = []
    for corners, mid in zip(board.getObjPoints(), board.getIds().ravel()):
        c = np.asarray(corners)[:, :2]
        cxm, cym = c.mean(0)
        has_marker[int(cym // square_mm), int(cxm // square_mm)] = True
        rects += marker_rects(d, int(mid), ox + c[0, 0], oy + c[0, 1], marker_mm)
    rects += [(ox + c * square_mm, oy + r * square_mm, square_mm, square_mm)
              for r in range(ny) for c in range(nx) if not has_marker[r, c]]
    fill(ax, rects)
    scale_bar(ax, (A4_W - 100) / 2, A4_H - 5)
    note(ax, ox, 3,
         f"CHARUCO  DICT_5X5_100  {nx}x{ny} squares  square {square_mm:g} mm  marker {marker_mm:g} mm (nominal)"
         "  -> measure into configs/camera.yaml; glue flat to rigid board", fontsize=5.5)
    return fig, board, (ox, oy)


# --------------------------------------------------------------------------- fold-up cube net
#
# Cross-shaped net, printed side out. Paper grid (col, row), rows top to bottom:
#
#             [24]                 24 = top, printed upside down so its top edge faces 22
#       [20]  [22]  [21]  [23]     side tags upright, left->right = counter-clockwise from above
#             [25]                 25 = bottom
#
# NET_PARENT is the hinge tree (child -> parent); every other shared cube edge gets a glue tab.
NET_CELLS = {20: (0, 1), 22: (1, 1), 21: (2, 1), 23: (3, 1), 24: (1, 0), 25: (1, 2)}
NET_HALF_TURN = {24}
NET_PARENT = {20: 22, 21: 22, 23: 21, 24: 22, 25: 22}
NET_TABS = [(20, "top"), (20, "bottom"), (21, "top"), (21, "bottom"), (23, "top"), (23, "bottom"), (23, "right")]


def net_fold(face_mm: float = 1.0) -> dict[int, dict[str, np.ndarray]]:
    """Fold the net in 3D (each hinge 90 deg, printed side ending up outside) and return, per
    face, its outward normal and the 3D directions of its printed tag's 'up' and 'right',
    expressed in the box frame defined by face 22 (normal +y, tag up +z)."""

    def rot(axis, ang):
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K

    n0 = np.array([0.0, 0.0, -1.0])  # page coords x right, y down => printed side faces -z
    pose = {22: (np.eye(3), np.zeros(3))}  # face -> (R, t): paper point p -> R p + t

    def corners(fid):
        c, r = NET_CELLS[fid]
        return np.array([[c, r, 0], [c + 1, r, 0], [c + 1, r + 1, 0], [c, r + 1, 0]], float) * face_mm

    def place(fid):
        if fid in pose:
            return pose[fid]
        R_p, t_p = place(NET_PARENT[fid])
        shared = [p for p in corners(fid) if any(np.allclose(p, q) for q in corners(NET_PARENT[fid]))]
        a, b = (R_p @ shared[0] + t_p), (R_p @ shared[1] + t_p)
        center = corners(fid).mean(0)
        for ang in (np.pi / 2, -np.pi / 2):
            Rh = rot(b - a, ang)
            R, t = Rh @ R_p, Rh @ (t_p - a) + a
            # the child must fold behind its parent's printed side
            if ((R @ center + t) - (a + b) / 2) @ (R_p @ n0) < 0:
                pose[fid] = (R, t)
                return pose[fid]
        raise AssertionError("no valid fold")

    out = {}
    for fid in NET_CELLS:
        R, t = place(fid)
        # printed tag's 'up' and 'right' on the page (y down); a half turn flips both
        sgn = -1.0 if fid in NET_HALF_TURN else 1.0
        up, right = sgn * np.array([0.0, -1.0, 0.0]), sgn * np.array([1.0, 0.0, 0.0])
        out[fid] = {"normal": R @ n0, "up": R @ up, "right": R @ right, "center": R @ corners(fid).mean(0) + t}
    # Box frame: face 22 normal -> +y, face 22 tag up -> +z.
    a, b = out[22]["normal"], out[22]["up"]
    A = np.stack([a, b, np.cross(a, b)], axis=1)
    B = np.stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]], axis=1).astype(float)
    R_align = B @ A.T
    cube_center = np.mean([v["center"] for v in out.values()], axis=0)
    for v in out.values():
        assert (v["center"] - cube_center) @ v["normal"] > 0, "printed side folded inward"
        for k in ("normal", "up", "right"):
            v[k] = R_align @ v[k]
    return out


def box_net_page(face_mm: float, tag_mm: float, tab_mm: float = 8.0, first_id: int = 20):
    if tag_mm > face_mm - 2 * tag_mm / 8:
        raise SystemExit(f"{tag_mm} mm tag leaves < 1 bit of white border on a {face_mm} mm face")
    d = cv2.aruco.getPredefinedDictionary(APRIL)
    w, h = 4 * face_mm + tab_mm, 3 * face_mm + 2 * tab_mm
    if w > A4_W - 16 or h > A4_H - 30:
        raise SystemExit(f"net {w:.0f}x{h:.0f} mm does not fit A4; use a smaller --cube-mm")
    ox, oy = (A4_W - w) / 2, 8.0 + tab_mm
    fig, ax = new_page()
    rects, ids = [], []
    for fid, (c, r) in NET_CELLS.items():
        mid = fid - 20 + first_id
        x0, y0 = ox + c * face_mm, oy + r * face_mm
        m = (face_mm - tag_mm) / 2
        rects += marker_rects(d, mid, x0 + m, y0 + m, tag_mm, half_turn=fid in NET_HALF_TURN)
        ids.append(mid)
        # each face edge: fold (dashed) if it meets another face or a tab, else cut (solid)
        tabs = {side for f, side in NET_TABS if f == fid}
        nbr = {"top": (c, r - 1), "bottom": (c, r + 1), "left": (c - 1, r), "right": (c + 1, r)}
        seg = {"top": [(x0, y0), (x0 + face_mm, y0)], "bottom": [(x0, y0 + face_mm), (x0 + face_mm, y0 + face_mm)],
               "left": [(x0, y0), (x0, y0 + face_mm)], "right": [(x0 + face_mm, y0), (x0 + face_mm, y0 + face_mm)]}
        for side, (p0, p1) in seg.items():
            fold = side in tabs or nbr[side] in NET_CELLS.values()
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.55" if fold else "black",
                    lw=0.3 if fold else 0.5, ls=(0, (2, 2)) if fold else "-")
    fill(ax, rects)
    # glue tabs (trapezoids, light grey), drawn outside the face they belong to
    for fid, side in NET_TABS:
        c, r = NET_CELLS[fid]
        x0, y0, f, t = ox + c * face_mm, oy + r * face_mm, face_mm, tab_mm
        poly = {"top": [(x0, y0), (x0 + t, y0 - t), (x0 + f - t, y0 - t), (x0 + f, y0)],
                "bottom": [(x0, y0 + f), (x0 + t, y0 + f + t), (x0 + f - t, y0 + f + t), (x0 + f, y0 + f)],
                "right": [(x0 + f, y0), (x0 + f + t, y0 + t), (x0 + f + t, y0 + f - t), (x0 + f, y0 + f)]}[side]
        ax.add_patch(matplotlib.patches.Polygon(poly, closed=False, fc="0.92", ec="black", lw=0.5))
    scale_bar(ax, A4_W - 112, A4_H - 8)
    note(ax, 10, oy + 3 * face_mm + tab_mm + 6,
         f"BOX NET  tag36h11 ids {first_id}-{first_id + 5}  cube {face_mm:g} mm  tag {tag_mm:g} mm (nominal)\n"
         "1. Cut along the SOLID lines (outline incl. grey tabs). Dashed = fold.\n"
         "2. Fold every dashed line and every tab AWAY from you (printed side stays outside).\n"
         "3. Fill the cube so it can't dent (foam / wood block, or packed card), glue tabs inside.\n"
         "4. Measure tags + cube edges -> measurements.yaml. Orientation is built in; no choices.",
         fontsize=5.5)
    return fig, ids


def check_net(face_map: dict) -> None:
    """The folded net must put every tag where configs/box.yaml says it is."""
    axes = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1]}
    folded = net_fold()
    for fid, (normal, up) in face_map.items():
        f = folded[int(fid)]
        n, u = np.array(axes[normal], float), np.array(axes[up], float)
        assert np.allclose(f["normal"], n, atol=1e-9), f"tag {fid}: folds to normal {f['normal']}, config {normal}"
        assert np.allclose(f["up"], u, atol=1e-9), f"tag {fid}: up {f['up']}, config {up}"
        assert np.allclose(f["right"], np.cross(u, n), atol=1e-9), f"tag {fid}: mirrored"
    print(f"  box net: folding puts all {len(face_map)} tags on the faces/orientations in box.yaml")


# --------------------------------------------------------------------------- self-check


def render(fig, px_per_mm=8.0) -> np.ndarray:
    fig.set_dpi(px_per_mm * MM_PER_IN)
    fig.canvas.draw()
    return cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2GRAY)


def check_april(img, expected_ids, tag_mm, px_per_mm, what):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(APRIL))
    corners, ids, _ = det.detectMarkers(img)
    got = sorted(ids.ravel().tolist()) if ids is not None else []
    assert got == sorted(expected_ids), f"{what}: detected {got}, expected {sorted(expected_ids)}"
    edges = [np.linalg.norm(np.diff(np.r_[c.reshape(4, 2), c.reshape(4, 2)[:1]], axis=0), axis=1).mean()
             for c in corners]
    size = np.mean(edges) / px_per_mm
    assert abs(size - tag_mm) < 0.3, f"{what}: rendered tag edge {size:.2f} mm != {tag_mm} mm"
    print(f"  {what}: ids {got[0]}-{got[-1]} detected, rendered edge {size:.2f} mm")


def check_board_pose(img, board, origin_mm, px_per_mm):
    """Board corners must map to page positions exactly as OpenCV's board layout says."""
    det = cv2.aruco.ArucoDetector(board.getDictionary())
    corners, ids, _ = det.detectMarkers(img)
    obj, imgp = board.matchImagePoints(corners, ids)
    pred = (obj.reshape(-1, 3)[:, :2] + origin_mm) * px_per_mm
    err = np.abs(pred - imgp.reshape(-1, 2)).max() / px_per_mm
    assert err < 0.3, f"board layout mismatch {err:.2f} mm"
    print(f"  board: layout matches getObjPoints to {err:.2f} mm")


def check_charuco(img, board):
    det = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = det.detectBoard(img)
    n_exp = (board.getChessboardSize()[0] - 1) * (board.getChessboardSize()[1] - 1)
    n = 0 if ch_ids is None else len(ch_ids)
    assert n == n_exp, f"charuco: {n}/{n_exp} corners detected"
    print(f"  charuco: {n}/{n_exp} corners detected")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("tags"))
    ap.add_argument("--board-tag-mm", type=float, default=55.0)
    ap.add_argument("--board-gap-mm", type=float, default=15.0)
    ap.add_argument("--box-style", choices=["net", "separate"], default="net",
                    help="net: fold-up paper cube; separate: six tags to stick on a solid box")
    ap.add_argument("--cube-mm", type=float, default=50.0, help="net: cube edge")
    ap.add_argument("--box-tag-mm", type=float, default=40.0)
    ap.add_argument("--box-dims-mm", type=float, nargs=3, default=(50.0, 50.0, 90.0), help="separate: box edges")
    ap.add_argument("--charuco-squares", type=int, nargs=2, default=(7, 5))
    ap.add_argument("--charuco-square-mm", type=float, default=36.0)
    ap.add_argument("--charuco-marker-mm", type=float, default=27.0)
    ap.add_argument("--check", action="store_true", help="render each page and re-detect its targets")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    px = 8.0

    fig, board, origin = board_page(a.board_tag_mm, a.board_gap_mm)
    fig.savefig(a.out / "board_A4.pdf")
    if a.check:
        img = render(fig, px)
        check_april(img, list(range(12)), a.board_tag_mm, px, "board")
        check_board_pose(img, board, np.array(origin), px)
    plt.close(fig)

    box_pdf = "box_net_A4.pdf" if a.box_style == "net" else "box_tags_A4.pdf"
    if a.box_style == "net":
        fig, ids = box_net_page(a.cube_mm, a.box_tag_mm)
    else:
        fig, ids = box_tags_page(a.box_tag_mm, tuple(a.box_dims_mm))
    fig.savefig(a.out / box_pdf)
    if a.check:
        check_april(render(fig, px), ids, a.box_tag_mm, px, "box tags")
        if a.box_style == "net":
            import yaml
            check_net(yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/box.yaml"))["faces"])
    plt.close(fig)

    fig, cboard, _ = charuco_page(tuple(a.charuco_squares), a.charuco_square_mm, a.charuco_marker_mm)
    fig.savefig(a.out / "charuco_A4.pdf")
    if a.check:
        check_charuco(render(fig, px), cboard)
    plt.close(fig)
    print(f"wrote {', '.join(str(a.out / n) for n in ('board_A4.pdf', box_pdf, 'charuco_A4.pdf'))}")


if __name__ == "__main__":
    main()
