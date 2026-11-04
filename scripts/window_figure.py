"""Figure of one frame of intersection A on the precise annotation: left, the moving cars' SAM masks and their
LiDAR-fitted cuboids (image-clock pose); right, the body triangulation of every window whose reference frame this is,
each solved in 3D (make_dumps.py). Usage: [ANN=docs/precise_ann_Asam] python window_figure.py K OUT_JPG DUMPS_DIR ..."""
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
os.chdir(os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
import ego_sequence as es  # noqa: E402
import v2x_seq_shape as vs  # noqa: E402

K, OUT, DUMPS = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
ANN = os.environ.get("ANN", "docs/precise_ann_Asam")
PAL = [(255, 127, 14), (31, 119, 180), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194),
       (23, 190, 207)]
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def corners_of(center, yaw, lwh):
    c, s = np.cos(yaw), np.sin(yaw)
    f, n = np.array([c, s]), np.array([-s, c])
    L, W_, H_ = lwh
    return np.array([[*(np.array(center[:2]) + dx * f + dy * n), center[2] + dz]
                     for dx in (-L / 2, L / 2) for dy in (-W_ / 2, W_ / 2) for dz in (-H_ / 2, H_ / 2)])


def draw_cuboid(img, uv8, col):
    """Draws a projected 3D box on img in place (corners in corners_of's order)."""
    for i, j in EDGES:
        cv2.line(img, tuple(np.round(uv8[i]).astype(int)), tuple(np.round(uv8[j]).astype(int)), col, 2, cv2.LINE_AA)
    return img


def strip(tile, text):
    t = cv2.copyMakeBorder(tile, 46, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(t, text, (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (30, 30, 30), 2, cv2.LINE_AA)
    return t


seq, ground = es.open_sequence()
img, fid = seq.frames[K], seq.fids[K]
H, W = img.shape[:2]
masks = cv2.imread(f"{ANN}/masks/{fid}.png", cv2.IMREAD_UNCHANGED)
cubs = json.load(open(f"{ANN}/cuboids/{fid}.json"))
dets = []
for d in DUMPS:
    if os.path.exists(f"{d}/{K:04d}.pkl"):
        dets += pickle.load(open(f"{d}/{K:04d}.pkl", "rb"))
moving = list(dict.fromkeys(d["track_tid"] for d in dets))
col = {t: PAL[i % len(PAL)][::-1] for i, t in enumerate(moving)}

left, right = img.copy(), (0.55 * img).astype(np.uint8)
xy = []
for i, e in enumerate(cubs):
    if e["track"] not in col:
        continue
    c = col[e["track"]]
    m = masks == i + 1
    left[m] = (0.35 * left[m] + 0.65 * np.array(c)).astype(np.uint8)
    cv2.drawContours(left, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1,
                     (255, 255, 255), 2)
    p = e.get("image_clock", e)
    uv, z = vs.project(seq.cams[K], corners_of(p["center"], p["yaw"], e["lwh"]))
    if (z > 0.3).all():
        draw_cuboid(left, uv, c)
        xy.append(uv)
    yy, xx = np.nonzero(m)
    xy.append(np.c_[xx, yy])
for d in dets:
    T = d["tri"][d["body"][d["tri"]].all(1)]
    for t in T:
        cv2.polylines(right, [np.round(d["uv"][t]).astype(np.int32)], True, col[d["track_tid"]], 1, cv2.LINE_AA)
    xy.append(d["uv"])
xy = np.vstack(xy)
x0, y0 = np.maximum(np.floor(xy.min(0)).astype(int) - 40, 0)
x1, y1 = np.minimum(np.ceil(xy.max(0)).astype(int) + 40, [W, H])
out = cv2.vconcat([strip(left[y0:y1, x0:x1], f"frame {K}: SAM masks and LiDAR-fitted cuboids"),
                   strip(right[y0:y1, x0:x1], "body triangulation of each window, solved in 3D")])
cv2.imwrite(OUT, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(OUT, len(moving), "cars")
