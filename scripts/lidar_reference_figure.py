"""Figure: the LiDAR reference of two cars of intersection A - raw accumulated height field, gaps closed, and the
labelled box scored against both (uses the 3dseg code; evaluation data only)."""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
os.chdir(os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lidar_masks as lm
import v2x_seq_demo as vd
import v2x_seq_shape as vs
import whole_scene_cars as w

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "lidar_reference.png")
seq = vs.Sequence()
seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
ground = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))[2]
shapes = lm.LidarShapes(seq, ground)
rows = {r["label"]: r for r in json.load(open("docs/whole_scene_cars_abl_A3.json")) if r.get("label")}
cars = sys.argv[1:] or ["003976", "003985"]
xs, ys = np.arange(-3, 3, 0.1) + 0.05, np.arange(-1.5, 1.5, 0.1) + 0.05
fig, axes = plt.subplots(len(cars), 3, figsize=(9.5, 3.3 * len(cars)))
for row, tid in zip(axes, cars):
    r = rows[tid]
    Lt, lx, ly, n = w.lidar_heights(shapes, tid)
    Lc = w.close_lidar_holes(Lt, lx[1] - lx[0])
    box = np.where((np.abs(xs)[:, None] < r["label_length"] / 2) & (np.abs(ys)[None] < r["label_width"] / 2),
                   r["label_height"], 0.0)
    s_raw = w.shape_scores(box, xs, ys, Lt, lx, ly)["iou"]
    s_cl = w.shape_scores(box, xs, ys, Lc, lx, ly)["iou"]
    ext = [ly[0], ly[-1], lx[0], lx[-1]]
    for a, F, title in ((row[0], Lt, f"raw accumulated LiDAR ({n} pts)"), (row[1], Lc, "gaps closed (r = 15 cm)"),
                        (row[2], None, f"labelled box: IoU {s_raw:.2f} raw / {s_cl:.2f} closed")):
        if F is None:
            a.imshow(box, origin="lower", extent=[ys[0], ys[-1], xs[0], xs[-1]], vmin=0, vmax=2, cmap="viridis")
        else:
            a.imshow(F, origin="lower", extent=ext, vmin=0, vmax=2, cmap="viridis")
        a.set_title(title, fontsize=9, loc="left")
        a.set_xlim(-1.5, 1.5)
        a.set_ylim(-3, 3)
        a.tick_params(labelsize=7)
    row[0].set_ylabel(f"car {tid} (m)", fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=110, facecolor="white")
print(OUT)
