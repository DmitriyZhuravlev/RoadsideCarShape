"""
Car shape from a fixed roadside camera by mixing classical techniques (V2X-Seq, intersection A):

  1. silhouette hull   : voxels inside the motion silhouette of the frames, mirror symmetry, columns filled from the
                         ground (v2x_seq_shape) - the outline, but too big where no view looks past the car (the space
                         over the trunk of a car seen from behind);
  2. flow triangulation: points on the car's surface from how triangles move and deform between consecutive frames
                         (car_fusion: a new Delaunay triangulation every frame, plane-induced homography with the car's
                         motion), 10-14 cm from the LiDAR;
  3. mix               : free-space carving - the segment from the camera to every triangulated point is empty, so
                         hull voxels crossed by >= FREE_MIN such segments (up to MARGIN before the point) are removed;
                         this cuts exactly what silhouettes cannot (concavities, the space behind the visible surface);
  4. recursive touching cuboids on the result (v2x_seq_shape.patch_decompose: straight to depth 3, then may turn).
Poses: the labelled position and heading (the initialisation used throughout); the masks and the flow are the method's.
Checked against the intersection's LiDAR points on the same cars (in the car's frame, over the whole track):
  precision    : distance from the model's surface to the LiDAR points;
  completeness : distance from the LiDAR points to the model's surface;
  roof         : model top vs LiDAR top (95th percentile of height);
  held-out IoU : the model's silhouette on frames not used (every 4th).

Writes docs/mix3d_views.png, docs/mix3d_scores.png and docs/mix3d_metrics.json.
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

import car_flow_planes as cfp
import car_fusion as cfu
import flow_planes as fp
import v2x_seq_demo as vd
import v2x_seq_shape as vs

TRACKS = ("003981", "004005", "004012")
MARGIN = 0.25             # m: free space stops this far before a triangulated point (its error is 10-15 cm)
FREE_MIN = 2              # segments needed to free a voxel
INK, MUTED = "#1f1f1e", "#8a8984"
MODELS = ("labelled box", "silhouette hull", "hull + triangulated free space", "cuboids depth 3", "cuboids depth 8")
COLORS = dict(zip(MODELS, ("#c3c2b7", "#8a8984", "#2a78d6", "#eb6834", "#d6457a")))


def free_space_counts(points, centres, axes, margin=MARGIN, reach=4.0, step=0.03):
    """Per voxel, the number of camera-to-point segments through it (only the last `reach` m, stopping `margin` m
    before the point; each segment counts once per voxel)."""
    xs, ys, zs = axes
    v = xs[1] - xs[0]
    lo = np.array([xs[0], ys[0], zs[0]])
    shape = (len(xs), len(ys), len(zs))
    counts = np.zeros(int(np.prod(shape)), np.int32)
    s_rel = np.arange(-reach, -margin, step)                         # distances back from the point
    for chunk in range(0, len(points), 2000):
        P = points[chunk:chunk + 2000]
        C = centres[chunk:chunk + 2000]
        d = P - C
        L = np.linalg.norm(d, axis=1, keepdims=True)
        u = d / L
        S = P[:, None, :] + s_rel[None, :, None] * u[:, None, :]    # (n, m, 3)
        idx = np.round((S - lo) / v).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=2)
        ok &= (L + s_rel[None, :]) > 0                               # not behind the camera
        flat = np.where(ok, np.ravel_multi_index(np.clip(idx, 0, np.array(shape) - 1).transpose(2, 0, 1), shape), -1)
        rows = np.repeat(np.arange(len(P)), flat.shape[1])
        pairs = np.unique(np.c_[rows, flat.ravel()], axis=0)          # once per segment and voxel
        pairs = pairs[pairs[:, 1] >= 0]
        np.add.at(counts, pairs[:, 1], 1)
    return counts.reshape(shape)


def to_ground_frame(V, o):
    """car_flow_planes object frame (origin at the box centre) -> v2x_seq_shape object frame (origin on the ground)."""
    V = V.copy()
    V[..., 2] += o["z"]
    return V


def camera_in_object(cam, o):
    C = -np.linalg.solve(cam[1], cam[2])          # the calibration's R is a general matrix, not a rotation
    c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return Rz.T @ (C - np.array([o["pos"][0], o["pos"][1], 0.0]))


def surface_samples_boxes(boxes, step=0.04):
    return np.vstack([vs.box_surface_samples(b["c"], b["R"], b["h"], step) for b in boxes])


def evaluate(model_pts, L, inside_fn=None):
    tL = cKDTree(L)
    tM = cKDTree(model_pts)
    prec = tL.query(model_pts)[0]
    comp = tM.query(L)[0]
    return float(np.median(prec)), float(np.median(comp))


def roof(points):
    return float(np.percentile(points[:, 2], 95))


def run_car(seq, tid, ground_z):
    obs = [o for o in seq.tracks[tid] if seq.usable(o)]
    by_k = {o["k"]: o for o in obs}
    l, w = (float(np.median([o["dims"][i] for o in obs])) for i in (0, 1))
    h_lab = float(np.median([o["dims"][2] for o in obs]))
    axes = vs.grid(l, w, 1.6)
    train = [o for i, o in enumerate(obs) if i % 4]
    test = [o for i, o in enumerate(obs) if i % 4 == 0]
    # 1. silhouette hull
    votes = vs.carve_votes(axes, [seq.view(o) for o in train])
    hull = vs.height_field(votes >= vs.VOTE)
    # 2. triangulated points from consecutive frames (train frames only), with the camera that saw them
    pts, cams_obj, lidar = [], [], []
    for o in train:
        if o["k"] + 1 in by_k:
            f = cfu.frame_model(seq, o, by_k[o["k"] + 1])
            if len(f["V"]):
                V = to_ground_frame(f["V"], o)
                pts.append(V.mean(1))
                cams_obj.append(np.repeat(camera_in_object(seq.cams[o["k"]], o)[None], len(V), 0))
    for o in obs:
        Lo = cfp.lidar_on_car(seq, o, ground_z)
        lidar.append(to_ground_frame(Lo, o))
    L = np.vstack(lidar)
    pts, cams_obj = np.vstack(pts), np.vstack(cams_obj)
    # 3. free-space carving of the hull by the camera-to-point segments
    free = free_space_counts(pts, cams_obj, axes)
    mixed = vs.height_field(hull & (free < FREE_MIN))
    models = {"silhouette hull": hull, "hull + triangulated free space": mixed}
    # 4. recursive cuboids on the mixed shape
    vs.VOX, vs.FLAT = 0.05, 0.10
    surf = vs.surface_points(mixed, axes)
    cub = {3: vs.patch_decompose(surf, 3), 8: vs.patch_decompose(surf, 8)}
    # evaluation
    res = {}
    lab_box = {"c": np.array([0, 0, h_lab / 2]), "R": np.eye(3), "h": np.array([l / 2, w / 2, h_lab / 2])}
    shapes = {"labelled box": surface_samples_boxes([lab_box]),
              "silhouette hull": vs.surface_points(hull, axes),
              "hull + triangulated free space": surf,
              "cuboids depth 3": surface_samples_boxes(cub[3]),
              "cuboids depth 8": surface_samples_boxes(cub[8])}
    shape_px = seq.shape
    for name, S in shapes.items():
        prec, comp = evaluate(S, L)
        if name in models:
            occ = models[name]
            sil = lambda o: vs.silhouette(occ, axes, seq.cams[o["k"]], o["pos"], o["yaw"], shape_px) > 0
        elif name == "labelled box":
            sil = lambda o: vs.label_box_silhouette(o, seq.cams[o["k"]], shape_px) > 0
        else:
            bx = cub[3] if "3" in name else cub[8]
            sil = lambda o, bx=bx: vs.patch_silhouette(bx, seq.cams[o["k"]], o["pos"], o["yaw"], shape_px) > 0
        iou = float(np.mean([vs.iou(sil(o), seq.mask(o) > 0) for o in test]))
        res[name] = {"precision_cm": 100 * prec, "completeness_cm": 100 * comp, "roof_err_cm": 100 * (roof(S) - roof(L)),
                     "held_out_iou": iou}
    res["counts"] = {"frames": len(obs), "triangulated_points": int(len(pts)), "lidar_points": int(len(L)),
                     "hull_voxels": int(hull.sum()), "mixed_voxels": int(mixed.sum()),
                     "cuboids_3": len(cub[3]), "cuboids_8": len(cub[8])}
    return res, {"axes": axes, "hull": hull, "mixed": mixed, "pts": pts, "cub": cub, "L": L, "dims": (l, w, h_lab),
                 "frame": test[len(test) // 2]}


def draw_voxels(ax, occ, axes, color):
    v = axes[0][1] - axes[0][0]
    S = vs.surface_points(occ, axes)
    faces = []
    for p in S[::2]:
        faces.append([p + [-v / 2, -v / 2, v / 2], p + [v / 2, -v / 2, v / 2], p + [v / 2, v / 2, v / 2], p + [-v / 2, v / 2, v / 2]])
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors="none", alpha=0.55))


def draw_boxes(ax, boxes):
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces_idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    polys, cols = [], []
    for b in boxes:
        C = b["c"] + (signs * b["h"]) @ b["R"].T
        for f in faces_idx:
            polys.append(C[f])
            cols.append(vs.LEVEL_COLORS[b["level"] % len(vs.LEVEL_COLORS)])
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="#1f1f1e", linewidths=0.2, alpha=0.6))


def views_figure(path, seq, runs):
    cols = ("photo", "silhouette hull", "+ triangulated points", "hull + free space", "cuboids depth 3", "cuboids depth 8")
    fig = plt.figure(figsize=(3.4 * len(cols), 3.4 * len(runs)))
    for r, (tid, (res, d)) in enumerate(runs.items()):
        l, w, h = d["dims"]
        L = d["L"][np.random.default_rng(0).permutation(len(d["L"]))[:2500]]
        for c, name in enumerate(cols):
            if c == 0:
                ax = fig.add_subplot(len(runs), len(cols), r * len(cols) + 1)
                o = d["frame"]
                ax.imshow(cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2RGB))
                x0, y0, x1, y1 = o["box"]
                ax.set_xlim(x0 - 20, x1 + 20)
                ax.set_ylim(y1 + 20, y0 - 20)
                ax.axis("off")
                ax.set_title(f"car {tid}, frame {o['k']}", fontsize=9.5, loc="left", color=INK)
                continue
            ax = fig.add_subplot(len(runs), len(cols), r * len(cols) + c + 1, projection="3d")
            if name == "silhouette hull":
                draw_voxels(ax, d["hull"], d["axes"], "#9fb9d8")
            elif name == "+ triangulated points":
                draw_voxels(ax, d["hull"], d["axes"], "#dfe6ee")
                P = d["pts"][np.random.default_rng(1).permutation(len(d["pts"]))[:3000]]
                ax.scatter(*P.T, s=0.6, c="#eb6834", alpha=0.6)
            elif name == "hull + free space":
                draw_voxels(ax, d["mixed"], d["axes"], "#2a78d6")
            else:
                draw_boxes(ax, d["cub"][3 if "3" in name else 8])
            ax.scatter(*L.T, s=0.25, c="k", alpha=0.25)
            lim = max(l, w) / 2 + 0.3
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(0, 2 * lim)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(18, -60)
            ax.axis("off")
            key = {"silhouette hull": "silhouette hull", "hull + free space": "hull + triangulated free space",
                   "cuboids depth 3": "cuboids depth 3", "cuboids depth 8": "cuboids depth 8"}.get(name)
            extra = f"\nroof {res[key]['roof_err_cm']:+.0f} cm, held-out IoU {res[key]['held_out_iou']:.2f}" if key else \
                f"\n{res['counts']['triangulated_points']} points (orange)"
            ax.set_title(name + extra, fontsize=8.5, loc="left", color=INK)
    fig.suptitle("Silhouette carving + triangulation from flow (free space) + recursive cuboids; black = LiDAR on the car",
                 fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def scores_figure(path, runs):
    cars = list(runs)
    keys = (("precision_cm", "model -> LiDAR, median (cm)"), ("completeness_cm", "LiDAR -> model, median (cm)"),
            ("roof_err_cm", "roof height error (cm)"), ("held_out_iou", "held-out silhouette IoU"))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    width = 0.16
    for ax, (key, title) in zip(axes, keys):
        for j, m in enumerate(MODELS):
            v = [runs[c][0][m][key] for c in cars]
            bars = ax.bar(np.arange(len(cars)) + (j - 2) * width, v, width * 0.92, color=COLORS[m], zorder=2, label=m)
            for b, x in zip(bars, v):
                ax.annotate(f"{x:.2f}" if key == "held_out_iou" else f"{x:.0f}", (b.get_x() + b.get_width() / 2, x),
                            xytext=(0, 2 if x >= 0 else -9), textcoords="offset points", ha="center", fontsize=7,
                            color=INK)
        ax.set_xticks(range(len(cars)), [f"car {c}" for c in cars], fontsize=8.5)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        vd.style(ax)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    seq = vs.Sequence()
    seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    plane = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))
    runs, metrics = {}, {}
    for tid in TRACKS:
        res, d = run_car(seq, tid, plane[2])
        runs[tid] = (res, d)
        metrics[tid] = res
        print(tid, json.dumps(res["counts"]))
        for m in MODELS:
            r = res[m]
            print(f"   {m:32s} precision {r['precision_cm']:5.1f} cm | completeness {r['completeness_cm']:5.1f} cm | "
                  f"roof {r['roof_err_cm']:+5.0f} cm | held-out IoU {r['held_out_iou']:.3f}", flush=True)
    json.dump(metrics, open(os.path.join(vd.OUT_DIR, "mix3d_metrics.json"), "w"), indent=1)
    views_figure(os.path.join(vd.OUT_DIR, "mix3d_views.png"), seq, runs)
    scores_figure(os.path.join(vd.OUT_DIR, "mix3d_scores.png"), runs)


if __name__ == "__main__":
    main()
