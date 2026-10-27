"""
Car-shaped masks from a real measurement of the car (V2X-Seq has no pixel masks; the projected 3D box is a cuboid).

Per car:
  1. its LiDAR points of every frame of its track, in the car's own frame (the labelled pose of each frame);
  2. mirrored across the car's centre plane - the roadside LiDAR sees mostly the near side, the car is symmetric;
  3. voxels (VOX m) with points; every column filled from the ground up to its highest point (from above a car is a
     height field) - a car-shaped solid: roof line, bonnet, trunk, nothing outside the body;
  4. that solid projected into each frame with the frame's labelled pose: the car's silhouette.

Writes docs/lidar_masks.png (cuboid masks vs LiDAR car masks vs our masks).
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import car_flow_planes as cfp
import mix3d as mx
import v2x_seq_shape as vs

VOX = 0.05
INK = "#1f1f1e"


def points_to_occupancy(P, axes, min_points=1):
    """Voxels of the grid `axes` that contain at least min_points points (points outside are ignored)."""
    v = axes[0][1] - axes[0][0]
    shape = tuple(len(a) for a in axes)
    idx = np.round((P - np.array([a[0] for a in axes])) / v).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    counts = np.bincount(np.ravel_multi_index(idx[ok].T, shape), minlength=int(np.prod(shape)))
    return counts.reshape(shape) >= min_points


class LidarShapes:
    """The car-shaped LiDAR solid of every track (cached) and its silhouette in any frame."""

    def __init__(self, seq, ground):
        self.seq, self.ground = seq, ground
        self.solids = {}

    def solid(self, tid, step=2):
        if tid not in self.solids:
            q = self.seq.tracks[tid]
            pts = [mx.to_ground_frame(cfp.lidar_on_car(self.seq, o, self.ground), o) for o in q[::step]]
            P = np.vstack([p for p in pts if len(p)]) if any(len(p) for p in pts) else np.zeros((0, 3))
            P = np.vstack([P, P * [1, -1, 1]])                  # the car is mirror-symmetric
            l, w, h = (float(np.median([o["dims"][i] for o in q])) for i in range(3))
            vs_vox = vs.VOX
            vs.VOX = VOX
            axes = vs.grid(l, w, max(h, 1.6))
            vs.VOX = vs_vox
            occ = points_to_occupancy(P[P[:, 2] > 0.05], axes)  # ground returns excluded
            occ = vs.height_field(occ)
            self.solids[tid] = (occ, axes, len(P))
        return self.solids[tid]

    def mask(self, tid, o):
        occ, axes, n = self.solid(tid)
        if not occ.any():
            return np.zeros(self.seq.shape, bool)
        return vs.silhouette(occ, axes, self.seq.cams[o["k"]], o["pos"], o["yaw"], self.seq.shape) > 0


def figure(path, seq, shapes, good, tids, ks=None):
    """Cuboid mask vs LiDAR car mask vs our masks for a few cars."""
    rows = []
    for tid in tids:
        q = seq.tracks[tid]
        o = q[len(q) // 2] if ks is None else next(p for p in q if p["k"] == ks[tid])
        rows.append((tid, o))
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 2.9 * len(rows)))
    titles = ("frame", "cuboid: labelled 3D box projected", "car: accumulated LiDAR solid projected",
              "ours: median subtraction + reflection cut")
    for r, (tid, o) in enumerate(rows):
        x0, y0, x1, y1 = np.array(o["box"]).round().astype(int)
        pad = 20
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad
        rgb = cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2RGB)[Y0:Y1, X0:X1]
        ms = (None, vs.label_box_silhouette(o, seq.cams[o["k"]], seq.shape) > 0, shapes.mask(tid, o), good(o) > 0)
        for c, m in enumerate(ms):
            ax = axes[r, c]
            if m is None:
                ax.imshow(rgb)
            else:
                mm = m[Y0:Y1, X0:X1]
                ov = rgb.astype(float) / 255
                ov[mm] = 0.45 * ov[mm] + 0.55 * np.array([0.16, 0.47, 0.84])
                ax.imshow(ov)
                if mm.any():
                    ax.contour(mm, levels=[0.5], colors="#eb6834", linewidths=0.8)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=8.5, loc="left", color=INK)
        axes[r, 0].set_title((titles[0] + "\n" if r == 0 else "") + f"car {tid}, frame {o['k']}", fontsize=8, loc="left",
                             color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor="white")
    plt.close(fig)
