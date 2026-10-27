"""
Hierarchical graph-based video segmentation (after Grundmann, Kwatra, Han, Essa, CVPR 2010), written for the
whole-scene reconstruction: colour AND the full optical flow (direction and magnitude), on a short window of frames.

  1. every frame of the window (reduced resolution, rows below the horizon) is over-segmented with the graph-based
     method of Felzenszwalb & Huttenlocher on [L, a, b, WF * flow_x, WF * flow_y] (dense DIS optical flow);
  2. region graph over the whole window: spatial edges between touching segments of a frame, temporal edges between
     a segment and the segments its pixels flow into in the next frame;
  3. hierarchy: the Felzenszwalb merge criterion on the region graph - merge A, B when
     d(A, B) <= min(Int(A) + tau / |A|, Int(B) + tau / |B|) - repeated with tau growing by TAU_STEP per level,
     region means recomputed after every level;
     d = |mean Lab difference| + W_FLOW * |mean flow difference| / (1 + FLOW_REL * the larger flow)
     (the relative term: perspective makes the near part of a car flow faster than its far part).
Labels of every level in every frame of the window; regions are consistent through time.

Writes docs/supervoxels_<k>.png when run as a script.
"""
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import felzenszwalb

SCALE = 0.25                     # working resolution
FZ_SCALE, FZ_SIGMA, FZ_MIN = 60.0, 0.6, 20
WF = 2.0                         # feature units per full-resolution px/frame of flow (per-frame segments)
W_FLOW, FLOW_REL = 5.0, 0.15     # region distance: weight of the flow difference, relative to the flow size
TAU0, TAU_STEP, LEVELS = 3000.0, 2.0, 8
MIN_TEMPORAL = 4                 # px of a segment that must flow into another to link them
INK = "#1f1f1e"


def dis_flows(grays):
    """Forward DIS flow j -> j+1 for every frame of the window (full-resolution px)."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return [dis.calc(grays[j], grays[j + 1], None) / SCALE for j in range(len(grays) - 1)]


def frame_motion(fwd, j):
    """Motion per frame at frame j: mean of the steps into and out of it."""
    f = [fwd[i] for i in (j - 1, j) if 0 <= i < len(fwd)]
    return sum(f) / len(f)


def union_find(n):
    return np.arange(n)


def find(p, a):
    r = a
    while p[r] != r:
        r = p[r]
    while p[a] != r:
        p[a], a = r, p[a]
    return r


def adjacency_2d(lab):
    """Unique touching pairs (a < b) of a 2D label image."""
    keys = []
    n = int(lab.max()) + 1
    for a, b in ((lab[:-1], lab[1:]), (lab[:, :-1], lab[:, 1:])):
        m = a != b
        lo, hi = np.minimum(a[m], b[m]).astype(np.int64), np.maximum(a[m], b[m]).astype(np.int64)
        keys.append(lo * n + hi)
    k = np.unique(np.concatenate(keys))
    return np.stack([k // n, k % n], 1)


def distance(lab_a, lab_b, fl_a, fl_b):
    big = np.maximum(np.linalg.norm(fl_a, axis=-1), np.linalg.norm(fl_b, axis=-1))
    return np.linalg.norm(lab_a - lab_b, axis=-1) + W_FLOW * np.linalg.norm(fl_a - fl_b, axis=-1) / (1 + FLOW_REL * big)


class VideoSegmentation:
    """Labels (levels, T, h, w) of a window of BGR frames; y0 = first image row used (the horizon)."""

    def __init__(self, frames, y0=0):
        self.y0 = y0
        small = [cv2.resize(f[y0:], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA) for f in frames]
        self.small = small
        grays = [cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) for s in small]
        fwd = dis_flows(grays)
        T = len(small)
        self.flow = np.stack([frame_motion(fwd, j) for j in range(T)])                 # (T, h, w, 2) full-res px
        labs = np.stack([cv2.cvtColor(s.astype(np.float32) / 255, cv2.COLOR_BGR2LAB) for s in small])
        # 1. per-frame graph segments on colour + flow
        base, offs = [], 0
        for j in range(T):
            feat = np.dstack([labs[j], WF * self.flow[j]])
            seg = felzenszwalb(feat, scale=FZ_SCALE, sigma=FZ_SIGMA, min_size=FZ_MIN, channel_axis=-1)
            base.append(seg + offs)
            offs += int(seg.max()) + 1
        base = np.stack(base)                                                            # (T, h, w) global ids
        n = offs
        cnt = np.bincount(base.ravel(), minlength=n).astype(float)
        mean_lab = np.stack([np.bincount(base.ravel(), labs[..., c].ravel(), minlength=n) for c in range(3)], 1) / cnt[:, None]
        mean_fl = np.stack([np.bincount(base.ravel(), self.flow[..., c].ravel(), minlength=n) for c in range(2)], 1) / cnt[:, None]
        # 2. region graph: spatial + temporal (along the flow)
        pairs = [adjacency_2d(base[j]) for j in range(T)]
        h, w = base.shape[1:]
        yy, xx = np.mgrid[0:h, 0:w]
        for j in range(T - 1):
            f = fwd[j] * SCALE
            x2 = np.clip(np.round(xx + f[..., 0]).astype(int), 0, w - 1)
            y2 = np.clip(np.round(yy + f[..., 1]).astype(int), 0, h - 1)
            a, b = base[j].ravel().astype(np.int64), base[j + 1][y2, x2].ravel().astype(np.int64)
            k, c = np.unique(a * n + b, return_counts=True)
            k = k[c >= MIN_TEMPORAL]
            pairs.append(np.stack([k // n, k % n], 1))
        edges = np.unique(np.sort(np.vstack(pairs), axis=1), axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]
        # 3. hierarchy: Felzenszwalb merging on the region graph, level by level
        self.levels = [base]
        node_of = np.arange(n)                     # base id -> current region id
        size, lab_m, fl_m = cnt.copy(), mean_lab, mean_fl
        tau = TAU0
        for _ in range(LEVELS):
            m = len(size)
            d = distance(lab_m[edges[:, 0]], lab_m[edges[:, 1]], fl_m[edges[:, 0]], fl_m[edges[:, 1]])
            order = np.argsort(d, kind="stable")
            p = union_find(m)
            internal = np.zeros(m)
            sz = size.copy()
            for e in order:
                a, b = find(p, edges[e, 0]), find(p, edges[e, 1])
                if a == b:
                    continue
                if d[e] <= min(internal[a] + tau / sz[a], internal[b] + tau / sz[b]):
                    p[b] = a
                    sz[a] += sz[b]
                    internal[a] = max(internal[a], internal[b], d[e])
            roots = np.array([find(p, i) for i in range(m)])
            uniq, new = np.unique(roots, return_inverse=True)
            k = len(uniq)
            size2 = np.bincount(new, size, minlength=k)
            lab_m = np.stack([np.bincount(new, lab_m[:, c] * size, minlength=k) for c in range(3)], 1) / size2[:, None]
            fl_m = np.stack([np.bincount(new, fl_m[:, c] * size, minlength=k) for c in range(2)], 1) / size2[:, None]
            size = size2
            edges = np.unique(np.sort(new[edges], axis=1), axis=0)
            edges = edges[edges[:, 0] != edges[:, 1]]
            node_of = new[node_of]
            self.levels.append(node_of[base])
            tau *= TAU_STEP
        self.region_flow = fl_m

    def full_res(self, level, j, shape):
        """Labels of one level in frame j at full resolution (rows above y0 = -1)."""
        H, W = shape
        lab = cv2.resize(self.levels[level][j].astype(np.int32), (W, H - self.y0), interpolation=cv2.INTER_NEAREST)
        out = -np.ones((H, W), np.int32)
        out[self.y0:] = lab
        return out

    def flow_full(self, j, shape):
        H, W = shape
        f = cv2.resize(self.flow[j], (W, H - self.y0), interpolation=cv2.INTER_LINEAR)
        out = np.zeros((H, W, 2), np.float32)
        out[self.y0:] = f
        return out


def colour_labels(lab, seed=3):
    rng = np.random.default_rng(seed)
    pal = rng.uniform(0.15, 0.95, (int(lab.max()) + 2, 3))
    return pal[lab]


def figure(path, seg, j, show_levels=(0, 3, 5, 8)):
    rgb = cv2.cvtColor(seg.small[j], cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 3, figsize=(18, 6.4))
    axes = axes.ravel()
    axes[0].imshow(rgb)
    axes[0].set_title("frame (reduced, below the horizon)", fontsize=9, loc="left", color=INK)
    mag = np.linalg.norm(seg.flow[j], axis=-1)
    axes[1].imshow(mag, cmap="magma", vmin=0, vmax=25)
    axes[1].set_title("dense flow |px / frame| (DIS)", fontsize=9, loc="left", color=INK)
    for ax, L in zip(axes[2:], show_levels):
        lab = seg.levels[L][j]
        ax.imshow(0.35 * rgb / 255 + 0.65 * colour_labels(lab))
        ax.set_title(f"level {L}: {len(np.unique(lab))} regions in this frame", fontsize=9, loc="left", color=INK)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import time

    import v2x_seq_shape as vs
    import v2x_seq_demo as vd
    seq = vs.Sequence()
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 70
    t0 = time.time()
    seg = VideoSegmentation([seq.frames[j] for j in range(k - 8, k + 9)], y0=240)
    print(f"{time.time() - t0:.1f} s; regions per level in the window:", [len(np.unique(L)) for L in seg.levels])
    figure(os.path.join(vd.OUT_DIR, f"supervoxels_{k}.png"), seg, 8)
