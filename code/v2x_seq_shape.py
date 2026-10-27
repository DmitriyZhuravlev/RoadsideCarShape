"""
Shape of a car from recursive touching cuboids (V2X-Seq roadside camera).

1. Start from the car's cuboid (pose from the labels here; from lifting.py in the full pipeline).
2. Visual hull inside it: a voxel (5 cm) of the cuboid can belong to the car only if it projects
   inside the car's silhouette (background subtraction) in the frames that see it. A moving car
   is seen from many directions, so its frames act as many cameras. Cars are mirror-symmetric
   about their long axis, so every frame is used twice (y -> -y). Every column is filled from
   the ground up to its highest surviving voxel.
3. Recursive contact decomposition: tight cuboid of the hull -> contact set on each of its 6 faces
   (hull voxels where the cuboid touches the object) -> cut planes through the ends of each contact
   set -> keep the cut that shrinks the two tight child cuboids most -> recurse. Children touch
   along the cut plane and rest on the ground or on each other.

Evaluation:
  synthetic: a car-shaped solid (sloped hood and windscreen, narrower cabin) rendered along a real
             V2X-Seq track; voxel IoU with the true shape.
  real:      3 moving V2X-Seq cars; every 4th frame is held out; IoU of the projected shape with the
             held-out silhouette, against the labelled 3D box.

Left-right symmetry (main): mirrored views in the hull and cuboids centred on the mirror plane,
compared with no symmetry, from 1 frame and from all frames.

Surface patches (main): only the outer cuboid (depth 0) is symmetric and stands on the ground; deeper
cuboids keep their parent's axes, lie flush against a parent face and turn into flat slabs.
Writes docs/shape_patches_3d.png, docs/shape_patches_scores.png and docs/shape_views_<car>.png
(original vs reconstruction from the real camera and from 6 canonical sides) (main); recursion_3d_figure and
symmetry_scores_figure wrote docs/shape_symmetry_3d.png and docs/shape_symmetry_scores.png; synthetic_figure,
real_figure and scores_figure produced docs/shape_synthetic.png, shape_v2x_seq.png, shape_scores.png.
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import v2x_seq_classical as vc
import v2x_seq_demo as vd

VOX = 0.05
VOTE = 0.9                       # share of the views that must see a voxel inside the silhouette
TRACKS = ("003981", "004005", "004012")
EDGES = vd.EDGES
INK, MUTED = "#1f1f1e", "#8a8984"
LEVEL_COLORS = ["#2a78d6", "#eb6834", "#2e9e6b", "#8a5cc2", "#c9a227", "#d6457a", "#38b2c9", "#7a4b2a"]


# --- geometry --------------------------------------------------------------------------------
def grid(l, w, h, margin=0.1, zmargin=1.0):
    """Voxel centres, centred on the object so that x and y are mirror-symmetric."""
    nx, ny = (int(np.ceil((s / 2 + margin) / VOX - 1e-9)) for s in (l, w))
    return ((np.arange(-nx, nx) + 0.5) * VOX, (np.arange(-ny, ny) + 0.5) * VOX,
            np.arange(0, h + zmargin, VOX) + VOX / 2)


def world(P, center, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.c_[center[0] + c * P[:, 0] - s * P[:, 1], center[1] + s * P[:, 0] + c * P[:, 1], P[:, 2]]


def project(cam, X):
    K, R, t = cam
    x = K @ (R @ X.T + t[:, None])
    return (x[:2] / x[2]).T, x[2]


def carve_votes(axes, views, symmetric=True):
    """Share of the views (that see the voxel) whose silhouette contains it; views = (cam, centre, yaw, mask)."""
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    hits, seen = np.zeros(X.size), np.zeros(X.size)
    for mirror in ((False, True) if symmetric else (False,)):
        P = np.c_[X.ravel(), (-Y if mirror else Y).ravel(), Z.ravel()]
        for cam, center, yaw, mask in views:
            uv, depth = project(cam, world(P, center, yaw))
            u, v = uv.round().astype(int).T
            ok = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0]) & (depth > 0)
            seen += ok
            hits[ok] += mask[v[ok], u[ok]] > 0
    return (hits / np.maximum(seen, 1)).reshape(X.shape)


def height_field(occ):
    """Fill every column from the ground up to its highest occupied voxel (cars are solid below their surface)."""
    k = occ.shape[2]
    top = np.where(occ.any(2), k - np.argmax(occ[:, :, ::-1], axis=2), 0)
    return np.arange(k)[None, None, :] < top[:, :, None]


# --- recursive contact decomposition ----------------------------------------------------------
def tight(occ, r):
    """Tight index box of the hull voxels inside region r = ((i0, i1), (j0, j1), (k0, k1)), or None."""
    sub = occ[tuple(slice(*x) for x in r)]
    if not sub.any():
        return None
    return tuple((x[0] + a.min(), x[0] + a.max() + 1) for x, a in zip(r, np.nonzero(sub)))


def symmetric_tight(occ, r):
    """Tight index box of the True voxels of a 3D boolean array inside region r, made symmetric
    about the centre of axis 1 (the object's left-right mirror plane).

    occ: np.ndarray of bool, shape (nx, ny, nz).
    r: tuple of three (start, stop) integer pairs ((i0, i1), (j0, j1), (k0, k1)), half-open, indexing occ.
    Returns None if occ[i0:i1, j0:j1, k0:k1] has no True voxel. Otherwise returns
    ((a0, a1), (b0, b1), (c0, c1)) where (a0, a1) and (c0, c1) are the tight min/max+1 index ranges of the
    True voxels along axes 0 and 2 (in absolute indices of occ), and (b0, b1) is the tight range along axis 1
    widened so that it is symmetric: b0 = min(t0, ny - t1), b1 = ny - b0, where (t0, t1) is the tight
    absolute range along axis 1 and ny = occ.shape[1].
    """
    b = tight(occ, r)
    if b is None:
        return None
    ny = occ.shape[1]
    b0 = min(b[1][0], ny - b[1][1])
    return b[0], (b0, ny - b0), b[2]


def volume(b):
    return np.prod([b1 - b0 for b0, b1 in b])


def contact_cuts(occ, b, min_size=4, axes=(0, 1, 2)):
    """Cut planes (along the given axes) through the ends of the contact sets of the 6 faces of the tight box b."""
    sub = occ[tuple(slice(*x) for x in b)]
    cuts = set()
    for axis in range(3):
        for side in (0, -1):
            face = np.take(sub, side, axis=axis)
            for f_ax, ax in enumerate(a for a in range(3) if a != axis):
                hit = np.nonzero(face.any(axis=1 - f_ax))[0]
                for c in ((hit.min(), hit.max() + 1) if len(hit) else ()):
                    if ax in axes and min_size <= c <= b[ax][1] - b[ax][0] - min_size:
                        cuts.add((ax, b[ax][0] + c))
    return cuts


def split(b, ax, c):
    lo, hi = list(b), list(b)
    lo[ax], hi[ax] = (b[ax][0], c), (c, b[ax][1])
    return tuple(lo), tuple(hi)


def decompose(occ, depth=3, min_gain=0.05, region=None, level=0, out=None, symmetric=False):
    """Tight cuboid -> best cut through a contact point -> two touching tight cuboids -> recurse.
    symmetric: every cuboid is centred on the left-right mirror plane (y = 0), so only cuts across the
    length (x) or the height (z) are allowed and each cuboid has 5 instead of 6 free parameters."""
    out = [] if out is None else out
    fit = symmetric_tight if symmetric else tight
    b = fit(occ, region or tuple((0, n) for n in occ.shape))
    if b is None:
        return out
    best = None
    if level < depth:
        for ax, c in contact_cuts(occ, b, axes=(0, 2) if symmetric else (0, 1, 2)):
            v = sum(volume(k) for k in (fit(occ, r) for r in split(b, ax, c)) if k is not None)
            if best is None or v < best[0]:
                best = (v, ax, c)
    if best is None or best[0] > (1 - min_gain) * volume(b):
        out.append({"b": b, "level": level})
        return out
    for r in split(b, best[1], best[2]):
        decompose(occ, depth, min_gain, r, level + 1, out, symmetric)
    return out


def boxes_volume(boxes, shape):
    v = np.zeros(shape, bool)
    for d in boxes:
        v[tuple(slice(*x) for x in d["b"])] = True
    return v


def box_corners(b, axes):
    (x0, x1), (y0, y1), (z0, z1) = [(a[i0] - VOX / 2, a[i1 - 1] + VOX / 2) for a, (i0, i1) in zip(axes, b)]
    return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                     [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def draw_boxes_3d(ax, corners_list, colors, alpha=0.4, edge="#1f1f1e"):
    """Draw solid cuboids on a matplotlib 3D axis.

    ax: a matplotlib Axes3D (created with projection="3d").
    corners_list: list of np.ndarray of shape (8, 3); corner order is
        0..3 = bottom face (x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0) and
        4..7 = the same four corners on the top face (z1).
    colors: list of matplotlib colour strings, same length as corners_list.
    Draws the 6 faces of each cuboid as one Poly3DCollection with facecolor=color, edgecolor=edge,
    alpha=alpha, linewidths=0.6. Returns None.
    """
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    for corners, color in zip(corners_list, colors):
        ax.add_collection3d(Poly3DCollection([corners[f] for f in faces], facecolor=color, edgecolor=edge,
                                             alpha=alpha, linewidths=0.6))


def style_3d(ax, l=5.0, h=2.5):
    ax.set_xlim(-l / 2, l / 2)
    ax.set_ylim(-l / 2, l / 2)
    ax.set_zlim(0, h)
    ax.set_box_aspect((l, l, h))
    ax.view_init(22, -58)
    ax.tick_params(labelsize=7, colors=MUTED)


# --- data ------------------------------------------------------------------------------------
class Sequence:
    def __init__(self):
        self.fids, self.times, self.cams, self.tracks, self.camera_xy = vd.load_sequence()
        # a frame whose image is missing (gaps in the recording) stays None: frame index = 10 Hz time slot
        self.frames = [cv2.imread(os.path.join(vd.DATA, "image", f"{f}.jpg")) for f in self.fids]
        have = [f for f in self.frames if f is not None]
        self.bg = np.median(np.stack(have[::2]), axis=0).astype(np.uint8)
        self.shape = have[0].shape[:2]

    def background(self, k):
        """The fixed camera: the video's median frame for every frame."""
        return self.bg

    def static_flow(self, k, m):
        """Image motion of a region that stands still: none for a fixed camera."""
        return np.zeros((int(np.count_nonzero(m)), 2))

    def mask(self, o, pad=8):
        """Largest foreground component inside the (padded) labelled 2D box."""
        H, W = self.shape
        x0, y0, x1, y1 = np.array(o["box"]).round().astype(int)
        x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
        fg = vc.foreground(self.frames[o["k"]][y0:y1, x0:x1], self.bg[y0:y1, x0:x1])
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
        m = np.zeros((H, W), np.uint8)
        if n > 1:
            m[y0:y1, x0:x1] = lab == 1 + np.argmax(stats[1:, 4])
        return m

    def usable(self, o):
        """Not truncated, away from the image border, not overlapping another labelled box."""
        H, W = self.shape
        b = np.array(o["box"])
        if o["truncated"] or b[0] < 5 or b[1] < 5 or b[2] > W - 5 or b[3] > H - 5:
            return False
        for t in self.tracks.values():
            for c in (np.array(p["box"]) for p in t if p["k"] == o["k"]):
                if not np.allclose(c, b):
                    ix = max(0, min(b[2], c[2]) - max(b[0], c[0]))
                    iy = max(0, min(b[3], c[3]) - max(b[1], c[1]))
                    if ix * iy > 0.02 * (b[2] - b[0]) * (b[3] - b[1]):
                        return False
        return True

    def view(self, o, mask=None):
        return self.cams[o["k"]], o["pos"], o["yaw"], self.mask(o) if mask is None else mask


def silhouette(occ, axes, cam, center, yaw, shape):
    ii = np.argwhere(occ)
    m = np.zeros(shape, np.uint8)
    if len(ii):
        P = np.c_[axes[0][ii[:, 0]], axes[1][ii[:, 1]], axes[2][ii[:, 2]]]
        u, v = project(cam, world(P, center, yaw))[0].round().astype(int).T
        ok = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
        m[v[ok], u[ok]] = 1
    return cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def label_box_silhouette(o, cam, shape):
    l, w, h = o["dims"]
    C = np.array([[x, y, z] for x in (-l / 2, l / 2) for y in (-w / 2, w / 2) for z in (o["z"] - h / 2, o["z"] + h / 2)])
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(project(cam, world(C, o["pos"], o["yaw"]))[0].round().astype(np.int32)), 1)
    return m


# --- synthetic car ---------------------------------------------------------------------------
PROFILE = np.array([(-2.3, 0), (-2.3, 0.85), (-1.9, 0.95), (-1.2, 1.0), (-0.9, 1.45), (0.5, 1.45), (1.3, 0.95),
                    (2.3, 0.75), (2.3, 0)])    # side profile (x forward, z up), m


def synthetic_inside(P):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    in_profile = Path(PROFILE).contains_points(np.c_[x.ravel(), z.ravel()]).reshape(x.shape)
    return in_profile & (np.abs(y) <= np.where(z < 0.95, 0.9, 0.9 - 0.3 * (z - 0.95) / 0.5))


def synthetic_experiment(seq, tid=TRACKS[0]):
    axes = grid(4.6, 1.8, 1.45, margin=0.0, zmargin=0.8)
    truth = synthetic_inside(np.stack(np.meshgrid(*axes, indexing="ij"), -1))
    fine = np.stack(np.meshgrid(np.arange(-2.3, 2.3, 0.02), np.arange(-0.9, 0.9, 0.02), np.arange(0, 1.46, 0.02),
                                indexing="ij"), -1)
    surface = fine[synthetic_inside(fine)]
    path = seq.tracks[tid]

    def render(o):
        u, v = project(seq.cams[o["k"]], world(surface, o["pos"], o["yaw"]))[0].round().astype(int).T
        ok = (u >= 0) & (u < seq.shape[1]) & (v >= 0) & (v < seq.shape[0])
        m = np.zeros(seq.shape, np.uint8)
        m[v[ok], u[ok]] = 1
        return cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    results, models = {}, {}
    for n_views in (1, 3, 10, 30):
        picks = [len(path) // 2] if n_views == 1 else np.linspace(0, len(path) - 1, n_views).round().astype(int)
        views = [seq.view(path[k], render(path[k])) for k in picks]
        hull = height_field(carve_votes(axes, views) >= 1.0)
        results[n_views] = {"hull": iou(hull, truth)}
        for d in range(5):
            boxes = decompose(hull, d)
            results[n_views][d] = (iou(boxes_volume(boxes, hull.shape), truth), len(boxes))
            models[n_views, d] = boxes
        models[n_views, "hull"] = hull
        models[n_views, "frame"] = (path[picks[len(picks) // 2]], views[len(picks) // 2][3])
    return axes, truth, results, models


# --- real cars ---------------------------------------------------------------------------------
def real_experiment(seq, tid):
    obs = [o for o in seq.tracks[tid] if seq.usable(o)]
    l, w = (float(np.median([o["dims"][i] for o in obs])) for i in (0, 1))
    axes = grid(l, w, 1.6)
    train = [o for i, o in enumerate(obs) if i % 4]
    test = [o for i, o in enumerate(obs) if i % 4 == 0]
    test_masks = [seq.mask(o) for o in test]
    out = {"obs": obs, "test": test, "axes": axes, "n_train": len(train)}
    scores = {}

    def score(occ):
        return float(np.mean([iou(silhouette(occ, axes, seq.cams[o["k"]], o["pos"], o["yaw"], seq.shape) > 0, m > 0)
                              for o, m in zip(test, test_masks)]))
    scores["labelled 3D box"] = float(np.mean([iou(label_box_silhouette(o, seq.cams[o["k"]], seq.shape) > 0, m > 0)
                                               for o, m in zip(test, test_masks)]))
    one = height_field(carve_votes(axes, [seq.view(train[len(train) // 2])]) >= VOTE)
    scores["hull, 1 frame"] = score(one)
    hull = height_field(carve_votes(axes, [seq.view(o) for o in train]) >= VOTE)
    for d in (0, 1, 2, 4):
        boxes = decompose(hull, d)
        scores[f"{len(boxes)} cuboid{'s' if len(boxes) > 1 else ''} (depth {d})"] = score(boxes_volume(boxes, hull.shape))
        out[d] = boxes
    scores[f"hull, {len(train)} frames"] = score(hull)
    out.update(hull=hull, scores=scores,
               height=float(axes[2][np.nonzero(hull.any((0, 1)))[0].max()] + VOX / 2),
               test_masks=test_masks)
    return out


# --- figures -----------------------------------------------------------------------------------
def draw_boxes_image(ax, boxes, axes, cam, center, yaw):
    for b in boxes:
        uv = project(cam, world(box_corners(b["b"], axes), center, yaw))[0]
        for p, q in EDGES:
            ax.plot(*uv[[p, q]].T, color=LEVEL_COLORS[b["level"] % len(LEVEL_COLORS)], linewidth=1.3)


def crop(ax, box, pad=25):
    x0, y0, x1, y1 = box
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y1 + pad, y0 - pad)
    ax.axis("off")


def synthetic_figure(path, seq, axes, truth, results, models):
    fig = plt.figure(figsize=(17, 8.2))
    o, m = models[10, "frame"]
    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(np.where(m[..., None] > 0, 215, 70).astype(np.uint8).repeat(3, axis=2))
    draw_boxes_image(ax, models[10, 3], axes, seq.cams[o["k"]], o["pos"], o["yaw"])
    crop(ax, cv2.boundingRect(m)[:2] + tuple(np.add(cv2.boundingRect(m)[:2], cv2.boundingRect(m)[2:])))
    ax.set_title("rendered silhouette + cuboids (depth 3)", fontsize=10, loc="left", color=INK)
    small = tuple(slice(None, None, 2) for _ in range(3))
    for k, (title, vol) in enumerate((("true shape", truth), ("visual hull, 10 frames", models[10, "hull"]))):
        ax = fig.add_subplot(2, 4, 2 + k, projection="3d")
        X, Y, Z = (a[s] for a, s in zip(np.meshgrid(*[np.r_[a[::2] - VOX, a[-1] + VOX] for a in axes], indexing="ij"),
                                         (small,) * 3))
        ax.voxels(*np.meshgrid(*[np.r_[a[::2] - VOX / 2, a[::2][-1] + 1.5 * VOX] for a in axes], indexing="ij"),
                  vol[small], facecolors="#9fb9d8", edgecolor="#5b7fa8", linewidth=0.1)
        style_3d(ax)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax = fig.add_subplot(2, 4, 4, projection="3d")
    boxes = models[10, 3]
    draw_boxes_3d(ax, [box_corners(b["b"], axes) for b in boxes],
                  [LEVEL_COLORS[b["level"] % len(LEVEL_COLORS)] for b in boxes])
    style_3d(ax)
    ax.set_title(f"{len(boxes)} touching cuboids (depth 3)", fontsize=10, loc="left", color=INK)

    for k, d in enumerate((0, 1, 2)):
        ax = fig.add_subplot(2, 4, 5 + k, projection="3d")
        boxes = models[10, d]
        draw_boxes_3d(ax, [box_corners(b["b"], axes) for b in boxes],
                      [LEVEL_COLORS[b["level"] % len(LEVEL_COLORS)] for b in boxes])
        style_3d(ax)
        ax.set_title(f"depth {d}: {len(boxes)} cuboid{'s' if len(boxes) > 1 else ''}, "
                     f"IoU {results[10][d][0]:.2f}", fontsize=10, loc="left", color=INK)
    ax = fig.add_subplot(2, 4, 8)
    for n_views, color in zip((1, 3, 10, 30), ("#c3c2b7", MUTED, "#2a78d6", "#eb6834")):
        ax.plot(range(5), [results[n_views][d][0] for d in range(5)], "o-", color=color, linewidth=2, markersize=6,
                label=f"{n_views} frame{'s' if n_views > 1 else ''}")
        ax.axhline(results[n_views]["hull"], color=color, linewidth=1, linestyle=(0, (3, 2)))
    ax.set_xticks(range(5))
    ax.set_xlabel("recursion depth", color=INK)
    ax.set_ylabel("voxel IoU with the true shape", color=INK)
    ax.set_ylim(0.3, 1.0)
    ax.set_title("cuboids (solid) vs visual hull (dashed)", fontsize=10, loc="left", color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    vd.style(ax)
    fig.suptitle("Synthetic car (sloped hood and windscreen, narrower cabin) moved along V2X-Seq track "
                 f"{TRACKS[0]}, real camera", fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=105, facecolor="white")
    plt.close(fig)


def real_figure(path, seq, runs):
    fig = plt.figure(figsize=(17, 4.3 * len(runs)))
    for r, (tid, res) in enumerate(runs.items()):
        axes, boxes = res["axes"], res[2]
        picks = [0, len(res["test"]) // 2, len(res["test"]) - 1]
        for c, k in enumerate(picks):
            o, m = res["test"][k], res["test_masks"][k]
            ax = fig.add_subplot(len(runs), 5, 5 * r + c + 1)
            ax.imshow(cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2RGB))
            for cnt in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
                ax.plot(*np.vstack([cnt[:, 0], cnt[:1, 0]]).T, color="white", linewidth=1.1, linestyle=(0, (2, 2)))
            draw_boxes_image(ax, boxes, axes, seq.cams[o["k"]], o["pos"], o["yaw"])
            crop(ax, o["box"])
            dist = np.linalg.norm(o["pos"] - seq.camera_xy)
            ax.set_title(f"car {tid}, held-out frame {o['k']} ({dist:.0f} m)", fontsize=9.5, loc="left", color=INK)
        ax = fig.add_subplot(len(runs), 5, 5 * r + 4, projection="3d")
        draw_boxes_3d(ax, [box_corners(b["b"], axes) for b in boxes],
                      [LEVEL_COLORS[b["level"] % len(LEVEL_COLORS)] for b in boxes])
        style_3d(ax)
        ax.set_title(f"{len(boxes)} touching cuboids (depth 2)", fontsize=9.5, loc="left", color=INK)
        ax = fig.add_subplot(len(runs), 5, 5 * r + 5)
        side = res["hull"].any(1).T
        ax.imshow(side, origin="lower", cmap="Greys", vmin=0, vmax=2.2, aspect="equal",
                  extent=(axes[0][0] - VOX / 2, axes[0][-1] + VOX / 2, 0, axes[2][-1] + VOX / 2))
        for b in boxes:
            C = box_corners(b["b"], axes)
            ax.add_patch(plt.Rectangle((C[0, 0], C[0, 2]), C[1, 0] - C[0, 0], C[4, 2] - C[0, 2], fill=False,
                                       edgecolor=LEVEL_COLORS[b["level"] % len(LEVEL_COLORS)], linewidth=1.5))
        ax.set_xlabel("along the car: rear → front (m)", color=INK, fontsize=9)
        ax.set_ylabel("height (m)", color=INK, fontsize=9)
        ax.set_title(f"side view: hull from {res['n_train']} frames (grey)", fontsize=9.5, loc="left", color=INK)
        ax.tick_params(labelsize=8, colors=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=95, facecolor="white")
    plt.close(fig)


def scores_figure(path, runs):
    names = list(next(iter(runs.values()))["scores"])
    fig, ax = plt.subplots(figsize=(13, 4.6))
    colors = ["#5b5a55", "#c3c2b7", "#9fb9d8", "#6f9fd6", "#2a78d6", "#1d4f8f", "#eb6834"]
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        vals = [list(r["scores"].values())[k] for r in runs.values()]
        bars = ax.bar(np.arange(len(runs)) + (k - (len(names) - 1) / 2) * width, vals, width * 0.92,
                      color=colors[k % len(colors)], zorder=2,
                      label=name.split(",")[0] + (", 1 frame" if "1 frame" in name else ", all frames" if "hull" in name else ""))
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(runs)), [f"car {tid} ({r['n_train']} frames, {len(r['test'])} held out)"
                                     for tid, r in runs.items()])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("IoU with the held-out silhouette", color=INK)
    ax.set_title("V2X-Seq: how well each shape predicts frames it was not built from", fontsize=11, loc="left",
                 color=INK)
    ax.legend(fontsize=8.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


# --- left-right symmetry ------------------------------------------------------------------------
DEPTHS = (0, 1, 2, 3)


def reconstruct(axes, views, symmetric, threshold):
    """Hull and recursive cuboids per depth; symmetric = mirrored views in the hull + centred cuboids."""
    hull = height_field(carve_votes(axes, views, symmetric) >= threshold)
    return {"hull": hull, **{d: decompose(hull, d, symmetric=symmetric) for d in DEPTHS}}


def symmetry_experiments(seq):
    """Synthetic car (true shape known) and the real cars, each without and with symmetry, 1 frame and all."""
    out = {}
    axes = grid(4.6, 1.8, 1.45, margin=0.0, zmargin=0.8)
    truth = synthetic_inside(np.stack(np.meshgrid(*axes, indexing="ij"), -1))
    fine = np.stack(np.meshgrid(np.arange(-2.3, 2.3, 0.02), np.arange(-0.9, 0.9, 0.02), np.arange(0, 1.46, 0.02),
                                indexing="ij"), -1)
    surface = fine[synthetic_inside(fine)]
    path = seq.tracks[TRACKS[0]]

    def render(o):
        u, v = project(seq.cams[o["k"]], world(surface, o["pos"], o["yaw"]))[0].round().astype(int).T
        ok = (u >= 0) & (u < seq.shape[1]) & (v >= 0) & (v < seq.shape[0])
        m = np.zeros(seq.shape, np.uint8)
        m[v[ok], u[ok]] = 1
        return cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    for n in (1, 10):
        picks = [len(path) // 2] if n == 1 else np.linspace(0, len(path) - 1, n).round().astype(int)
        views = [seq.view(path[k], render(path[k])) for k in picks]
        for sym in (False, True):
            rec = reconstruct(axes, views, sym, 1.0)
            rec["score"] = {d: iou(boxes_volume(rec[d], truth.shape), truth) for d in DEPTHS}
            rec["score"]["hull"] = iou(rec["hull"], truth)
            out["synthetic", n, sym] = rec
    out["synthetic"] = {"axes": axes, "truth": truth, "frame": path[len(path) // 2],
                        "mask": render(path[len(path) // 2])}
    for tid in TRACKS:
        obs = [o for o in seq.tracks[tid] if seq.usable(o)]
        l, w = (float(np.median([o["dims"][i] for o in obs])) for i in (0, 1))
        axes = grid(l, w, 1.6)
        train = [o for i, o in enumerate(obs) if i % 4]
        test = [o for i, o in enumerate(obs) if i % 4 == 0]
        masks = [seq.mask(o) > 0 for o in test]

        def score(occ):
            return float(np.mean([iou(silhouette(occ, axes, seq.cams[o["k"]], o["pos"], o["yaw"], seq.shape) > 0, m)
                                  for o, m in zip(test, masks)]))
        for n, chosen in ((1, [train[len(train) // 2]]), (len(train), train)):
            views = [seq.view(o) for o in chosen]
            for sym in (False, True):
                rec = reconstruct(axes, views, sym, VOTE)
                rec["score"] = {d: score(boxes_volume(rec[d], rec["hull"].shape)) for d in DEPTHS}
                rec["score"]["hull"] = score(rec["hull"])
                rec["height"] = float(axes[2][np.nonzero(rec["hull"].any((0, 1)))[0].max()] + VOX / 2)
                out[tid, "one" if n == 1 else "all", sym] = rec
        out[tid] = {"axes": axes, "test": test, "n_train": len(train), "frame": test[len(test) // 2]}
    return out


def hull_voxels(ax, occ, axes, color="#9fb9d8"):
    step = 2
    edges = [np.r_[a[::step] - VOX / 2, a[::step][-1] + (step - 0.5) * VOX] for a in axes]
    ax.voxels(*np.meshgrid(*edges, indexing="ij"), occ[::step, ::step, ::step], facecolors=color,
              edgecolor="#5b7fa8", linewidth=0.1)


def boxes_3d(ax, boxes, axes):
    draw_boxes_3d(ax, [box_corners(b["b"], axes) for b in boxes],
                  [LEVEL_COLORS[b["level"] % len(LEVEL_COLORS)] for b in boxes], alpha=0.55)


def recursion_3d_figure(path, seq, res):
    rows = [("synthetic, 10 frames", "synthetic", 10), ("synthetic, 1 frame", "synthetic", 1)] + \
           [(f"car {tid}, {res[tid]['n_train']} frames", tid, "all") for tid in TRACKS] + \
           [(f"car {TRACKS[0]}, 1 frame", TRACKS[0], "one")]
    cols = 7
    fig = plt.figure(figsize=(22, 3.3 * len(rows)))
    for r, (name, key, n) in enumerate(rows):
        info, sym, asym = res[key], res[key, n, True], res[key, n, False]
        axes = info["axes"]
        ax = fig.add_subplot(len(rows), cols, cols * r + 1, projection=None if key != "synthetic" else "3d")
        if key == "synthetic":
            hull_voxels(ax, info["truth"], axes, "#c9d9b0")
            style_3d(ax)
            ax.set_title("true shape", fontsize=9.5, loc="left", color=INK)
        else:
            o = info["frame"]
            ax.imshow(cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2RGB))
            draw_boxes_image(ax, sym[3], axes, seq.cams[o["k"]], o["pos"], o["yaw"])
            crop(ax, o["box"])
            ax.set_title(f"held-out frame {o['k']}, depth 3", fontsize=9.5, loc="left", color=INK)
        ax.text2D(-0.02, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10.5,
                  color=INK) if key == "synthetic" else ax.text(-0.04, 0.5, name, transform=ax.transAxes, rotation=90,
                                                                va="center", ha="right", fontsize=10.5, color=INK)
        ax = fig.add_subplot(len(rows), cols, cols * r + 2, projection="3d")
        hull_voxels(ax, sym["hull"], axes)
        style_3d(ax)
        ax.set_title("symmetric hull" + (f" (IoU {sym['score']['hull']:.2f})" if key == "synthetic" else
                                         f", top {sym['height']:.2f} m"), fontsize=9.5, loc="left", color=INK)
        for c, d in enumerate(DEPTHS):
            ax = fig.add_subplot(len(rows), cols, cols * r + 3 + c, projection="3d")
            boxes_3d(ax, sym[d], axes)
            style_3d(ax)
            ax.set_title(f"depth {d}: {len(sym[d])} cuboid{'s' if len(sym[d]) > 1 else ''}, "
                         f"{'IoU' if key == 'synthetic' else 'held-out'} {sym['score'][d]:.2f}", fontsize=9.5,
                         loc="left", color=INK)
        ax = fig.add_subplot(len(rows), cols, cols * r + 7, projection="3d")
        boxes_3d(ax, asym[3], axes)
        style_3d(ax)
        ax.set_title(f"no symmetry, depth 3: {asym['score'][3]:.2f}", fontsize=9.5, loc="left", color=MUTED)
    fig.suptitle("Recursive touching cuboids with left-right symmetry (colour = recursion level; "
                 "synthetic: voxel IoU with the true shape, real: IoU with held-out silhouettes)",
                 fontsize=12, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=90, facecolor="white")
    plt.close(fig)


def symmetry_scores_figure(path, res):
    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    for n, color in ((1, "#eb6834"), (10, "#2a78d6")):
        for sym, ls in ((False, (0, (3, 2))), (True, "-")):
            sc = res["synthetic", n, sym]["score"]
            ax_s.plot(DEPTHS, [sc[d] for d in DEPTHS], marker="o", color=color, linestyle=ls, linewidth=2,
                      label=f"{n} frame{'s' if n > 1 else ''}, {'symmetric' if sym else 'no symmetry'}")
    ax_s.set_xticks(DEPTHS)
    ax_s.set_xlabel("recursion depth", color=INK)
    ax_s.set_ylabel("voxel IoU with the true shape", color=INK)
    ax_s.set_ylim(0.3, 1.0)
    ax_s.set_title("Synthetic car", fontsize=10.5, loc="left", color=INK)
    ax_s.legend(fontsize=8.5, frameon=False, loc="lower right")
    vd.style(ax_s)
    groups = [(tid, n) for tid in TRACKS for n in ("one", "all")]
    labels = [f"{tid}\n{'1 frame' if n == 'one' else str(res[tid]['n_train']) + ' frames'}" for tid, n in groups]
    width = 0.2
    for k, (sym, key, color, name) in enumerate(((False, 3, "#c3c2b7", "cuboids depth 3, no symmetry"),
                                                 (True, 3, "#2a78d6", "cuboids depth 3, symmetric"),
                                                 (False, "hull", "#e8b9a3", "hull, no symmetry"),
                                                 (True, "hull", "#eb6834", "hull, symmetric"))):
        vals = [res[tid, n, sym]["score"][key] for tid, n in groups]
        bars = ax_r.bar(np.arange(len(groups)) + (k - 1.5) * width, vals, width * 0.92, color=color, label=name,
                        zorder=2)
        for b, v in zip(bars, vals):
            ax_r.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                          ha="center", fontsize=7, color=INK)
    ax_r.set_xticks(range(len(groups)), labels, fontsize=8.5)
    ax_r.set_ylim(0, 1.0)
    ax_r.set_ylabel("IoU with held-out silhouettes", color=INK)
    ax_r.set_title("Real V2X-Seq cars", fontsize=10.5, loc="left", color=INK)
    ax_r.legend(fontsize=8.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    vd.style(ax_r)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


# --- oriented surface patches -------------------------------------------------------------------
FLAT = 0.10                      # m: a cuboid thinner than this is a flat surface patch
ROTATE_FROM = 4                  # cuboids of depth 1..3 keep the parent's axes, deeper ones may turn


def pca_frame(P):
    """Principal axes of a point set as a right-handed rotation matrix (columns: longest .. thinnest)."""
    if len(P) < 3:
        return np.eye(3)
    R = np.linalg.eigh(np.cov((P - P.mean(0)).T))[1][:, ::-1].copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R
PATCH_DEPTHS = (0, 1, 2, 3, 5, 8)


def box_surface_samples(center, R, half, step=0.03):
    """Points on the 6 faces of an oriented box.
    center: (3,) box centre; R: (3, 3) rotation, columns are the box axes; half: (3,) half extents along those axes.
    Local points L (m, 3) are mapped to world as center + L @ R.T. Returns one np.ndarray (N, 3)."""
    faces = []
    for a in range(3):
        b, c = (i for i in range(3) if i != a)
        U, V = np.meshgrid(*(np.linspace(-half[k], half[k], max(2, int(np.ceil(2 * half[k] / step)) + 1))
                             for k in (b, c)))
        for sign in (-1, 1):
            L = np.zeros((U.size, 3))
            L[:, a], L[:, b], L[:, c] = sign * half[a], U.ravel(), V.ravel()
            faces.append(center + L @ R.T)
    return np.vstack(faces)


def surface_points(occ, axes):
    """Centres of hull voxels with an empty neighbour to the side or above (the ground side is never seen)."""
    p = np.pad(occ, ((1, 1), (1, 1), (0, 1)))
    empty = (~p[2:, 1:-1, :-1] | ~p[:-2, 1:-1, :-1] | ~p[1:-1, 2:, :-1] | ~p[1:-1, :-2, :-1] | ~p[1:-1, 1:-1, 1:])
    ii = np.argwhere(occ & empty)
    return np.c_[axes[0][ii[:, 0]], axes[1][ii[:, 1]], axes[2][ii[:, 2]]]


def oriented_box(P, R):
    L = P @ R
    lo, hi = L.min(0) - VOX / 2, L.max(0) + VOX / 2
    return {"c": R @ ((lo + hi) / 2), "R": R, "h": (hi - lo) / 2}


def touches_parent(child, parent, tol):
    """True if the child box lies flush against at least one face of the parent box.
    Boxes are dicts with keys "c" (np.ndarray (3,) centre), "R" (np.ndarray (3, 3) rotation, columns are the
    box axes) and "h" (np.ndarray (3,) half extents along those axes). The child has the same axes as the parent.
    Returns True if, in the parent frame, one of the child's faces lies within tol of the parent's face on the
    same side of the same axis."""
    d = parent["R"].T @ (child["c"] - parent["c"])
    return bool(np.any((np.abs(d - child["h"] + parent["h"]) <= tol) | (np.abs(d + child["h"] - parent["h"]) <= tol)))


def child_boxes(P, masks, parent, rotate=False):
    """Tight boxes of the pieces. rotate=False: in the parent's axes, and None if one of them would not lie
    flush against a face of the parent. rotate=True: the tighter of the parent-aligned box and the box along
    the piece's principal axes (a small piece of a sloped surface turns to follow it)."""
    if rotate:
        return [min((oriented_box(P[m], parent["R"]), oriented_box(P[m], pca_frame(P[m]))),
                    key=lambda b: np.prod(b["h"])) for m in masks]
    kids = [oriented_box(P[m], parent["R"]) for m in masks]
    return kids if all(touches_parent(k, parent, 0.51 * VOX) for k in kids) else None


def outer_box(P):
    """Depth 0: axis-aligned in the object frame, centred on the mirror plane y = 0, standing on the ground."""
    lo, hi = P.min(0) - VOX / 2, P.max(0) + VOX / 2
    y = max(-lo[1], hi[1])
    lo, hi = np.array([lo[0], -y, 0.0]), np.array([hi[0], y, hi[2]])
    return {"c": (lo + hi) / 2, "R": np.eye(3), "h": (hi - lo) / 2}


def patch_decompose(P, depth, min_gain=0.05, min_points=6, level=0, box=None, out=None, rotate_from="default"):
    """Recursive touching cuboids over the hull surface. Depth 0 is the symmetric outer cuboid standing on
    the ground. Every deeper cuboid keeps its parent's axes and is the tight box of its piece of surface,
    which must lie flush against a face of the parent; a piece of surface along a parent face becomes a
    flat slab on that face. Cuts go through the ends of the contact sets (surface points touching a face
    of the cuboid), across any of its axes; the cut whose two child cuboids have the least volume wins.
    If no contact cut shrinks the volume by min_gain, the piece is halved along its longest axis
    (counted as 'mid'). rotate_from: depth from which cuboids may turn (None = never; default ROTATE_FROM)."""
    rotate_from = ROTATE_FROM if rotate_from == "default" else rotate_from
    rotate = rotate_from is not None and level + 1 >= rotate_from
    out = [] if out is None else out
    box = box or outer_box(P)
    box.update(level=level, n=len(P))
    thin = 2 * box["h"].min() <= FLAT
    if level >= depth or len(P) < 2 * min_points or (level > 0 and thin):
        box["flat"] = thin
        out.append(box)
        return out
    L = (P - box["c"]) @ box["R"]
    candidates = set()
    for f in range(3):
        for side in (-1, 1):
            touching = np.abs(L[:, f] - side * box["h"][f]) <= 0.75 * VOX
            for a in range(3):
                if a != f and touching.any():
                    candidates.update((a, v) for v in (L[touching, a].min() - VOX / 2, L[touching, a].max() + VOX / 2))
    best = None
    for a, v in candidates:
        lo = L[:, a] < v
        if min(lo.sum(), (~lo).sum()) < min_points:
            continue
        kids = child_boxes(P, (lo, ~lo), box, rotate)
        if kids is None:
            continue
        vol = sum(np.prod(2 * k["h"]) for k in kids)
        if best is None or vol < best[0]:
            best = (vol, lo, kids, "contact")
    if best is None or best[0] > (1 - min_gain) * np.prod(2 * box["h"]):
        a = int(np.argmax(box["h"]))
        lo = L[:, a] < np.median(L[:, a])
        kids = child_boxes(P, (lo, ~lo), box, rotate) if min(lo.sum(), (~lo).sum()) >= min_points else None
        if kids is None:
            box["flat"] = thin
            out.append(box)
            return out
        best = (None, lo, kids, "mid")
    for m, kid in zip((best[1], ~best[1]), best[2]):
        kid["cut"] = best[3]
        patch_decompose(P[m], depth, min_gain, min_points, level + 1, kid, out, rotate_from)
    return out


def patch_points(boxes, step=0.03):
    return np.vstack([box_surface_samples(b["c"], b["R"], b["h"], step) for b in boxes])


def chamfer(A, B):
    from scipy.spatial import cKDTree
    return 0.5 * (cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean())


def patch_silhouette(boxes, cam, center, yaw, shape):
    m = np.zeros(shape, np.uint8)
    for b in boxes:
        C = b["c"] + np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * b["h"] @ b["R"].T
        cv2.fillConvexPoly(m, cv2.convexHull(project(cam, world(C, center, yaw))[0].round().astype(np.int32)), 1)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, 1, -1)
    return filled


def patch_experiments(seq):
    res = {}
    axes = grid(4.6, 1.8, 1.45, margin=0.0, zmargin=0.8)
    truth = synthetic_inside(np.stack(np.meshgrid(*axes, indexing="ij"), -1))
    true_surface = surface_points(truth, axes)
    fine = np.stack(np.meshgrid(np.arange(-2.3, 2.3, 0.02), np.arange(-0.9, 0.9, 0.02), np.arange(0, 1.46, 0.02),
                                indexing="ij"), -1)
    surface = fine[synthetic_inside(fine)]
    path = seq.tracks[TRACKS[0]]

    def render(o):
        u, v = project(seq.cams[o["k"]], world(surface, o["pos"], o["yaw"]))[0].round().astype(int).T
        ok = (u >= 0) & (u < seq.shape[1]) & (v >= 0) & (v < seq.shape[0])
        m = np.zeros(seq.shape, np.uint8)
        m[v[ok], u[ok]] = 1
        return cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    picks = set(np.linspace(0, len(path) - 1, 10).round().astype(int).tolist())
    views = [seq.view(path[k], render(path[k])) for k in sorted(picks)]
    hull = height_field(carve_votes(axes, views, True) >= 1.0)
    P = surface_points(hull, axes)
    rec = {"P": P, "hull_score": chamfer(P, true_surface), "truth_P": true_surface, "n_train": len(views),
           "truth_boxes": patch_decompose(true_surface, 8),
           "frames": [(o, render(o) > 0, k in picks) for k, o in enumerate(path) if seq.usable(o)]}
    for d in PATCH_DEPTHS:
        boxes = patch_decompose(P, d)
        rec[d] = boxes
        rec["score", d] = chamfer(patch_points(boxes), true_surface)
    res["synthetic"] = rec
    for tid in TRACKS:
        obs = [o for o in seq.tracks[tid] if seq.usable(o)]
        l, w = (float(np.median([o["dims"][i] for o in obs])) for i in (0, 1))
        axes = grid(l, w, 1.6)
        train = [o for i, o in enumerate(obs) if i % 4]
        test = [o for i, o in enumerate(obs) if i % 4 == 0]
        masks = [seq.mask(o) > 0 for o in test]
        hull = height_field(carve_votes(axes, [seq.view(o) for o in train], True) >= VOTE)
        P = surface_points(hull, axes)
        rec = {"P": P, "test": test, "n_train": len(train), "frame": test[len(test) // 2],
               "frames": [(o, seq.mask(o) > 0, i % 4 != 0) for i, o in enumerate(obs)],
               "hull_score": float(np.mean([iou(silhouette(hull, axes, seq.cams[o["k"]], o["pos"], o["yaw"],
                                                           seq.shape) > 0, m) for o, m in zip(test, masks)]))}
        for d in PATCH_DEPTHS:
            boxes = patch_decompose(P, d)
            rec[d] = boxes
            rec["score", d] = float(np.mean([iou(patch_silhouette(boxes, seq.cams[o["k"]], o["pos"], o["yaw"],
                                                                  seq.shape) > 0, m) for o, m in zip(test, masks)]))
        res[tid] = rec
    return res


def draw_patches_3d(ax, boxes, color_by="level"):
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    for b in boxes:
        C = b["c"] + (signs * b["h"]) @ b["R"].T
        if color_by == "normal" and b.get("flat"):
            n = np.abs(b["R"][:, 2])
            color = tuple(0.25 + 0.7 * n[[0, 2, 1]] / n.max())
        else:
            color = LEVEL_COLORS[b["level"] % len(LEVEL_COLORS)]
        ax.add_collection3d(Poly3DCollection([C[f] for f in faces], facecolor=color, edgecolor="#1f1f1e",
                                             alpha=0.75 if b.get("flat") else 0.45, linewidths=0.3))


def draw_patches_image(ax, boxes, cam, center, yaw, color_by="level"):
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    for b in boxes:
        uv = project(cam, world(b["c"] + (signs * b["h"]) @ b["R"].T, center, yaw))[0]
        color = LEVEL_COLORS[b["level"] % len(LEVEL_COLORS)] if color_by == "level" else "#f2c94c"
        for p, q in edges:
            ax.plot(*uv[[p, q]].T, color=color, linewidth=0.9 if b.get("flat") else 1.3)


def patches_figure(path, seq, res):
    rows = ["synthetic"] + list(TRACKS)
    cols = 2 + len(PATCH_DEPTHS)
    fig = plt.figure(figsize=(3.05 * cols, 3.3 * len(rows)))
    for r, key in enumerate(rows):
        rec = res[key]
        ax = fig.add_subplot(len(rows), cols, cols * r + 1, projection="3d" if key == "synthetic" else None)
        deep = rec[PATCH_DEPTHS[-1]]
        if key == "synthetic":
            ax.scatter(*rec["truth_P"][::2].T, s=0.6, color="#7a9a5a")
            style_3d(ax)
            ax.set_title("true surface", fontsize=9.5, loc="left", color=INK)
            name = "synthetic car, 10 frames"
        else:
            o = rec["frame"]
            ax.imshow(cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2RGB))
            draw_patches_image(ax, deep, seq.cams[o["k"]], o["pos"], o["yaw"], color_by="flat")
            crop(ax, o["box"])
            ax.set_title(f"held-out frame {o['k']}, depth {PATCH_DEPTHS[-1]}", fontsize=9.5, loc="left", color=INK)
            name = f"car {key}, {rec['n_train']} frames"
        (ax.text2D if key == "synthetic" else ax.text)(-0.06, 0.5, name, transform=ax.transAxes, rotation=90,
                                                         va="center", ha="right", fontsize=10.5, color=INK)
        ax = fig.add_subplot(len(rows), cols, cols * r + 2, projection="3d")
        ax.scatter(*rec["P"][::2].T, s=0.6, color="#5b7fa8")
        style_3d(ax)
        ax.set_title("hull surface (symmetric)", fontsize=9.5, loc="left", color=INK)
        for c, d in enumerate(PATCH_DEPTHS):
            boxes = rec[d]
            ax = fig.add_subplot(len(rows), cols, cols * r + 3 + c, projection="3d")
            draw_patches_3d(ax, boxes, color_by="normal" if d == PATCH_DEPTHS[-1] else "level")
            style_3d(ax)
            flat = sum(b.get("flat", False) for b in boxes)
            metric = (f"{100 * rec['score', d]:.0f} cm" if key == "synthetic" else f"{rec['score', d]:.2f}")
            ax.set_title(f"depth {d}: {len(boxes)} cuboids, {flat} flat\n"
                         f"{'surface error' if key == 'synthetic' else 'held-out IoU'} {metric}", fontsize=9,
                         loc="left", color=INK)
    fig.suptitle("Recursive touching cuboids: symmetric outer cuboid (depth 0), straight children flush against the parent (depth 1-3), rotated below; flat "
                 f"surface patches (thinner than {100 * FLAT:.0f} cm) at depth; last column coloured by patch normal",
                 fontsize=12, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=90, facecolor="white")
    plt.close(fig)


def patch_scores_figure(path, res):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axs[0]
    rec = res["synthetic"]
    ax.plot(PATCH_DEPTHS, [100 * rec["score", d] for d in PATCH_DEPTHS], "o-", color="#2a78d6", linewidth=2,
            label="recursive cuboids")
    ax.axhline(100 * rec["hull_score"], color="#eb6834", linestyle=(0, (3, 2)), linewidth=1.5,
               label="hull surface (limit)")
    ax.set_xlabel("recursion depth", color=INK)
    ax.set_ylabel("mean surface distance to the true car (cm)", color=INK)
    ax.set_title("Synthetic car: surface error", fontsize=10.5, loc="left", color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    vd.style(ax)
    ax = axs[1]
    for key, color in zip(["synthetic"] + list(TRACKS), ("#5b5a55", "#2a78d6", "#eb6834", "#2e9e6b")):
        thick = [np.median([2 * b["h"].min() for b in res[key][d]]) * 100 for d in PATCH_DEPTHS]
        ax.plot(PATCH_DEPTHS, thick, "o-", color=color, linewidth=2, label=key if key == "synthetic" else f"car {key}")
    ax.axhline(100 * FLAT, color=MUTED, linestyle=(0, (3, 2)), linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("recursion depth", color=INK)
    ax.set_ylabel("median cuboid thickness (cm)", color=INK)
    ax.set_title("Cuboids become flat patches", fontsize=10.5, loc="left", color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    vd.style(ax)
    ax = axs[2]
    for tid, color in zip(TRACKS, ("#2a78d6", "#eb6834", "#2e9e6b")):
        ax.plot(PATCH_DEPTHS, [res[tid]["score", d] for d in PATCH_DEPTHS], "o-", color=color, linewidth=2,
                label=f"car {tid}")
        ax.axhline(res[tid]["hull_score"], color=color, linestyle=(0, (3, 2)), linewidth=1)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("recursion depth", color=INK)
    ax.set_ylabel("IoU with held-out silhouettes", color=INK)
    ax.set_title("Real cars (dashed: hull)", fontsize=10.5, loc="left", color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


# --- original vs reconstruction from several sides ----------------------------------------------
LIGHT = np.array([0.35, 0.25, 0.9]) / np.linalg.norm([0.35, 0.25, 0.9])
CANONICAL = (("front", 8, 0), ("rear", 8, 180), ("side", 8, 90), ("top", 89, -90), ("3/4 front", 25, 40),
             ("3/4 rear", 25, 220))


def box_faces(center, R, half):
    """The 6 faces of an oriented box as polygons with their outward unit normals.
    center: np.ndarray (3,); R: np.ndarray (3, 3) rotation whose columns are the box axes;
    half: np.ndarray (3,) half extents along those axes.
    Returns a list of 6 tuples (polygon, normal): polygon is np.ndarray (4, 3) of world points in
    cyclic order (no self-crossing), normal is np.ndarray (3,) = sign * R[:, a] for the face at local
    coordinate a = sign * half[a]."""
    faces = []
    for a in range(3):
        b, c = (k for k in range(3) if k != a)
        for sign in (-1, 1):
            local = np.zeros((4, 3))
            local[:, a] = sign * half[a]
            local[:, b] = [-half[b], half[b], half[b], -half[b]]
            local[:, c] = [-half[c], -half[c], half[c], half[c]]
            faces.append((center + local @ R.T, sign * R[:, a]))
    return faces


def shade(base, k):
    return tuple(np.clip(np.array(matplotlib.colors.to_rgb(base)) * k, 0, 1))


def render_in_camera(ax, boxes, cam, center, yaw, base="#b9c7d8"):
    """Painter's-algorithm rendering of the cuboids seen by the real camera, shaded by the view angle."""
    from matplotlib.collections import PolyCollection
    K, Rc, t = cam
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    polys, depths, colors = [], [], []
    for b in boxes:
        for poly, n in box_faces(b["c"], b["R"], b["h"]):
            Xc = Rc @ world(poly, center, yaw).T + t[:, None]
            uv = (K @ Xc)[:2] / (K @ Xc)[2]
            ray = Xc.mean(1) / np.linalg.norm(Xc.mean(1))
            polys.append(uv.T)
            depths.append(Xc[2].mean())
            colors.append(shade(base, 0.35 + 0.65 * abs((Rc @ Rz @ n) @ ray)))
    order = np.argsort(depths)[::-1]
    ax.add_collection(PolyCollection([polys[k] for k in order], facecolors=[colors[k] for k in order],
                                     edgecolors=[shade(colors[k], 0.7) for k in order], linewidths=0.3))


def render_canonical(ax, boxes, elev, azim, base="#b9c7d8", extent=(5.0, 2.4)):
    faces, colors = [], []
    for b in boxes:
        for poly, n in box_faces(b["c"], b["R"], b["h"]):
            faces.append(poly)
            colors.append(shade(base, 0.4 + 0.6 * abs(n @ LIGHT)))
    ax.add_collection3d(Poly3DCollection(faces, facecolors=colors, edgecolors=[shade(c, 0.7) for c in colors],
                                         linewidths=0.25))
    l, h = extent
    ax.set_xlim(-l / 2, l / 2)
    ax.set_ylim(-l / 2, l / 2)
    ax.set_zlim(0, h)
    ax.set_box_aspect((l, l, h), zoom=1.45)
    ax.set_proj_type("ortho")
    ax.view_init(elev, azim)
    ax.axis("off")


def view_azimuth(o, camera_xy):
    """Direction of the camera seen from the car, relative to its heading (0 = from the front)."""
    d = camera_xy - o["pos"]
    return (np.degrees(np.arctan2(d[1], d[0]) - o["yaw"]) + 180) % 360 - 180


def side_name(a):
    a = abs(a)
    return "front" if a < 30 else "front-quarter" if a < 70 else "side" if a < 110 else "rear-quarter" if a < 150 \
        else "rear"


def views_figure(path, seq, res, key, depth=8):
    rec = res[key]
    boxes = rec[depth]
    synthetic = key == "synthetic"
    frames = rec["frames"]
    az = np.array([view_azimuth(o, seq.camera_xy) for o, _, _ in frames])
    order = np.argsort(np.abs(az))
    picks = [int(order[int(round(q * (len(order) - 1)))]) for q in (0.0, 0.5, 1.0)]
    rows = 3 if synthetic else 2
    fig = plt.figure(figsize=(19, 3.6 * rows + 0.6))
    for c, k in enumerate(picks):
        o, m, used = frames[k]
        ax = fig.add_subplot(rows, 6, 2 * c + 1)
        x0, y0, x1, y1 = o["box"]
        if synthetic:
            ax.imshow(np.full(seq.shape + (3,), 235, np.uint8))
            render_in_camera(ax, res["synthetic"]["truth_boxes"], seq.cams[o["k"]], o["pos"], o["yaw"], "#c9d9b0")
            ax.set_title(f"true shape, {side_name(az[k])} view", fontsize=9.5, loc="left", color=INK)
        else:
            ax.imshow(cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2RGB))
            ax.set_title(f"original, frame {o['k']} ({side_name(az[k])}, {'used' if used else 'held out'})",
                         fontsize=9.5, loc="left", color=INK)
        crop(ax, (x0, y0, x1, y1), pad=15)
        ax = fig.add_subplot(rows, 6, 2 * c + 2)
        ax.imshow(np.full(seq.shape + (3,), 235, np.uint8))
        render_in_camera(ax, boxes, seq.cams[o["k"]], o["pos"], o["yaw"])
        for cnt in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            ax.plot(*np.vstack([cnt[:, 0], cnt[:1, 0]]).T, color=ALT_INK, linewidth=1.0, linestyle=(0, (3, 2)))
        crop(ax, (x0, y0, x1, y1), pad=15)
        ax.set_title("reconstruction, same camera", fontsize=9.5, loc="left", color=INK)
    canon_rows = [("reconstruction", boxes, "#b9c7d8")] + ([("true shape", rec["truth_boxes"], "#c9d9b0")]
                                                         if synthetic else [])
    for r, (label, bx, base) in enumerate(canon_rows):
        for c, (name, elev, azim) in enumerate(CANONICAL):
            ax = fig.add_subplot(rows, 6, 6 * (r + 1) + c + 1, projection="3d")
            render_canonical(ax, bx, elev, azim, base)
            ax.set_title(f"{label}: {name}", fontsize=9.5, loc="left", color=INK)
    title = ("Synthetic car" if synthetic else f"V2X-Seq car {key}") + \
        f": {len(boxes)} recursive cuboids (depth {depth}) from {rec['n_train']} frames" + \
        ("" if synthetic else f"; held-out silhouette IoU {rec['score', depth]:.2f}") + \
        ("; dashed = observed silhouette" if not synthetic else "; dashed = rendered true silhouette")
    fig.suptitle(title, fontsize=12, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=95, facecolor="white")
    plt.close(fig)


ALT_INK = "#d6457a"


def main():
    os.makedirs(vd.OUT_DIR, exist_ok=True)
    seq = Sequence()
    res = patch_experiments(seq)
    for key in ["synthetic"] + list(TRACKS):
        rec = res[key]
        line = []
        for d in PATCH_DEPTHS:
            boxes = rec[d]
            mid = sum(b.get("cut") == "mid" for b in boxes)
            line.append(f"d{d}: {len(boxes)} boxes ({sum(b.get('flat', False) for b in boxes)} flat, {mid} mid-cut) "
                        f"{rec['score', d]:.3f}")
        print(f"{key}: hull {rec['hull_score']:.3f} | " + " | ".join(line))
    patches_figure(os.path.join(vd.OUT_DIR, "shape_patches_3d.png"), seq, res)
    patch_scores_figure(os.path.join(vd.OUT_DIR, "shape_patches_scores.png"), res)
    for key in ["synthetic"] + list(TRACKS):
        views_figure(os.path.join(vd.OUT_DIR, f"shape_views_{key}.png"), seq, res, key)


if __name__ == "__main__":
    main()
