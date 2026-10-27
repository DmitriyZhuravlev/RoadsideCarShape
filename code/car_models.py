"""
The best classical 3D model of every car seen by the fixed intersection camera (V2X-Seq, intersection A).

  0. silhouette hull from the motion masks, shaved by MASK_SHAVE px (background subtraction leaves a halo of
     blurred pixels round each car; at grazing roof rays every pixel of halo is centimetres of false height);
  1. cuboid        - the tight box of the hull, on the ground, symmetric: the first bound;
  2. triangulation - the convex polyhedron of the points where the car touches the cuboid;
  3. recursion     - every triangle split into four, LEVELS times; each vertex pulled onto the hull, and on surfaces
                     facing up (where an elevated camera never sees past the roof) further in, to the evidence of the
                     car's own motion (mesh_refine.refine):
        a) points triangulated from optical flow between consecutive frames (car_fusion);
        b) wide-baseline photo-consistency: a 5x5 tangent-plane patch at every upward vertex is moved inwards along
           the normal and projected into all training frames (seconds apart, metres of baseline); best mean NCC;
  4. every vertex coloured with the median colour of the frames that see it (unseen ones from their mirror image).
For comparison only, a silhouette-guarded variant: the pulled-in top may only remove volume that stays hidden behind
the rest of the car (an uncovered training-view pixel deeper than GUARD_TOL px inside the mask brings back the farthest
carved voxel on it). It restores nearly the whole hull: the masks and this calibration demand ~25 cm more height
than the motion parallax and the LiDAR, so the guard cannot referee the roof.
Poses: labelled positions and headings (the initialisation used throughout). LiDAR only for the evaluation.

Writes docs/models_cars_*.png, docs/models_scores.png, docs/models_mean_car.png, docs/models_metrics.json and
docs/models_meshes.json (for the interactive viewer).
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

import car_flow_planes as cfp
import car_fusion as cfu
import mesh_refine as mr
import mix3d as mx
import v2x_seq_demo as vd
import v2x_seq_shape as vs

LEVELS = 3
PHOTO_S = np.arange(-0.2, 0.9, 0.025)     # m: depths searched along the inward normal
PEAK, MARGIN = 0.5, 0.15                  # photo-consistency: peak NCC and its margin over the median
MASK_SHAVE = 2                            # px eroded from every motion mask before carving
GUARD_TOL = 3                             # px: uncovered pixels closer than this to the mask edge are ignored
MIN_TOP = 50                              # evidence points above 0.8 m for a car to count as "own evidence"
INK, MUTED = "#1f1f1e", "#8a8984"


def tangent(n):
    a = np.where(np.abs(n[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return t1, np.cross(n, t1)


def photo_depth(seq, gray, V, N, views, s=PHOTO_S, half=2, spacing=0.035):
    """Mean pairwise NCC of 5x5 tangent-plane patches over all views, for every depth s along -N.
    Returns score (n, S) (nan with < 4 views) and the number of views."""
    t1, t2 = tangent(N)
    g = np.arange(-half, half + 1) * spacing
    a, b = (x.ravel() for x in np.meshgrid(g, g, indexing="ij"))
    P = (V[:, None, None] - s[None, :, None, None] * N[:, None, None]
         + a[None, None, :, None] * t1[:, None, None] + b[None, None, :, None] * t2[:, None, None])
    n_, S_, m_ = P.shape[:3]
    acc, sq, cnt = np.zeros((n_, S_, m_)), np.zeros((n_, S_)), np.zeros((n_, S_))
    for o in views:
        cam = seq.cams[o["k"]]
        C = mx.camera_in_object(cam, o)
        vis = np.einsum("ij,ij->i", N, C - V) / np.linalg.norm(C - V, axis=1) > 0.25
        mask = cv2.dilate(seq.mask(o).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        uv = vs.project(cam, vs.world(P.reshape(-1, 3), o["pos"], o["yaw"]))[0].reshape(n_, S_, m_, 2)
        val = map_coordinates(gray[o["k"]], [uv[..., 1].ravel(), uv[..., 0].ravel()], order=1,
                              mode="nearest").reshape(n_, S_, m_)
        cu = np.clip(uv[:, :, m_ // 2].round().astype(int), 0, [seq.shape[1] - 1, seq.shape[0] - 1])
        sd = val.std(2)
        ok = vis[:, None] & mask[cu[..., 1], cu[..., 0]] & (sd > 2.0)
        acc += (val - val.mean(2, keepdims=True)) / (sd[..., None] + 1e-6) * ok[..., None]
        sq += ok * m_
        cnt += ok
    with np.errstate(invalid="ignore"):
        score = ((acc ** 2).sum(2) - sq) / (m_ * np.maximum(cnt * (cnt - 1), 1))
    score[cnt < 4] = np.nan
    return score, cnt


def photo_points(seq, gray, V, F, views):
    N = mr.vertex_normals(V, F)
    up = N[:, 2] > 0.3
    score, _ = photo_depth(seq, gray, V[up], N[up], views)
    filled = np.nan_to_num(score, nan=-9)
    best = filled.argmax(1)
    peak = filled[np.arange(len(best)), best]
    with np.errstate(all="ignore"):
        conf = (peak > PEAK) & (peak - np.nanmedian(score, 1) > MARGIN) & (best > 0) & (best < len(PHOTO_S) - 1)
    return (V[up] - PHOTO_S[best][:, None] * N[up])[conf]


def top_heights(occ, axes):
    v = axes[2][1] - axes[2][0]
    k = occ.shape[2]
    return np.where(occ.any(2), (k - np.argmax(occ[:, :, ::-1], axis=2)) * v, 0.0)


def mesh_to_heightfield(S, axes):
    """Voxels below the highest surface sample of their column (cars are height fields seen from above)."""
    v = axes[0][1] - axes[0][0]
    i = np.round((S[:, 0] - axes[0][0]) / v).astype(int)
    j = np.round((S[:, 1] - axes[1][0]) / v).astype(int)
    ok = (i >= 0) & (i < len(axes[0])) & (j >= 0) & (j < len(axes[1]))
    top = np.full((len(axes[0]), len(axes[1])), -np.inf)
    np.maximum.at(top, (i[ok], j[ok]), S[ok, 2])
    return np.asarray(axes[2])[None, None, :] < top[:, :, None]


def shaved(seq, o, px):
    m = seq.mask(o)
    return cv2.erode(m.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) if px else m


def coverage_map(uv, shape):
    """Pixels covered by projected voxel centres, closed like v2x_seq_shape.silhouette."""
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    uv = np.round(uv).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    m[uv[ok, 1], uv[ok, 0]] = 1
    m = cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return m > 0


def silhouette_guard(occ, hull, axes, seq, views, iterations=12):
    """Carving may only remove volume that stays hidden behind the rest of the car. For every training-view pixel
    that the hull covered inside the motion mask and the carved model no longer covers, the carved voxel on that
    pixel farthest from the camera comes back: the car touches a silhouette ray at its far end, while the hull
    volume nearer to the camera on that ray is the part no view can see; repeat."""
    ii = np.argwhere(hull)
    P = np.c_[axes[0][ii[:, 0]], axes[1][ii[:, 1]], axes[2][ii[:, 2]]]
    prep = []
    for o in views:
        uv, depth = vs.project(seq.cams[o["k"]], vs.world(P, o["pos"], o["yaw"]))
        px = np.clip(np.round(uv).astype(int), 0, [seq.shape[1] - 1, seq.shape[0] - 1])
        inner = shaved(seq, o, GUARD_TOL) > 0
        prep.append((uv, depth, px[:, 1] * seq.shape[1] + px[:, 0], coverage_map(uv, seq.shape) & inner))
    for _ in range(iterations):
        keep = occ[tuple(ii.T)]
        carved = np.where(~keep)[0]
        if not len(carved):
            break
        back = np.zeros(len(ii), bool)
        for uv, depth, pid, full in prep:
            lost = (full & ~coverage_map(uv[keep], seq.shape)).ravel()
            hit = lost[pid[carved]]
            if not hit.any():
                continue
            c, pixel = carved[hit], pid[carved][hit]
            order = np.lexsort((-depth[c], pixel))           # per pixel, the farthest carved voxel first
            first = np.r_[True, pixel[order][1:] != pixel[order][:-1]]
            back[c[order][first]] = True
        if not back.any():
            break
        occ = occ.copy()
        occ[tuple(ii[back].T)] = True
        occ = vs.height_field(occ)
    return occ


def vertex_colours(seq, V, N, views):
    """Median colour of the frames whose camera faces the vertex and whose motion mask contains it."""
    cols = []
    for o in views:
        cam = seq.cams[o["k"]]
        C = mx.camera_in_object(cam, o)
        vis = np.einsum("ij,ij->i", N, C - V) / np.linalg.norm(C - V, axis=1) > 0.2
        uv = vs.project(cam, vs.world(V, o["pos"], o["yaw"]))[0]
        u, w = np.clip(uv.round().astype(int), 0, [seq.shape[1] - 1, seq.shape[0] - 1]).T
        inm = seq.mask(o)[w, u] > 0
        c = seq.frames[o["k"]][w, u][:, ::-1].astype(float)
        c[~(vis & inm)] = np.nan
        cols.append(c)
    with np.errstate(all="ignore"):
        cols = np.nanmedian(np.stack(cols), axis=0) / 255
    return mirror_fill(V, cols)


def mirror_fill(V, C):
    """Unseen vertices take the colour of their mirror image (cars are symmetric), else the car's median colour."""
    C = C.copy()
    known = ~np.isnan(C).any(1)
    if not known.any():
        return np.full((len(V), 3), 0.5)
    unknown = np.where(~known)[0]
    d, j = cKDTree(V[known]).query(V[unknown] * [1, -1, 1])
    C[unknown[d < 0.15]] = C[known][j[d < 0.15]]
    C[np.isnan(C).any(1)] = np.median(C[known], axis=0)
    return C


def silhouette_iou(V, F, seq, test):
    return float(np.mean([vs.iou(mr.mesh_silhouette(V, F, seq.cams[o["k"]], o["pos"], o["yaw"], seq.shape),
                                 seq.mask(o) > 0) for o in test]))


def evidence(seq, gray, tid):
    """The hull and everything the car's motion says about its surface."""
    obs = [o for o in seq.tracks[tid] if seq.usable(o)]
    by_k = {o["k"]: o for o in obs}
    l, w = (float(np.median([o["dims"][i] for o in obs])) for i in (0, 1))
    axes = vs.grid(l, w, 1.6)
    train = [o for i, o in enumerate(obs) if i % 4]
    test = [o for i, o in enumerate(obs) if i % 4 == 0]
    hull = vs.height_field(vs.carve_votes(axes, [seq.view(o, shaved(seq, o, MASK_SHAVE)) for o in train]) >= vs.VOTE)
    V0, F0, box, _ = mr.coarse_mesh(hull, axes)
    V2, F2 = mr.refine(V0, F0, hull, axes, np.zeros((0, 3)), box, levels=2)[-1]
    photo = photo_points(seq, gray, V2, F2, train)
    flow = []
    for o in train:
        if o["k"] + 1 in by_k:
            f = cfu.frame_model(seq, o, by_k[o["k"] + 1])
            if len(f["V"]):
                flow.append(mx.to_ground_frame(f["V"], o).mean(1))
    flow = np.vstack(flow) if flow else np.zeros((0, 3))
    E = np.vstack([photo, flow])
    return {"obs": obs, "train": train, "test": test, "axes": axes, "hull": hull, "dims": (l, w), "photo": photo,
            "flow": flow, "E": E, "n_top": int((E[:, 2] > 0.8).sum())}


def build(seq, ev, ground_z):
    """Cuboid -> contact triangulation -> recursive subdivision pulled by the motion evidence -> silhouette guard
    -> the same chain on the guarded volume -> colours; and the scores of every stage."""
    axes, hull = ev["axes"], ev["hull"]
    V0, F0, box, contacts = mr.coarse_mesh(hull, axes)
    steps = mr.refine(V0, F0, hull, axes, ev["E"], box, levels=LEVELS)
    V, F = steps[-1]
    # comparison only: the same mesh after the silhouette guard (it trusts the masks over the motion evidence)
    occ = mesh_to_heightfield(mr.sample_surface(V, F, per=20), axes) & hull
    occ = silhouette_guard(occ, hull, axes, seq, ev["train"])
    occ = occ | occ[:, ::-1]
    G0, GF0, gbox, _ = mr.coarse_mesh(occ, axes)
    guarded = mr.refine(G0, GF0, occ, axes, np.zeros((0, 3)), gbox, levels=LEVELS)[-1]
    L = np.vstack([mx.to_ground_frame(cfp.lidar_on_car(seq, o, ground_z), o) for o in ev["obs"]])
    colours = vertex_colours(seq, V, mr.vertex_normals(V, F), ev["train"])
    roof_l = float(np.percentile(L[:, 2], 99))
    cell = 0.1
    key = np.floor(L[:, :2] / cell).astype(int)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    ltop = np.full(inv.max() + 1, -np.inf)
    np.maximum.at(ltop, inv, L[:, 2])
    cnt = np.bincount(inv)
    cells = np.array([L[inv == i, :2].mean(0) for i in range(len(cnt))])
    seen = (cnt >= 3) & (ltop > 0.8)                   # cells where the LiDAR hit a top surface (hood, roof, trunk)

    def top_err(S):
        """Model top minus LiDAR top over the cells where LiDAR saw a top surface (median, cm)."""
        t = cKDTree(S[:, :2])
        idx = t.query_ball_point(cells[seen], cell)
        mt = np.array([S[i, 2].max() if len(i) else np.nan for i in idx])
        return 100 * float(np.nanmedian(mt - ltop[seen])) if seen.any() and np.isfinite(mt).any() else 0.0

    def height(S):                                      # 99th percentiles: the roadside LiDAR rarely sees the far roof
        return {"roof99_err_cm": 100 * (float(np.percentile(S[:, 2], 99)) - roof_l), "top_err_cm": top_err(S)}
    rows = {"silhouette hull": {**mr.evaluate(vs.surface_points(hull, axes), L), **height(vs.surface_points(hull, axes)),
                                "iou": float(np.mean([vs.iou(vs.silhouette(hull, axes, seq.cams[o["k"]], o["pos"],
                                                                           o["yaw"], seq.shape) > 0, seq.mask(o) > 0)
                                                      for o in ev["test"]]))}}
    rows["silhouette-guarded"] = {**mr.evaluate(mr.sample_surface(*guarded), L), **height(mr.sample_surface(*guarded)),
                                  "iou": silhouette_iou(*guarded, seq, ev["test"])}
    for lev, (Vl, Fl) in enumerate(steps):
        rows[f"level {lev}"] = {**mr.evaluate(mr.sample_surface(Vl, Fl), L), **height(mr.sample_surface(Vl, Fl)), "iou": silhouette_iou(Vl, Fl, seq, ev["test"]),
                                "triangles": int(len(Fl))}
    return {"V": V, "F": F, "colours": colours, "steps": steps, "box": box, "contacts": contacts, "L": L, "rows": rows,
            "occ": occ}


# --- figures --------------------------------------------------------------------------------------------
def shaded(V, F, colours):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    light = np.array([0.35, -0.45, 0.82])
    shade = 0.45 + 0.55 * np.clip(np.abs(fn @ light), 0, 1)
    return np.clip(colours[F].mean(1) * shade[:, None], 0, 1), fn


def draw_model(ax, V, F, colours, dims, elev, azim, L=None):
    fc, _ = shaded(V, F, colours)
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc, edgecolors="none"))
    if L is not None:
        Ls = L[np.random.default_rng(0).permutation(len(L))[:1500]]
        ax.scatter(*Ls.T, s=0.3, c="#d03b3b", alpha=0.35, depthshade=False)
    lim = max(dims) / 2 + 0.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, lim)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev, azim)
    ax.axis("off")


def render_into_photo(img, V, F, colours, cam, o, alpha=0.85):
    """Painter's-algorithm rendering of the textured, shaded mesh into the camera image."""
    Xw = vs.world(V, o["pos"], o["yaw"])
    uv, depth = vs.project(cam, Xw)
    fc, _ = shaded(V, F, colours)
    order = np.argsort(-depth[F].mean(1))
    out = img.copy()
    layer = img.copy()
    for i in order:
        cv2.fillConvexPoly(layer, uv[F[i]].round().astype(np.int32), (fc[i][::-1] * 255).tolist(), lineType=cv2.LINE_AA)
    m = np.zeros(img.shape[:2], np.uint8)
    for t in uv[F].round().astype(np.int32):
        cv2.fillConvexPoly(m, t, 1)
    out[m > 0] = (alpha * layer[m > 0] + (1 - alpha) * img[m > 0]).astype(np.uint8)
    return out


def cars_figure(path, seq, models, tids):
    cols = ("photo", "model in the photo (held-out frame)", "3D model, 3/4 view", "side view, red = LiDAR", "front view")
    fig = plt.figure(figsize=(3.4 * len(cols), 2.7 * len(tids)))
    for r, tid in enumerate(tids):
        m, ev = models[tid]
        o = ev["test"][len(ev["test"]) // 2]
        x0, y0, x1, y1 = np.array(o["box"]).round().astype(int)
        pad = 12
        crop = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
        img = seq.frames[o["k"]]
        ax = fig.add_subplot(len(tids), len(cols), r * len(cols) + 1)
        ax.imshow(cv2.cvtColor(img[crop], cv2.COLOR_BGR2RGB))
        ax.axis("off")
        fr = m["rows"][f"level {LEVELS}"]
        ax.set_title(f"car {tid}: {len(ev['obs'])} frames, {ev['n_top']} motion points on the top", fontsize=8, loc="left",
                     color=INK)
        ax = fig.add_subplot(len(tids), len(cols), r * len(cols) + 2)
        ax.imshow(cv2.cvtColor(render_into_photo(img, m["V"], m["F"], m["colours"], seq.cams[o["k"]], o)[crop],
                               cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"IoU {fr['iou']:.2f}, roof {fr['roof_err_cm']:+.0f} cm, compl. {fr['completeness_cm']:.0f} cm",
                     fontsize=8, loc="left", color=INK)
        C = mx.camera_in_object(seq.cams[o["k"]], o)
        seen_az = np.degrees(np.arctan2(C[1], C[0]))
        side = -90 if C[1] < 0 else 90
        for c, (elev, azim, lid) in enumerate(((22, seen_az + 25, False), (2, side, True), (2, 0 if C[0] > 0 else 180, True))):
            ax = fig.add_subplot(len(tids), len(cols), r * len(cols) + 3 + c, projection="3d")
            draw_model(ax, m["V"], m["F"], m["colours"], ev["dims"], elev, azim, m["L"] if lid else None)
            if r == 0:
                ax.set_title(cols[2 + c], fontsize=8.5, loc="left", color=INK)
    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def scores_figure(path, models):
    groups = [("silhouette hull", lambda t: models[t][0]["rows"]["silhouette hull"], "#8a8984"),
              ("silhouette-guarded", lambda t: models[t][0]["rows"]["silhouette-guarded"], "#c3c2b7"),
              (f"new: level {LEVELS}", lambda t: models[t][0]["rows"][f"level {LEVELS}"], "#2a78d6")]
    keys = (("completeness_cm", "LiDAR -> model, median (cm)"),
            ("top_err_cm", "top height error where LiDAR saw the top (cm)"), ("iou", "held-out silhouette IoU"))
    tids = list(models)
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
    x = np.arange(len(tids))
    for ax, (key, title) in zip(axes, keys):
        for g, (name, get, col) in enumerate(groups):
            vals = np.array([get(t).get(key, np.nan) if get(t) else np.nan for t in tids])
            ax.bar(x + (g - 1) * 0.27, vals, 0.25, color=col, zorder=2,
                   label=f"{name} (mean {np.nanmean(vals):.2f})" if key == "iou" else f"{name} (mean {np.nanmean(vals):.0f})")
        ax.set_xticks(x, tids, rotation=60, fontsize=7.5)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right" if key == "iou" else "upper right")
        if key == "iou":
            ax.set_ylim(0.5, 0.95)
        vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def mean_car_figure(path, models):
    from skimage.measure import marching_cubes
    occs = [mr.normalised_occupancy(m["V"], m["F"], m["box"]) for m, _ in models.values()]
    occ = np.mean(occs, axis=0)
    dims = np.mean([m["box"][1] - m["box"][0] for m, _ in models.values()], axis=0)
    n = occ.shape
    v, f, _, _ = marching_cubes(np.pad(occ, 1), 0.5)
    v = ((v - 1) / (np.array(n) - 1) - [0.5, 0.5, 0.0]) * dims
    grey = np.full((len(v), 3), 0.62)
    fig = plt.figure(figsize=(18, 4.4))
    for c, (name, elev, azim) in enumerate((("side", 2, -90), ("3/4 front", 22, 35), ("top", 85, -90), ("3/4 rear", 22, 215))):
        ax = fig.add_subplot(1, 4, c + 1, projection="3d")
        draw_model(ax, v, f, grey, dims[:2], elev, azim)
        ax.set_title(f"mean of {len(models)} cars: {name}", fontsize=9.5, loc="left", color=INK)
    fig.suptitle(f"Average car (size-normalised occupancy, 0.5 level): {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} m",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def export_meshes(path, models):
    out = {}
    for tid, (m, ev) in models.items():
        V, F, cols = m["V"], m["F"], m["colours"]
        out[tid] = {"v": np.round(V, 3).ravel().tolist(), "f": F.ravel().tolist(),
                    "c": np.round(cols, 2).ravel().tolist(),
                    "lidar": np.round(m["L"][np.random.default_rng(0).permutation(len(m["L"]))[:2000]], 3).ravel().tolist(),
                    "frames": len(ev["obs"]), "source": "own" if ev["n_top"] >= MIN_TOP else "prior",
                    "evidence": ev["n_top"], "size": (m["box"][1] - m["box"][0]).round(2).tolist(),
                    "scores": m["rows"][f"level {LEVELS}"], "hull": m["rows"]["silhouette hull"]}
    json.dump(out, open(path, "w"))


def main():
    seq = vs.Sequence()
    seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    ground = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))[2]
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in seq.frames]
    models = {}
    for tid in mr.suitable_cars(seq):
        ev = evidence(seq, gray, tid)
        m = build(seq, ev, ground)
        models[tid] = (m, ev)
        r = m["rows"]
        print(f"car {tid}: {len(ev['photo'])} photo + {len(ev['flow'])} flow points ({ev['n_top']} on the top) | " +
              " | ".join(f"{k}: compl {r[k]['completeness_cm']:.1f} top {r[k]['top_err_cm']:+.0f} IoU {r[k]['iou']:.2f}"
                         for k in ("silhouette hull", "silhouette-guarded", f"level {LEVELS}")), flush=True)
    names = ["silhouette hull", "silhouette-guarded"] + [f"level {k}" for k in range(LEVELS + 1)]
    summary = {name: {k: {"mean": float(np.mean([m["rows"][name][k] for m, _ in models.values()])),
                          "sd": float(np.std([m["rows"][name][k] for m, _ in models.values()]))}
                      for k in ("precision_cm", "completeness_cm", "roof99_err_cm", "top_err_cm", "iou")}
               for name in names}
    for name, s_ in summary.items():
        print(f"{name:32s} " + " | ".join(f"{k} {v['mean']:.2f} ± {v['sd']:.2f}" for k, v in s_.items()))
    json.dump({"summary": summary, "per_car": {t: m["rows"] for t, (m, _) in models.items()}},
              open(os.path.join(vd.OUT_DIR, "models_metrics.json"), "w"), indent=1)
    tids = list(models)
    for i in range(0, len(tids), 8):
        cars_figure(os.path.join(vd.OUT_DIR, f"models_cars_{i // 8 + 1}.png"), seq, models, tids[i:i + 8])
    scores_figure(os.path.join(vd.OUT_DIR, "models_scores.png"), models)
    mean_car_figure(os.path.join(vd.OUT_DIR, "models_mean_car.png"), models)
    export_meshes(os.path.join(vd.OUT_DIR, "models_meshes.json"), models)
    os.makedirs(os.path.join(vd.OUT_DIR, "viewer", "photos"), exist_ok=True)
    for tid, (m, ev) in models.items():                 # photo crops for the viewer
        o = ev["test"][len(ev["test"]) // 2]
        x0, y0, x1, y1 = np.array(o["box"]).round().astype(int)
        crop = seq.frames[o["k"]][max(0, y0 - 12):y1 + 12, max(0, x0 - 12):x1 + 12]
        cv2.imwrite(os.path.join(vd.OUT_DIR, "viewer", "photos", f"{tid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(summary, open(os.path.join(vd.OUT_DIR, "viewer", "summary.json"), "w"))


if __name__ == "__main__":
    main()
