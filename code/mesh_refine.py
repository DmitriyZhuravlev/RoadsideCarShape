"""
Coarse-to-fine car meshes from a fixed roadside camera (V2X-Seq, intersection A), for every suitable car.

  1. cuboid        : the tight box of the silhouette hull (v2x_seq_shape), on the ground, symmetric;
  2. contact points: hull surface voxels on the faces of that cuboid - where the car touches its box;
  3. coarse mesh   : the convex polyhedron of the contact points (it touches every face where the car does);
  4. recursion     : every triangle is split into four; every vertex moves inwards along its normal to the evidence
                     of the car - the silhouette hull's boundary, or deeper if the points triangulated from optical
                     flow over all frames (car_fusion: a plane per triangle from its deformation between consecutive
                     frames) lie further in (concavities silhouettes cannot see); a light Laplacian smoothing;
  5. check against the intersection's LiDAR points on each car (precision, completeness, roof height) and held-out
     silhouette IoU at every level; then averages over all cars and the mean car shape (every final mesh in
     size-normalised coordinates, occupancy averaged, 0.5 iso-surface).
Poses: the labelled position and heading (the initialisation used throughout).

Writes docs/refine_steps.png, docs/refine_cars.png, docs/refine_scores.png, docs/refine_mean_car.png and
docs/refine_metrics.json.
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.sparse as sp
from scipy.spatial import ConvexHull, cKDTree
from skimage.measure import marching_cubes

import car_flow_planes as cfp
import car_fusion as cfu
import mix3d as mx
import v2x_seq_demo as vd
import v2x_seq_shape as vs

LEVELS = 4
STEP_S = 0.025            # m: search step along the normal
RADIUS = 0.15             # m: triangulated points this close to the normal line are evidence
MIN_VOTES = 5
UP = 0.5                  # flow evidence carves only surfaces facing up (roof, hood, trunk)
DEEPER = 0.05
FLOW_DEPTH = 1.0          # m: how far inside the hull boundary flow points are looked for             # m: the flow points must lie at least this much inside the hull boundary
INK, MUTED = "#1f1f1e", "#8a8984"


def subdivide(V, F):
    """Midpoint (1-to-4) subdivision; shared edges share their midpoint; orientation kept."""
    V = [v for v in V]
    mid = {}

    def m(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            mid[key] = len(V)
            V.append((V[a] + V[b]) / 2)
        return mid[key]
    out = []
    for a, b, c in F:
        ab, bc, ca = m(a, b), m(b, c), m(c, a)
        out += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
    return np.array(V), np.array(out)


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def adjacency(F, n):
    """Row-normalised vertex adjacency (each edge once)."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.unique(np.vstack([E, E[:, ::-1]]), axis=0)
    A = sp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    return sp.diags(1 / np.maximum(deg, 1)) @ A


def laplacian(V, A, lam=0.5, mu=-0.53, iterations=2):
    """Taubin smoothing: a shrinking and an inflating step, so the mesh is smoothed without shrinking."""
    for _ in range(iterations):
        V = V + lam * (A @ V - V)
        V = V + mu * (A @ V - V)
    return V


def occupied(occ, axes, P):
    v = axes[0][1] - axes[0][0]
    lo = np.array([axes[0][0], axes[1][0], axes[2][0]])
    idx = np.round((P - lo) / v).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(occ.shape)), axis=-1)
    out = np.zeros(P.shape[:-1], bool)
    out[ok] = occ[tuple(idx[ok].T)]
    return out


def coarse_mesh(hull, axes):
    S = vs.surface_points(hull, axes)
    v = axes[0][1] - axes[0][0]
    lo = np.array([S[:, 0].min(), -np.abs(S[:, 1]).max(), 0.0])
    hi = np.array([S[:, 0].max(), np.abs(S[:, 1]).max(), S[:, 2].max()])
    touch = np.zeros(len(S), bool)
    for a in range(3):
        touch |= np.abs(S[:, a] - lo[a]) <= 1.5 * v
        touch |= np.abs(S[:, a] - hi[a]) <= 1.5 * v
    C = S[touch]
    C = np.vstack([C, C * [1, -1, 1]])                    # the car is mirror-symmetric
    ch = ConvexHull(C)
    keep = np.unique(ch.simplices)
    remap = -np.ones(len(C), int)
    remap[keep] = np.arange(len(keep))
    V, F = C[keep], remap[ch.simplices]
    centre = V.mean(0)
    for i, (a, b, c) in enumerate(F):                   # outward orientation
        n = np.cross(V[b] - V[a], V[c] - V[a])
        if n @ (V[a] - centre) < 0:
            F[i] = (a, c, b)
    return V, F, (lo, hi), C


def refine(V, F, hull, axes, pts, box, levels=LEVELS):
    lo, hi = box
    tree = cKDTree(pts) if len(pts) else None
    steps = [(V.copy(), F.copy())]
    for lev in range(levels):
        V, F = subdivide(V, F)
        N = vertex_normals(V, F)
        s = np.arange(-0.8, 1.5, STEP_S)
        line = V[:, None, :] - s[None, :, None] * N[:, None, :]            # s > 0: inwards
        inside = occupied(hull, axes, line)
        # the hull boundary nearest to the vertex along its normal, on either side: an empty -> occupied step;
        # a vertex already carved inside the hull is not pushed back out by it
        enter = inside[:, 1:] & ~inside[:, :-1]
        cand = np.where(enter, np.abs(s[1:])[None, :], np.inf)
        j = cand.argmin(1)
        s_h = np.where(np.isfinite(cand[np.arange(len(V)), j]), s[1:][j], 0.0)
        target = s_h.copy()
        if tree is not None:
            for i, (p, n) in enumerate(zip(V, N)):
                if n[2] < UP:          # sides are bounded by the silhouettes; only the top is left unseen
                    continue
                w0, w1 = min(s_h[i], 0.0) - 0.3, s_h[i] + FLOW_DEPTH
                idx = tree.query_ball_point(p - n * (w0 + w1) / 2, (w1 - w0) / 2 + RADIUS)
                if len(idx) < MIN_VOTES:
                    continue
                Q = pts[idx] - p
                along = -(Q @ n)
                perp = np.linalg.norm(Q + along[:, None] * n, axis=1)
                # all points near the normal line vote (not only the deeper ones, which would bias every vertex
                # inwards); the vertex follows their median only if it lies clearly inside the hull boundary
                sel = (perp < RADIUS) & (along > w0) & (along < w1)
                if sel.sum() >= MIN_VOTES:
                    med = float(np.median(along[sel]))
                    if med > s_h[i] + DEEPER:
                        target[i] = med
        # the displacement is a field on the surface: capped by the triangle size and smoothed over neighbours,
        # so that neighbouring vertices move together and triangles do not fold
        A = adjacency(F, len(V))
        edge = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1).mean()
        target = np.clip(target, -max(1.5 * edge, 0.1), max(1.5 * edge, 0.1))
        for _ in range(4):
            target = 0.5 * target + 0.5 * (A @ target)
        V = V - target[:, None] * N
        V = laplacian(V, A)
        V = np.clip(V, lo, hi)
        V[:, 2] = np.maximum(V[:, 2], 0.0)
        steps.append((V.copy(), F.copy()))
    return steps


def sample_surface(V, F, per=6, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.random((2, len(F), per))
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    P = A[:, None] + a[..., None] * (B - A)[:, None] + b[..., None] * (C - A)[:, None]
    return np.vstack([P.reshape(-1, 3), V])


def mesh_silhouette(V, F, cam, center, yaw, shape):
    uv = vs.project(cam, vs.world(V, center, yaw))[0]
    m = np.zeros(shape, np.uint8)
    for t in uv[F].round().astype(np.int32):          # one by one: fillPoly would XOR overlapping triangles
        cv2.fillConvexPoly(m, t, 1)
    return m > 0


def evaluate(S, L):
    p, c = mx.evaluate(S, L)
    return {"precision_cm": 100 * p, "completeness_cm": 100 * c, "roof_err_cm": 100 * (mx.roof(S) - mx.roof(L))}


def suitable_cars(seq, min_frames=15):
    out = []
    for tid, q in seq.tracks.items():
        obs = [o for o in q if seq.usable(o)]
        if len(obs) < min_frames:
            continue
        P = np.array([o["pos"] for o in obs])
        if np.linalg.norm(P[-1] - P[0]) < 5 or np.median([o["dims"][0] for o in obs]) < 3.2:
            continue
        if min(np.linalg.norm(P - seq.camera_xy, axis=1)) > 70:
            continue
        out.append(tid)
    return out


def run_car(seq, tid, ground_z):
    obs = [o for o in seq.tracks[tid] if seq.usable(o)]
    by_k = {o["k"]: o for o in obs}
    l, w = (float(np.median([o["dims"][i] for o in obs])) for i in (0, 1))
    axes = vs.grid(l, w, 1.6)
    train = [o for i, o in enumerate(obs) if i % 4]
    test = [o for i, o in enumerate(obs) if i % 4 == 0]
    hull = vs.height_field(vs.carve_votes(axes, [seq.view(o) for o in train]) >= vs.VOTE)
    if hull.sum() < 100:
        return None
    pts = []
    for o in train:
        if o["k"] + 1 in by_k:
            f = cfu.frame_model(seq, o, by_k[o["k"] + 1])
            if len(f["V"]):
                pts.append(mx.to_ground_frame(f["V"], o).mean(1))
    pts = np.vstack(pts) if pts else np.zeros((0, 3))
    L = np.vstack([mx.to_ground_frame(cfp.lidar_on_car(seq, o, ground_z), o) for o in obs])
    if len(L) < 200:
        return None
    V0, F0, box, contacts = coarse_mesh(hull, axes)
    steps = refine(V0, F0, hull, axes, pts, box)
    masks = {o["k"]: seq.mask(o) > 0 for o in test}
    lo, hi = box
    cub = {"c": (lo + hi) / 2, "R": np.eye(3), "h": (hi - lo) / 2}
    rows = {"cuboid": {**evaluate(vs.box_surface_samples(cub["c"], cub["R"], cub["h"], 0.04), L),
                       "iou": float(np.mean([vs.iou(vs.patch_silhouette([cub], seq.cams[o["k"]], o["pos"], o["yaw"],
                                                                         seq.shape) > 0, masks[o["k"]]) for o in test]))},
            "silhouette hull": {**evaluate(vs.surface_points(hull, axes), L),
                                "iou": float(np.mean([vs.iou(vs.silhouette(hull, axes, seq.cams[o["k"]], o["pos"], o["yaw"],
                                                                           seq.shape) > 0, masks[o["k"]]) for o in test]))}}
    for lev, (V, F) in enumerate(steps):
        rows[f"level {lev}"] = {**evaluate(sample_surface(V, F), L),
                                "iou": float(np.mean([vs.iou(mesh_silhouette(V, F, seq.cams[o["k"]], o["pos"], o["yaw"],
                                                                             seq.shape), masks[o["k"]]) for o in test])),
                                "triangles": int(len(F))}
    return {"rows": rows, "steps": steps, "box": box, "contacts": contacts, "L": L, "pts": pts, "dims": (l, w),
            "frame": test[len(test) // 2], "frames": len(obs)}


# --- mean car -------------------------------------------------------------------------------------------
def normalised_occupancy(V, F, box, n=(48, 24, 20)):
    lo, hi = box
    g = [np.linspace(-0.5, 0.5, k) for k in n[:2]] + [np.linspace(0, 1, n[2])]
    G = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    size = hi - lo
    centre = np.array([(lo[0] + hi[0]) / 2, 0.0, 0.0])
    P = centre + G * [size[0], size[1], size[2]]
    S = sample_surface(V, F, per=4)
    N_s = np.zeros_like(S)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    N_s[:len(F) * 4] = np.repeat(fn, 4, 0)
    N_s[len(F) * 4:] = vertex_normals(V, F)
    d, idx = cKDTree(S).query(P)
    inside = np.einsum("ij,ij->i", P - S[idx], N_s[idx]) < 0
    return inside.reshape(n).astype(float)


# --- figures --------------------------------------------------------------------------------------------
def draw_mesh(ax, V, F, alpha=0.85):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.5, 0.77])
    shade = 0.35 + 0.65 * np.clip(np.abs(fn @ light), 0, 1)
    base = np.array([0x2a, 0x78, 0xd6]) / 255
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=np.c_[shade[:, None] * base, np.full(len(F), alpha)],
                                         edgecolors="none" if len(F) > 4000 else (0.12, 0.12, 0.12, 0.35),
                                         linewidths=0.15))


def style3d(ax, dims, elev=18, azim=-60):
    l, w = dims
    lim = max(l, w) / 2 + 0.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, lim)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev, azim)
    ax.axis("off")


def steps_figure(path, seq, tid, r):
    cols = ["photo", "cuboid + contact points"] + [f"level {k}" for k in range(LEVELS + 1)]
    fig = plt.figure(figsize=(3.2 * len(cols), 6.6))
    Ls = r["L"][np.random.default_rng(0).permutation(len(r["L"]))[:2500]]
    for row, (elev, azim) in enumerate(((18, -60), (5, -90))):
        for c, name in enumerate(cols):
            if c == 0:
                if row == 0:
                    ax = fig.add_subplot(2, len(cols), 1)
                    o = r["frame"]
                    ax.imshow(cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2RGB))
                    V, F = r["steps"][-1]
                    sil = mesh_silhouette(V, F, seq.cams[o["k"]], o["pos"], o["yaw"], seq.shape).astype(np.uint8)
                    for cnt in cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
                        ax.plot(*np.r_[cnt[:, 0], cnt[:1, 0]].T, color="#2a78d6", linewidth=1.2)
                    m = seq.mask(o).astype(np.uint8)
                    for cnt in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
                        ax.plot(*np.r_[cnt[:, 0], cnt[:1, 0]].T, color="#eb6834", linewidth=0.8, linestyle="--")
                    x0, y0, x1, y1 = o["box"]
                    ax.set_xlim(x0 - 15, x1 + 15)
                    ax.set_ylim(y1 + 15, y0 - 15)
                    ax.axis("off")
                    ax.set_title(f"car {tid}, frame {o['k']} (held out)\nblue: final mesh, orange: motion mask", fontsize=9, loc="left", color=INK)
                continue
            ax = fig.add_subplot(2, len(cols), row * len(cols) + c + 1, projection="3d")
            if c == 1:
                lo, hi = r["box"]
                Cb = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                for a, b in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
                    ax.plot(*Cb[[a, b]].T, color="#eb6834", linewidth=1)
                ax.scatter(*r["contacts"].T, s=1.5, c="#2a78d6")
            else:
                V, F = r["steps"][c - 2]
                draw_mesh(ax, V, F)
                if row == 0:
                    rr = r["rows"][name]
                    ax.set_title(f"{name}: {len(F)} triangles\ncompl. {rr['completeness_cm']:.0f} cm, roof "
                                 f"{rr['roof_err_cm']:+.0f} cm, IoU {rr['iou']:.2f}", fontsize=8, loc="left", color=INK)
            ax.scatter(*Ls.T, s=0.2, c="k", alpha=0.25)
            style3d(ax, r["dims"], elev, azim)
            if c == 1 and row == 0:
                ax.set_title("cuboid (orange) + contact points (blue)", fontsize=8, loc="left", color=INK)
    fig.suptitle("Cuboid -> contact points -> convex triangulation -> recursive subdivision pulled to the silhouette hull and "
                 "the flow-triangulated points; black = LiDAR", fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def cars_figure(path, seq, results):
    n = len(results)
    cols = 6
    rows = int(np.ceil(n / cols)) * 2
    fig = plt.figure(figsize=(3.1 * cols, 3.0 * rows))
    for i, (tid, r) in enumerate(results.items()):
        rr, cc = (i // cols) * 2, i % cols
        ax = fig.add_subplot(rows, cols, rr * cols + cc + 1)
        o = r["frame"]
        ax.imshow(cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2RGB))
        x0, y0, x1, y1 = o["box"]
        ax.set_xlim(x0 - 10, x1 + 10)
        ax.set_ylim(y1 + 10, y0 - 10)
        ax.axis("off")
        ax.set_title(f"car {tid} ({r['frames']} frames)", fontsize=8.5, loc="left", color=INK)
        ax = fig.add_subplot(rows, cols, (rr + 1) * cols + cc + 1, projection="3d")
        V, F = r["steps"][-1]
        draw_mesh(ax, V, F)
        Ls = r["L"][np.random.default_rng(0).permutation(len(r["L"]))[:1500]]
        ax.scatter(*Ls.T, s=0.2, c="k", alpha=0.25)
        style3d(ax, r["dims"])
        fr = r["rows"][f"level {LEVELS}"]
        ax.set_title(f"compl. {fr['completeness_cm']:.0f} cm, roof {fr['roof_err_cm']:+.0f} cm", fontsize=8, loc="left",
                     color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor="white")
    plt.close(fig)


def scores_figure(path, results):
    names = ["cuboid", "silhouette hull"] + [f"level {k}" for k in range(LEVELS + 1)]
    keys = (("precision_cm", "mesh -> LiDAR (cm)"), ("completeness_cm", "LiDAR -> mesh (cm)"),
            ("roof_err_cm", "roof height error (cm)"), ("iou", "held-out silhouette IoU"))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for ax, (key, title) in zip(axes, keys):
        vals = np.array([[r["rows"][n][key] for n in names] for r in results.values()])
        mean, sd = vals.mean(0), vals.std(0)
        colors = ["#c3c2b7", "#8a8984"] + ["#2a78d6"] * (LEVELS + 1)
        bars = ax.bar(range(len(names)), mean, 0.65, yerr=sd, color=colors, zorder=2, capsize=3,
                      error_kw=dict(ecolor=MUTED, lw=1))
        for b, x in zip(bars, mean):
            ax.annotate(f"{x:.2f}" if key == "iou" else f"{x:.0f}", (b.get_x() + b.get_width() / 2, max(x, 0)),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(names)), [n.replace("silhouette ", "") for n in names], rotation=30, fontsize=8)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(f"{title}, mean ± sd over {len(results)} cars", fontsize=9.5, loc="left", color=INK)
        vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def mean_car_figure(path, results):
    occ = np.mean([normalised_occupancy(*r["steps"][-1], r["box"]) for r in results.values()], axis=0)
    dims = np.mean([[r["box"][1][a] - r["box"][0][a] for a in range(3)] for r in results.values()], axis=0)
    n = occ.shape
    v, f, _, _ = marching_cubes(np.pad(occ, 1), 0.5)
    v = (v - 1) / (np.array(n) - 1)
    v = (v - [0.5, 0.5, 0.0]) * dims
    fig = plt.figure(figsize=(18, 4.4))
    for c, (name, elev, azim) in enumerate((("side", 4, 90), ("3/4 front", 22, 35), ("top", 85, -90), ("3/4 rear", 22, 215))):
        ax = fig.add_subplot(1, 4, c + 1, projection="3d")
        draw_mesh(ax, v, f, alpha=0.95)
        style3d(ax, dims[:2], elev, azim)
        ax.set_title(f"mean car of {len(results)} cars: {name}", fontsize=9.5, loc="left", color=INK)
    fig.suptitle(f"Average shape of all reconstructed cars (size-normalised occupancy, 0.5 level), mean size "
                 f"{dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} m", fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main():
    seq = vs.Sequence()
    seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    ground = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))[2]
    cars = suitable_cars(seq)
    print(f"{len(cars)} suitable cars: {cars}", flush=True)
    results = {}
    for tid in cars:
        r = run_car(seq, tid, ground)
        if r is None:
            continue
        results[tid] = r
        fr = r["rows"][f"level {LEVELS}"]
        hr = r["rows"]["silhouette hull"]
        print(f"car {tid}: {r['frames']} frames, {len(r['pts'])} flow points | hull compl {hr['completeness_cm']:.1f} roof "
              f"{hr['roof_err_cm']:+.0f} IoU {hr['iou']:.2f} | level {LEVELS} compl {fr['completeness_cm']:.1f} prec "
              f"{fr['precision_cm']:.1f} roof {fr['roof_err_cm']:+.0f} IoU {fr['iou']:.2f}", flush=True)
    summary = {}
    for name in ["cuboid", "silhouette hull"] + [f"level {k}" for k in range(LEVELS + 1)]:
        vals = {k: [r["rows"][name][k] for r in results.values()] for k in ("precision_cm", "completeness_cm",
                                                                             "roof_err_cm", "iou")}
        summary[name] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in vals.items()}
    json.dump({"cars": list(results), "summary": summary,
               "per_car": {t: r["rows"] for t, r in results.items()}},
              open(os.path.join(vd.OUT_DIR, "refine_metrics.json"), "w"), indent=1)
    for name, s in summary.items():
        print(f"{name:16s} " + " | ".join(f"{k} {v['mean']:.2f} ± {v['sd']:.2f}" for k, v in s.items()))
    best = max(results, key=lambda t: len(results[t]["pts"]))
    steps_figure(os.path.join(vd.OUT_DIR, "refine_steps.png"), seq, best, results[best])
    cars_figure(os.path.join(vd.OUT_DIR, "refine_cars.png"), seq, results)
    scores_figure(os.path.join(vd.OUT_DIR, "refine_scores.png"), results)
    mean_car_figure(os.path.join(vd.OUT_DIR, "refine_mean_car.png"), results)


if __name__ == "__main__":
    main()
