"""
Triangle normals of a car with a smoothness condition where two triangles are stitched (V2X-Seq, intersection A).

tri_tracks.py solves every vertex of a reference triangulation independently (its rays from all frames of its track,
the smoothed cuboid motion); a triangle's normal then comes from its own three noisy vertices: 10 cm of depth noise on
a 30 cm triangle tilts it by 30-50 deg. Here all vertices are solved together:
  unknowns   : one depth per vertex along its ray in the reference frame, X_v = C + s_v d_v;
  data       : reprojection of every vertex into every frame of its track (px);
  smoothness : for every edge shared by two triangles, LAMBDA * (n_1 - n_2), the difference of their unit normals
               (3 residuals), under a soft-L1 loss so that real creases (bonnet / windscreen, the car's outline)
               can survive while noise is flattened; vertices with too short a track are filled from their
               neighbours by this term alone.
Robust least squares (trf, sparse Jacobian), started from the independent triangulation. Checked against the LiDAR:
the depth of every vertex along its ray, the normal of every triangle against the local LiDAR plane - for the
independent triangles, the 0.5 m patch normals of tri_tracks, and the stitched solution for several LAMBDA.

Writes docs/tri_smooth_<tid>.png and docs/tri_smooth.json.
"""
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import tri_tracks as tt
import v2x_seq_demo as vd
import v2x_seq_shape as vs

SPAN = 8
LAMBDAS = (0.0, 2.0, 5.0, 10.0, 20.0)
PATCH = 0.5
INK, MUTED = "#1f1f1e", "#8a8984"


def edge_pairs(tri):
    """(a, b, c, d): triangles (a, b, c) and (b, a, d) of the reference triangulation share the edge (a, b)."""
    seen, pairs = {}, []
    for t in tri:
        for i in range(3):
            a, b, c = t[i], t[(i + 1) % 3], t[(i + 2) % 3]
            key = (min(a, b), max(a, b))
            if key in seen:
                a2, b2, c2 = seen[key]
                pairs.append((a2, b2, c2, c))
            else:
                seen[key] = (a, b, c)
    return np.array(pairs, int).reshape(-1, 4)


def smooth_sparsity(n, obs_vertex, pairs):
    m, p = len(obs_vertex), len(pairs)
    rows = [np.arange(2 * m), 2 * m + np.repeat(np.arange(3 * p), 4)]
    cols = [np.repeat(obs_vertex, 2), np.repeat(pairs, 3, axis=0).ravel()]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * m + 3 * p, n)).tocsr()


def normals(X, tri):
    n = np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]])
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)


def stitched(seq, obs, P, tri, uv, s0, Cr, Dr, lam, ref=None, bounds=None, vertex_lab=None, de_soft=15.0,
             weak=0.3):
    """vertex_lab (n, 3): image colour at every vertex; the smoothness between two triangles is lam when their mean
    colours are within de_soft (one body panel), weak * lam across a colour boundary (window / body, lights), where
    real creases are."""
    """Joint depths of all vertices: reprojection + LAMBDA * normal differences across shared edges."""
    n = len(P)
    vi, fi = np.nonzero(~np.isnan(uv[:, :, 0]))
    keep = fi != (len(obs) // 2 if ref is None else ref)                      # the reference frame fixes nothing
    vi, fi = vi[keep], fi[keep]
    obs_uv = uv[vi, fi].astype(float)
    pairs = edge_pairs(tri)
    # every pair's two triangles in the image orientation of the reference triangulation
    t1 = np.stack([pairs[:, 0], pairs[:, 1], pairs[:, 2]], 1)
    t2 = np.stack([pairs[:, 1], pairs[:, 0], pairs[:, 3]], 1)
    w = np.ones(len(pairs))
    if vertex_lab is not None and len(pairs):
        de = np.linalg.norm(vertex_lab[t1].mean(1) - vertex_lab[t2].mean(1), axis=1)
        w = np.where(de < de_soft, 1.0, weak)
    frames = {}
    for j in np.unique(fi):
        o = obs[j]
        frames[j] = (seq.cams[o["k"]], o)

    def fun(s):
        X = Cr + s[:, None] * Dr
        r = np.zeros((len(vi), 2))
        for j, (cam, o) in frames.items():
            sel = fi == j
            r[sel] = vs.project(cam, vs.world(X[vi[sel]], o["pos"], o["yaw"]))[0] - obs_uv[sel]
        sm = lam * w[:, None] * (normals(X, t1) - normals(X, t2))
        return np.r_[r.ravel(), sm.ravel()]
    if lam == 0:
        return s0
    kw = {} if bounds is None else {"bounds": bounds}
    sol = least_squares(fun, s0, jac_sparsity=smooth_sparsity(n, vi, pairs), loss="soft_l1", f_scale=1.0,
                        method="trf", tr_solver="lsmr", x_scale="jac", max_nfev=80, **kw)
    return sol.x


def evaluate(X, tri, G, gt_range, Cr, Dr, L, tL, Gc):
    err = np.einsum("ni,ni->n", X - Cr, Dr) - gt_range
    n = normals(X, tri)
    nerr = np.full(len(tri), np.nan)
    for i in range(len(tri)):
        if np.isnan(Gc[i]).any() or np.isnan(n[i]).any():
            continue
        idx = tL.query_ball_point(Gc[i], 0.35)
        if len(idx) < 6:
            continue
        gn = np.linalg.svd(L[idx] - L[idx].mean(0), full_matrices=False)[2][2]
        nerr[i] = np.degrees(np.arccos(min(1.0, abs(gn @ n[i]))))
    ok = ~np.isnan(err)
    return {"depth_err_cm": float(100 * np.nanmedian(np.abs(err))), "normal_err_deg": float(np.nanmedian(nerr)),
            "normals_within_20deg": float(np.mean(nerr[~np.isnan(nerr)] < 20)), "checked": int((~np.isnan(nerr)).sum()),
            "depths_checked": int(ok.sum())}, nerr


def patch_normals(X, tri, ok_v):
    out = np.full((len(tri), 3), np.nan)
    tX = cKDTree(X[ok_v])
    Xo = X[ok_v]
    for i, t in enumerate(tri):
        c = X[t].mean(0)
        if np.isnan(c).any():
            continue
        idx = tX.query_ball_point(c, PATCH)
        if len(idx) >= 6:
            out[i] = np.linalg.svd(Xo[idx] - Xo[idx].mean(0), full_matrices=False)[2][2]
    return out


def run(seq, tid, ground):
    obs = tt.smooth_poses([o for o in seq.tracks[tid] if seq.usable(o)])
    r = len(obs) // 2
    o_r = obs[r]
    cam_r = seq.cams[o_r["k"]]
    gray = cv2.cvtColor(seq.frames[o_r["k"]], cv2.COLOR_BGR2GRAY)
    P, tri = tt.reference_mesh(gray, seq.mask(o_r) > 0)
    uv = tt.track(seq, obs, r, P, span=SPAN)
    C = np.full(uv.shape[:2] + (3,), np.nan)
    D = np.full(uv.shape[:2] + (3,), np.nan)
    for j, o in enumerate(obs):
        ok = ~np.isnan(uv[:, j, 0])
        if ok.any():
            C[ok, j], D[ok, j] = tt.rays(seq.cams[o["k"]], o, uv[ok, j].astype(float))
    X0, _ = tt.intersect_rays(C, D)
    Cr, Dr = tt.rays(cam_r, o_r, P.astype(float))
    Cr = np.array(Cr)
    Dr = Dr / np.linalg.norm(Dr, axis=1, keepdims=True)
    s0 = np.einsum("ni,ni->n", X0 - Cr, Dr)
    tracked = ~np.isnan(s0)
    # untracked vertices start at the median depth of their neighbours
    if (~tracked).any():
        nb = cKDTree(P[tracked])
        _, j = nb.query(P[~tracked], k=min(4, tracked.sum()))
        s0[~tracked] = np.median(s0[tracked][np.atleast_2d(j)], axis=1)
    G, L = tt.lidar_points(seq, o_r, ground, cam_r, P)
    gt_range = np.linalg.norm(G - Cr, axis=1)
    Gc, _ = tt.lidar_points(seq, o_r, ground, cam_r, P[tri].mean(1), radius=6)
    tL = cKDTree(L)
    res, sols = {}, {}
    for lam in LAMBDAS:
        s = stitched(seq, obs, P, tri, uv, s0.copy(), Cr, Dr, lam)
        X = Cr + s[:, None] * Dr
        if lam == 0:
            X[~tracked] = np.nan                                 # independent triangles: no fill-in
        res[f"lambda {lam:g}"], ne = evaluate(X, tri, G, gt_range, Cr, Dr, L, tL, Gc)
        sols[lam] = (X, ne)
    Xi = sols[0.0][0]
    pn = patch_normals(Xi, tri, ~np.isnan(Xi[:, 0]))
    pe = np.full(len(tri), np.nan)
    for i in range(len(tri)):
        if np.isnan(Gc[i]).any() or np.isnan(pn[i]).any():
            continue
        idx = tL.query_ball_point(Gc[i], 0.35)
        if len(idx) >= 6:
            gn = np.linalg.svd(L[idx] - L[idx].mean(0), full_matrices=False)[2][2]
            pe[i] = np.degrees(np.arccos(min(1.0, abs(gn @ pn[i]))))
    res["patch normals 0.5 m"] = {"normal_err_deg": float(np.nanmedian(pe)), "normals_within_20deg": float(np.mean(pe[~np.isnan(pe)] < 20))}
    return res, sols, P, tri, o_r, L


def figure(path, seq, tid, res, sols, P, tri, o_r, L):
    best = min((k for k in res if k.startswith("lambda") and k != "lambda 0"),
               key=lambda k: res[k]["normal_err_deg"])
    lam_best = float(best.split()[1])
    fig = plt.figure(figsize=(20, 9.5))
    gs = fig.add_gridspec(2, 4)
    img = cv2.cvtColor(seq.frames[o_r["k"]], cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = o_r["box"]
    for c, (lam, name) in enumerate(((0.0, "independent triangles"), (lam_best, f"stitched, smoothness lambda {lam_best:g}"))):
        ax = fig.add_subplot(gs[0, 2 * c])
        ax.imshow(img)
        for t, e in zip(tri, sols[lam][1]):
            if not np.isnan(e):
                ax.fill(*P[t].T, color=plt.cm.RdYlGn_r(min(e, 60) / 60), alpha=0.75, linewidth=0)
        ax.triplot(P[:, 0], P[:, 1], tri, color="white", linewidth=0.3)
        ax.set_xlim(x0 - 10, x1 + 10)
        ax.set_ylim(y1 + 10, y0 - 10)
        ax.axis("off")
        r = res["lambda 0" if lam == 0 else best]
        ax.set_title(f"{name}: normal error median {r['normal_err_deg']:.0f} deg,\n{100 * r['normals_within_20deg']:.0f}% "
                     f"within 20 deg (green 0 ... red >= 60 deg)", fontsize=9, loc="left", color=INK)
        ax = fig.add_subplot(gs[0, 2 * c + 1], projection="3d")
        X = sols[lam][0]
        okt = ~np.isnan(X[tri]).any((1, 2))
        n = normals(X, tri[okt])
        shade = 0.4 + 0.6 * np.abs(n @ np.array([0.3, -0.5, 0.8]))
        ax.add_collection3d(Poly3DCollection(X[tri[okt]], facecolors=plt.cm.Blues(0.35 + 0.6 * shade),
                                             edgecolors="#1f1f1e", linewidths=0.2))
        Ls = L[np.random.default_rng(0).permutation(len(L))[:2500]]
        ax.scatter(*Ls.T, s=0.3, c="#d03b3b", alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-0.3, 2.5)
        ax.set_box_aspect((1, 1, 0.47))
        ax.view_init(18, -60)
        ax.tick_params(labelsize=6)
        ax.set_title(f"in 3D (red = LiDAR): depth error {r['depth_err_cm']:.0f} cm", fontsize=9, loc="left", color=INK)
    ax = fig.add_subplot(gs[1, :2])
    keys = [k for k in res if k.startswith("lambda")]
    lam = [float(k.split()[1]) for k in keys]
    ax.plot(lam, [res[k]["normal_err_deg"] for k in keys], "o-", color="#2a78d6", label="normal error, median (deg)")
    ax.plot(lam, [100 * res[k]["normals_within_20deg"] for k in keys], "s-", color="#2f7d4f", label="% normals within 20 deg")
    ax.plot(lam, [res[k]["depth_err_cm"] for k in keys], "^-", color="#eb6834", label="depth error, median (cm)")
    ax.axhline(res["patch normals 0.5 m"]["normal_err_deg"], color=MUTED, linestyle="--",
               label=f"patch normals 0.5 m: {res['patch normals 0.5 m']['normal_err_deg']:.0f} deg")
    ax.set_xlabel("smoothness weight lambda (0 = independent triangles)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"car {tid}: what the smoothness condition changes", fontsize=10, loc="left", color=INK)
    vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def main(tids):
    seq = vs.Sequence()
    seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    ground = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))[2]
    report = {}
    for tid in tids:
        res, sols, P, tri, o_r, L = run(seq, tid, ground)
        report[tid] = res
        print(f"car {tid}:", flush=True)
        for k, v in res.items():
            print(f"   {k:20s} " + " | ".join(f"{a} {b:.2f}" if isinstance(b, float) else f"{a} {b}" for a, b in v.items()),
                  flush=True)
        figure(os.path.join(vd.OUT_DIR, f"tri_smooth_{tid}.png"), seq, tid, res, sols, P, tri, o_r, L)
    json.dump(report, open(os.path.join(vd.OUT_DIR, "tri_smooth.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["004012", "004005", "003981", "003892"])
