"""
Triangle normals of a car from tracked vertices and the known motion of its cuboid (V2X-Seq, intersection A).

  1. one reference frame: corners inside the (eroded) motion mask, Delaunay triangulation;
  2. every vertex tracked with pyramidal Lucas-Kanade forwards and backwards through the clip (forward-backward
     check, the track ends when the point leaves the mask);
  3. in the car's own frame the car is static and the fixed camera moves around it (the labelled cuboid pose of each
     frame), so every vertex is the least-squares intersection of its rays from all frames where it was tracked -
     with known motion each track point adds one useful equation (depth along the ray; the second one is the epipolar
     check), three per triangle fix its plane;
  4. the same with the tracks cut to +-N frames around the reference, N = 1 (two frames) ... all, to show how the
     accuracy grows with the baseline;
  5. check against the LiDAR of the reference frame: the depth of every vertex along its ray versus the nearest LiDAR
     return in that pixel, and every triangle's normal versus the local LiDAR plane.

Writes docs/tri_tracks_<tid>.png and docs/tri_tracks.json.
"""
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay, cKDTree

import car_flow_planes as cfp
import mix3d as mx
import v2x_seq_demo as vd
import v2x_seq_shape as vs

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FB_TOL = 0.7              # px: forward-backward tolerance per step
SPANS = (1, 2, 4, 8, 16, 999)
PATCH = float(os.environ.get("PATCH", "0.5"))     # m: radius of the patch normals
SMOOTH = os.environ.get("SMOOTH", "0") == "1"   # smoothed cuboid trajectory instead of the per-frame labels
INK, MUTED = "#1f1f1e", "#8a8984"


def intersect_rays(C, D):
    """Least-squares intersection of each point's rays. C, D: (n, k, 3) origins and directions, NaN rows = no ray.
    Returns X (n, 3) and the largest angle (deg) between a ray and the mean ray (the triangulation baseline)."""
    ok = ~np.isnan(D).any(2)
    D = np.where(ok[..., None], D, 0.0)
    C = np.where(ok[..., None], C, 0.0)
    D = D / np.maximum(np.linalg.norm(D, axis=2, keepdims=True), 1e-12)
    A = ok[..., None, None] * (np.eye(3) - np.einsum("nki,nkj->nkij", D, D))
    S = A.sum(1) + 1e-9 * np.eye(3)
    X = np.linalg.solve(S, np.einsum("nkij,nkj->ni", A, C)[..., None])[..., 0]
    m = D.sum(1)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    ang = np.degrees(np.arccos(np.clip(np.einsum("nki,ni->nk", D, m), -1, 1)))
    ang = np.where(ok, ang, 0).max(1)
    few = ok.sum(1) < 2
    X[few] = np.nan
    ang[few] = 0
    return X, ang


def rays(cam, o, uv):
    """Camera centre and pixel rays in the car's ground frame (origin at the box centre on the ground).
    The dataset's R is a general 3x3 matrix (orthonormality error ~0.3), so back-projection uses its inverse."""
    K, R, t = cam
    c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    C = Rz.T @ (-np.linalg.solve(R, t) - np.array([o["pos"][0], o["pos"][1], 0.0]))
    d = np.linalg.solve(K @ R, np.c_[uv, np.ones(len(uv))].T).T
    d *= np.sign((R @ d.T)[2])[:, None]                                 # in front of the camera
    return np.broadcast_to(C, d.shape), d @ Rz


def smooth_poses(obs, deg=3):
    """A rigid car's path is smooth: low-order polynomial fits of position and heading over time."""
    k = np.array([o["k"] for o in obs], float)
    P = np.array([o["pos"] for o in obs])
    yaw = np.unwrap([o["yaw"] for o in obs])
    d = min(deg, len(obs) - 1)
    out = []
    for i, o in enumerate(obs):
        q = dict(o)
        q["pos"] = np.array([np.polyval(np.polyfit(k, P[:, a], d), k[i]) for a in (0, 1)])
        q["yaw"] = float(np.polyval(np.polyfit(k, yaw, d), k[i]))
        out.append(q)
    return out


def reprojection(X, uv, obs, cams):
    """Median distance (px) between every tracked point and the projection of its triangulated vertex."""
    res = []
    for j, o in enumerate(obs):
        ok = ~np.isnan(uv[:, j, 0]) & ~np.isnan(X[:, 0])
        if ok.any():
            q = vs.project(cams[o["k"]], vs.world(X[ok], o["pos"], o["yaw"]))[0]
            res.append(np.linalg.norm(q - uv[ok, j], axis=1))
    return float(np.median(np.concatenate(res))) if res else float("nan")


def reference_mesh(gray, mask, erode=9, top_frac=1.0):
    """Corners inside the mask eroded by `erode` px (and in its upper `top_frac` of height), Delaunay triangulation."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((erode, erode), np.uint8))
    if top_frac < 1.0:
        ys = np.nonzero(mask)[0]
        inner[int(ys.min() + top_frac * (ys.max() - ys.min())):] = 0
    area = inner.sum()
    pts = cv2.goodFeaturesToTrack(gray, 600, 0.003, max(3, int(np.sqrt(area / 500))), mask=inner)
    P = pts[:, 0].astype(np.float32)
    tri = Delaunay(P).simplices
    cen = P[tri].mean(1).round().astype(int)
    edge = np.max([np.linalg.norm(P[tri[:, a]] - P[tri[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    keep = inner[cen[:, 1], cen[:, 0]] > 0
    keep &= edge < 4 * np.median(edge)
    return P, tri[keep]


def track(seq, obs, r, P, span=None):
    """uv (n, len(obs), 2) with NaN where the track is lost; tracked from obs[r] both ways (at most `span` frames)."""
    n = len(P)
    uv = np.full((n, len(obs), 2), np.nan, np.float32)
    uv[:, r] = P
    gray = {o["k"]: cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2GRAY) for o in obs}
    for step in (1, -1):
        alive = np.ones(n, bool)
        cur = P.copy()
        i = r
        while 0 <= i + step < len(obs) and alive.any() and (span is None or abs(i + step - r) <= span):
            a, b = obs[i], obs[i + step]
            if abs(b["k"] - a["k"]) != 1:
                break
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[a["k"]], gray[b["k"]], cur[:, None], None, **LK)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[b["k"]], gray[a["k"]], nxt, None, **LK)
            fb = np.linalg.norm(back[:, 0] - cur, axis=1)
            nxt = nxt[:, 0]
            mask = cv2.dilate(seq.mask(b).astype(np.uint8), np.ones((5, 5), np.uint8))
            q = np.clip(nxt.round().astype(int), 0, [seq.shape[1] - 1, seq.shape[0] - 1])
            alive &= (st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < FB_TOL) & (mask[q[:, 1], q[:, 0]] > 0)
            i += step
            uv[alive, i] = nxt[alive]
            cur = nxt
    return uv


def lidar_points(seq, o, ground_z, cam, uv, radius=4):
    """For each pixel: the front-most LiDAR point of this car projecting within `radius` px (object frame), else NaN."""
    L = mx.to_ground_frame(cfp.lidar_on_car(seq, o, ground_z), o)
    q, z = vs.project(cam, vs.world(L, o["pos"], o["yaw"]))
    t = cKDTree(q)
    out = np.full((len(uv), 3), np.nan)
    for i, idx in enumerate(t.query_ball_point(uv, radius)):
        if len(idx) >= 2:
            out[i] = L[idx[int(np.argmin(z[idx]))]]
    return out, L


def main(tids):
    seq = vs.Sequence()
    seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    ground = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))[2]
    report = {}
    for tid in tids:
        obs = [o for o in seq.tracks[tid] if seq.usable(o)]
        if SMOOTH:
            obs = smooth_poses(obs)
        r = len(obs) // 2
        o_r = obs[r]
        cam_r = seq.cams[o_r["k"]]
        gray = cv2.cvtColor(seq.frames[o_r["k"]], cv2.COLOR_BGR2GRAY)
        P, tri = reference_mesh(gray, seq.mask(o_r) > 0)
        uv = track(seq, obs, r, P)
        C = np.full(uv.shape[:2] + (3,), np.nan)
        D = np.full(uv.shape[:2] + (3,), np.nan)
        for j, o in enumerate(obs):
            ok = ~np.isnan(uv[:, j, 0])
            if ok.any():
                C[ok, j], D[ok, j] = rays(seq.cams[o["k"]], o, uv[ok, j].astype(float))
        # ground truth along the reference rays
        G, L = lidar_points(seq, o_r, ground, cam_r, P)
        Cr, Dr = rays(cam_r, o_r, P.astype(float))
        Dr = Dr / np.linalg.norm(Dr, axis=1, keepdims=True)
        gt_range = np.linalg.norm(G - Cr, axis=1)
        tL = cKDTree(L)
        res = {}
        for span in SPANS:
            sel = np.abs(np.arange(len(obs)) - r) <= span
            X, ang = intersect_rays(np.where(sel[None, :, None], C, np.nan), np.where(sel[None, :, None], D, np.nan))
            err = np.einsum("ni,ni->n", X - Cr, Dr) - gt_range               # metric range along the reference ray
            Xt = X[tri]
            n = np.cross(Xt[:, 1] - Xt[:, 0], Xt[:, 2] - Xt[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            nerr = []
            cen_uv = P[tri].mean(1)
            Gc, _ = lidar_points(seq, o_r, ground, cam_r, cen_uv, radius=6)
            for i in range(len(tri)):
                if np.isnan(Gc[i]).any() or np.isnan(n[i]).any():
                    nerr.append(np.nan)
                    continue
                g = Gc[i]
                idx = tL.query_ball_point(g, 0.35)
                if len(idx) < 6:
                    nerr.append(np.nan)
                    continue
                Q = L[idx] - L[idx].mean(0)
                gn = np.linalg.svd(Q, full_matrices=False)[2][2]
                nerr.append(np.degrees(np.arccos(min(1.0, abs(gn @ n[i])))))
            nerr = np.array(nerr)
            # patch normals: a plane through every triangulated vertex within PATCH m of the triangle centre
            okX = ~np.isnan(X[:, 0])
            tX = cKDTree(X[okX])
            perr = np.full(len(tri), np.nan)
            for i in range(len(tri)):
                if np.isnan(nerr[i]) or np.isnan(Xt[i]).any():
                    continue
                idx = tX.query_ball_point(Xt[i].mean(0), PATCH)
                if len(idx) < 6:
                    continue
                Q = X[okX][idx] - X[okX][idx].mean(0)
                pn = np.linalg.svd(Q, full_matrices=False)[2][2]
                g = Gc[i]
                idl = tL.query_ball_point(g, 0.35)
                gn = np.linalg.svd(L[idl] - L[idl].mean(0), full_matrices=False)[2][2]
                perr[i] = np.degrees(np.arccos(min(1.0, abs(gn @ pn))))
            good = ~np.isnan(err)
            res[span] = {"frames": int(sel.sum()), "tracked": int((~np.isnan(X[:, 0])).sum()),
                         "reproj_px": reprojection(X, np.where(sel[None, :, None], uv, np.nan), obs, seq.cams),
                         "angle_deg": float(np.nanmedian(ang[ang > 0])) if (ang > 0).any() else 0.0,
                         "depth_err_cm": float(100 * np.nanmedian(np.abs(err[good]))) if good.any() else float("nan"),
                         "depth_bias_cm": float(100 * np.nanmedian(err[good])) if good.any() else float("nan"),
                         "normal_err_deg": float(np.nanmedian(nerr)) if (~np.isnan(nerr)).any() else float("nan"),
                         "normals_checked": int((~np.isnan(nerr)).sum()), "X": X, "nerr": nerr, "err": err,
                         "patch_normal_err_deg": float(np.nanmedian(perr)) if (~np.isnan(perr)).any() else float("nan")}
            print(f"car {tid} +-{span:3d} frames ({res[span]['frames']:2d} used): {res[span]['tracked']} of {len(P)} "
                  f"vertices, baseline {res[span]['angle_deg']:.1f} deg, reprojection {res[span]['reproj_px']:.2f} px | depth error {res[span]['depth_err_cm']:.0f} cm "
                  f"(bias {res[span]['depth_bias_cm']:+.0f}) | normal error {res[span]['normal_err_deg']:.0f} deg, "
                  f"{PATCH} m patch {res[span]['patch_normal_err_deg']:.0f} deg "
                  f"({res[span]['normals_checked']} triangles)", flush=True)
        report[tid] = {str(s): {k: v for k, v in d.items() if k not in ("X", "nerr", "err")} for s, d in res.items()}
        figure(os.path.join(vd.OUT_DIR, f"tri_tracks_{tid}{'_smooth' if SMOOTH else ''}.png"), seq, tid, o_r, P, tri, uv, res, L, r)
    json.dump(report, open(os.path.join(vd.OUT_DIR, f"tri_tracks{'_smooth' if SMOOTH else ''}.json"), "w"), indent=1)


def figure(path, seq, tid, o_r, P, tri, uv, res, L, r):
    fig = plt.figure(figsize=(20, 9.5))
    gs = fig.add_gridspec(2, 4)
    x0, y0, x1, y1 = np.array(o_r["box"]).round().astype(int)
    img = cv2.cvtColor(seq.frames[o_r["k"]], cv2.COLOR_BGR2RGB)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    ax.triplot(P[:, 0], P[:, 1], tri, color="#2a78d6", linewidth=0.5)
    for i in range(0, len(P), max(1, len(P) // 60)):
        t = uv[i][~np.isnan(uv[i, :, 0])]
        ax.plot(t[:, 0], t[:, 1], color="#eb6834", linewidth=0.6)
    ax.set_xlim(x0 - 60, x1 + 60)
    ax.set_ylim(y1 + 40, y0 - 40)
    ax.axis("off")
    ax.set_title(f"car {tid}: reference triangulation (blue), vertex tracks (orange)", fontsize=9, loc="left", color=INK)
    best = SPANS[-1]
    for j, span in enumerate((1, best)):
        ax = fig.add_subplot(gs[0, 1 + j])
        ax.imshow(img)
        ne = res[span]["nerr"]
        for t, e in zip(tri, ne):
            if not np.isnan(e):
                ax.fill(*P[t].T, color=plt.cm.RdYlGn_r(min(e, 60) / 60), alpha=0.75, linewidth=0)
        ax.set_xlim(x0 - 10, x1 + 10)
        ax.set_ylim(y1 + 10, y0 - 10)
        ax.axis("off")
        lab = "two frames" if span == 1 else f"all {res[span]['frames']} frames"
        ax.set_title(f"{lab}: normal error vs LiDAR plane, median {res[span]['normal_err_deg']:.0f} deg\n"
                     f"(green 0 deg ... red >= 60 deg)", fontsize=9, loc="left", color=INK)
    ax = fig.add_subplot(gs[0, 3])
    spans = [s for s in SPANS]
    fr = [res[s]["frames"] for s in spans]
    ax.plot(fr, [res[s]["depth_err_cm"] for s in spans], "o-", color="#2a78d6", label="depth error (cm)")
    ax.plot(fr, [res[s]["normal_err_deg"] for s in spans], "s-", color="#eb6834", label="normal error (deg)")
    ax.set_xscale("log")
    ax.set_xlabel("frames in the track (log)", fontsize=9)
    ax.set_title("median errors vs number of frames", fontsize=9.5, loc="left", color=INK)
    ax.legend(fontsize=8, frameon=False)
    vd.style(ax)
    for j, span in enumerate((1, best)):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2], projection="3d")
        X = res[span]["X"]
        okt = ~np.isnan(X[tri]).any((1, 2))
        Xt = X[tri[okt]]
        far = np.linalg.norm(Xt.mean(1)[:, :2], axis=1) < 4
        ax.add_collection3d(Poly3DCollection(Xt[far], facecolors="#2a78d6", edgecolors="#1f1f1e", linewidths=0.2,
                                             alpha=0.6))
        Ls = L[np.random.default_rng(0).permutation(len(L))[:2500]]
        ax.scatter(*Ls.T, s=0.4, c="#d03b3b", alpha=0.4)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-0.5, 2.5)
        ax.set_box_aspect((1, 1, 0.5))
        ax.view_init(18, -60)
        ax.set_title(f"{'two frames' if span == 1 else 'all frames'}: triangulated triangles (blue) vs LiDAR (red), "
                     f"depth error {res[span]['depth_err_cm']:.0f} cm", fontsize=9.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or ["004012"])
