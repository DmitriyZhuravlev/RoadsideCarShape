"""
A car's 3D surface from the fixed roadside camera: a new triangulation every frame, a plane per triangle from how it
moves and deforms into the next frame, one 3D model per frame, and the mean surface of all of them (V2X-Seq, A).

  1. every frame k: Delaunay triangulation of the region around the tracked car (10 px grid + corners);
  2. images of the triangles in frame k+1: DIS optical flow (forward-backward checked);
     a triangle that does not move (median flow < STATIC_PX) is a static plane of the background and is left out;
     moving triangles inside the car's (padded) box are the car;
  3. per moving triangle: plane m = n/d from its deformation, H = K (R + t m^T) K^-1, with (R, t) the car's motion
     between the frames (label poses, as the initialisation) - flow_planes.solve_triangles;
  4. every frame's triangles are moved into the car's own frame: one 3D model per frame;
  5. mean surface: all frames' triangles as oriented points (vertices and centre, normal oriented away from the car's
     centre), averaged in 8 cm voxels (mean position, mean normal, kept if seen >= 3 times with coherent normals);
     a mesh by marching cubes of the signed distance to the nearest fused point along its normal (Hoppe et al. 1992);
     normals also from the shape of the mean surface (local plane of the fused points within 0.3 m);
  6. check against the intersection's LiDAR points on the same car (in its frame, over the whole track): per-frame
     models vs the fused surface, distance and normal angle to the local LiDAR plane.

Writes docs/carfusion_frames.png, docs/carfusion_3d.png, docs/carfusion_accuracy.png, docs/carfusion_metrics.json.
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay, cKDTree
from skimage.measure import marching_cubes

import car_flow_planes as cfp
import flow_planes as fp
import v2x_seq_demo as vd
import v2x_seq_shape as vs

TRACKS = ("003981", "004005", "004012")
STEP = 1                  # consecutive frames (0.1 s)
STATIC_PX = 0.6           # median flow below this: a static plane, not the car
VOXEL = 0.08
INK, MUTED = "#1f1f1e", "#8a8984"


def fuse_oriented_points(points, normals, lo, voxel, shape, min_count=3, min_coherence=0.4):
    """Mean position and mean normal of oriented points per voxel (count >= min_count, coherent normals)."""
    idx = np.floor((points - lo) / voxel).astype(int)
    inside = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    idx, points, normals = idx[inside], points[inside], normals[inside]
    flat = np.ravel_multi_index(idx.T, shape)
    n = int(np.prod(shape))
    count = np.bincount(flat, minlength=n)
    psum = np.stack([np.bincount(flat, points[:, c], minlength=n) for c in range(3)], 1)
    nsum = np.stack([np.bincount(flat, normals[:, c], minlength=n) for c in range(3)], 1)
    keep = count >= min_count
    keep &= np.linalg.norm(nsum, axis=1) >= min_coherence * np.maximum(count, 1)
    P = psum[keep] / count[keep, None]
    N = nsum[keep] / np.linalg.norm(nsum[keep], axis=1, keepdims=True)
    return P, N, count[keep]


def region_triangles(gray, box1, box2, grid=10, pad=30):
    H, W = gray.shape
    x0, y0 = int(max(0, min(box1[0], box2[0]) - pad)), int(max(0, min(box1[1], box2[1]) - pad))
    x1, y1 = int(min(W - 1, max(box1[2], box2[2]) + pad)), int(min(H - 1, max(box1[3], box2[3]) + pad))
    gx, gy = np.meshgrid(np.arange(x0, x1, grid), np.arange(y0, y1, grid))
    pts = [np.c_[gx.ravel(), gy.ravel()]]
    sub = np.zeros_like(gray)
    sub[y0:y1, x0:x1] = 255
    c = cv2.goodFeaturesToTrack(gray, 300, 0.01, 5, mask=sub)
    if c is not None:
        pts.append(c.reshape(-1, 2))
    P = np.vstack(pts).astype(float)
    _, u = np.unique((P // 3).astype(int), axis=0, return_index=True)
    P = P[np.sort(u)]
    tri = Delaunay(P).simplices
    return P, tri


def frame_model(seq, o, o2):
    k, j = o["k"], o2["k"]
    K, Rc, tc = seq.cams[k]
    A_k = cfp.object_to_camera(Rc, tc, o["pos"], o["yaw"], o["z"])
    A_j = cfp.object_to_camera(seq.cams[j][1], seq.cams[j][2], o2["pos"], o2["yaw"], o2["z"])
    rel = A_j @ np.linalg.inv(A_k)
    g1 = cv2.cvtColor(seq.frames[k], cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(seq.frames[j], cv2.COLOR_BGR2GRAY)
    P, tri = region_triangles(g1, o["box"], o2["box"])
    flow, ok = cfp.crop_flow(g1, g2, o["box"], o2["box"])
    H, W = g1.shape
    pts = np.clip(P[tri].mean(1).round().astype(int), 0, [W - 1, H - 1])
    # moving vs static: median flow of every triangle (sampled at its centre and vertices)
    samp = np.clip(np.concatenate([P[tri], P[tri].mean(1, keepdims=True)], 1).round().astype(int), 0, [W - 1, H - 1])
    mag = np.median(np.linalg.norm(flow[samp[..., 1], samp[..., 0]], axis=2), axis=1)
    x0, y0, x1, y1 = o["box"]
    in_box = (pts[:, 0] >= x0 - 5) & (pts[:, 0] <= x1 + 5) & (pts[:, 1] >= y0 - 5) & (pts[:, 1] <= y1 + 5)
    moving = (mag >= STATIC_PX) & in_box
    lab = np.full((H, W), -1, np.int32)
    car_ids = np.where(moving)[0]
    for i in car_ids:
        cv2.fillConvexPoly(lab, P[tri[i]].round().astype(np.int32), int(i))
    M, info = fp.solve_triangles(P, tri, lab, flow, ok, K, rel[:3, :3], rel[:3, 3], [], max_px=300)
    V, N = fp.vertices_from_planes(P, tri, M, K)
    good = ~np.isnan(V).any((1, 2)) & moving
    inv = np.linalg.inv(A_k)
    Vo = V[good] @ inv[:3, :3].T + inv[:3, 3]
    No = N[good] @ inv[:3, :3].T
    l, w, h = o["dims"]
    inside = np.all((np.abs(Vo[..., 0]) < l / 2 + 0.6) & (np.abs(Vo[..., 1]) < w / 2 + 0.6) &
                    (np.abs(Vo[..., 2]) < h / 2 + 0.6), axis=1)
    return {"k": k, "P": P, "tri": tri, "moving": moving, "static": (mag < STATIC_PX) & in_box, "good": good,
            "V": Vo[inside], "N": No[inside], "N_img": N, "flow": flow, "box": o["box"]}


def evaluate_points(Pts, Nrm, L, tree):
    d, _ = tree.query(Pts)
    ang = []
    for p, n in zip(Pts[::2], Nrm[::2]):
        idx = tree.query_ball_point(p, 0.3)
        if len(idx) >= 6:
            Q = L[idx] - L[idx].mean(0)
            _, sv, Vt = np.linalg.svd(Q)
            if sv[2] < 0.3 * sv[1]:
                ang.append(np.degrees(np.arccos(min(1.0, abs(Vt[2] @ n)))))
    return d, np.array(ang)


def mean_surface(P, N, dims, voxel=VOXEL / 2):
    """Mesh of the zero level of the signed distance to the nearest fused point along its normal."""
    l, w, h = dims
    lo = -np.array([l / 2 + 0.4, w / 2 + 0.4, h / 2 + 0.4])
    hi = -lo
    axes = [np.arange(a, b, voxel) for a, b in zip(lo, hi)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    tree = cKDTree(P)
    dist, idx = tree.query(G, k=4)
    sd = np.einsum("nkj,nkj->nk", G[:, None, :] - P[idx], N[idx]).mean(1)
    sd = np.where(dist[:, 0] < 0.35, sd, np.where(np.einsum("nj,nj->n", G - P[idx[:, 0]], N[idx[:, 0]]) > 0, 0.3, -0.3))
    vol = np.clip(sd, -0.3, 0.3).reshape([len(a) for a in axes])
    verts, faces, normals, _ = marching_cubes(vol, 0.0, spacing=(voxel, voxel, voxel))
    return verts + lo, faces, normals


def run_car(seq, tid, ground_z):
    obs = [o for o in seq.tracks[tid] if seq.usable(o)]
    by_k = {o["k"]: o for o in obs}
    frames_, lidar = [], []
    for o in obs:
        if o["k"] + STEP in by_k:
            frames_.append(frame_model(seq, o, by_k[o["k"] + STEP]))
            lidar.append(cfp.lidar_on_car(seq, o, ground_z))
    L = np.vstack(lidar)
    tree = cKDTree(L)
    dims = obs[0]["dims"]
    # per-frame accuracy
    per = []
    for f in frames_:
        if len(f["V"]) < 5:
            continue
        pts = np.vstack([f["V"].reshape(-1, 3), f["V"].mean(1)])
        nrm = np.vstack([np.repeat(f["N"], 3, 0), f["N"]])
        d, a = evaluate_points(pts, nrm, L, tree)
        per.append((np.median(d), np.median(a) if len(a) else np.nan, len(f["V"])))
    # fusion: oriented points, normals away from the car's centre
    pts = np.vstack([np.vstack([f["V"].reshape(-1, 3), f["V"].mean(1)]) for f in frames_ if len(f["V"])])
    nrm = np.vstack([np.vstack([np.repeat(f["N"], 3, 0), f["N"]]) for f in frames_ if len(f["V"])])
    flip = np.einsum("ij,ij->i", nrm, pts) < 0
    nrm[flip] *= -1
    l, w, h = dims
    lo = -np.array([l / 2 + 0.6, w / 2 + 0.6, h / 2 + 0.6])
    shape = tuple(np.ceil(-2 * lo / VOXEL).astype(int))
    FP, FN, FC = fuse_oriented_points(pts, nrm, lo, VOXEL, shape)
    d_f, a_f = evaluate_points(FP, FN, L, tree)
    # normals from the geometry of the fused points: local plane (PCA) of the mean surface points within 0.3 m
    ftree = cKDTree(FP)
    FN_geo = FN.copy()
    for i, p in enumerate(FP):
        idx = ftree.query_ball_point(p, 0.3)
        if len(idx) >= 8:
            Q = FP[idx] - FP[idx].mean(0)
            n = np.linalg.svd(Q)[2][2]
            FN_geo[i] = n if n @ FN[i] >= 0 else -n
    _, a_g = evaluate_points(FP, FN_geo, L, tree)
    surf = mean_surface(FP, FN, dims) if len(FP) > 50 else None
    d_s = tree.query(surf[0])[0] if surf is not None else np.array([np.nan])
    per = np.array(per)
    m = {"frames": len(frames_), "per_frame_triangles_median": float(np.median(per[:, 2])),
         "per_frame_dist_median_cm": float(100 * np.nanmedian(per[:, 0])),
         "per_frame_normal_median_deg": float(np.nanmedian(per[:, 1])),
         "fused_points": int(len(FP)), "fused_dist_median_cm": float(100 * np.median(d_f)),
         "fused_normal_median_deg": float(np.median(a_f)) if len(a_f) else None,
         "fused_normal_within_20": float(np.mean(a_f < 20)) if len(a_f) else None,
         "fused_geometry_normal_median_deg": float(np.median(a_g)) if len(a_g) else None,
         "fused_geometry_normal_within_20": float(np.mean(a_g < 20)) if len(a_g) else None,
         "surface_dist_median_cm": float(100 * np.median(d_s)), "lidar_points": int(len(L))}
    return m, frames_, (FP, FN, FC), surf, L, dims


# --- figures -------------------------------------------------------------------------------------------
def frames_figure(path, seq, picks):
    fig, axes = plt.subplots(len(picks), 3, figsize=(18, 4.6 * len(picks)))
    for r, (tid, frs) in enumerate(picks.items()):
        for c, f in enumerate(frs):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(seq.frames[f["k"]], cv2.COLOR_BGR2RGB))
            for i, t in enumerate(f["tri"]):
                if f["good"][i]:
                    col = np.clip(f["N_img"][i] * 0.5 + 0.5, 0, 1)
                    ax.fill(*f["P"][t].T, facecolor=col, edgecolor="white", linewidth=0.15, alpha=0.8)
                elif f["static"][i]:
                    ax.fill(*f["P"][t].T, facecolor="none", edgecolor="#9a9a9a", linewidth=0.2)
                elif f["moving"][i]:
                    ax.fill(*f["P"][t].T, facecolor="none", edgecolor="#eb6834", linewidth=0.3)
            x0, y0, x1, y1 = f["box"]
            ax.set_xlim(x0 - 30, x1 + 30)
            ax.set_ylim(y1 + 30, y0 - 30)
            ax.axis("off")
            ax.set_title(f"car {tid}, frame {f['k']}: new triangulation; grey = static plane, colour = normal of a "
                         f"moving triangle", fontsize=8.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def views_figure(path, results):
    views = (("side", 5, 90), ("3/4 front", 25, 35), ("top", 85, -90), ("3/4 rear", 25, 215))
    fig = plt.figure(figsize=(19, 4.0 * 2 * len(results)))
    rows = 2 * len(results)
    for r, (tid, (m, frames_, fused, surf, L, dims)) in enumerate(results.items()):
        one = max(frames_, key=lambda f: len(f["V"]))
        for c, (name, elev, azim) in enumerate(views):
            for rr, kind in enumerate(("one frame", "mean surface")):
                ax = fig.add_subplot(rows, len(views), (2 * r + rr) * len(views) + c + 1, projection="3d")
                if kind == "one frame" and len(one["V"]):
                    ax.add_collection3d(Poly3DCollection(one["V"], facecolors=np.clip(one["N"] * 0.5 + 0.5, 0, 1),
                                                         edgecolors="none", alpha=0.85))
                if kind == "mean surface" and surf is not None:
                    v, f_, n = surf
                    tri = v[f_]
                    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
                    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(np.abs(fn) * 0.8 + 0.15, 0, 1),
                                                         edgecolors="none", alpha=0.9))
                Ls = L[np.random.default_rng(1).permutation(len(L))[:2500]]
                ax.scatter(*Ls.T, s=0.3, c="k", alpha=0.3)
                Cb, E = cfp.box_edges(*dims)
                for a, b in E:
                    ax.plot(*Cb[[a, b]].T, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
                lim = max(dims) / 2 + 0.3
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-lim, lim)
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev, azim)
                ax.axis("off")
                ttl = (f"car {tid}, one frame ({len(one['V'])} triangles)" if kind == "one frame"
                       else f"car {tid}, mean surface of {m['frames']} frames")
                ax.set_title(f"{ttl}: {name}" if c == 0 else name, fontsize=9, loc="left", color=INK)
    fig.suptitle("Per-frame 3D models from triangle deformation vs their mean surface; black = LiDAR on the car, dashed "
                 "= labelled box", fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor="white")
    plt.close(fig)


def accuracy_figure(path, metrics):
    cars = list(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (k1, k2, title) in zip(axes, (("per_frame_dist_median_cm", "fused_dist_median_cm",
                                          "median distance to the LiDAR points on the car (cm)"),
                                         ("per_frame_normal_median_deg", "fused_normal_median_deg",
                                          "median angle to the local LiDAR plane (°)"))):
        series = [(k1, "one frame", "#c3c2b7"), (k2, "mean surface of all frames", "#2a78d6")]
        if "normal" in k1:
            series.append(("fused_geometry_normal_median_deg", "normal from the mean surface's shape", "#eb6834"))
        for j, (key, name, color) in enumerate(series):
            v = [metrics[c][key] if metrics[c][key] is not None else np.nan for c in cars]
            bars = ax.bar(np.arange(len(cars)) + (j - (len(series) - 1) / 2) * 0.27, v, 0.25, color=color, zorder=2,
                          label=name)
            for b, x in zip(bars, v):
                if np.isfinite(x):
                    ax.annotate(f"{x:.0f}", (b.get_x() + b.get_width() / 2, x), xytext=(0, 2),
                                textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(cars)), [f"car {c}" for c in cars])
        ax.set_ylabel(title, color=INK)
        ax.legend(fontsize=8.5, frameon=False)
        vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    seq = vs.Sequence()
    seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    plane = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))
    results, metrics, picks = {}, {}, {}
    for tid in TRACKS:
        m, frames_, fused, surf, L, dims = run_car(seq, tid, plane[2])
        results[tid] = (m, frames_, fused, surf, L, dims)
        metrics[tid] = m
        frs = sorted(frames_, key=lambda f: -len(f["V"]))[:3]
        picks[tid] = sorted(frs, key=lambda f: f["k"])
        print(tid, json.dumps(m), flush=True)
    json.dump(metrics, open(os.path.join(vd.OUT_DIR, "carfusion_metrics.json"), "w"), indent=1)
    frames_figure(os.path.join(vd.OUT_DIR, "carfusion_frames.png"), seq, picks)
    views_figure(os.path.join(vd.OUT_DIR, "carfusion_3d.png"), results)
    accuracy_figure(os.path.join(vd.OUT_DIR, "carfusion_accuracy.png"), metrics)


if __name__ == "__main__":
    main()
