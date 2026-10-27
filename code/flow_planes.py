"""
Normals and 3D of image triangles from how they deform between two frames of a moving camera (classical only).

A planar patch seen from two calibrated views related by X2 = R X1 + t maps by the plane-induced homography
H = K (R + t m^T) K^-1, with m = n / d its normal over its distance (plane m . X = 1). So with the camera motion known,
the way a triangle moves and changes shape between the frames determines its plane: every optical-flow pixel inside
it gives two linear equations in m (cross(q2, R q + t (m . q)) = 0, q = K^-1 x), three unknowns per triangle.
Then n = -m / |m| (towards the camera), the vertices are X = q / (m . q).

Data: the V2X-Seq-SPD vehicle-side camera (moving car, 10 Hz), its pose per frame from the calibration chain
camera <- LiDAR <- NovAtel <- world, images undistorted with the camera's distortion model; the same car's LiDAR is
the ground truth.
  * triangles: Delaunay triangulation of frame k (vertices at the ends of LSD segments, a grid and the border);
  * flow: DIS optical flow k -> j and j -> k, forward-backward check;
  * per triangle: plane from all its consistent flow pixels (dense, least squares with two re-weighting passes),
    rejected when the parallax (flow minus the flow of the plane at infinity) is too small - near the focus of
    expansion depth is not observable;
  * baseline for comparison: the three vertices triangulated separately from their flow (normal = cross product);
  * accuracy against LiDAR: depth along the ray for every LiDAR point, and the angle between each triangle's normal
    and the plane fitted to the LiDAR points inside it.

Writes docs/flowtri_motion.png, docs/flowtri_maps.png, docs/flowtri_3d.png, docs/flowtri_accuracy.png and
docs/flowtri_metrics.json.
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

import v2x_seq_demo as vd
import world3d as w3

DATA = os.path.join(os.path.dirname(vd.DATA), "vehicle-side")
INK, MUTED = "#1f1f1e", "#8a8984"
MIN_PARALLAX = 2.0          # px: below this the triangle's depth is not observable
FB_TOL = 1.0                # px: forward-backward flow consistency


def plane_from_flow(q, q2, R, t, weights=None):
    """Plane m (m . X = 1) of a patch from correspondences q -> q2 (normalised, (N, 3)) with X2 = R X1 + t.
    Linear least squares on cross(q2, t) (q . m) = -cross(q2, R q). Returns (m, rms) or (None, inf)."""
    if len(q) < 3:
        return None, np.inf
    a = np.cross(q2, t[None, :])                                  # (N, 3)
    A = np.einsum("ni,nj->nij", a, q).reshape(-1, 3)
    b = (-np.cross(q2, q @ R.T)).reshape(-1)
    if weights is not None:
        w = np.repeat(np.sqrt(weights), 3)
        A, b = A * w[:, None], b * w
    m = np.linalg.lstsq(A, b, rcond=None)[0]
    return m, float(np.sqrt(np.mean((A @ m - b) ** 2)))


def triangulate_midpoint(q, q2, R, t):
    """Points X1 (in view 1) from rays q (view 1) and q2 (view 2) with X2 = R X1 + t (midpoint method)."""
    d1 = q / np.linalg.norm(q, axis=1, keepdims=True)
    d2 = (q2 @ R) / np.linalg.norm(q2, axis=1, keepdims=True)    # view-2 rays in view-1 coordinates
    c2 = -R.T @ t
    out = []
    for a, b in zip(d1, d2):
        M = np.array([[a @ a, -a @ b], [a @ b, -b @ b]])
        if abs(np.linalg.det(M)) < 1e-10:
            out.append(np.full(3, np.nan))
            continue
        s, u = np.linalg.solve(M, [a @ c2, b @ c2])
        out.append((s * a + c2 + u * b) / 2)
    return np.array(out)


# --- data ---------------------------------------------------------------------------------------------
def pose(d):
    """World -> camera 4x4 for one frame (camera <- LiDAR <- NovAtel <- world)."""
    def T(path, key=None):
        j = json.load(open(os.path.join(DATA, path)))
        j = j.get("transform", j)
        M = np.eye(4)
        M[:3, :3] = np.array(j["rotation"], float)
        M[:3, 3] = np.array(j["translation"], float).ravel()
        return M
    l2c, l2n, n2w = T(d["calib_lidar_to_camera_path"]), T(d["calib_lidar_to_novatel_path"]), T(d["calib_novatel_to_world_path"])
    return l2c @ np.linalg.inv(l2n) @ np.linalg.inv(n2w), l2c


def load():
    info = sorted(json.load(open(os.path.join(DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    intr = json.load(open(os.path.join(DATA, info[0]["calib_camera_intrinsic_path"])))
    K = np.array(intr["cam_K"], float).reshape(3, 3)
    D = np.array(intr["cam_D"], float)
    return info, K, D


def frame(d, K, D, maps={}):
    img = cv2.imread(os.path.join(DATA, d["image_path"]))
    key = img.shape
    if key not in maps:
        maps[key] = cv2.initUndistortRectifyMap(K, D, None, K, (img.shape[1], img.shape[0]), cv2.CV_32FC1)
    return cv2.remap(img, *maps[key], cv2.INTER_LINEAR)


def object_boxes(d):
    out = []
    for o in json.load(open(os.path.join(DATA, d["label_camera_std_path"]))):
        b = o["2d_box"]
        out.append([float(b[k]) for k in ("xmin", "ymin", "xmax", "ymax")])
    return out


# --- triangles and flow ---------------------------------------------------------------------------------
def triangulation(gray, grid=40):
    H, W = gray.shape
    s = cv2.createLineSegmentDetector().detect(gray)[0].reshape(-1, 4)
    s = s[np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) > 15]
    gx, gy = np.meshgrid(np.arange(0, W, grid), np.arange(0, H, grid))
    P = np.vstack([s[:, :2], s[:, 2:], np.c_[gx.ravel(), gy.ravel()],
                   np.c_[np.linspace(0, W - 1, 49), np.zeros(49)], np.c_[np.linspace(0, W - 1, 49), np.full(49, H - 1)],
                   np.c_[np.zeros(28), np.linspace(0, H - 1, 28)], np.c_[np.full(28, W - 1), np.linspace(0, H - 1, 28)]])
    P[:, 0], P[:, 1] = np.clip(P[:, 0], 0, W - 1), np.clip(P[:, 1], 0, H - 1)
    _, idx = np.unique((P // 6).astype(int), axis=0, return_index=True)
    P = P[np.sort(idx)]
    tri = Delaunay(P).simplices
    lab = np.full((H, W), -1, np.int32)
    for i, t in enumerate(tri):
        cv2.fillConvexPoly(lab, P[t].round().astype(np.int32), int(i))
    return P, tri, lab


def flows(g1, g2):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f12, f21 = dis.calc(g1, g2, None), dis.calc(g2, g1, None)
    H, W = g1.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    back = cv2.remap(f21, xx + f12[..., 0], yy + f12[..., 1], cv2.INTER_LINEAR)
    ok = np.linalg.norm(f12 + back, axis=2) < FB_TOL
    return f12, ok


# --- per-triangle planes ------------------------------------------------------------------------------
def solve_triangles(P, tri, lab, flow, ok, K, R, t, objects, max_px=400, seed=0):
    rng = np.random.default_rng(seed)
    Kinv = np.linalg.inv(K)
    H, W = lab.shape
    excl = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in objects:                     # labelled cars and people may move: not a static plane
        excl[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = True
    ys, xs = np.nonzero((lab >= 0) & ok & ~excl)
    ids = lab[ys, xs]
    order = np.argsort(ids)
    ys, xs, ids = ys[order], xs[order], ids[order]
    starts = np.searchsorted(ids, np.arange(len(tri) + 1))
    M = np.full((len(tri), 3), np.nan)
    info = np.zeros((len(tri), 3))                    # parallax px, residual, pixels
    for i in range(len(tri)):
        a, b = starts[i], starts[i + 1]
        if b - a < 30:
            continue
        sel = np.arange(a, b)
        if len(sel) > max_px:
            sel = rng.choice(sel, max_px, replace=False)
        x = np.c_[xs[sel], ys[sel]].astype(float)
        x2 = x + flow[ys[sel], xs[sel]]
        q = np.c_[x, np.ones(len(x))] @ Kinv.T
        q2 = np.c_[x2, np.ones(len(x))] @ Kinv.T
        rot = q @ R.T                                  # plane at infinity: image of the rotation alone
        rot_px = (rot @ K.T)[:, :2] / (rot @ K.T)[:, 2:]
        parallax = float(np.median(np.linalg.norm(x2 - rot_px, axis=1)))
        info[i, 0], info[i, 2] = parallax, len(sel)
        if parallax < MIN_PARALLAX:
            continue
        m, _ = plane_from_flow(q, q2, R, t)
        for _ in range(2):                             # re-weight: Huber on the reprojection error
            Hm = K @ (R + np.outer(t, m)) @ Kinv
            pr = np.c_[x, np.ones(len(x))] @ Hm.T
            err = np.linalg.norm(pr[:, :2] / pr[:, 2:] - x2, axis=1)
            w = np.where(err < 1.5, 1.0, 1.5 / np.maximum(err, 1e-9))
            m, _ = plane_from_flow(q, q2, R, t, w)
        Hm = K @ (R + np.outer(t, m)) @ Kinv
        pr = np.c_[x, np.ones(len(x))] @ Hm.T
        info[i, 1] = float(np.median(np.linalg.norm(pr[:, :2] / pr[:, 2:] - x2, axis=1)))
        Qv = np.c_[P[tri[i]], np.ones(3)] @ Kinv.T
        if np.all(Qv @ m > 1e-4) and info[i, 1] < 2.0:
            M[i] = m
    return M, info


def vertices_from_planes(P, tri, M, K):
    Kinv = np.linalg.inv(K)
    V = np.full((len(tri), 3, 3), np.nan)
    for i, m in enumerate(M):
        if np.isnan(m).any():
            continue
        Q = np.c_[P[tri[i]], np.ones(3)] @ Kinv.T
        V[i] = Q / (Q @ m)[:, None]
    N = -M / np.linalg.norm(M, axis=1, keepdims=True)
    return V, N


def vertices_from_points(P, tri, flow, K, R, t):
    """Baseline: each vertex triangulated on its own from the flow at the vertex."""
    Kinv = np.linalg.inv(K)
    H, W = flow.shape[:2]
    pts = np.clip(P.round().astype(int), 0, [W - 1, H - 1])
    x2 = P + flow[pts[:, 1], pts[:, 0]]
    q, q2 = np.c_[P, np.ones(len(P))] @ Kinv.T, np.c_[x2, np.ones(len(P))] @ Kinv.T
    X = triangulate_midpoint(q, q2, R, t)
    V = X[tri]
    N = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    N[np.einsum("ij,ij->i", N, V.mean(1)) > 0] *= -1
    return V, N


# --- LiDAR check --------------------------------------------------------------------------------------
def read_xyz(path):
    """x, y, z of a binary_compressed PCD whose fields may have different sizes (stored field after field)."""
    import struct
    raw = open(path, "rb").read()
    tag = b"DATA binary_compressed\n"
    start = raw.index(tag) + len(tag)
    header = raw[:start].decode()
    n = int(next(line for line in header.splitlines() if line.startswith("POINTS")).split()[1])
    csize, usize = struct.unpack("<II", raw[start:start + 8])
    data = vd.lzf_decompress(raw[start + 8:start + 8 + csize], usize)
    return np.frombuffer(data[:12 * n], np.float32).reshape(3, n).T


def lidar_in_camera(d, l2c):
    X = read_xyz(os.path.join(DATA, d["pointcloud_path"]))
    X = X[np.isfinite(X).all(1)]
    Xc = X @ l2c[:3, :3].T + l2c[:3, 3]
    return Xc[Xc[:, 2] > 1.0]


def evaluate(P, tri, lab, V, N, K, Xc, objects):
    H, W = lab.shape
    uv = (Xc @ K.T)[:, :2] / Xc[:, 2:]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    for x0, y0, x1, y1 in objects:
        ok &= ~((uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1))
    uv, Xc = uv[ok], Xc[ok]
    tid = lab[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    rows, normals = [], []
    for i in np.unique(tid[tid >= 0]):
        if np.isnan(V[i]).any():
            continue
        sel = tid == i
        X = Xc[sel]
        r = np.linalg.norm(X, axis=1)
        rays = X / r[:, None]
        n = np.cross(V[i, 1] - V[i, 0], V[i, 2] - V[i, 0])
        den = rays @ n
        s = np.where(np.abs(den) > 1e-9, (n @ V[i, 0]) / np.where(np.abs(den) > 1e-9, den, 1), np.nan)
        for a, b in zip(s, r):
            if np.isfinite(a) and a > 0:
                rows.append((b, abs(a - b) / b))
        if sel.sum() >= 6:
            c = X.mean(0)
            _, sv, Vt = np.linalg.svd(X - c)
            if sv[2] < 0.25 * sv[1]:                   # the LiDAR points are really planar
                normals.append((float(np.linalg.norm(c)),
                                float(np.degrees(np.arccos(min(1.0, abs(Vt[2] @ N[i])))))))
    return np.array(rows).reshape(-1, 2), np.array(normals).reshape(-1, 2)


# --- figures -------------------------------------------------------------------------------------------
def motion_figure(path, img1, img2, P, tri, flow, M, info):
    H, W = img1.shape[:2]
    pts = np.clip(P.round().astype(int), 0, [W - 1, H - 1])
    P2 = P + flow[pts[:, 1], pts[:, 0]]
    fig, axes = plt.subplots(1, 3, figsize=(21, 4.6))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    rng = np.random.default_rng(1)
    show = rng.choice(len(tri), min(350, len(tri)), replace=False)
    for i in show:
        color = "#2a78d6" if not np.isnan(M[i]).any() else "#eb6834"
        axes[0].fill(*P[tri[i]].T, facecolor="none", edgecolor=color, linewidth=0.6)
        axes[1].fill(*P2[tri[i]].T, facecolor="none", edgecolor=color, linewidth=0.6)
    for k in range(0, len(P), 9):
        axes[0].annotate("", xy=P2[k], xytext=P[k], arrowprops=dict(arrowstyle="->", color="#f2c94c", linewidth=0.6))
    axes[0].set_title("frame k: triangles and their optical flow (yellow)", fontsize=10, loc="left", color=INK)
    axes[1].set_title("frame k+Δ: the same triangles moved and deformed by the flow (blue = solved, orange = rejected)",
                      fontsize=10, loc="left", color=INK)
    par = info[:, 0]
    pm = np.full((H, W), np.nan)
    img = np.zeros((H, W), float)
    for i, t in enumerate(tri):
        cv2.fillConvexPoly(img, P[t].round().astype(np.int32), float(par[i]))
    a = axes[2].imshow(img, cmap="magma", vmin=0, vmax=25)
    plt.colorbar(a, ax=axes[2], fraction=0.03)
    axes[2].set_title(f"parallax per triangle (px, flow minus rotation); < {MIN_PARALLAX:.0f} px: depth not observable "
                      f"(focus of expansion)", fontsize=10, loc="left", color=INK)
    for ax in axes:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor="white")
    plt.close(fig)


def maps_figure(path, img1, P, tri, V, N, Xc, K, objects):
    H, W = img1.shape[:2]
    nmap = np.full((H, W, 3), 255, np.uint8)
    dmap = np.full((H, W), np.nan)
    for i, t in enumerate(tri):
        if np.isnan(V[i]).any():
            continue
        pts = P[t].round().astype(np.int32)
        cv2.fillConvexPoly(nmap, pts, tuple(int(v) for v in ((N[i] * 0.5 + 0.5) * 255).clip(0, 255)))
        tmp = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(tmp, pts, 1)
        dmap[tmp > 0] = np.linalg.norm(V[i].mean(0))
    uv = (Xc @ K.T)[:, :2] / Xc[:, 2:]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    fig, axes = plt.subplots(1, 3, figsize=(21, 4.6))
    axes[0].imshow(nmap)
    axes[0].set_title("normal of every triangle from its deformation (colour = x, y, z in the camera frame)",
                      fontsize=10, loc="left", color=INK)
    a = axes[1].imshow(dmap, cmap="viridis_r", vmin=0, vmax=60)
    axes[1].set_title("distance of every triangle from its deformation (m)", fontsize=10, loc="left", color=INK)
    axes[2].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) // 2 + 60)
    axes[2].scatter(uv[ok, 0], uv[ok, 1], c=np.linalg.norm(Xc[ok], axis=1), s=1, cmap="viridis_r", vmin=0, vmax=60)
    axes[2].set_title("LiDAR distance (m), same colour scale", fontsize=10, loc="left", color=INK)
    plt.colorbar(a, ax=axes[1], fraction=0.03)
    for ax in axes:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor="white")
    plt.close(fig)


def render_3d(path, img1, P, tri, V, N, Xc):
    """Textured triangles in the camera-k frame seen from above-behind and from the side, LiDAR as dots."""
    import scene_mesh as sm
    # camera frame (x right, y down, z forward) -> a z-up frame for viewing: X = x, Y = z, Z = -y
    to_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])
    Vu = V @ to_up.T
    Nu = N @ to_up.T
    labels = np.zeros(len(tri), int)
    rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    views = [("from above and behind the car", w3.look_at(np.array([0.0, -14.0, 16.0]), np.array([0.0, 22.0, -2.0]))),
             ("from the left side", w3.look_at(np.array([-28.0, 14.0, 6.0]), np.array([0.0, 18.0, -1.0])))]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    Lu = Xc @ to_up.T
    for c, (name, view) in enumerate(views):
        axes[0, c].imshow(sm.render_mesh(view, rgb, P, tri, Vu, Nu, labels, "texture"))
        axes[0, c].set_title(f"textured triangles placed by their deformation, {name}", fontsize=10, loc="left",
                             color=INK)
        im = sm.render_mesh(view, rgb, P, tri, Vu, Nu, labels, "normals")
        uv, dd = w3.project(view, Lu[::7])
        ok = (dd > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 720)
        axes[1, c].imshow(im)
        axes[1, c].scatter(uv[ok, 0], uv[ok, 1], s=0.3, c="k", alpha=0.35)
        axes[1, c].set_title(f"coloured by normal, black = LiDAR, {name}", fontsize=10, loc="left", color=INK)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def accuracy_figure(path, res):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    bins = ((0, 10), (10, 20), (20, 40), (40, 80))
    for j, (name, color) in enumerate((("three vertices triangulated", "#c3c2b7"), ("whole-triangle deformation", "#2a78d6"))):
        D, Nn = res[name]
        v = [100 * np.median(D[(D[:, 0] >= lo) & (D[:, 0] < hi), 1]) if ((D[:, 0] >= lo) & (D[:, 0] < hi)).any()
             else np.nan for lo, hi in bins]
        bars = axes[0].bar(np.arange(len(bins)) + (j - 0.5) * 0.38, v, 0.36, color=color, zorder=2, label=name)
        for b, x in zip(bars, v):
            if np.isfinite(x):
                axes[0].annotate(f"{x:.1f}", (b.get_x() + b.get_width() / 2, x), xytext=(0, 2),
                                 textcoords="offset points", ha="center", fontsize=8, color=INK)
        if len(Nn):
            axes[1].hist(Nn[:, 1], bins=np.linspace(0, 90, 31), color=color, alpha=0.7,
                         label=f"{name}: median {np.median(Nn[:, 1]):.1f}°, < 20° {np.mean(Nn[:, 1] < 20):.0%}")
    axes[0].set_xticks(range(len(bins)), [f"{lo}-{hi} m" for lo, hi in bins])
    axes[0].set_ylabel("median depth error vs LiDAR (% of distance)", color=INK)
    axes[0].set_title("3D position", fontsize=10.5, loc="left", color=INK)
    axes[0].legend(fontsize=8.5, frameon=False)
    axes[1].set_xlabel("angle between the triangle normal and the LiDAR plane (°)", color=INK)
    axes[1].set_title("normals (triangles with >= 6 planar LiDAR points)", fontsize=10.5, loc="left", color=INK)
    axes[1].legend(fontsize=8.5, frameon=False)
    for ax in axes:
        vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def synthetic_test():
    """Known plane, known motion, exact flow -> the solver must return the plane."""
    K = np.array([[800, 0, 960], [0, 800, 540], [0, 0, 1.0]])
    n, d = np.array([0.0, -1.0, 0.0]), 1.6                     # ground 1.6 m below the camera (y down)
    m = n / d
    a = np.radians(2)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    t = np.array([0.1, 0.0, -2.0])
    x = np.c_[np.random.default_rng(0).uniform(200, 1700, 50), np.random.default_rng(1).uniform(700, 1050, 50)]
    q = np.c_[x, np.ones(50)] @ np.linalg.inv(K).T
    X = q / (q @ m)[:, None]
    X2 = X @ R.T + t
    q2 = X2 / X2[:, 2:]
    est, rms = plane_from_flow(q, q2, R, t)
    return float(np.linalg.norm(est - m)), rms


def main():
    err, rms = synthetic_test()
    print(f"synthetic plane: |m - m_true| = {err:.2e}, residual {rms:.2e}", flush=True)
    info, K, D = load()
    poses = [pose(d) for d in info]
    centres = np.array([-np.linalg.inv(T[:3, :3]) @ T[:3, 3] for T, _ in poses])
    # frame pairs with 2-5 m of camera motion
    pairs = []
    for k in range(0, len(info) - 1, 6):
        for j in range(k + 1, min(k + 12, len(info))):
            b = np.linalg.norm(centres[j] - centres[k])
            if 2.0 <= b <= 5.0:
                pairs.append((k, j, b))
                break
    print(f"{len(pairs)} frame pairs with a 2-5 m baseline", flush=True)
    res = {"three vertices triangulated": ([], []), "whole-triangle deformation": ([], [])}
    shown = None
    stats = []
    for k, j, b in pairs:
        T1, l2c = poses[k]
        T2, _ = poses[j]
        Trel = T2 @ np.linalg.inv(T1)
        R, t = Trel[:3, :3], Trel[:3, 3]
        img1, img2 = frame(info[k], K, D), frame(info[j], K, D)
        g1, g2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        P, tri, lab = triangulation(g1)
        flow, ok = flows(g1, g2)
        objects = object_boxes(info[k])
        M, inf = solve_triangles(P, tri, lab, flow, ok, K, R, t, objects)
        Vd, Nd = vertices_from_planes(P, tri, M, K)
        Vp, Np = vertices_from_points(P, tri, flow, K, R, t)
        Vp[np.isnan(M).any(1)] = np.nan                   # same triangles for both methods
        Xc = lidar_in_camera(info[k], l2c)
        for name, V, N in (("whole-triangle deformation", Vd, Nd), ("three vertices triangulated", Vp, Np)):
            dep, nor = evaluate(P, tri, lab, V, N, K, Xc, objects)
            res[name][0].append(dep)
            res[name][1].append(nor)
        stats.append((len(tri), int((~np.isnan(M).any(1)).sum()), b))
        solved = int((~np.isnan(M).any(1)).sum())
        if shown is None or solved > shown[0]:
            shown = (solved, (img1, img2, P, tri, flow, M, inf, Vd, Nd, Xc, objects))
    res = {k: (np.vstack(v[0]), np.vstack(v[1])) for k, v in res.items()}
    metrics = {"synthetic_plane_error": err, "pairs": len(pairs),
               "triangles_per_pair": float(np.mean([s[0] for s in stats])),
               "solved_share": float(np.mean([s[1] / s[0] for s in stats]))}
    for name, (Dd, Nn) in res.items():
        metrics[name] = {"lidar_points": int(len(Dd)), "depth_err_median_pct": float(100 * np.median(Dd[:, 1])),
                         "depth_within_10pct": float(np.mean(Dd[:, 1] < 0.1)),
                         "normal_err_median_deg": float(np.median(Nn[:, 1])) if len(Nn) else None,
                         "normal_within_20deg": float(np.mean(Nn[:, 1] < 20)) if len(Nn) else None,
                         "planar_triangles": int(len(Nn))}
    print(json.dumps(metrics, indent=1), flush=True)
    json.dump(metrics, open(os.path.join(vd.OUT_DIR, "flowtri_metrics.json"), "w"), indent=1)
    img1, img2, P, tri, flow, M, inf, Vd, Nd, Xc, objects = shown[1]
    motion_figure(os.path.join(vd.OUT_DIR, "flowtri_motion.png"), img1, img2, P, tri, flow, M, inf)
    maps_figure(os.path.join(vd.OUT_DIR, "flowtri_maps.png"), img1, P, tri, Vd, Nd, Xc, K, objects)
    render_3d(os.path.join(vd.OUT_DIR, "flowtri_3d.png"), img1, P, tri, Vd, Nd, Xc)
    accuracy_figure(os.path.join(vd.OUT_DIR, "flowtri_accuracy.png"), res)


if __name__ == "__main__":
    main()
