"""Points of a pedestrian / cyclist triangulated over its whole pass, with the motion known from its cuboid track
(vru_lidar.py's poses): corners inside the masks tracked frame to frame (forward-backward checked), each track one
point fixed in the object's frame, solved by DLT over all its frames; the non-rigid parts (pedalling legs, spokes)
fail the reprojection test and drop out.  Compared with the object's own accumulated LiDAR points.

    V2X_SEQ_DIR=<seq> python tri_vru.py PKL OUT_PNG
"""
import os, sys, pickle
import numpy as np, cv2
from scipy.spatial import cKDTree
os.chdir(os.path.expanduser("~/Documents/MyArticles/code/3dseg")); sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import v2x_seq_shape as vs

PKL, OUT = sys.argv[1], sys.argv[2]
d = pickle.load(open(PKL, "rb"))
seq = vs.Sequence()
pose, L, W, H = d["pose"], *d["dims"]
ks = sorted(k for k in pose if d["masks"].get(k) is not None)
HH, WW = seq.shape


def triangulate_dlt(Ps, uv):
    """Linear (DLT) triangulation from F projection matrices (F, 3, 4) and pixels (F, 2): the point (3,) and its
    reprojection error per observation (F,) in pixels."""
    F = len(Ps)
    A = np.zeros((2 * F, 4))
    for i in range(F):
        A[2 * i] = uv[i, 0] * Ps[i][2] - Ps[i][0]
        A[2 * i + 1] = uv[i, 1] * Ps[i][2] - Ps[i][1]
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    Xh = np.linalg.svd(A, full_matrices=False)[2][-1]
    if abs(Xh[3]) < 1e-12:
        return np.full(3, np.nan), np.full(F, np.inf)
    X = Xh[:3] / Xh[3]
    q = Ps @ np.append(X, 1)
    return X, np.linalg.norm(q[:, :2] / q[:, 2:3] - uv, axis=1)


def full_mask(k):
    y0, x0, m = d["masks"][k]
    out = np.zeros((HH, WW), np.uint8); out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    return out


def P_obj(k):
    """3x4: object-frame point -> pixel in frame k (the object's planar pose, then the camera)."""
    K, R, t = seq.cams[k]
    p, a = pose[k]
    T = np.array([[np.cos(a), -np.sin(a), 0, p[0]], [np.sin(a), np.cos(a), 0, p[1]], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    return K @ np.c_[R, t] @ T


gray = {k: cv2.cvtColor(seq.frames[k], cv2.COLOR_BGR2GRAY) for k in ks}
masks = {k: full_mask(k) for k in ks}
lk = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
SPAN = int(os.environ.get("SPAN", "12"))                 # frames tracked on each side of a seed frame
tracks = []                                              # [{k: uv}], and the seed colour
for r in ks[::3]:
    er = cv2.erode(masks[r], np.ones((3, 3), np.uint8))
    P0 = cv2.goodFeaturesToTrack(gray[r], 400, 0.005, 3, mask=er)
    if P0 is None:
        continue
    P0 = P0.reshape(-1, 2).astype(np.float32)
    obs = [{r: p} for p in P0]
    for direction in (1, -1):
        cur, alive, kp = P0.copy(), np.ones(len(P0), bool), r
        for step in range(1, SPAN + 1):
            kn = r + direction * step
            if kn not in gray or not alive.any():
                break
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[kp], gray[kn], cur, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[kn], gray[kp], nxt, None, **lk)
            fb = np.linalg.norm(back - cur, axis=1)
            q = np.round(nxt).astype(int)
            inb = (q[:, 0] >= 0) & (q[:, 0] < WW) & (q[:, 1] >= 0) & (q[:, 1] < HH)
            inm = np.zeros(len(q), bool)
            inm[inb] = cv2.dilate(masks[kn], np.ones((5, 5), np.uint8))[q[inb, 1], q[inb, 0]] > 0
            alive &= (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5) & inm
            for i in np.nonzero(alive)[0]:
                obs[i][kn] = nxt[i].copy()
            cur, kp = nxt, kn
    col = seq.frames[r][np.round(P0[:, 1]).astype(int), np.round(P0[:, 0]).astype(int)]
    tracks += [(o, c) for o, c in zip(obs, col) if len(o) >= 7]

pts, cols, res_med = [], [], []
Pk = {k: P_obj(k) for k in ks}
for o, c in tracks:
    kk = sorted(o)
    X, res = triangulate_dlt(np.array([Pk[k] for k in kk]), np.array([o[k] for k in kk]))
    if not np.isfinite(X).all():
        continue
    res_med.append(np.median(res))
    inside = abs(X[0]) <= L / 2 + 0.2 and abs(X[1]) <= W / 2 + 0.2 and -0.1 <= X[2] <= H + 0.3
    if np.median(res) <= 1.0 and res.max() <= 3.0 and inside:
        pts.append(X); cols.append(c[::-1] / 255.0)
pts, cols = np.array(pts), np.array(cols)

# the viewing directions over the pass (camera azimuth in the object frame) - the baseline of the triangulation
az = []
for k in ks:
    K, R, t = seq.cams[k]; C = -np.linalg.solve(R, t); p, a = pose[k]
    c = (C[:2] - p) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    az.append(np.degrees(np.arctan2(c[1], c[0])))
cl = d["cl"]
dist = cKDTree(cl).query(pts)[0] if len(pts) else np.zeros(0)
print(f"{d['typ']}: {len(ks)} frames, camera azimuth in the object frame {min(az):.0f}..{max(az):.0f} deg; "
      f"{len(tracks)} tracks, {len(pts)} points kept (median reprojection <= 1 px, inside the box +0.2 m); "
      f"track residual median {np.median(res_med):.2f} px; distance to the LiDAR points median {np.median(dist):.3f} m, "
      f"p90 {np.percentile(dist, 90):.3f} m" if len(pts) else "no points")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (i, j, nm) in zip(axes, ((0, 2, "side (x along the travel, z up)"), (0, 1, "from above (x, y)"), (1, 2, "from the front (y, z)"))):
    ax.scatter(cl[:, i], cl[:, j], s=2, c="#b8b8b8", label="its LiDAR (accumulated, mirrored)")
    if len(pts):
        ax.scatter(pts[:, i], pts[:, j], s=6, c=cols, label="triangulated points")
    ax.set_aspect("equal"); ax.set_title(nm, fontsize=10, loc="left"); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle(f"{d['typ']}: points triangulated over the pass with the known motion ({len(pts)} points)", fontsize=11, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(OUT, dpi=90); print("->", OUT)

# --- the points as the near surface, the far one by symmetry: on the side plane (x along the travel, z up) the
# half-width |y| of the surface at every cell, the median of the triangulated points within R_CELL; cells without
# points take the nearest measured one within 0.2 m, farther ones a thin default (wheels, frame: a prior); the carve
# (the side silhouettes) bounded by +-that half-width
import whole_scene_cars as w
from scipy.ndimage import median_filter, distance_transform_edt
occ, (xs, ys, zs) = d["occ"], d["axes"]
R_CELL, THIN = float(os.environ.get("R_CELL", "0.06")), float(os.environ.get("THIN", "0.10"))
side = occ.any(1)                                          # the carve's side silhouette (x, z)
hw = np.full(side.shape, np.nan)
tree = cKDTree(pts[:, [0, 2]])
for ix, x in enumerate(xs):
    for iz, z in enumerate(zs):
        if side[ix, iz]:
            nb = tree.query_ball_point([x, z], R_CELL)
            if len(nb) >= 3:
                hw[ix, iz] = np.median(np.abs(pts[nb, 1]))
have = np.isfinite(hw)
dist_c, (jx, jz) = distance_transform_edt(~have, return_indices=True)
step_ = xs[1] - xs[0]
fill = np.where(dist_c * step_ <= 0.2, hw[jx, jz], THIN)
hw = np.where(have, hw, fill)
hw = median_filter(np.where(side, hw, 0.0), size=5)
occ2 = occ & (np.abs(ys)[None, :, None] <= hw[:, None, :] + 0.03)
print(f"surface cells measured {have.sum()} of {side.sum()} ({have.sum() / max(side.sum(), 1):.0%}); half-width median {np.median(hw[side]):.2f} m")
gs = 0.05; rx, ry = np.arange(-1.5, 1.5, gs) + gs / 2, np.arange(-1.0, 1.0, gs) + gs / 2
def hf(P_):
    Hm = np.zeros((len(rx), len(ry))); ix = np.floor((P_[:, 0] + 1.5) / gs).astype(int); iy = np.floor((P_[:, 1] + 1.0) / gs).astype(int)
    k_ = (ix >= 0) & (ix < len(rx)) & (iy >= 0) & (iy < len(ry)); np.maximum.at(Hm, (ix[k_], iy[k_]), P_[k_, 2]); return Hm
Lr = w.close_lidar_holes(hf(cl), gs, radius=0.08)
X3, Y3, Z3 = np.meshgrid(xs, ys, zs, indexing="ij")
for nm, o_ in (("carve", occ), ("carve + surface (symmetric)", occ2)):
    iou = w.shape_scores(hf(np.c_[X3[o_], Y3[o_], Z3[o_]]), rx, ry, Lr, rx, ry)["iou"]
    print(f"{nm:28s}: width {np.ptp(Y3[o_]) + (ys[1] - ys[0]):.2f} m, voxels {o_.sum()}, IoU vs its LiDAR {iou:.3f}")
np.save(OUT.replace(".png", "_occ.npy"), occ2)
