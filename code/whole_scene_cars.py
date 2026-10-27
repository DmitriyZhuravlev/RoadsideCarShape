"""
3D car models from the whole video by averaging the per-window reconstructions (no masks, no labels).

whole_scene.py solves every moving object in every window of +-8 frames on its own (DUMP=1 saves them). Here:
  1. tracks: objects of successive windows are linked by their ground position, predicted with their own solved
     velocity (Hungarian assignment inside a gate that grows with the range);
  2. car frame: the track's footprint positions are smoothed over time; the heading is the direction of the smoothed
     path (a car drives along its heading); every window's points are moved into the car frame (x forward, y left,
     z up), then refined by trimmed ICP on the ground plane (x, y, yaw) against the model of all other windows;
  3. symmetry: the car's centre plane y = y0 is the one that best maps the points onto their own mirror image;
     after centring, every point is also used mirrored (the unseen side);
  4. averaging: in every 10 cm column of the car frame each window gives the height of its highest point there; the
     model height is the MEDIAN over the windows that saw the column (at least MIN_VIEWS of them); colour = median
     colour of those top points; columns lower than 0.25 m are ground;
  5. mesh: the height field filled down to the ground, marching cubes, Taubin smoothing.
Checks (evaluation only - the labels and the LiDAR never enter 1-5): length / width / height against the labelled
box; the car's position and heading per window against the label; the shape against the accumulated, mirrored LiDAR
of the same car (lidar_masks): surface distances both ways and the IoU of the solids, after centring, also against
the LiDAR field with its scan holes closed ("fused closed"); and the same
for the best single window without averaging and without symmetry.

Writes docs/whole_scene_cars.png, docs/whole_scene_cars.json and docs/cars_viewer/meshes.json.
"""
import glob
import json
import os
import pickle

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import grey_closing, median_filter, label as cc_label
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from skimage.measure import marching_cubes

import mesh_refine as mr
import v2x_seq_demo as vd
import v2x_seq_shape as vs

CELL = 0.1                                   # m
MIN_VIEWS = 2
MIN_TRACK = 5                                # windows
GATE, GATE_REL, MAX_GAP = 2.5, 0.04, 6       # m, share of the range, frames
GROUND_H = 0.25
VOTE = float(os.environ.get("VOTE", "0.2"))      # share of the windows that must see a column
DENS_REL = float(os.environ.get("DENS_REL", "0.25"))
SYMMETRY = os.environ.get("SYMMETRY", "1") == "1"
SYM_SEARCH = float(os.environ.get("SYM_SEARCH", "1.2"))   # m: how far the symmetry plane may sit from the centre
VIDEO = os.environ.get("VIDEO", "1") == "1"
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")
RIDE_TABLE_OUT = os.environ.get("RIDE_TABLE_OUT", "")
CABIN = os.environ.get("CABIN", "1") == "1"         # the cabin from the full silhouettes (visual hull)
CARVE_VOTE = float(os.environ.get("CARVE_VOTE", "0.95"))   # share of the windows that must see a voxel inside
CARVE_DILATE = int(os.environ.get("CARVE_DILATE", "0"))      # cells the body footprint is widened for the cabin
CARVE_FOOT = os.environ.get("CARVE_FOOT", "0") == "1"        # restrict the visual hull to the body footprint
CARVE_ZMAX = float(os.environ.get("CARVE_ZMAX", "1.9"))   # m: the silhouette cone smears up toward the camera
LEN_MAX = float(os.environ.get("LEN_MAX", "0"))           # m: trim longer cars at the camera end (0: off)
BODY_FIELD = os.environ.get("BODY_FIELD", "1") == "1"        # blend the triangulated body field in (0: hull only)
# lower edge of the colour body above the road (no tyres, the darker lower part falls out of the body cluster), m
CLASSIFY = os.environ.get("CLASSIFY", "image")    # body type from the side-view silhouette, or "3d" profile
TYPE_RIDE = {"sedan": 0.40, "hatchback": 0.40, "SUV": 0.52, "van/MPV": 0.48, "unknown": 0.45}
TYPE_W = {"sedan": 1.90, "hatchback": 1.85, "SUV": 1.95, "van/MPV": 2.05, "unknown": 2.10}
TYPE_WIDTH = os.environ.get("TYPE_WIDTH", "1") == "1"      # clip the width to the body type's prior
RANGE_HULL = float(os.environ.get("RANGE_HULL", "40"))     # m: nearer tracks keep the silhouette hull only
W_MEDIAN = os.environ.get("W_MEDIAN", "1") == "1"          # weight windows by their reprojection error
SIL_SNAP = os.environ.get("SIL_SNAP", "1") == "1"          # snap silhouettes to image edges (sub-pixel hull)
ICP_ROUNDS = int(os.environ.get("ICP_ROUNDS", "3"))
MATCH_GATE = float(os.environ.get("MATCH_GATE", "3.0"))   # m: evaluation only, label match per window
SIL_POSE = os.environ.get("SIL_POSE", "0") == "1"          # refine window poses to the silhouettes (off: worse in 3D)
SIL_SEARCH = tuple(float(v) for v in os.environ.get("SIL_SEARCH", "1.0,2.0,4.0").split(","))   # m along, m across, deg
SIL_SEARCH = (SIL_SEARCH[0], SIL_SEARCH[1], np.radians(SIL_SEARCH[2]))  # search half-widths of the pose refinement
SIL_SMOOTH = 5                                             # windows: running median of the pose corrections
SIL_SIDE = float(os.environ.get("SIL_SIDE", "0.7"))        # refine only windows that see the car this side-on
CAR_COLOUR = os.environ.get("CAR_COLOUR", "best")  # "best": each vertex the original pixel of its best view; "blend": cos^4 average
PANELS = os.environ.get("PANELS", "whole_scene_video_full_*")
W_MAX = 2.1                                  # m: widest car (class prior used with the symmetry plane)
INK, MUTED = "#1f1f1e", "#8a8984"
DUMPS = os.environ.get("DUMPS", os.path.join(vd.OUT_DIR, "whole_scene_objects"))


# --- 1. tracks --------------------------------------------------------------------------------------------------
def load_dets():
    frames = []
    for path in sorted(glob.glob(os.path.join(DUMPS, "*.pkl"))):
        dets = pickle.load(open(path, "rb"))
        for d in dets:
            body = d["body"] & ~d["filled"]
            d["c"] = np.median(d["X"][body, :2], axis=0) if body.any() else np.median(d["X"][:, :2], axis=0)
            d["range"] = float(np.linalg.norm(d["c"] - d["camera"][:2]))
        k = int(os.path.basename(path)[:4])
        frames.append((k, dets))
    return frames


def link_tracks(frames):
    """Hungarian linking frame by frame; a track predicts with the velocity of its last detection."""
    open_t, done = [], []
    for k, dets in frames:
        still = []
        for t in open_t:
            (still if k - t[-1][0] <= MAX_GAP else done).append(t)
        open_t = still
        if not dets:
            continue
        cost = np.full((len(open_t), len(dets)), 1e6)
        for i, t in enumerate(open_t):
            k0, d0 = t[-1]
            pred = d0["c"] + d0["vel"] * (k - k0)
            for j, d in enumerate(dets):
                c = np.linalg.norm(pred - d["c"])
                if c <= GATE + GATE_REL * d["range"]:
                    cost[i, j] = c
        used = set()
        if len(open_t):
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e6:
                    open_t[i].append((k, dets[j]))
                    used.add(j)
        open_t += [[(k, d)] for j, d in enumerate(dets) if j not in used]
    done += open_t
    return sorted([t for t in done if len(t) >= MIN_TRACK], key=len, reverse=True)


# --- 2. car frame ------------------------------------------------------------------------------------------------
def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def smooth_path(track):
    """Smoothed footprint positions and headings (direction of the smoothed path) of every window."""
    k = np.array([t[0] for t in track], float)
    c = np.array([t[1]["c"] for t in track])
    deg = min(3, len(k) - 1)
    pc = [np.polyfit(k, c[:, a], deg) for a in (0, 1)]
    pos = np.stack([np.polyval(p, k) for p in pc], 1)
    # heading: the solved velocities of the windows (joint fits of many tracks), smoothed over time; the path of the
    # footprint centre moves with the visible part of the car and is only the fallback
    v = np.array([t[1]["vel"] for t in track])
    vs_ = np.stack([np.polyval(np.polyfit(k, v[:, a], min(2, len(k) - 1)), k) for a in (0, 1)], 1)
    d = np.stack([np.polyval(np.polyder(p), k) for p in pc], 1)
    d = np.where(np.linalg.norm(vs_, axis=1, keepdims=True) > 0.05, vs_, d)
    return pos, np.arctan2(d[:, 1], d[:, 0])


def to_car(X, pos, th):
    return np.c_[(X[:, :2] - pos) @ rot(th), X[:, 2]]


def window_points(det, spacing=0.05):
    """Dense samples of the window's surface: every triangle of the car body (not the mirror image) sampled at
    about `spacing`, colours interpolated (the vertices alone cover only a few cells of the height field)."""
    T = det["tri"]
    on = det["body"][T].all(1)
    T = T[on]
    tri_col = det["tri_bgr"][on].astype(float) if "tri_bgr" in det else None   # mean image colour per triangle
    X, col = det["X"], det["bgr"].astype(float)
    if not len(T):
        return X[det["body"]], col[det["body"]]
    A, B, C = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    ok = area < 1.5                                            # long slivers across the outline are no surface
    n = np.clip(np.ceil(area / spacing ** 2), 1, 400).astype(int) * ok
    idx = np.repeat(np.arange(len(T)), n)
    rng = np.random.default_rng(0)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    w = np.c_[1 - u - v, u, v]
    P = (w[:, :, None] * X[T[idx]]).sum(1)
    Cc = tri_col[idx] if tri_col is not None else (w[:, :, None] * col[T[idx]]).sum(1)
    return np.vstack([P, X[det["body"]]]), np.vstack([Cc, col[det["body"]]])


def icp_2d(A, B, iters=8, trim=0.7, max_d=0.6):
    """Planar rigid (x, y, yaw) alignment of the points A (n, 3) onto the model B (m, 3), heights compared too."""
    tree = cKDTree(B)
    th, t = 0.0, np.zeros(2)
    for _ in range(iters):
        A2 = np.c_[A[:, :2] @ rot(th).T + t, A[:, 2]]
        d, j = tree.query(A2)
        ok = d < max_d
        if ok.sum() < 20:
            break
        ok &= d <= np.quantile(d[ok], trim)
        P, Q = A[ok, :2], B[j[ok], :2]
        cp, cq = P.mean(0), Q.mean(0)
        H = (P - cp).T @ (Q - cq)
        th = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
        t = cq - rot(th) @ cp
    return th, t


def track_in_car_frame(track):
    pos, head = smooth_path(track)
    parts = []
    for (k, d), p, h in zip(track, pos, head):
        X, col = window_points(d)
        # every window centred on its own surface: the along-ray depth error of a window (metres at 60-70 m) would
        # otherwise smear the car sideways; the view-dependent part is left to the ICP below
        Y = to_car(X, d["c"], h)
        med = np.median(Y[:, :2], axis=0)
        ok = np.all(np.abs(Y[:, :2] - med) < [3.2, 2.0], axis=1)            # outliers of the window
        A = rot(h).T
        parts.append([Y[ok], col[ok], k, A, -A @ d["c"]])                   # car xy = A @ world xy + b
    for _ in range(ICP_ROUNDS):
        for i in range(len(parts)):
            others = np.vstack([q[0] for j, q in enumerate(parts) if j != i])
            th, t = icp_2d(parts[i][0], others)
            if abs(th) < np.radians(15) and np.linalg.norm(t) < 2.5:
                parts[i][0] = np.c_[parts[i][0][:, :2] @ rot(th).T + t, parts[i][0][:, 2]]
                parts[i][3] = rot(th) @ parts[i][3]
                parts[i][4] = rot(th) @ parts[i][4] + t
    return parts, pos, head


# --- 3. symmetry -------------------------------------------------------------------------------------------------
def symmetry_plane(Y, search=1.2, step=0.02):
    """y0 that maps the points best onto their own mirror image (share of mirrored points with a neighbour)."""
    tree = cKDTree(Y)
    best, y_best = -1, 0.0
    for y0 in np.arange(-search, search + 1e-9, step):
        M = Y * [1, -1, 1] + [0, 2 * y0, 0]
        s = np.mean(tree.query(M, distance_upper_bound=0.12)[0] < 0.12)
        if s > best:
            best, y_best = s, y0
    return y_best, best


# --- 4. averaging ------------------------------------------------------------------------------------------------
def wmedian(T, w):
    """Per-column weighted median of T (m, n) with weights w (m,); NaN entries ignored, NaN where no data."""
    order = np.argsort(T, axis=0, kind="stable")          # NaNs sort last
    Tw = np.take_along_axis(T, order, 0)
    ws = np.where(np.isfinite(Tw), w[order], 0.0)         # a window that did not see the cell carries no weight
    cw = np.cumsum(ws, 0)
    tot = cw[-1]
    idx = np.argmax(cw >= 0.5 * tot, 0)                   # first crossing of half the total weight
    return np.where(tot > 0, Tw[idx, np.arange(T.shape[1])], np.nan)


def height_field(parts, mirror, min_views=MIN_VIEWS, w=None):
    """Per column the median over windows of each window's highest point; colours alike."""
    allp = np.vstack([p[0] for p in parts])
    lo = np.percentile(allp[:, :2], 0.5, axis=0) - 0.6
    hi = np.percentile(allp[:, :2], 99.5, axis=0) + 0.6
    if mirror:
        m = max(abs(lo[1]), abs(hi[1]))
        lo[1], hi[1] = -m, m
    nx, ny = (int(np.ceil((hi[a] - lo[a]) / CELL)) for a in (0, 1))
    tops = []
    cols = []
    dens = np.zeros(nx * ny)
    for P, col, *_ in parts:
        Ps, Cs = (np.vstack([P, P * [1, -1, 1]]), np.vstack([col, col])) if mirror else (P, col)
        ix = np.floor((Ps[:, :2] - lo) / CELL).astype(int)
        ok = (ix[:, 0] >= 0) & (ix[:, 1] >= 0) & (ix[:, 0] < nx) & (ix[:, 1] < ny)
        top = np.full(nx * ny, np.nan)
        flat = ix[ok, 0] * ny + ix[ok, 1]
        z = Ps[ok, 2]
        order = np.argsort(-z)                   # highest first: np.unique keeps each column's first = highest
        cell, first, cnt = np.unique(flat[order], return_index=True, return_counts=True)
        dens[cell] += cnt / max(len(flat), 1)                      # every window has the same weight
        top[cell] = z[order][first]
        c = np.full((nx * ny, 3), np.nan)
        c[cell] = Cs[ok][order][first]
        tops.append(top)
        cols.append(c)
    T = np.stack(tops)
    views = np.sum(~np.isnan(T), axis=0)
    with np.errstate(all="ignore"):
        Hm = np.where(views >= min_views, wmedian(T, w) if w is not None else np.nanmedian(T, axis=0), np.nan)
    with np.errstate(all="ignore"):
        Cm = np.nanmedian(np.stack(cols), axis=0)
    Hm = Hm.reshape(nx, ny)
    Cm = Cm.reshape(nx, ny, 3)
    # the true surface collects points from every window, the along-ray depth noise spreads them thin: keep the
    # cells whose density is at least DENS_REL of the densest cell of the same cross-section of the car
    D = median_filter(dens.reshape(nx, ny), size=3)
    Hm = np.where(D >= DENS_REL * D.max(axis=1, keepdims=True), Hm, np.nan)
    Hm = np.where(np.isnan(Hm), 0.0, Hm)
    Hm = median_filter(Hm, size=3)
    occ = Hm > GROUND_H
    lab, n = cc_label(occ)
    if n > 1:
        big = 1 + np.argmax(np.bincount(lab.ravel())[1:])
        occ = lab == big
    Hm = np.where(occ, Hm, 0.0)
    xs = lo[0] + (np.arange(nx) + 0.5) * CELL
    ys = lo[1] + (np.arange(ny) + 0.5) * CELL
    return Hm, Cm, xs, ys, views.reshape(nx, ny)


def convex_hull_model(Hm, Cm, xs, ys):
    """The car as the convex hull of the averaged cells: every occupied cell's top point and its footprint on the
    road. Cars are nearly convex; the hull is closed, has no holes and ignores the noise inside the body.
    (V, F, C per triangle in BGR, ConvexHull)"""
    occ = Hm > GROUND_H
    if occ.sum() < 8:
        return None
    ix, iy = np.nonzero(occ)
    top = np.c_[xs[ix], ys[iy], Hm[ix, iy]]
    pts = np.vstack([top, top * [1, 1, 0]])
    h = ConvexHull(pts)
    V = pts
    F = h.simplices
    cen = V[F].mean(1)
    jx = np.clip(np.round((cen[:, 0] - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1)
    jy = np.clip(np.round((cen[:, 1] - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1)
    C = Cm[jx, jy]
    fill = np.nanmedian(Cm[occ], axis=0) if np.isfinite(Cm[occ]).any() else np.array([140.0, 140, 140])
    C = np.where(np.isfinite(C), C, fill)
    used, inv = np.unique(F, return_inverse=True)
    return V[used], inv.reshape(-1, 3), C, h


def coloured_hull(hull, parts, mirror, levels=2, reach=0.3):
    """The hull subdivided into small triangles (1-to-4, `levels` times), every triangle coloured by the average
    (median) colour of the surface samples of ALL windows near it (and their mirror images); triangles nobody saw
    take the colour of the nearest coloured one."""
    V, F, C, h = hull
    for _ in range(levels):
        V, F = mr.subdivide(V, F)
    P = np.vstack([p[0] for p in parts])
    Q = np.vstack([p[1] for p in parts]).astype(float)
    if mirror:
        P, Q = np.vstack([P, P * [1, -1, 1]]), np.vstack([Q, Q])
    cen = V[F].mean(1)
    d, j = cKDTree(cen).query(P, distance_upper_bound=reach)
    ok = np.isfinite(d)
    col = np.full((len(F), 3), np.nan)
    order = np.argsort(j[ok], kind="stable")
    jj, qq = j[ok][order], Q[ok][order]
    if len(jj):
        starts = np.r_[0, np.nonzero(np.diff(jj))[0] + 1]
        for a, b in zip(starts, np.r_[starts[1:], len(jj)]):
            col[jj[a]] = np.median(qq[a:b], axis=0)
    seen = np.isfinite(col[:, 0])
    if seen.any() and (~seen).any():
        col[~seen] = col[seen][cKDTree(cen[seen]).query(cen[~seen])[1]]
    col[~np.isfinite(col[:, 0])] = 140.0
    return V, F, col


# --- the cabin: visual hull of the full silhouettes; colours: video pixels ---------------------------------------------
def snap_silhouette(gray, sil, reach=3.0):
    """Snap a background-subtraction silhouette onto the strongest nearby image edge: every contour point moves
    +-reach px along its normal to the local gradient maximum (the blob boundary is threshold-quantized by 1-2 px)."""
    m0 = sil.astype(np.uint8)
    if int(m0.sum()) < 100 or not SIL_SNAP:
        return sil
    H, W = m0.shape
    g = np.hypot(*np.gradient(cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)))
    out = np.zeros_like(m0)
    for c in cv2.findContours(m0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
        P = c[:, 0].astype(float)
        if len(P) < 12:
            cv2.fillPoly(out, [c], 1)
            continue
        d = np.roll(P, -1, 0) - np.roll(P, 1, 0)
        n = np.c_[-d[:, 1], d[:, 0]]
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        best, bg = P.copy(), np.full(len(P), -1.0)
        for s in np.linspace(-reach, reach, 13):
            Q = P + s * n
            qi = np.clip(np.round(Q).astype(int), 0, [W - 1, H - 1])
            v = g[qi[:, 1], qi[:, 0]]
            up = v > bg
            bg, best = np.where(up, v, bg), np.where(up[:, None], Q, best)
        cv2.fillPoly(out, [np.round(best).astype(np.int32)], 1)
    return out > 0


def full_silhouette(seq, k, d):
    """The car's full silhouette in frame k (windows, roof and wheels included): the foreground against the video's
    median background around the body, the component holding the body, wet-road reflection and cast shadow cut.
    (mask crop, X0, Y0) or None."""
    import ipm_track_models as itm_
    import v2x_seq_classical as vc_
    frame = seq.frames[k]
    uv = d["uv"][d["body"]]
    x0, y0 = np.floor(uv.min(0)).astype(int)
    x1, y1 = np.ceil(uv.max(0)).astype(int)
    hgt = y1 - y0
    X0, Y0 = max(0, x0 - 20), max(0, y0 - 2 * hgt)
    X1, Y1 = min(frame.shape[1], x1 + 20), min(frame.shape[0], y1 + hgt)
    fg = vc_.foreground(frame[Y0:Y1, X0:X1], seq.background(k)[Y0:Y1, X0:X1])
    n, lab, _, _ = cv2.connectedComponentsWithStats(fg)
    core = lab[np.clip(uv[:, 1].astype(int) - Y0, 0, Y1 - Y0 - 1), np.clip(uv[:, 0].astype(int) - X0, 0, X1 - X0 - 1)]
    core = core[core > 0]
    if n < 2 or not len(core):
        return None
    b = vc_.make_blob((lab == np.bincount(core).argmax()).astype(np.uint8), X0, Y0)
    if b is None:
        return None
    b = vc_.cut_reflection(b, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    bx0, by0, bx1, by1 = b["box"]
    sil = itm_.shadow_free(frame[by0:by1, bx0:bx1], seq.background(k)[by0:by1, bx0:bx1], b["m"] > 0)
    sil = snap_silhouette(cv2.cvtColor(frame[by0:by1, bx0:bx1], cv2.COLOR_BGR2GRAY), sil)
    return sil, bx0, by0


def to_world(f, k, Y):
    """Car-frame points -> world points of window k (planar pose of that window)."""
    A, b = f["poses"][k]
    return np.c_[(Y[:, :2] - b) @ A, Y[:, 2]]


def carve(seq, f, sils, vote=CARVE_VOTE, zmax=CARVE_ZMAX, use_foot=None):
    """Visual hull in the car frame: a voxel stays when it projects inside the car's silhouette in at least `vote` of
    the windows that see it (it and its mirror image - the car is symmetric), inside the averaged body's footprint
    (dilated) - the silhouettes of a few similar views smear along the viewing ray, the footprint cuts that off.
    Returns the carved height field on the averaged grid."""
    xs, ys = f["xs"], f["ys"]
    zs = np.arange(CELL / 2, zmax, CELL)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    hits, seen = np.zeros(len(P)), np.zeros(len(P))
    for k, sl in sils.items():
        if sl is None or k not in f["poses"]:
            continue
        m, x0, y0 = sl
        for Q in (P, P * [1, -1, 1]):
            q, z = vs.project(seq.cams[k], to_world(f, k, Q))
            u, v = np.round(q[:, 0] - x0).astype(int), np.round(q[:, 1] - y0).astype(int)
            inb = (z > 0) & (u >= 0) & (v >= 0) & (u < m.shape[1]) & (v < m.shape[0])
            seen += inb
            hits[inb] += m[v[inb], u[inb]]
    occ = (seen > 0) & (hits >= vote * np.maximum(seen, 1))
    occ = occ.reshape(X.shape)
    use_foot = CARVE_FOOT if use_foot is None else use_foot
    if use_foot:
        foot = f["H"] > GROUND_H
        foot = foot | foot[:, ::-1]
        from scipy.ndimage import binary_dilation
        if CARVE_DILATE:
            foot = binary_dilation(foot, iterations=CARVE_DILATE)
        occ &= foot[:, :, None]
    Hc = np.where(occ.any(2), (len(zs) - np.argmax(occ[:, :, ::-1], axis=2)) * CELL, 0.0)
    Hc = median_filter(Hc, size=3)
    lab_, n = cc_label(Hc > GROUND_H)
    if n > 1:
        Hc = np.where(lab_ == 1 + np.argmax(np.bincount(lab_.ravel())[1:]), Hc, 0.0)
    if LEN_MAX:
        Hc = trim_camera_end(seq, f, Hc, LEN_MAX)
    return Hc


def trim_camera_end(seq, f, Hc, lmax):
    """The silhouette cone of a car seen along its length smears toward the camera: a car longer than lmax is cut at
    its camera end (the far end is bounded by the rays grazing it). Trims the carved field and the body field."""
    xs = f["xs"]
    cx = [(A @ -np.linalg.solve(seq.cams[k][1], seq.cams[k][2])[:2] + b)[0] for k, (A, b) in f["poses"].items()]
    occ = (Hc > GROUND_H) | (f["H"] > GROUND_H)
    ix = np.nonzero(occ.any(1))[0]
    if not len(ix) or xs[ix[-1]] - xs[ix[0]] + CELL <= lmax:
        return Hc
    if np.median(cx) > 0:
        keep = xs <= xs[ix[0]] + lmax - CELL / 2
    else:
        keep = xs >= xs[ix[-1]] - lmax + CELL / 2
    f["H"] = np.where(keep[:, None], f["H"], 0.0)
    return np.where(keep[:, None], Hc, 0.0)


def mask_iou_of_polygon(points_uv, sil, x0, y0, margin=40):
    """IoU between the convex hull of projected points and a silhouette crop (top-left pixel at x0, y0)."""
    if len(points_uv) < 3:
        return 0.0
    h, w = sil.shape
    canvas = np.zeros((h + 2 * margin, w + 2 * margin), np.uint8)
    canvas[margin:margin + h, margin:margin + w] = sil.astype(np.uint8)
    hull = np.zeros_like(canvas)
    q = points_uv - np.array([x0 - margin, y0 - margin])
    cv2.fillConvexPoly(hull, cv2.convexHull(np.round(q).astype(np.int32)), 1)
    inter = np.sum(canvas & hull)
    return float(inter) / max(float(np.sum(canvas | hull)), 1.0)


def model_hull_points(H, xs, ys):
    """The corners of the model's convex hull: top and foot of every occupied column."""
    ix, iy = np.nonzero(H > GROUND_H)
    if len(ix) < 4:
        return None
    top = np.c_[xs[ix], ys[iy], H[ix, iy]]
    P = np.vstack([top, top * [1, 1, 0]])
    return P[ConvexHull(P).vertices]


def refine_poses(seq, f, sils, H):
    """Every window's planar pose moved so that the model's projection overlaps the car's silhouette best (IoU of the
    masks): the ICP poses align point clouds whose depth along the viewing ray is the least certain; the silhouette
    and the model's contact with the road fix it. Coarse-to-fine search over (along, across, yaw) in the car frame,
    then a running median of the corrections over time (the car moves smoothly). Returns (IoU before, after)."""
    P = model_hull_points(H, f["xs"], f["ys"])
    if P is None:
        return None
    # only windows that see the car from the side: there the silhouette's contact line with the road is long and fixes
    # the position along the viewing ray; seen end-on the silhouette is long along the ray and would pull the car
    side = {}
    for (k, _), p, h in zip(f["track"], f["pos"], f["head"]):
        C = -np.linalg.solve(seq.cams[k][1], seq.cams[k][2])[:2]
        side[k] = abs(np.sin(h - np.arctan2(p[1] - C[1], p[0] - C[0])))
    ks = sorted(k for k in f["poses"] if sils.get(k) is not None and side.get(k, 0) >= SIL_SIDE)
    corr, before = {}, []
    for k in ks:
        m, x0, y0 = sils[k]
        A, b = f["poses"][k]

        def score(d):
            c, s_ = np.cos(d[2]), np.sin(d[2])
            Q = np.c_[P[:, :2] @ np.array([[c, -s_], [s_, c]]).T + d[:2], P[:, 2]]
            return mask_iou_of_polygon(vs.project(seq.cams[k], to_world(f, k, Q))[0], m, x0, y0)

        best = np.zeros(3)
        s0 = score(best)
        before.append(s0)
        span = np.array(SIL_SEARCH)
        for _ in range(4):                                   # coarse to fine
            grid = [best + span * np.array([a, c, e]) for a in (-1, -0.5, 0, 0.5, 1) for c in (-1, -0.5, 0, 0.5, 1)
                    for e in (-1, 0, 1)]
            sc = [score(d) for d in grid]
            best = grid[int(np.argmax(sc))]
            span = span / 2
        corr[k] = best if score(best) > s0 + 0.01 else np.zeros(3)
    if not ks:
        return None
    C = np.array([corr[k] for k in ks])
    C = np.stack([median_filter(C[:, a], size=min(SIL_SMOOTH, len(ks)), mode="nearest") for a in range(3)], 1)
    after = []
    for k, d in zip(ks, C):
        c, s_ = np.cos(d[2]), np.sin(d[2])
        R = np.array([[c, -s_], [s_, c]])
        A, b = f["poses"][k]
        f["poses"][k] = (R.T @ A, R.T @ (b - d[:2]))          # car = R^T (A world + b - t)
        m, x0, y0 = sils[k]
        after.append(mask_iou_of_polygon(vs.project(seq.cams[k], to_world(f, k, P))[0], m, x0, y0))
    return float(np.mean(before)), float(np.mean(after))


def orient_outward(V, F):
    """Every triangle of a closed convex mesh facing outward."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    inward = np.sum(n * (V[F].mean(1) - V.mean(0)), axis=1) < 0
    F = F.copy()
    F[inward, 1], F[inward, 2] = F[inward, 2], F[inward, 1]
    return F


def pixel_colours(seq, f, sils, V, F, min_cos=0.15, sharp=4.0):
    if CAR_COLOUR == "best":
        return best_view_colours(seq, f, sils, V, F, min_cos)
    return blend_colours(seq, f, sils, V, F, min_cos, sharp)


def best_view_colours(seq, f, sils, V, F, min_cos=0.15):
    """Colour of every vertex = the ORIGINAL PIXEL of the one window that sees it most directly and largest (cos of the
    angle to the camera over the distance), where it faces the camera, is not hidden behind another part of the car
    (z-buffer of the model's own vertices) and lies inside the car's silhouette; sampled bilinearly, not averaged, so
    the video's colours stay as they are. A vertex no window sees takes its mirror image's (the car is symmetric); still
    unseen: the nearest seen vertex's."""
    N = mr.vertex_normals(V, F)
    edge = float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)))
    best = np.full((2, len(V)), -1.0)
    col = np.zeros((2, len(V), 3))
    for k, sl in sils.items():
        if sl is None or k not in f["poses"]:
            continue
        m, x0, y0 = sl
        A, b = f["poses"][k]
        K, R, t = seq.cams[k]
        Cc = -np.linalg.solve(R, t)
        for i, sign in enumerate((1, -1)):
            Q, Nq = V * [1, sign, 1], N * [1, sign, 1]
            W = to_world(f, k, Q)
            Nw = np.c_[Nq[:, :2] @ A, Nq[:, 2]]
            d = Cc - W
            dist = np.linalg.norm(d, axis=1)
            cosang = np.sum(Nw * d, axis=1) / dist
            q, z = vs.project(seq.cams[k], W)
            r = int(np.ceil(edge * K[0, 0] / np.median(z[z > 0]) / 2)) + 1 if (z > 0).any() else 1
            ok = (cosang > min_cos) & vs.visible(q, z, seq.shape, r, 1.5 * edge)
            mu, mv = np.round(q[:, 0]).astype(int) - x0, np.round(q[:, 1]).astype(int) - y0
            ok &= (mu >= 0) & (mv >= 0) & (mu < m.shape[1]) & (mv < m.shape[0])
            ok[ok] = m[mv[ok], mu[ok]] > 0
            score = cosang / dist
            upd = ok & (score > best[i])
            if upd.any():
                col[i, upd] = vs.bilinear(seq.frames[k], q[upd])
                best[i, upd] = score[upd]
    own = best[0] >= 0
    out = np.where(own[:, None], col[0], col[1])
    seen = own | (best[1] >= 0)
    if not seen.any():
        return np.full((len(V), 3), 150.0)
    if not seen.all():
        from scipy.spatial import cKDTree
        out[~seen] = out[seen][cKDTree(V[seen]).query(V[~seen])[1]]
    return out


def blend_colours(seq, f, sils, V, F, min_cos=0.15, sharp=4.0):
    """Colour of every vertex = the VIDEO PIXELS of every window that sees it, weighted by cos^sharp of the angle
    between its normal and the direction to the camera (head-on views dominate, single-pixel noise cancels); a
    vertex no window sees takes its mirror image's colour (the car is symmetric); still unseen: grey."""
    N = mr.vertex_normals(V, F)
    col = np.full((len(V), 3), np.nan)
    for mirror in (False, True):
        todo = np.isnan(col[:, 0])
        if not todo.any():
            break
        Q = V[todo] * ([1, -1, 1] if mirror else 1)
        Nq = N[todo] * ([1, -1, 1] if mirror else 1)
        wsum = np.zeros(len(Q))
        csum = np.zeros((len(Q), 3))
        for k, sl in sils.items():
            if sl is None or k not in f["poses"]:
                continue
            m, x0, y0 = sl
            A, b = f["poses"][k]
            W = to_world(f, k, Q)
            Nw = np.c_[Nq[:, :2] @ A, Nq[:, 2]]
            K, R, t = seq.cams[k]
            Cc = -np.linalg.solve(R, t)
            d = Cc - W
            cosang = np.sum(Nw * d, axis=1) / np.linalg.norm(d, axis=1)
            q, z = vs.project(seq.cams[k], W)
            u, v = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
            mu, mv = u - x0, v - y0
            inside = (z > 0) & (mu >= 0) & (mv >= 0) & (mu < m.shape[1]) & (mv < m.shape[0])
            inside[inside] = m[mv[inside], mu[inside]]
            w = np.where(inside, np.clip(cosang - min_cos, 0.0, None) ** sharp, 0.0)
            if w.any():
                uc = np.clip(u, 0, seq.shape[1] - 1)
                vc = np.clip(v, 0, seq.shape[0] - 1)
                px = seq.frames[k][vc, uc].astype(float)
                wsum += w
                csum += w[:, None] * px
        got = np.where((wsum > 0)[:, None], csum / np.maximum(wsum, 1e-9)[:, None], np.nan)
        col[np.nonzero(todo)[0]] = got
    col[np.isnan(col[:, 0])] = 150.0
    return col


def hull_heights(hull, xs, ys):
    """Height field of the convex hull on the grid xs, ys: the lowest upward facet plane above each cell, inside the
    footprint."""
    if hull is None:
        return np.zeros((len(xs), len(ys)))
    h = hull[3]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    E = h.equations                                         # n . p + d <= 0 inside
    up = E[E[:, 2] > 1e-6]
    z = np.min((-up[:, 3][None, None] - up[:, 0][None, None] * X[..., None] - up[:, 1][None, None] * Y[..., None])
               / up[:, 2][None, None], axis=-1)
    fp = Delaunay(h.points[h.vertices][:, :2])
    inside = fp.find_simplex(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape) >= 0
    return np.where(inside, np.maximum(z, 0), 0.0)


# --- body type ---------------------------------------------------------------------------------------------------
def body_type(H, xs, ground_h=GROUND_H):
    """Body type from the side profile of the averaged car (x = length axis, +x = direction of travel = front).
    Mostly scale-free features: aspect = length / height, roof = share of the length under the roof (> 85 % of the
    height), rear / front = mean height of the rear / front fifth relative to the height. Rules: a box with a high
    rear and a long roof is a van/MPV; a high rear and a short, tall body an SUV; a low rear (a boot) a sedan;
    otherwise a hatchback."""
    occ = H > ground_h
    cols = np.nonzero(occ.any(1))[0]
    if len(cols) < 5:
        return "unknown", {}
    prof = np.where(occ, H, 0).max(1)[cols[0]:cols[-1] + 1]
    L = float(xs[cols[-1]] - xs[cols[0]] + (xs[1] - xs[0]))
    Hc = float(np.percentile(H[occ], 98))
    n5 = max(1, len(prof) // 5)
    f = {"L": L, "H": Hc, "aspect": L / Hc, "roof": float(np.mean(prof > 0.85 * Hc)),
         "rear": float(prof[:n5].mean() / Hc), "front": float(prof[-n5:].mean() / Hc)}
    if f["roof"] >= 0.55 and f["rear"] >= 0.75 and f["aspect"] < 3.0:
        return "van/MPV", f
    if f["rear"] >= 0.65 and f["aspect"] < 2.9:
        return "SUV", f
    if f["rear"] < 0.65:
        return "sedan", f
    return "hatchback", f


def profile_type(prof, aspect):
    """The same body-type rules on a normalised side profile (rear first, front last)."""
    n5 = max(1, len(prof) // 5)
    f = {"aspect": float(aspect), "roof": float(np.mean(prof > 0.85)), "rear": float(prof[:n5].mean()),
         "front": float(prof[-n5:].mean())}
    # the height of the rear fifth (a boot step) separates the types; the length/height ratio of an oblique side
    # view is foreshortened and is not used
    if f["roof"] >= 0.6 and f["rear"] >= 0.85:
        return "van/MPV", f
    if f["rear"] >= 0.85:
        return "SUV", f
    if f["rear"] < 0.78:
        return "sedan", f
    return "hatchback", f


def side_view_type(seq, f, n_views=3):
    """Body type from the image: in the windows where the car is seen most from the side (heading across the viewing
    ray), its full silhouette (background subtraction with windows and wheels, reflection and shadow cut) gives the
    side profile - scale-free and independent of the 3D shape. Majority vote over the n_views most side-on windows."""
    import ipm_track_models as itm_
    import v2x_seq_classical as vc_
    cand = []
    for (k, d), p, h in zip(f["track"], f["pos"], f["head"]):
        K, R, t = seq.cams[k]
        C = -np.linalg.solve(R, t)
        ray = np.arctan2(p[1] - C[1], p[0] - C[0])
        cand.append((abs(np.sin(h - ray)), k, d, p, h))
    cand = sorted(cand, key=lambda c: -c[0])[:n_views]
    votes, feats, used = [], [], []
    for side, k, d, p, h in cand:
        if side < 0.8:
            continue
        frame = seq.frames[k]
        uv = d["uv"][d["body"]]
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int)
        hgt = y1 - y0
        X0, Y0 = max(0, x0 - 15), max(0, y0 - hgt)
        X1, Y1 = min(frame.shape[1], x1 + 15), min(frame.shape[0], y1 + hgt)
        fg = vc_.foreground(frame[Y0:Y1, X0:X1], seq.background(k)[Y0:Y1, X0:X1])
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
        if n < 2:
            continue
        core = lab[np.clip(uv[:, 1].astype(int) - Y0, 0, Y1 - Y0 - 1), np.clip(uv[:, 0].astype(int) - X0, 0, X1 - X0 - 1)]
        core = core[core > 0]
        if not len(core):
            continue
        m = (lab == np.bincount(core).argmax()).astype(np.uint8)
        b = vc_.make_blob(m, X0, Y0)
        if b is None:
            continue
        b = vc_.cut_reflection(b, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        bx0, by0, bx1, by1 = b["box"]
        sil = itm_.shadow_free(frame[by0:by1, bx0:bx1], seq.background(k)[by0:by1, bx0:bx1], b["m"] > 0)
        sil = snap_silhouette(cv2.cvtColor(frame[by0:by1, bx0:bx1], cv2.COLOR_BGR2GRAY), sil)
        cols = np.nonzero(sil.any(0))[0]
        if len(cols) < 20:
            continue
        top = np.argmax(sil[:, cols], axis=0)
        bottom = sil.shape[0] - 1 - np.argmax(sil[::-1, cols], axis=0)
        base = np.percentile(bottom, 90)
        prof = np.clip(base - top, 0, None)
        prof = prof / max(prof.max(), 1)
        # which image side is the front: project the car's front and rear on the ground
        q = vs.project(seq.cams[k], np.c_[np.vstack([p + 2 * np.r_[np.cos(h), np.sin(h)],
                                                       p - 2 * np.r_[np.cos(h), np.sin(h)]]), np.zeros(2)])[0]
        if q[0, 0] < q[1, 0]:
            prof = prof[::-1]                               # rear first, front last
        t_, ft = profile_type(prof, len(cols) / max(base - top.min(), 1))
        votes.append(t_)
        feats.append(ft)
        used.append(k)
    if not votes:
        return "unknown", {}, []
    # majority; a tie goes to the most side-on view (votes are in that order). Not max(set(votes)): the iteration
    # order of a set of strings changes from run to run (hash randomisation), so ties were decided at random
    best = max(votes, key=votes.count)
    return best, feats[votes.index(best)], used


def label_types(seq):
    """The labelled type (Car / Van ...) of every track (evaluation only)."""
    out = {}
    for f in seq.fids[::10]:
        for o in json.load(open(os.path.join(vd.DATA, "label", "camera", f"{f}.json"))):
            out.setdefault(o["track_id"], o["type"])
    return out


# --- 5. mesh -------------------------------------------------------------------------------------------------------
def mesh_from_heights(Hm, Cm, xs, ys, vox=CELL):
    nz = int(np.ceil(max(Hm.max(), 0.5) / vox)) + 2
    zc = (np.arange(nz) + 0.5) * vox
    vol = (zc[None, None, :] < Hm[:, :, None]).astype(np.float32)
    vol = np.pad(vol, 1)
    if vol.sum() < 10:
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3))
    V, F, _, _ = marching_cubes(vol, 0.5)
    V = np.c_[xs[0] + (V[:, 0] - 1) * vox, ys[0] + (V[:, 1] - 1) * vox, (V[:, 2] - 0.5) * vox]   # padded indices
    V[:, 2] = np.maximum(V[:, 2], 0.0)
    V = mr.laplacian(V, mr.adjacency(F, len(V)), iterations=10)
    ix = np.clip(np.round((V[:, 0] - xs[0]) / vox).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round((V[:, 1] - ys[0]) / vox).astype(int), 0, len(ys) - 1)
    C = Cm[ix, iy][:, ::-1] / 255.0
    C = np.where(np.isnan(C), 0.55, C)
    return V, F, C


# --- checks (labels and LiDAR: evaluation only) --------------------------------------------------------------------
def match_label(seq, track, pos):
    """Evaluation only: the labelled track nearest to the car's path in at least half of its windows."""
    votes = {}
    for (k, d), p in zip(track, pos):
        best, bd = None, MATCH_GATE
        for tid, q in seq.tracks.items():
            for o in q:
                if o["k"] == k and np.linalg.norm(o["pos"] - p) < bd:
                    best, bd = tid, np.linalg.norm(o["pos"] - p)
        if best:
            votes[best] = votes.get(best, 0) + 1
    if not votes:
        return None
    tid = max(votes, key=votes.get)
    return tid if votes[tid] >= 0.5 * len(track) else None


def dims_of(Hm, xs, ys):
    occ = Hm > GROUND_H
    if not occ.any():
        return (np.nan,) * 3
    ix, iy = np.nonzero(occ)
    return (float(xs[ix].max() - xs[ix].min() + CELL), float(ys[iy].max() - ys[iy].min() + CELL),
            float(np.percentile(Hm[occ], 98)))


def lidar_heights(shapes, tid):
    occ, axes, n = shapes.solid(tid)
    k = occ.shape[2]
    top = np.where(occ.any(2), (k - np.argmax(occ[:, :, ::-1], axis=2)) * (axes[2][1] - axes[2][0]), 0.0)
    return top, axes[0], axes[1], n


def close_lidar_holes(top, step, radius=0.15):
    """Evaluation only: close the sparse-scan holes of a LiDAR height field (a 60 m scan leaves gaps between the
    returns; against the raw field even the labelled box scores IoU 0.39, against the closed one 0.59)."""
    size = max(3, 2 * int(round(radius / step)) + 1)
    return grey_closing(top, size=(size, size))


def resample(Hm, xs, ys, xq, yq):
    ix = np.round((xq[:, None] - xs[0]) / (xs[1] - xs[0])).astype(int)
    iy = np.round((yq[None, :] - ys[0]) / (ys[1] - ys[0])).astype(int)
    ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
    return np.where(ok, Hm[np.clip(ix, 0, len(xs) - 1), np.clip(iy, 0, len(ys) - 1)], 0.0)


def shape_scores(Hm, xs, ys, Lt, lx, ly):
    """After centring both footprints and trying both headings: solid IoU and mean |height difference|."""
    def centred(H, x, y):
        occ = H > GROUND_H
        if not occ.any():
            return x, y
        ix, iy = np.nonzero(occ)
        return x - (x[ix.min()] + x[ix.max()]) / 2, y - (y[iy.min()] + y[iy.max()]) / 2
    xs2, ys2 = centred(Hm, xs, ys)
    lx2, ly2 = centred(Lt, lx, ly)
    best = None
    for flip in (1, -1):
        H = resample(Hm, flip * xs2[::flip], ys2, lx2, ly2) if flip == 1 else resample(Hm[::-1], -xs2[::-1], ys2, lx2, ly2)
        a, b = H > GROUND_H, Lt > GROUND_H
        inter = np.minimum(np.where(a, H, 0), np.where(b, Lt, 0)).sum()
        union = np.maximum(np.where(a, H, 0), np.where(b, Lt, 0)).sum()
        both = a & b
        s = {"iou": float(inter / max(union, 1e-9)),
             "height_err_cm": float(100 * np.median(np.abs(H[both] - Lt[both]))) if both.any() else float("nan"),
             "footprint_iou": float((a & b).sum() / max((a | b).sum(), 1))}
        s["flip"] = flip
        if best is None or s["iou"] > best["iou"]:
            best = s
    return best


def pose_errors(seq, tid, track, pos, head, x_off):
    """Our car centre (path + recentring offset along the heading) and heading against the label, per window."""
    q = {o["k"]: o for o in seq.tracks[tid]}
    dp, dh = [], []
    for (k, _), p, h in zip(track, pos, head):
        if k not in q:
            continue
        c = p + rot(h) @ np.array([x_off, 0.0])
        dp.append(np.linalg.norm(c - q[k]["pos"]))
        e = (h - q[k]["yaw"]) % np.pi
        dh.append(np.degrees(min(e, np.pi - e)))
    return float(np.median(dp)) if dp else np.nan, float(np.median(dh)) if dh else np.nan


# --- run -----------------------------------------------------------------------------------------------------------
def fuse_track(track):
    parts, pos, head = track_in_car_frame(track)
    allp = np.vstack([p[0] for p in parts])
    pts = allp[allp[:, 2] > 0.3] if SYMMETRY else np.zeros((0, 3))
    y0, sym_score = symmetry_plane(pts, search=SYM_SEARCH) if len(pts) else (0.0, np.nan)
    for p in parts:
        p[0] = p[0] - [0, y0, 0]
        p[4] = p[4] - [0, y0]
        if SYMMETRY:                         # a car is symmetric about its centre plane and at most W_MAX wide
            keep = np.abs(p[0][:, 1]) <= W_MAX / 2
            p[0], p[1] = p[0][keep], p[1][keep]
    Hm, Cm, xs, ys, views = height_field(parts, mirror=SYMMETRY, min_views=max(MIN_VIEWS, int(VOTE * len(parts))),
                                         w=np.array([1.0 / (0.5 + t[1]["reprojection_px"]) for t in track])
                                         if W_MEDIAN else None)
    Hn, Cn, xsn, ysn, _ = height_field(parts, mirror=False, min_views=max(MIN_VIEWS, int(VOTE * len(parts))),
                                       w=np.array([1.0 / (0.5 + t[1]["reprojection_px"]) for t in track])
                                       if W_MEDIAN else None)
    occ = Hm > GROUND_H
    x_off = 0.0
    if occ.any():
        ix = np.nonzero(occ.any(1))[0]
        x_off = (xs[ix[0]] + xs[ix[-1]]) / 2
    xs_c = xs - x_off
    for p in parts:
        p[4] = p[4] - [x_off, 0]
    hull = convex_hull_model(Hm, Cm, xs_c, ys)
    painted = coloured_hull(hull, parts, SYMMETRY) if hull is not None else None
    # the best single window, no averaging, no symmetry (for the comparison)
    best_single = max(range(len(parts)), key=lambda i: len(parts[i][0]))
    Hs, _, xss, yss, _ = height_field([parts[best_single]], mirror=False, min_views=1)
    return {"nosym": (Hn, Cn, xsn, ysn),
            "parts": parts, "pos": pos, "head": head, "y0": float(y0), "sym_score": float(sym_score),
            "H": Hm, "C": Cm, "xs": xs_c, "ys": ys, "x_off": float(x_off), "views": views,
            "single": (Hs, xss, yss), "hull": hull, "painted": painted, "track": track,
            "poses": {p[2]: (p[3], p[4]) for p in parts}}


def main():
    import ego_sequence as es
    seq, ground = es.open_sequence()
    import lidar_masks as lm
    shapes = lm.LidarShapes(seq, ground)
    ltypes = label_types(seq)
    frames = load_dets()
    tracks = link_tracks(frames)
    print(f"{sum(len(d) for _, d in frames)} objects in {len(frames)} windows -> {len(tracks)} tracks of >= {MIN_TRACK}")
    rows, models = [], {}
    for n, tr in enumerate(tracks):
        f = fuse_track(tr)
        if CABIN:
            sils = {k: full_silhouette(seq, k, d) for k, d in tr}
            near = float(np.median([d["range"] for _, d in tr])) < RANGE_HULL
            Hc = carve(seq, f, sils, use_foot=CARVE_FOOT and not near)
            if SIL_POSE:                             # poses to the silhouettes, then the cabin carved again
                H0 = np.maximum(Hc, np.where(f["H"] > GROUND_H, f["H"], 0)) if BODY_FIELD and not near else Hc
                f["sil_fit"] = refine_poses(seq, f, sils, H0)
                Hc = carve(seq, f, sils, use_foot=CARVE_FOOT and not near)
            f["H_body"] = f["H"]
            if BODY_FIELD and not near:
                f["H"] = np.maximum(Hc, np.where(f["H"] > GROUND_H, f["H"], 0))   # body field + carved cabin
            else:
                f["H"] = Hc                                                      # the silhouettes alone
            hull = convex_hull_model(f["H"], f["C"], f["xs"], f["ys"])
            f["hull"] = hull
            if hull is not None:
                V, Fh, _, _ = hull
                Fh = orient_outward(V, Fh)
                for _ in range(3):
                    V, Fh = mr.subdivide(V, Fh)
                vcol = pixel_colours(seq, f, sils, V, Fh)
                f["painted"] = (V, Fh, vcol[Fh].mean(1))
                f["vertex_colours"] = vcol
        btype3d, feats3d = body_type(f["H"], f["xs"])
        f["body_type_3d"] = btype3d
        btype, feats, f["side_windows"] = side_view_type(seq, f) if CLASSIFY == "image" else (btype3d, feats3d, [])
        if btype == "unknown":                  # never seen from the side: the averaged 3D profile decides
            btype, feats = btype3d, feats3d
        f["body_type"], f["features"] = btype, feats
        if TYPE_WIDTH:
            # the width of crossing traffic lies nearly along the viewing ray: weakly observed by one camera;
            # the body type's prior width bounds the field (centred on its own y extent)
            w = TYPE_W.get(btype, W_MAX)
            keep = np.abs(f["ys"]) <= w / 2          # the car frame is already centred on the symmetry plane
            f["H"][:, ~keep] = 0.0
            hull = convex_hull_model(f["H"], f["C"], f["xs"], f["ys"])
            f["hull"] = hull
            if hull is not None and CABIN:
                V, Fh, _, _ = hull
                Fh = orient_outward(V, Fh)
                for _ in range(3):
                    V, Fh = mr.subdivide(V, Fh)
                vcol = pixel_colours(seq, f, sils, V, Fh)
                f["painted"] = (V, Fh, vcol[Fh].mean(1))
                f["vertex_colours"] = vcol
        tid = match_label(seq, tr, f["pos"])
        L, W, Hh = dims_of(f["H"], f["xs"], f["ys"])
        if f.get("sil_fit"):
            f["sil_iou_before"], f["sil_iou_after"] = f["sil_fit"]
        row = {"track": n, "windows": len(tr), "frames": f"{tr[0][0]}-{tr[-1][0]}",
               **({"silhouette IoU before": f["sil_iou_before"], "silhouette IoU after": f["sil_iou_after"]}
                  if f.get("sil_fit") else {}),
               "range_m": float(np.median([d["range"] for _, d in tr])),
               "speed_mps": float(np.median([d["speed_mps"] for _, d in tr])),
               "length": L, "width": W, "height": Hh, "label": tid, "symmetry_y0": f["y0"], "body_type": btype,
               "body_type_3d": btype3d,
               **{f"feature {k}": v for k, v in feats.items()}}
        if tid:
            ld = np.median([o["dims"] for o in seq.tracks[tid]], axis=0)
            row.update({"label_length": float(ld[0]), "label_width": float(ld[1]), "label_height": float(ld[2])})
            dp, dh = pose_errors(seq, tid, tr, f["pos"], f["head"], f["x_off"])
            row.update({"centre_err_m": dp, "heading_err_deg": dh})
            # the model's centre (car-frame origin) placed by every window's pose, against the label: along / across
            # the viewing ray (evaluation only)
            q = {o["k"]: o for o in seq.tracks[tid]}
            ea, ec = [], []
            for k in f["poses"]:
                if k in q:
                    e = to_world(f, k, np.zeros((1, 3)))[0, :2] - q[k]["pos"]
                    ray = q[k]["pos"] - seq.camera_xy
                    ray = ray / np.linalg.norm(ray)
                    ea.append(e @ ray)
                    ec.append(e[0] * -ray[1] + e[1] * ray[0])
            if ea:
                row.update({"pose: along-ray err m": float(np.median(np.abs(ea))),
                            "pose: across-ray err m": float(np.median(np.abs(ec)))})
            Lt, lx, ly, nL = lidar_heights(shapes, tid)
            row["label_type"] = ltypes.get(tid, "?")
            if nL > 200:
                row["lidar_body_type"] = body_type(Lt, lx)[0]       # the same rules on the car's LiDAR (check)
            if nL > 200:
                sc = shape_scores(f["H"], f["xs"], f["ys"], Lt, lx, ly)
                row["flip"] = sc.pop("flip")
                row.update({f"fused: {k}": v for k, v in sc.items()})
                Lc = close_lidar_holes(Lt, lx[1] - lx[0])
                sc = shape_scores(f["H"], f["xs"], f["ys"], Lc, lx, ly)
                sc.pop("flip")
                row.update({f"fused closed: {k}": v for k, v in sc.items()})
                Hs, xss, yss = f["single"]
                sc = shape_scores(Hs, xss, yss, Lt, lx, ly)
                sc.pop("flip")
                row.update({f"single window: {k}": v for k, v in sc.items()})
                sc = shape_scores(Hs, xss, yss, Lc, lx, ly)
                sc.pop("flip")
                row.update({f"single closed: {k}": v for k, v in sc.items()})
                if f["hull"] is not None:
                    Hh = hull_heights(f["hull"], f["xs"], f["ys"])
                    sc = shape_scores(Hh, f["xs"], f["ys"], Lt, lx, ly)
                    sc.pop("flip")
                    row.update({f"hull: {k}": v for k, v in sc.items()})
                    Lh, Wh, HH = dims_of(Hh, f["xs"], f["ys"])
                    row.update({"hull_length": Lh, "hull_height": HH})
                f["lidar"] = (Lt, lx, ly)
        rows.append(row)
        models[n] = (f, row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    json.dump(rows, open(os.path.join(vd.OUT_DIR, f"whole_scene_cars{OUT_SUFFIX}.json"), "w"), indent=1)
    if RIDE_TABLE_OUT:                       # the second pass: every window of a car gets its body type's height
        table = {}
        for f, row in models.values():
            ride = TYPE_RIDE.get(f["body_type"], TYPE_RIDE["unknown"])
            for k, d in f["track"]:
                c = d["uv"][d["body"]].mean(0)
                table.setdefault(int(k), []).append([float(c[0]), float(c[1]), ride])
        json.dump(table, open(RIDE_TABLE_OUT, "w"))
    types_figure(os.path.join(vd.OUT_DIR, f"body_types{OUT_SUFFIX}.png"), seq, models)
    gallery(os.path.join(vd.OUT_DIR, f"whole_scene_cars{OUT_SUFFIX}.png"), models)
    triangulation_figure(os.path.join(vd.OUT_DIR, f"car_triangulations{OUT_SUFFIX}.png"), seq, models)
    export(os.path.join(vd.OUT_DIR, f"cars_viewer{OUT_SUFFIX}"), models)
    if VIDEO:
        scene_video(seq, ground, models, os.path.join(vd.OUT_DIR, f"whole_scene_full{OUT_SUFFIX}.mp4"))


def types_figure(path, seq, models, n=12):
    """Body types: the car in the image, its averaged side profile against the LiDAR's, ours / LiDAR-rule / label."""
    pick = sorted([m for m in models.values() if m[0]["body_type"] != "unknown"], key=lambda m: -m[1]["windows"])[:n]
    cols = 3
    rows_n = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows_n, 2 * cols, figsize=(19, 2.7 * rows_n), squeeze=False,
                             gridspec_kw={"width_ratios": [1.25, 1] * cols})
    for i, (f, row) in enumerate(pick):
        r, c = divmod(i, cols)
        k, d = max(f["track"], key=lambda t: len(t[1]["X"]))
        uv = d["uv"]
        x0, y0 = np.floor(uv.min(0)).astype(int) - 20
        x1, y1 = np.ceil(uv.max(0)).astype(int) + 20
        ax = axes[r, 2 * c]
        ax.imshow(cv2.cvtColor(seq.frames[k][max(0, y0):y1, max(0, x0):x1], cv2.COLOR_BGR2RGB))
        ax.axis("off")
        lab = f"label {row['label']} ({row.get('label_type', '?')})" if row.get("label") else "no label"
        ax.set_title(f"ours: {f['body_type']} (3D profile: {f['body_type_3d']})  |  {lab}", fontsize=8.5, loc="left",
                     color=INK)
        ax = axes[r, 2 * c + 1]
        occ = f["H"] > GROUND_H
        prof = np.where(occ.any(1), np.where(occ, f["H"], 0).max(1), np.nan)
        ax.plot(f["xs"], prof, color="#2a78d6", lw=1.8, label="ours (averaged)")
        if "lidar" in f:
            Lt, lx, ly = f["lidar"]
            lo = Lt > GROUND_H
            lp = np.where(lo.any(1), np.where(lo, Lt, 0).max(1), np.nan)
            jx = np.nonzero(lo.any(1))[0]
            sgn = row.get("flip", 1)
            ax.plot(sgn * (lx - (lx[jx[-1]] + lx[jx[0]]) / 2), lp, color="#c03a3a", lw=1.2, label="LiDAR (check)")
        ax.set_xlim(-3, 3)
        ax.set_ylim(0, 2.2)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ft = f["features"]
        ax.set_title(f"LiDAR rule: {row.get('lidar_body_type', '-')}  L/H {ft.get('aspect', np.nan):.1f} "
                     f"roof {ft.get('roof', np.nan):.2f} rear {ft.get('rear', np.nan):.2f}", fontsize=7.5, loc="left",
                     color=INK)
        ax.text(2.9, 0.1, "front ->", fontsize=7, ha="right", color=MUTED)
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    for j in range(len(pick), rows_n * cols):
        r, c = divmod(j, cols)
        axes[r, 2 * c].axis("off")
        axes[r, 2 * c + 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor="white")
    plt.close(fig)


def triangulation_figure(path, seq, models, n=6):
    """Per car: one window's triangulation in the image (every triangle filled with its mean image colour), the same
    triangulation in 3D in the car frame, and the model averaged over the whole video (subdivided convex hull, every
    triangle with the colour averaged over all windows); the LiDAR of the car (red) as a check."""
    pick = sorted([m for m in models.values() if m[0]["painted"] is not None and m[1].get("label")],
                  key=lambda m: -m[1]["windows"])[:n]
    fig = plt.figure(figsize=(17, 3.6 * len(pick)))
    for r, (f, row) in enumerate(pick):
        k, d = max(f["track"], key=lambda t: len(t[1]["X"]))                     # the window with most vertices
        A, b = f["poses"][k]
        body = d["body"][d["tri"]].all(1)
        T = d["tri"][body]
        col = d["tri_bgr"][body][:, ::-1] / 255.0
        # image
        ax = fig.add_subplot(len(pick), 4, 4 * r + 1)
        img = seq.frames[k].copy()
        for t in T:
            cv2.polylines(img, [np.round(d["uv"][t]).astype(np.int32)], True, (60, 200, 255), 1, cv2.LINE_AA)
        x0, y0 = np.floor(d["uv"].min(0)).astype(int) - 15
        x1, y1 = np.ceil(d["uv"].max(0)).astype(int) + 15
        ax.imshow(cv2.cvtColor(img[max(0, y0):y1, max(0, x0):x1], cv2.COLOR_BGR2RGB))
        ax.set_title(f"car {row['label']}, frame {k}: body triangulation on the video frame", fontsize=8.5,
                     loc="left", color=INK)
        ax.axis("off")
        # one window in 3D
        Y = np.c_[d["X"][:, :2] @ A.T + b, d["X"][:, 2]]
        ax = fig.add_subplot(len(pick), 4, 4 * r + 2, projection="3d")
        ax.add_collection3d(Poly3DCollection(Y[T], facecolors=col, edgecolor="#ffffff", linewidths=0.2))
        ax.set_title(f"the same triangles in 3D (one window, {len(T)} triangles)", fontsize=8.5, loc="left", color=INK)
        # averaged model
        V, F, C = f["painted"]
        ax2 = fig.add_subplot(len(pick), 4, 4 * r + 3, projection="3d")
        ax2.add_collection3d(Poly3DCollection(V[F], facecolors=np.clip(C[:, ::-1] / 255, 0, 1), edgecolor="none"))
        ax2.set_title(f"averaged over {row['windows']} windows, cabin from silhouettes; colours = video pixels",
                      fontsize=8.5, loc="left", color=INK)
        ax3 = fig.add_subplot(len(pick), 4, 4 * r + 4, projection="3d")
        ax3.add_collection3d(Poly3DCollection(V[F], facecolors=np.clip(C[:, ::-1] / 255, 0, 1), edgecolor="none",
                                              alpha=0.35))
        if "lidar" in f:
            Lt, lx, ly = f["lidar"]
            ix, iy = np.nonzero(Lt > GROUND_H)
            Lp = np.c_[lx[ix], ly[iy], Lt[ix, iy]]
            Lp[:, :2] -= (Lp[:, :2].max(0) + Lp[:, :2].min(0)) / 2
            jx, jy = np.nonzero(f["H"] > GROUND_H)
            Lp[:, :2] += [(f["xs"][jx].max() + f["xs"][jx].min()) / 2, (f["ys"][jy].max() + f["ys"][jy].min()) / 2]
            if row.get("flip", 1) < 0:
                Lp[:, 0] = -Lp[:, 0]
            ax3.scatter(Lp[::3, 0], Lp[::3, 1], Lp[::3, 2], s=1, c="#c03a3a")
        ax3.set_title(f"check: LiDAR top of the car (red), IoU {row.get('hull: iou', np.nan):.2f}", fontsize=8.5,
                      loc="left", color=INK)
        for a in (ax, ax2, ax3):
            if hasattr(a, "set_zlim"):
                a.set_xlim(-2.6, 2.6)
                a.set_ylim(-2.6, 2.6)
                a.set_zlim(0, 2.2)
                a.view_init(24, -58)
                a.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=75, facecolor="white")
    plt.close(fig)


def gallery(path, models, n=8):
    pick = [m for m in models.values() if m[1].get("label")][:n] or list(models.values())[:n]
    fig = plt.figure(figsize=(18, 3.4 * len(pick)))
    for r, (f, row) in enumerate(pick):
        ax = fig.add_subplot(len(pick), 4, 4 * r + 1, projection="3d")
        if f["painted"] is not None:
            V, F, C = f["painted"]
            ax.add_collection3d(Poly3DCollection(V[F], facecolors=np.clip(C[:, ::-1] / 255, 0, 1), edgecolor="none"))
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_zlim(0, 2.6)
        ax.view_init(22, -60)
        ax.set_axis_off()
        ax.set_title(f"track {row['track']} ({row['windows']} windows, frames {row['frames']}, "
                     f"{row['range_m']:.0f} m)", fontsize=8.5, loc="left", color=INK)
        ext = [f["ys"][0], f["ys"][-1], f["xs"][0], f["xs"][-1]]
        ax2 = fig.add_subplot(len(pick), 4, 4 * r + 2)
        ax2.imshow(f["H"], origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1.8)
        ax2.set_title(f"ours: averaged, mirrored (hull IoU {row.get('hull: iou', np.nan):.2f})", fontsize=8.5, loc="left",
                      color=INK)
        ax3 = fig.add_subplot(len(pick), 4, 4 * r + 3)
        Hs, xss, yss = f["single"]
        ax3.imshow(Hs, origin="lower", extent=[yss[0], yss[-1], xss[0], xss[-1]], cmap="viridis", vmin=0, vmax=1.8)
        ax3.set_title("one window only, no symmetry", fontsize=8.5, loc="left", color=INK)
        ax4 = fig.add_subplot(len(pick), 4, 4 * r + 4)
        if "lidar" in f:
            Lt, lx, ly = f["lidar"]
            ax4.imshow(Lt, origin="lower", extent=[ly[0], ly[-1], lx[0], lx[-1]], cmap="viridis", vmin=0, vmax=1.8)
            ax4.set_title(f"check: LiDAR of label {row['label']}  IoU fused {row.get('fused: iou', np.nan):.2f} / "
                          f"single {row.get('single window: iou', np.nan):.2f}", fontsize=8.5, loc="left", color=INK)
        for a in (ax2, ax3, ax4):
            a.set_xlim(-1.6, 1.6)
            a.set_ylim(-3.0, 3.0)
            a.tick_params(labelsize=7)
        dims = f"L {row['length']:.2f} W {row['width']:.2f} H {row['height']:.2f} m"
        if row.get("label"):
            dims += f"  (label {row['label_length']:.2f} {row['label_width']:.2f} {row['label_height']:.2f})"
        ax2.set_xlabel(dims, fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=75, facecolor="white")
    plt.close(fig)


def draw_meshes(img, cam, meshes, light=(0.3, -0.5, 0.8)):
    """Painter's rendering of flat-coloured, Lambert-shaded meshes [(V world, F, C BGR per face)] over img."""
    if not meshes:
        return img
    Vs, Fs, Cs, off = [], [], [], 0
    for V, F, C in meshes:
        Vs.append(V)
        Fs.append(F + off)
        Cs.append(C)
        off += len(V)
    V, F, C = np.vstack(Vs), np.vstack(Fs), np.vstack(Cs).astype(float)
    q, z = vs.project(cam, V)
    ok = (z[F] > 0.5).all(1)
    F, C = F[ok], C[ok]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array(light) / np.linalg.norm(light)
    col = np.clip(C * (0.35 + 0.65 * np.abs(n @ light))[:, None], 0, 255)
    for i in np.argsort(-z[F].mean(1)):
        pts = np.round(q[F[i]] * 4).astype(np.int32)
        cv2.fillConvexPoly(img, pts, tuple(float(c) for c in col[i]), lineType=cv2.LINE_AA, shift=2)
        cv2.polylines(img, [pts], True, tuple(float(c) * 0.6 for c in col[i]), 1, cv2.LINE_AA, shift=2)
    return img


def scene_video(seq, ground, models, out_path):
    """Final video: the four panels of whole_scene.py (segmentation, triangulation, surface, LiDAR) and two more -
    the convex-hull car models standing at their pose of every window on the textured road, and a bird's-eye check
    of those models against the LiDAR of the frame (red)."""
    import whole_scene as ws
    panels = {}
    for d in glob.glob(os.path.join(vd.OUT_DIR, PANELS)):
        for f in glob.glob(os.path.join(d, "*.jpg")):
            panels[int(os.path.basename(f)[:4])] = f
    out_dir = os.path.join(vd.OUT_DIR, f"whole_scene_video_final{OUT_SUFFIX}")
    os.makedirs(out_dir, exist_ok=True)
    H, W = seq.shape
    tw, th = 640, 360
    for k in sorted(panels):
        K, R, t = seq.cams[k]
        Cc = -np.linalg.solve(R, t)
        cam = ws.scene_camera(seq, k, Cc)
        shape = (int(H * ws.VIEW_SCALE), int(W * ws.VIEW_SCALE))
        # the road: the frame laid flat (IPM) below the horizon
        gx, gy = np.meshgrid(np.arange(0, W, 24), np.arange(int(0.24 * H), H, 24))
        P = np.c_[gx.ravel(), gy.ravel()].astype(float)
        G = np.c_[ws.itm.ground_point(seq.cams[k], P), np.zeros(len(P))]
        okg = np.linalg.norm(G[:, :2] - Cc[:2], axis=1) < ws.VIEW_RANGE
        tri = Delaunay(P).simplices
        tri = tri[okg[tri].all(1)]
        road, _ = ws.render(cam, shape, G, tri, P, seq.frames[k])
        meshes = []
        for f, row in models.values():
            if f["painted"] is None or k not in f["poses"]:
                continue
            A, b = f["poses"][k]
            V, F, C = f["painted"]
            meshes.append((np.c_[(V[:, :2] - b) @ A, V[:, 2]], F, C))
        scene = draw_meshes(road.copy(), cam, meshes)
        # bird's eye: models from above, LiDAR of the frame in red
        target = np.r_[ws.itm.ground_point(seq.cams[k], np.array([[W / 2, 0.5 * H]]))[0], 0.0]
        Kt = K.copy()
        Kt[:2] *= ws.VIEW_SCALE
        top = ws.look_at(Kt, target + np.r_[0.05, 0.05, 85.0], target, up=(0, 1.0, 0))
        bird = (0.55 * ws.render(top, shape, G, tri, P, seq.frames[k])[0] + 0.45 * 245).astype(np.uint8)
        bird = draw_meshes(bird, top, meshes)
        L, _, _ = ws.lidar_depth(seq, k, ground, seq.cams[k])
        L = L[(L[:, 2] > 0.3) & (L[:, 2] < 2.5) & (np.linalg.norm(L[:, :2] - Cc[:2], axis=1) < ws.VIEW_RANGE)]
        q, z = vs.project(top, L)
        inb = (z > 0) & (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < shape[1]) & (q[:, 1] < shape[0])
        for u, v in q[inb].astype(int):
            cv2.circle(bird, (int(u), int(v)), 1, (40, 40, 210), -1)
        pan = cv2.imread(panels[k])
        ph, pw = pan.shape[0] // 2, pan.shape[1] // 2
        tiles = [pan[:ph, :pw], pan[:ph, pw:], pan[ph:, :pw], pan[ph:, pw:]]
        ws.label_tag(scene, f"car models (convex hulls, averaged over the video) at their pose: {len(meshes)} cars")
        ws.label_tag(bird, "from above: models (colour) vs LiDAR points 0.3-2.5 m high (red), check")
        tiles = [cv2.resize(x, (tw, th), interpolation=cv2.INTER_AREA) for x in tiles + [scene, bird]]
        img = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        cv2.imwrite(os.path.join(out_dir, f"{k:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    os.system(f"ffmpeg -y -loglevel error -framerate 5 -pattern_type glob -i '{out_dir}/*.jpg' -c:v libx264 "
              f"-pix_fmt yuv420p -crf 24 {out_path}")


def export(dirname, models):
    os.makedirs(dirname, exist_ok=True)
    out = {}
    for n, (f, row) in models.items():
        if f["painted"] is None:
            continue
        V, F, C = f["painted"]
        if "vertex_colours" in f:                          # video pixels per vertex, shared vertices (smooth)
            C = f["vertex_colours"][:, ::-1] / 255.0
        else:
            V, F = V[F].reshape(-1, 3), np.arange(3 * len(F)).reshape(-1, 3)   # flat shading: own vertices per face
            C = np.repeat(C[:, ::-1] / 255.0, 3, axis=0)
        item = {"v": np.round(V, 3).ravel().tolist(), "f": F.ravel().tolist(), "c": np.round(C, 3).ravel().tolist(),
                "row": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}}
        Vs, Fs, Cs = mesh_from_heights(f["H"], f["C"], f["xs"], f["ys"])          # the averaged surface (no hull)
        item["vr"], item["fr"], item["cr"] = (np.round(Vs, 3).ravel().tolist(), Fs.ravel().tolist(),
                                              np.round(Cs, 3).ravel().tolist())
        if f.get("nosym") is not None:                       # the one-sided field, mirror off
            Hn, Cn, xsn, ysn = f["nosym"]
            Vn, Fn, Cn2 = mesh_from_heights(Hn, Cn, xsn, ysn)
            item["vn"], item["fn"], item["cn"] = (np.round(Vn, 3).ravel().tolist(), Fn.ravel().tolist(),
                                                  np.round(Cn2, 3).ravel().tolist())
        if "lidar" in f:
            # the LiDAR check shown in the same place: both footprints centred, the better of the two headings
            Lt, lx, ly = f["lidar"]
            ix, iy = np.nonzero(Lt > GROUND_H)
            P = np.c_[lx[ix], ly[iy], Lt[ix, iy]]
            P[:, :2] -= (P[:, :2].max(0) + P[:, :2].min(0)) / 2
            occ = f["H"] > GROUND_H
            if occ.any():
                jx, jy = np.nonzero(occ)
                c = np.array([(f["xs"][jx].max() + f["xs"][jx].min()) / 2, (f["ys"][jy].max() + f["ys"][jy].min()) / 2])
                P[:, :2] += c
                if row.get("flip", 1) < 0:
                    P[:, 0] = 2 * c[0] - P[:, 0]
            item["lidar_top"] = np.round(P, 2).ravel().tolist()
        out[f"track {n}"] = item
    json.dump(out, open(os.path.join(dirname, "meshes.json"), "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
