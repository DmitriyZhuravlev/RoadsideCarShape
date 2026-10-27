"""
The whole scene from one roadside video, no masks and no labels (V2X-Seq, intersection A).

  1. the WHOLE reference frame is triangulated: corners + a regular grid, Delaunay;
  2. every vertex is tracked through +-WIN frames with pyramidal Lucas-Kanade (forward-backward check);
  3. a vertex is STATIC when it does not move in the frames next to the reference (track displacement and patch
     change below a few px / grey levels), MOVING when its track moves; vertices without a usable track take the label
     of their neighbours of the same colour; a triangle is static / moving when all three vertices are, mixed
     triangles are torn off (the object boundary);
  4. static triangles do not move, so they carry no parallax: they are taken FLAT, on the ground plane z = 0 (IPM);
  5. moving triangles: connected regions = objects; per object a planar rigid motion (x, y, yaw) per frame and one
     depth per vertex along its reference ray are solved together - reprojection of every tracked vertex, the
     normal smoothness of tri_smooth across every shared edge (weaker across colour edges), the lowest vertices on
     the ground (they fix the scale), 0 <= z <= ZMAX for all, smooth motion in time; started from a rigid fit of the
     IPM points of the lowest vertices;
  6. checked against the LiDAR of the reference frame (evaluation only): depth of every vertex along its ray.

Writes docs/whole_scene_<k>.png, docs/whole_scene_<k>_views.png and docs/whole_scene.json.
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
import scipy.sparse.linalg  # noqa: F401  (sp.linalg)
from scipy.optimize import least_squares
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree

import ipm_track_models as itm
import body_masks as bmk
import colour_flow_split as cfs
import ego_sequence as es
import supervoxels as svx
import wheels as wh
import tri_smooth as ts
import tri_tracks as tt
import v2x_seq_classical as vc
import v2x_seq_demo as vd
import v2x_seq_shape as vs

WIN = int(os.environ.get("WIN", "8"))            # frames on each side of the reference
GRID = 20                                         # px: grid vertices (textureless regions)
STATIC_PX, MOVING_PX = 0.8, 2.0                   # px per frame next to the reference
PATCH_STILL = 8.0                                 # grey levels: mean |patch(k +- 1) - patch(k)| of a static vertex
DE_SAME = 14.0                                    # Lab: label propagation / smoothness weight
LAMBDA = 10.0
ZMAX = 3.2                                        # m: highest point of a road user
FAR = 150.0                                       # m: static ground beyond this is dropped
MIN_TRI = 15
SEG = os.environ.get("SEG", "sv")                 # sv: labels from the video segmentation; vertex: from single tracks
SV_LEVEL = 3                                      # hierarchy level used (cars = own regions, not merged into road)
MIN_SPEED = 1.0                                   # m/s: slower objects are taken as static
MIN_VERTS = int(os.environ.get("MIN_VERTS", "40"))
FLAT_H = 0.4                                      # m: an object whose 95th height percentile is lower is flat   # fewer solved vertices: not enough evidence, laid flat
WHEELS_JOIN = os.environ.get("WHEELS_JOIN", "1") == "1"   # regions with the car's mean flow join it (tyres)
RIDE, RIDE_TOL = 0.45, 0.1                        # m: the colour body's lower edge above the road, on average
RIDE_MEASURE = os.environ.get("RIDE_MEASURE", "0") == "1"   # per car from the mask's road contact (less reliable)
RIDE_TABLE = os.environ.get("RIDE_TABLE", "")
EDGE_BINS = 8                                     # slices along the body for its lower-edge anchors    # json {k: [[cx, cy, body height], ...]} from the first pass
MERGE = os.environ.get("MERGE", "affine")         # parts of a car: "affine" flow consistency, or "mean" flow
MERGE_GAIN, MERGE_TOL = 1.3, 0.3                  # one affine field within 30 % (+0.3 px) of two separate ones
BODY_VERTS = int(os.environ.get("BODY_VERTS", "400"))   # dense vertices per body at least (v7; v6: 150)
BODY_DE = 20.0                                    # Lab: colour clusters merged into the body
BODY_GRID = int(os.environ.get("BODY_GRID", "5"))      # px between the dense body vertices at most (v7; v6: 10)
SV_MOVE = 1.5                                     # px / frame: mean dense flow of a moving region
MOVE_REL = float(os.environ.get("MOVE_REL", "0.3"))   # moving camera: + this share of the ego-motion flow
HORIZON_MARGIN = int(os.environ.get("HORIZON_MARGIN", "30"))   # px above the horizon a moving camera keeps
OUTLINE_STEP = 8                                  # px between outline vertices
SPLIT = os.environ.get("SPLIT", "1") == "1"       # tear the moving mesh where the image motion changes
FLOW_TOL, FLOW_REL = 1.2, 0.2                     # px, share of the flow
TYRES = os.environ.get("TYRES", "1") == "1"        # tyre contacts fix the scale where tyres are visible
TYRE_REACH = 0.6                                  # share of the object's height searched below it for tyres
CONTACT_RES = 3.0                                 # px: a contact that moves differently from its car is wrong
W_CONTACT = 3.0                                   # weight of a contact's reprojection (px)
W_ANCHOR, W_MOTION = 30.0, 20.0                   # per metre: lowest vertices on the ground, smooth motion
NONHOLO = os.environ.get("NONHOLO", "1") == "1"   # a car moves along its heading
W_NONHOLO = 30.0                                  # per metre of sideways step
MIRROR = os.environ.get("MIRROR", "1") == "1"     # below the contact row: the wet-road mirror image, z < 0
PRUNE = int(os.environ.get("PRUNE", "2"))         # rounds of tearing off vertices that do not fit the motion
MAX_TRAVEL = float(os.environ.get("MAX_TRAVEL", "100"))   # px of image motion along a track
FILL = os.environ.get("FILL", "1") == "1"         # holes inside a solved object: depths from its neighbours
RES_MAX, RES_REL = 1.5, 2.5                       # px, times the object's median residual
CONTACT_LOW = 0.25                                # the contact row must lie in the lowest quarter of the object
BOTTOM = 0.12                                     # lowest share of an object's image height = ground contacts
SUFFIX = os.environ.get("SUFFIX", "")
FIGURES = os.environ.get("FIGURES", "1") == "1"
EXPORT = os.environ.get("EXPORT", "0") == "1"
VIDEO = os.environ.get("VIDEO", "0") == "1"
DUMP = os.environ.get("DUMP", "0") == "1"
DUMP_DIR = os.environ.get("DUMP_DIR", "whole_scene_objects")
VIEW_SCALE = 0.5                                  # novel views at half resolution
VIEW_RANGE = 95.0                                 # m: scene shown in the novel views
INK, MUTED = "#1f1f1e", "#8a8984"


# --- mesh and tracks ------------------------------------------------------------------------------------------
def scene_vertices(gray, horizon, bottom=None):
    """Corners + a regular grid between the horizon row and the bottom row (the ego car's hood on a moving camera)."""
    H, W = gray.shape
    bottom = H if bottom is None else bottom
    m = np.zeros_like(gray)
    m[horizon:bottom] = 255
    c = cv2.goodFeaturesToTrack(gray, 9000, 0.004, 7, mask=m)[:, 0]
    gx, gy = np.meshgrid(np.arange(GRID // 2, W, GRID), np.arange(horizon + GRID // 2, bottom, GRID))
    g = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)
    far = cKDTree(c).query(g)[0] > GRID / 3
    P = np.vstack([c, g[far]]).astype(np.float32)
    tri = Delaunay(P).simplices
    e = np.max([np.linalg.norm(P[tri[:, a]] - P[tri[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    return P, tri[e < 2.2 * GRID]


def track_all(grays, r, P):
    """uv (n, F, 2), NaN where lost: tracked from grays[r] through the clip, both ways, forward-backward check."""
    n, F = len(P), len(grays)
    uv = np.full((n, F, 2), np.nan, np.float32)
    uv[:, r] = P
    for step in (1, -1):
        alive = np.ones(n, bool)
        cur = P.copy()
        i = r
        while 0 <= i + step < F and alive.any():
            a, b = grays[i], grays[i + step]
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, cur[:, None], None, **tt.LK)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, **tt.LK)
            nxt = nxt[:, 0]
            inside = (nxt[:, 0] >= 0) & (nxt[:, 1] >= 0) & (nxt[:, 0] < a.shape[1]) & (nxt[:, 1] < a.shape[0])
            alive &= (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back[:, 0] - cur, axis=1) < tt.FB_TOL) & inside
            i += step
            uv[alive, i] = nxt[alive]
            cur = np.where(alive[:, None], nxt, cur)
    return uv


def patch_change(grays, r, P, half=3):
    """Mean |patch - reference patch| at the SAME pixel in the frames next to the reference (box filter)."""
    q = np.clip(P.round().astype(int), 0, [grays[r].shape[1] - 1, grays[r].shape[0] - 1])
    out = np.zeros(len(P))
    for j in (r - 1, r + 1):
        if 0 <= j < len(grays):
            d = cv2.blur(cv2.absdiff(grays[j], grays[r]).astype(np.float32), (2 * half + 1, 2 * half + 1))
            out = np.maximum(out, d[q[:, 1], q[:, 0]])
    return out


def vertex_lab(frame, P):
    lab = cv2.cvtColor(cv2.blur(frame, (5, 5)), cv2.COLOR_BGR2LAB).astype(np.float32)
    q = np.clip(P.round().astype(int), 0, [frame.shape[1] - 1, frame.shape[0] - 1])
    L = lab[q[:, 1], q[:, 0]]
    return np.c_[L[:, 0] * 100 / 255, L[:, 1] - 128, L[:, 2] - 128]


def classify(uv, r, change, tri, lab):
    """0 static, 1 moving, -1 unknown -> propagated over same-colour mesh edges."""
    near = [j for j in (r - 1, r + 1) if 0 <= j < uv.shape[1]]
    d = np.stack([np.linalg.norm(uv[:, j] - uv[:, r], axis=1) for j in near], 1)
    tracked = ~np.isnan(d)
    dmax = np.where(tracked, d, 0).max(1)
    lab_v = np.full(len(uv), -1)
    lab_v[tracked.all(1) & (dmax < STATIC_PX) & (change < 2 * PATCH_STILL)] = 0
    lab_v[(~tracked.any(1)) & (change < 0.5 * PATCH_STILL)] = 0     # textureless and unchanged
    lab_v[tracked.any(1) & (dmax > MOVING_PX)] = 1
    e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
    e = e[np.linalg.norm(lab[e[:, 0]] - lab[e[:, 1]], axis=1) < DE_SAME]
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(len(uv), len(uv))).tocsr()
    for _ in range(30):
        unk = lab_v < 0
        if not unk.any():
            break
        votes_m = A @ (lab_v == 1).astype(float)
        votes_s = A @ (lab_v == 0).astype(float)
        new = np.where(votes_m > votes_s, 1, np.where(votes_s > 0, 0, -1))
        changed = unk & (new >= 0)
        if not changed.any():
            break
        lab_v[changed] = new[changed]
    return lab_v


def vertex_flow(uv, r):
    """Image motion per frame around the reference (mean of the backward and forward steps), NaN if untracked."""
    f = [uv[:, j] - uv[:, r] if j > r else uv[:, r] - uv[:, j] for j in (r - 1, r + 1) if 0 <= j < uv.shape[1]]
    return np.nanmean(np.stack(f), axis=0) if len(f) > 1 else f[0]


def same_motion(fa, fb):
    """Two vertices move together: flows within FLOW_TOL px or FLOW_REL of their size (untracked: no evidence)."""
    d = np.linalg.norm(fa - fb, axis=-1)
    size = np.maximum(np.linalg.norm(fa, axis=-1), np.linalg.norm(fb, axis=-1))
    return ~(d > np.maximum(FLOW_TOL, FLOW_REL * size))


def objects(tri, moving_tri, min_tri=MIN_TRI, flow=None, lab=None):
    """Connected regions of the moving triangles; two triangles are connected through a shared edge whose two
    vertices move together (the mesh tears where the motion changes: one object per rigid motion)."""
    idx = np.nonzero(moving_tri)[0]
    t_of_edge = {}
    for i, t in enumerate(tri[idx]):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            t_of_edge.setdefault((min(a, b), max(a, b)), []).append(i)
    rows = [(v[0], v[1], e[0], e[1]) for e, v in t_of_edge.items() if len(v) == 2]
    rows = np.array(rows, int).reshape(-1, 4)
    if flow is not None and len(rows):
        # a triangle is linked to its neighbour when all its vertices move like the two shared ones
        ok = same_motion(flow[rows[:, 2]], flow[rows[:, 3]])
        for side in (0, 1):
            T = tri[idx[rows[:, side]]]
            for c in range(3):
                ok &= same_motion(flow[T[:, c]], flow[rows[:, 2]]) | np.isnan(flow[T[:, c], 0]) \
                    | np.isnan(flow[rows[:, 2], 0])
        rows = rows[ok]
    rows = rows[:, :2]
    G = sp.coo_matrix((np.ones(len(rows)), (rows[:, 0], rows[:, 1])), shape=(len(idx), len(idx)))
    n, lab = connected_components(G, directed=False)
    out = [idx[lab == c] for c in range(n) if (lab == c).sum() >= min_tri]
    return sorted(out, key=len, reverse=True)


def ground_at(cam, uv, z):
    """Pixels -> points on the horizontal plane at height z (IPM of a raised plane)."""
    if z == 0.0:
        return itm.ground_point(cam, uv)
    o0 = {"pos": np.zeros(2), "yaw": 0.0}
    C, D = tt.rays(cam, o0, uv)
    C = np.array(C)
    return (C + ray_at_height(C, D, z)[:, None] * D)[:, :2]


def body_mask(frame, mask, k=3, de_merge=15.0, seed=0):
    """The car body inside a car's foreground mask by colour: k-means in Lab, the largest cluster and the clusters
    close to it in colour; tyres, dark trims, windows of another colour and the remaining shadow fall out."""
    lab = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    X = lab[mask]
    if len(X) < 200:
        return mask.copy()
    cv2.setRNGSeed(seed)
    _, labels, cen = cv2.kmeans(X, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3,
                                cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    big = np.argmax(np.bincount(labels, minlength=k))
    keep = np.linalg.norm(cen - cen[big], axis=1) <= de_merge
    body = np.zeros(mask.shape, np.uint8)
    body[mask] = keep[labels]
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(body)
    if n < 2:
        return np.zeros(mask.shape, bool)
    body = cc == 1 + np.argmax(st[1:, 4])
    return bmk.fill_holes(body) & mask


_RIDES = None


def table_ride(k, body, reach=60.0):
    """The body height of this car from the first pass (car_fusion: its body type -> typical height), found by the
    body's image centroid in the same window; RIDE when the car was not modelled in the first pass."""
    global _RIDES
    if _RIDES is None:
        _RIDES = {int(a): b for a, b in json.load(open(RIDE_TABLE)).items()}
    ys, xs = np.nonzero(body)
    c = np.array([xs.mean(), ys.mean()])
    best, bd = RIDE, reach
    for cx, cy, ride in _RIDES.get(k, []):
        d = np.hypot(cx - c[0], cy - c[1])
        if d < bd:
            best, bd = ride, d
    return best


def body_height(cam, car, body, min_gap=2):
    """Height of the body's lower edge above the road, measured in the frame: in every column the car's mask (with
    tyres, reflection cut) ends on the road (z = 0); the body's lowest pixel lies on the vertical line through that
    road point, its ray meets that line at the body's height. Median over the columns with a gap; RIDE if none."""
    cols = np.nonzero(body.any(0) & car.any(0))[0]
    if not len(cols):
        return RIDE
    H = car.shape[0]
    yb = H - 1 - np.argmax(body[::-1, cols], axis=0)
    yc = H - 1 - np.argmax(car[::-1, cols], axis=0)
    ok = yc - yb >= min_gap
    if ok.sum() < 5:
        return RIDE
    G = np.c_[itm.ground_point(cam, np.c_[cols[ok], yc[ok]].astype(float)), np.zeros(ok.sum())]
    o0 = {"pos": np.zeros(2), "yaw": 0.0}
    Cb, Db = tt.rays(cam, o0, np.c_[cols[ok], yb[ok]].astype(float))
    Cb = np.array(Cb)
    # closest approach of the ray C + s D and the vertical line G + t z: t = the body's height there
    z = np.array([0.0, 0.0, 1.0])
    w0 = Cb - G
    a, b, c = (Db * Db).sum(1), Db @ z, 1.0
    d, e = (Db * w0).sum(1), w0 @ z
    t = (a * e - b * d) / np.maximum(a * c - b * b, 1e-9)
    return float(np.clip(np.median(t), 0.05, 1.0))


def affine_flow_rms(Ff, ys, xs):
    """RMS residual (px) of the best affine flow field u = A [x, y, 1] over the pixels."""
    M = np.c_[xs, ys, np.ones(len(xs))].astype(float)
    F = Ff[ys, xs].astype(float)
    A, *_ = np.linalg.lstsq(M, F, rcond=None)
    return float(np.sqrt(np.mean(np.sum((M @ A - F) ** 2, axis=1))))


def one_motion(Ff, ma, mb, sample=4000, seed=0):
    """Two touching parts move as one rigid car when ONE affine flow field (rotation, perspective, translation of a
    turning car) explains both nearly as well as two separate ones."""
    rng = np.random.default_rng(seed)
    pts = []
    for m in (ma, mb):
        ys, xs = np.nonzero(m)
        i = rng.choice(len(xs), min(sample, len(xs)), replace=False)
        pts.append((ys[i], xs[i]))
    ra, rb = affine_flow_rms(Ff, *pts[0]), affine_flow_rms(Ff, *pts[1])
    ru = affine_flow_rms(Ff, np.r_[pts[0][0], pts[1][0]], np.r_[pts[0][1], pts[1][1]])
    return ru <= MERGE_GAIN * max(ra, rb) + MERGE_TOL


def merge_parts(dets, Ff, shape):
    """Full-frame masks of the detections; touching parts whose mean flows agree (within max(1.5 px, 35 %)) are one
    car again - the flow clusters also cut a near, turning car whose image motion varies across its body."""
    H, W = shape
    ms = []
    for b in dets:
        x0, y0, x1, y1 = b["box"]
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = b["m"] > 0
        ms.append(m)
    changed = True
    while changed and len(ms) > 1:
        changed = False
        fl = [Ff[m].mean(0) for m in ms]
        for a in range(len(ms)):
            grown = cv2.dilate(ms[a].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            for b in range(a + 1, len(ms)):
                v = max(np.linalg.norm(fl[a]), np.linalg.norm(fl[b]))
                same = one_motion(Ff, ms[a], ms[b]) if MERGE == "affine" else \
                    np.linalg.norm(fl[a] - fl[b]) <= max(1.5, 0.35 * v)
                if (grown & ms[b]).any() and same:
                    ms[a] = ms[a] | ms[b]
                    del ms[b]
                    changed = True
                    break
            if changed:
                break
    return ms


def mask_labels(seq, ks, r, horizon, P, uv, grays):
    """Cars from background subtraction: the foreground against the video's median background, the wet-road
    reflection cut at the contact line, split into cars by optical-flow clusters (colour_flow_split), the cast shadow
    removed (HSV test), moving ones kept; the BODY of each by colour clustering. A dense triangulation of every body
    (grid + corners + outline 2 px inside); vertices of a car outside its body (tyres, windows, shadow) are torn off,
    everything else is static ground."""
    k = ks[r]
    frame = seq.frames[k]
    H, W = seq.shape
    # flow to the previous frame that has an image, per time slot (gaps in the recording)
    if r > 0:
        F = cfs.flows([seq.frames[ks[r - 1]], frame])[1] / (k - ks[r - 1])
    else:
        F = cfs.flows([frame, seq.frames[ks[r + 1]]])[0] / (ks[r + 1] - k)
    Ff = cfs.full_res(F, (H, W))
    bodies, rides, cars = [], [], np.zeros((H, W), bool)
    bg = seq.background(k)
    bottom = getattr(seq, "hood", H)
    if bottom < H:                                       # a camera at car height: the scenery above the horizon
        top = max(0, horizon - HORIZON_MARGIN)          # moves by parallax - no road user is that tall there
        bg = bg.copy()
        bg[:top] = frame[:top]
    for m in merge_parts(cfs.detect(frame, bg, F), Ff, (H, W)):
        ys_, xs_ = np.nonzero(m)
        x0, y0, x1, y1 = xs_.min(), ys_.min(), xs_.max() + 1, ys_.max() + 1
        if m[:horizon].any() and m[horizon:].sum() < 0.5 * m.sum():
            continue
        if y1 >= bottom - 3:
            continue                                     # cut by the ego car's hood: no road contact
        pad = 4
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
        sub = itm.shadow_free(frame[Y0:Y1, X0:X1], bg[Y0:Y1, X0:X1], m[Y0:Y1, X0:X1])
        m = np.zeros((H, W), bool)
        m[Y0:Y1, X0:X1] = sub
        # moving: its image motion differs from what it would have standing still on the road (zero for a fixed
        # camera; for a moving one the ego motion seen through every column's road contact)
        if m.sum() < cfs.MIN_AREA:
            continue
        sf = seq.static_flow(k, m)
        with np.errstate(all="ignore"):
            res = np.nanmean(np.linalg.norm(Ff[m] - sf, axis=1))
            ego = np.nanmean(np.linalg.norm(sf, axis=1))
        if not np.isfinite(res) or res < SV_MOVE + MOVE_REL * ego:
            continue                                     # a stopped car (no parallax), or no road contact
        body = body_mask(frame, m, de_merge=BODY_DE)
        if body.sum() < 0.5 * cfs.MIN_AREA:
            continue
        cars |= m
        bodies.append(body)
        rides.append(table_ride(k, body) if RIDE_TABLE else (body_height(seq.cams[k], m, body) if RIDE_MEASURE else RIDE))
    body_id = -np.ones((H, W), np.int32)
    for g, bm_ in enumerate(bodies):
        body_id[bm_ & (body_id < 0)] = g
    # vertices: the scene's own outside the cars, dense ones on every body
    q = np.clip(P.round().astype(int), 0, [W - 1, H - 1])
    keep = ~cars[q[:, 1], q[:, 0]]
    P, uv = P[keep], uv[keep]
    extra = []
    for g, bm_ in enumerate(bodies):
        inner = cv2.erode(bm_.astype(np.uint8), np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero(inner)
        if not len(xs):
            continue
        step = int(np.clip(np.sqrt(len(xs) / BODY_VERTS), 3, BODY_GRID))     # at least ~BODY_VERTS per body
        grid = np.c_[xs, ys][(xs % step == 0) & (ys % step == 0)]
        cn = cv2.goodFeaturesToTrack(grays[r], 400, 0.005, 5, mask=inner)
        cn = cn[:, 0] if cn is not None else np.zeros((0, 2))
        ol = [c[::OUTLINE_STEP, 0] for c in cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
              if len(c) >= 3 * OUTLINE_STEP]
        extra += [grid, cn] + ol
    if extra:
        E = np.vstack(extra).astype(np.float32)
        E = E[np.r_[True, cKDTree(E).query(E, k=2)[0][1:, 1] > 2.5]] if len(E) > 1 else E
        P = np.vstack([P, E]).astype(np.float32)
        uv = np.concatenate([uv, track_all(grays, r, E)])
    tri = Delaunay(P).simplices
    e = np.max([np.linalg.norm(P[tri[:, a]] - P[tri[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    tri = tri[e < 2.2 * GRID]
    q = np.clip(P.round().astype(int), 0, [W - 1, H - 1])
    body_of = body_id[q[:, 1], q[:, 0]]
    lv = np.where(body_of >= 0, 1, np.where(cars[q[:, 1], q[:, 0]], -1, 0))
    vis = {"regs": np.where(body_id >= 0, body_id, -1), "moving": np.ones(len(bodies), bool), "cars": cars,
           "rides": rides}
    return P, tri, uv, vertex_lab(seq.frames[k], P), lv, body_of, vis


def car_silhouette(gray, P, tri, idx, frame=None, bg=None):
    """The object's region in the reference frame, and its silhouette with the tyres: what differs from the video's
    median background inside the region's hull extended down by TYRE_REACH, the parts touching the region, the
    wet-road reflection cut off at the contact line (v2x_seq_classical). (region, silhouette or None)"""
    m = np.zeros(gray.shape, np.uint8)
    for i in idx:
        cv2.fillConvexPoly(m, P[tri[i]].round().astype(np.int32), 1)
    ys, xs = np.nonzero(m)
    if len(ys) < 50 or frame is None or bg is None:
        return m, None
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    area = cv2.fillConvexPoly(np.zeros_like(m), hull, 1)
    reach = int(TYRE_REACH * (ys.max() - ys.min()))
    area = cv2.dilate(area, np.ones((2 * reach + 1, 9), np.uint8), anchor=(4, 0))   # grows downwards only
    fg = vc.foreground(frame, bg) * area
    n, lab_, st, _ = cv2.connectedComponentsWithStats(fg)
    touch = np.unique(lab_[(m > 0) & (fg > 0)])
    sil = np.isin(lab_, touch[touch > 0]).astype(np.uint8)
    b = vc.make_blob(sil, 0, 0)
    if b is None:
        return m, None
    b = vc.cut_reflection(b, gray)
    sil = np.zeros_like(m)
    x0, y0, x1, y1 = b["box"]
    sil[y0:y1, x0:x1] = b["m"]
    return m, sil


def tyre_contacts(grays, r, P, tri, idx, frame=None, bg=None):
    """Ground contacts of the object's visible tyres in the reference frame (rims by Hough circles, the tyre's lower
    edge below them - wheels.py) found on its silhouette with the tyres, tracked through the window like any vertex.
    None when no tyre is found."""
    m, sil = car_silhouette(grays[r], P, tri, idx, frame, bg)
    cts = wh.wheel_contacts(grays[r], sil if sil is not None else m)
    if not cts:
        return None
    return track_all(grays, r, np.array(cts, np.float32))


def contact_y(gray, P, tri, idx):
    """Image row of the object's contact with the ground: the lowest sharp horizontal edge (v2x_seq_classical)."""
    m = np.zeros(gray.shape, np.uint8)
    for i in idx:
        cv2.fillConvexPoly(m, P[tri[i]].round().astype(np.int32), 1)
    b = vc.make_blob(m, 0, 0)
    if b is None:
        return None
    row = float(vc.contact_row(gray, b))
    y0, y1 = b["box"][1], b["box"][3]
    # a contact row in the upper body is a strong edge of the car itself (trunk, bumper seen from behind)
    return row if row >= y1 - CONTACT_LOW * (y1 - y0) else None


def reflection_split(gray, P, tri, idx):
    """An object's triangles above its contact row (sharp underside, v2x_seq_classical.contact_row) and below it
    (the blurred mirror image on the wet road: a virtual object BELOW the ground, it must not be anchored)."""
    m = np.zeros(gray.shape, np.uint8)
    for i in idx:
        cv2.fillConvexPoly(m, P[tri[i]].round().astype(np.int32), 1)
    b = vc.make_blob(m, 0, 0)
    if b is None:
        return idx, idx[:0]
    row = vc.contact_row(gray, b)
    below = P[tri[idx], 1].mean(1) > row + 1
    return idx[~below], idx[below]


def fill_holes(s, P, tri, idx0, cam):
    """Triangles of the object's original region whose vertices were torn off (tracks that did not fit) but lie
    inside the outline of the solved part: their depths along their own rays come from the solved neighbours (graph
    Laplace equation with the solved vertices fixed). The halo outside the outline stays torn off."""
    hull = cv2.convexHull(P[s["verts"]].astype(np.float32))
    cand = idx0[~np.isin(idx0, s["tri_idx"])]
    if not len(cand):
        return
    cen = P[tri[cand]].mean(1)
    inside = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in cen])
    cand = cand[inside]
    new_v = np.setdiff1d(np.unique(tri[cand]), s["verts"])
    if not len(new_v):
        return
    Cn, Dn = tt.rays(cam, {"pos": np.zeros(2), "yaw": 0.0}, P[new_v])
    Cn = np.array(Cn)
    Dn = Dn / np.linalg.norm(Dn, axis=1, keepdims=True)
    known = {v: i for i, v in enumerate(s["verts"])}
    unk = {v: i for i, v in enumerate(new_v)}
    Cr0 = Cn[0]
    s_known = np.linalg.norm(s["X"] - Cr0, axis=1)
    rows, cols, vals, rhs = [], [], [], np.zeros(len(new_v))
    edges = set()
    for t in tri[np.r_[s["tri_idx"], cand]]:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges.add((min(a, b), max(a, b)))
    deg = np.zeros(len(new_v))
    for a, b in edges:
        for x, y in ((a, b), (b, a)):
            if x in unk:
                deg[unk[x]] += 1
                if y in unk:
                    rows.append(unk[x]); cols.append(unk[y]); vals.append(-1.0)
                elif y in known:
                    rhs[unk[x]] += s_known[known[y]]
    A = sp.coo_matrix((np.r_[vals, deg + 1e-6], (np.r_[rows, np.arange(len(new_v))], np.r_[cols, np.arange(len(new_v))])),
                      shape=(len(new_v), len(new_v))).tocsr()
    depth = sp.linalg.spsolve(A, rhs)
    ok = np.isfinite(depth) & (depth > 1)
    Xn = Cn + depth[:, None] * Dn
    Xn[:, 2] = np.clip(Xn[:, 2], 0, ZMAX)
    keep_t = cand[np.isin(tri[cand], np.r_[s["verts"], new_v[ok]]).all(1)]
    s["verts"] = np.r_[s["verts"], new_v[ok]]
    s["X"] = np.vstack([s["X"], Xn[ok]])
    s["filled"] = new_v[ok]
    s["tri_idx"] = np.r_[s["tri_idx"], keep_t]


def largest_part(tri, idx):
    """The largest edge-connected part of the triangles idx."""
    if not len(idx):
        return idx
    parts = objects(tri, np.isin(np.arange(len(tri)), idx), min_tri=1)
    return parts[0] if parts else idx[:0]


# --- geometry -------------------------------------------------------------------------------------------------
def planar_procrustes(A, B, w=None):
    """Weighted 2D rigid fit B ~ R(theta) A + t."""
    if len(A) < 2:
        return 0.0, np.zeros(2)
    w = np.ones(len(A)) if w is None else w
    ca, cb = np.average(A, 0, w), np.average(B, 0, w)
    a, b = A - ca, B - cb
    th = np.arctan2(np.sum(w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])), np.sum(w * (a * b).sum(1)))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return float(th), cb - R @ ca


def ransac_procrustes(A, B, tol=0.35, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    best = np.zeros(len(A), bool)
    for _ in range(iters if len(A) > 2 else 0):
        i = rng.choice(len(A), 2, replace=False)
        th, t = planar_procrustes(A[i], B[i])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        inl = np.linalg.norm(A @ R.T + t - B, axis=1) < tol
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 3:
        return None
    return planar_procrustes(A[best], B[best])


def ray_at_height(C, D, z):
    """Depth along the ray (C + s D) where it reaches height z."""
    return (z - C[..., 2]) / np.where(np.abs(D[..., 2]) > 1e-9, D[..., 2], -1e-9)


def ray_interval(C, D, z_lo, z_hi, near=0.5):
    """Depths along the rays (C + s D) where they stay between heights z_lo and z_hi, in front of the camera (s >=
    near): rays may go down (a camera above the objects) or up (a car's camera below a van's roof). (lo, hi)"""
    a, b = ray_at_height(C, D, z_lo), ray_at_height(C, D, z_hi)
    up = D[..., 2] > 0
    lo = np.maximum(np.where(up, a, b), near)
    hi = np.where(up, b, a)
    hi = np.where(np.abs(D[..., 2]) > 1e-9, hi, np.inf)
    return lo, np.maximum(hi, lo + 1e-3)


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def move(X, m, c):
    """Planar rigid motion m = (theta, tx, ty) about the centre c (2,) of the object."""
    xy = (X[:, :2] - c) @ rot(m[0]).T + c + m[1:]
    return np.c_[xy, X[:, 2]]


def solve_object(cams, ks, r, uv, tri_o, lab, m_init=None, fix_motion=False, contact_y=None, contacts=None,
                 ride=None):
    """Joint planar motion of one object over the clip and depths of its vertices (reference rays)."""
    verts = np.unique(tri_o)
    remap = -np.ones(uv.shape[0], int)
    remap[verts] = np.arange(len(verts))
    T = remap[tri_o]
    U = uv[verts].copy()
    n, F = len(verts), len(ks)
    # long tracks of a near, fast object drift (moving highlights on its paint): keep +-half frames, at most
    # MAX_TRAVEL px of image motion
    f = np.nanmedian(np.linalg.norm(vertex_flow(uv[verts], r), axis=1))
    if not np.isfinite(f):                                  # nothing inside was tracked
        return None
    half = int(np.clip(MAX_TRAVEL / max(f, 1e-3), 3, F))
    for j in range(F):
        if abs(j - r) > half:
            U[:, j] = np.nan
    K, R, t = cams[ks[r]]
    o0 = {"pos": np.zeros(2), "yaw": 0.0}
    Cr, Dr = tt.rays(cams[ks[r]], o0, U[:, r])
    Cr = np.array(Cr)
    Dr = Dr / np.linalg.norm(Dr, axis=1, keepdims=True)
    s_lo = ray_at_height(Cr, Dr, ZMAX)
    ys = U[:, r, 1]
    span_y = max(ys.max() - ys.min(), 10)
    if contact_y is None and ride is not None:
        # the body's lower edge (a sill is horizontal) along its WHOLE length: the lowest vertex in each of
        # EDGE_BINS slices along the image x of the body - lowest rows alone bunch at one end and tip the car
        mirror = np.zeros(n, bool)
        xs_ = U[:, r, 0]
        edges = np.linspace(xs_.min(), xs_.max() + 1e-6, EDGE_BINS + 1)
        bottom = np.zeros(n, bool)
        for a, b in zip(edges[:-1], edges[1:]):
            sel = np.nonzero((xs_ >= a) & (xs_ < b))[0]
            if len(sel):
                bottom[sel[np.argmax(ys[sel])]] = True
    elif contact_y is None:
        mirror = np.zeros(n, bool)
        bottom = ys >= min(ys.max() - BOTTOM * span_y, np.sort(ys)[-min(4, n)])      # anchored on the ground
    else:
        # below the contact row: the mirror image on the wet road, a virtual object under the ground (z < 0);
        # the vertices at the contact row touch the ground
        mirror = ys > contact_y + 2
        bottom = np.abs(ys - contact_y) <= max(3.0, 0.06 * span_y)
        if bottom.sum() < 3:
            bottom = np.argsort(np.abs(ys - contact_y))[:min(4, n)]
            bottom = np.isin(np.arange(n), bottom)
    s_hi = np.where(mirror, ray_at_height(Cr, Dr, -ZMAX), ray_at_height(Cr, Dr, 0.0))
    s_gnd = ray_at_height(Cr, Dr, 0.0)
    body_lo, body_hi = (np.where(mirror, a_, b_) for a_, b_ in zip(ray_interval(Cr, Dr, -ZMAX, 0.0),
                                                                  ray_interval(Cr, Dr, 0.0, ZMAX)))
    # tyre contacts: fixed points of the car frame ON the ground; in the reference frame each one is exactly where
    # its pixel ray meets the road, its track fixes the motion and the scale - no body vertex is pinned then
    Xc, Uc = np.zeros((0, 3)), np.zeros((0, F, 2))
    if contacts is not None and len(contacts):
        good_c = (~np.isnan(contacts[:, :, 0])).sum(1) >= 4
        if good_c.any():
            Uc = contacts[good_c].astype(float)
            Cc, Dc = tt.rays(cams[ks[r]], o0, Uc[:, r])
            Cc = np.array(Cc)
            Dc = Dc / np.linalg.norm(Dc, axis=1, keepdims=True)
            Xc = Cc + ray_at_height(Cc, Dc, 0.0)[:, None] * Dc
            bottom = np.zeros(n, bool)
    ys_body = np.where(mirror, -np.inf, ys)
    low = ys_body >= min(ys_body.max() - 0.3 * span_y, np.sort(ys_body)[-min(8, (~mirror).sum())])  # initial motion
    # initial motion: rigid fit of the IPM points of the lowest vertices (they touch the ground)
    G = {j: np.full((n, 2), np.nan) for j in range(F)}
    for j in range(F):
        ok = ~np.isnan(U[:, j, 0])
        G[j][ok] = ground_at(cams[ks[j]], U[ok, j], ride if ride is not None else 0.0)
    if len(Xc):                                 # the contacts' own IPM points join the initial fit (weighted x4)
        for j in range(F):
            Gc = np.full((len(Uc), 2), np.nan)
            okc = ~np.isnan(Uc[:, j, 0])
            Gc[okc] = itm.ground_point(cams[ks[j]], Uc[okc, j])
            G[j] = np.vstack([G[j]] + [Gc] * 4)
        low = np.r_[low, np.ones(4 * len(Uc), bool)]
    c = np.nanmean(G[r][low], axis=0)
    tol = 0.3 + 0.015 * np.linalg.norm(c - Cr[0, :2])                  # IPM noise grows with the range
    m0 = np.full((F, 3), np.nan)
    m0[r] = 0
    for j in range(F):
        if j == r:
            continue
        ok = low & ~np.isnan(G[j][:, 0])
        if ok.sum() >= 3:
            f = ransac_procrustes(G[r][ok] - c, G[j][ok] - c, tol=tol)
            if f is not None:
                m0[j] = (f[0], *f[1])
    good = ~np.isnan(m0[:, 0])
    if good.sum() < 3:
        return None
    jj = np.arange(F)
    deg = min(2, good.sum() - 1)
    for a in (1, 2):
        p = np.polyfit(jj[good] - r, m0[good, a], deg, w=np.where(jj[good] == r, 100.0, 1.0))
        m0[:, a] = np.polyval(p, jj - r)
    m0[r] = 0
    # a car drives along its heading (no side slip): the yaw change is the change of the direction of its path
    vel = np.gradient(m0[:, 1:], axis=0)
    head = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    moved = np.linalg.norm(m0[-1, 1:] - m0[0, 1:]) > 0.5
    m0[:, 0] = (head - head[r]) if (NONHOLO and moved) else (m0[:, 0] if not NONHOLO else 0.0)
    m0[r] = 0
    u = vel[r] / max(np.linalg.norm(vel[r]), 1e-9)              # direction of travel in the reference frame
    if m_init is not None:                      # (checks only) a given motion: init or fixed
        m0 = m_init(c).copy()
        u = np.gradient(m0[:, 1:], axis=0)[r]
        u /= max(np.linalg.norm(u), 1e-9)
    frames = [j for j in range(F) if j != r and (~np.isnan(U[:, j, 0])).any()]
    fpos = {j: i for i, j in enumerate(frames)}
    # initial depths: ray intersection with the initial motion
    Cs = np.full((n, F, 3), np.nan)
    Ds = np.full((n, F, 3), np.nan)
    for j in range(F):
        ok = ~np.isnan(U[:, j, 0])
        if not ok.any():
            continue
        th, tx, ty = m0[j]
        pos = np.array([tx, ty]) + c - rot(th) @ c                  # world = R X + pos
        Cj, Dj = tt.rays(cams[ks[j]], {"pos": pos, "yaw": th}, U[ok, j])
        Cs[ok, j], Ds[ok, j] = Cj, Dj
    X0, _ = tt.intersect_rays(Cs, Ds)
    s0 = np.einsum("ni,ni->n", X0 - Cr, Dr)
    s_mid = ray_at_height(Cr, Dr, 0.8)
    s0 = np.where(np.isfinite(s0), s0, s_mid)
    s0 = np.where(mirror & ~np.isfinite(X0[:, 0]), ray_at_height(Cr, Dr, -0.8), s0)
    s0 = np.clip(s0, body_lo + 1e-3, body_hi - 1e-3)
    s0[bottom] = (ray_at_height(Cr, Dr, ride) if ride is not None else s_gnd)[bottom] - 1e-3
    vi, fi = np.nonzero(~np.isnan(U[:, :, 0]))
    keep = fi != r
    vi, fi = vi[keep], fi[keep]
    obs_uv = U[vi, fi].astype(float)
    pairs = ts.edge_pairs(T)
    t1 = np.stack([pairs[:, 0], pairs[:, 1], pairs[:, 2]], 1)
    t2 = np.stack([pairs[:, 1], pairs[:, 0], pairs[:, 3]], 1)
    lv = lab[verts]
    w = np.where(np.linalg.norm(lv[t1].mean(1) - lv[t2].mean(1), axis=1) < 15.0, 1.0, 0.3) if len(pairs) else np.ones(0)
    if len(pairs):
        w = w * np.where(mirror[t1].any(1) != mirror[t2].any(1), 0.0, 1.0)    # the crease at the contact line
    bi = np.nonzero(bottom)[0]
    z_anchor = ride if ride is not None else 0.0      # the body (no tyres) sits at its measured height above the road
    nm = 3 * len(frames)
    c_i, c_f = np.nonzero(~np.isnan(Uc[:, :, 0])) if len(Uc) else (np.zeros(0, int), np.zeros(0, int))
    keep_c = np.isin(c_f, frames)
    c_i, c_f = c_i[keep_c], c_f[keep_c]
    # temporal smoothness: second differences over consecutive frames (the reference motion is 0)
    trip = [(a, a + 1, a + 2) for a in range(F - 2)]

    def unpack(x):
        M = np.zeros((F, 3))
        for j, i in fpos.items():
            M[j] = x[3 * i:3 * i + 3]
        return M, x[nm:]

    def fun(x):
        M, s = unpack(x)
        X = Cr + s[:, None] * Dr
        res = np.zeros((len(vi), 2))
        for j in frames:
            sel = fi == j
            res[sel] = vs.project(cams[ks[j]], move(X[vi[sel]], M[j], c))[0] - obs_uv[sel]
        sm = LAMBDA * w[:, None] * (ts.normals(X, t1) - ts.normals(X, t2))
        an = W_ANCHOR * (X[bi, 2] - z_anchor)
        cr = np.zeros((len(c_i), 2))
        for j in set(c_f):
            sel = c_f == j
            cr[sel] = vs.project(cams[ks[j]], move(Xc[c_i[sel]], M[j], c))[0] - Uc[c_i[sel], j]
        cr = W_CONTACT * cr.ravel()
        # motion priors divided by the path length: scale-free, so that they cannot pull the object towards the
        # camera (a smaller copy closer to the camera moves less and would pay less)
        norm = path_ref / (np.linalg.norm(np.diff(M[:, 1:], axis=0), axis=1).sum() + 0.3)
        # second differences per time slot: the frames of a window may have gaps (missing images)
        acc = np.array([M[a] - 2 * M[b] + M[cc] if ks[cc] - ks[a] == 2 else
                        (M[cc] - M[b]) / (ks[cc] - ks[b]) - (M[b] - M[a]) / (ks[b] - ks[a])
                        for a, b, cc in trip]).ravel()
        acc = W_MOTION * acc * np.tile([8.0 / norm, 1.0, 1.0], len(trip)) * norm   # yaw in rad ~ 8 m at car size
        nh = np.zeros(F - 1)
        if NONHOLO:
            for a in range(F - 1):
                d = rot(-(M[a, 0] + M[a + 1, 0]) / 2) @ (M[a + 1, 1:] - M[a, 1:])
                nh[a] = W_NONHOLO * norm * (u[0] * d[1] - u[1] * d[0])  # sideways step in the car's frame
        return np.r_[res.ravel(), sm.ravel(), an, cr, acc, nh]

    # Jacobian sparsity
    m_rows = len(vi)
    rows, cols = [], []
    rr = np.arange(2 * m_rows)
    rows.append(np.repeat(rr, 4))
    vc = nm + np.repeat(vi, 2)
    mc = 3 * np.array([fpos[j] for j in fi])
    mc = np.repeat(mc, 2)
    cols.append(np.stack([vc, mc, mc + 1, mc + 2], 1).ravel())
    base = 2 * m_rows
    if len(pairs):
        rows.append(base + np.repeat(np.arange(3 * len(pairs)), 4))
        cols.append(nm + np.repeat(pairs, 3, axis=0).ravel())
    base += 3 * len(pairs)
    rows.append(base + np.arange(len(bi)))
    cols.append(nm + bi)
    base += len(bi)
    if len(c_i):
        cm = 3 * np.array([fpos[j] for j in c_f])
        rows.append(base + np.repeat(np.arange(2 * len(c_i)), 3))
        cols.append((np.repeat(cm, 2)[:, None] + np.arange(3)).ravel())
    base += 2 * len(c_i)
    nprior = 3 * len(trip) + F - 1                     # the path length couples every prior row to all motions
    rows.append(np.repeat(base + np.arange(nprior), nm))
    cols.append(np.tile(np.arange(nm), nprior))
    base += nprior
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    S = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(base, nm + n)).tocsr()
    x0 = np.r_[np.concatenate([m0[j] for j in frames]), s0]
    path_ref = np.linalg.norm(np.diff(m0[:, 1:], axis=0), axis=1).sum() + 0.3
    b_lo, b_hi = (ray_interval(Cr, Dr, max(ride - RIDE_TOL, 0.0), ride + RIDE_TOL) if ride is not None
                  else ray_interval(Cr, Dr, 0.0, 0.25))
    lo = np.r_[np.full(nm, -np.inf), np.where(bottom, b_lo, body_lo)]
    hi = np.r_[np.full(nm, np.inf), np.where(bottom, b_hi, body_hi)]
    if fix_motion:
        lo[:nm] = x0[:nm] - 1e-5
        hi[:nm] = x0[:nm] + 1e-5
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    sol = least_squares(fun, x0, jac_sparsity=S, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, method="trf",
                        tr_solver="lsmr", x_scale="jac", max_nfev=60)
    M, s = unpack(sol.x)
    X = Cr + s[:, None] * Dr
    res = fun(sol.x)[:2 * m_rows].reshape(-1, 2)
    e = np.linalg.norm(res, axis=1)
    cres = np.zeros(len(Xc))                             # per contact: median reprojection residual (px)
    for i in range(len(Xc)):
        d = [np.linalg.norm(vs.project(cams[ks[j]], move(Xc[i:i + 1], M[j], c))[0][0] - Uc[i, j])
             for j in range(F) if j != r and not np.isnan(Uc[i, j, 0])]
        cres[i] = np.median(d) if d else 0.0
    vres = np.zeros(n)
    for v in range(n):                                   # per vertex: median residual over its track
        sel = vi == v
        vres[v] = np.median(e[sel]) if sel.any() else 0.0
    return {"verts": verts, "X": X, "M": M, "centre": c, "tri": tri_o, "bottom": verts[bottom], "mirror": verts[mirror],
            "vertex_res": vres, "contacts": Xc, "contact_uv": Uc, "contact_res": cres,
            "reprojection_px": float(np.median(np.linalg.norm(res, axis=1))) if len(res) else float("nan"),
            "speed_mps": float(np.linalg.norm(M[min(F - 1, r + 1), 1:] - M[max(0, r - 1), 1:]) /
                               (min(F - 1, r + 1) - max(0, r - 1)) * 10.0 *               # time slots of 0.1 s
                               ((min(F - 1, r + 1) - max(0, r - 1)) / (ks[min(F - 1, r + 1)] - ks[max(0, r - 1)]))),
            "X_ipm": np.c_[itm.ground_point(cams[ks[r]], U[:, r]), np.zeros(n)]}


# --- LiDAR (evaluation only) ----------------------------------------------------------------------------------
def lidar_depth(seq, k, ground, cam):
    """Per pixel the nearest LiDAR return (distance from the camera), NaN elsewhere; and the world points."""
    if hasattr(seq, "lidar_world"):              # moving camera: LiDAR already in the road-plane world
        L = seq.lidar_world(k)
    else:
        L = vd.read_pcd(os.path.join(vd.DATA, seq.info[k]["pointcloud_path"]))
        L = L[np.isfinite(L).all(1)].copy()
        L[:, 2] -= ground
    q, z = vs.project(cam, L)
    H, W = seq.shape
    ok = (z > 1) & (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < W) & (q[:, 1] < H)
    C = -np.linalg.solve(cam[1], cam[2])
    return L[ok], q[ok], np.linalg.norm(L[ok] - C, axis=1)


def depth_errors(Xv, uv, C, Lq, Ld, radius=3.0):
    """Our distance from the camera minus the nearest-in-depth LiDAR return within `radius` px."""
    t = cKDTree(Lq)
    out = np.full(len(Xv), np.nan)
    ours = np.linalg.norm(Xv - C, axis=1)
    for i, idx in enumerate(t.query_ball_point(uv, radius)):
        if len(idx) >= 1 and np.isfinite(ours[i]):
            out[i] = ours[i] - Ld[idx].min()
    return out


# --- rendering ------------------------------------------------------------------------------------------------
def look_at(K, eye, target, up=(0, 0, 1.0)):
    f = target - eye
    f /= np.linalg.norm(f)
    rgt = np.cross(f, up)
    rgt /= np.linalg.norm(rgt)
    dwn = np.cross(f, rgt)
    R = np.stack([rgt, dwn, f])
    return K, R, -R @ eye


def render(cam, shape, X, tri, uv_src, src, bg=245):
    """Texture-mapped painter's rendering of the triangles X[tri] (source pixels uv_src[tri]) into the camera."""
    H, W = shape
    q, z = vs.project(cam, X)
    zt = z[tri].min(1)
    ok = (zt > 0.5) & np.isfinite(q[tri]).all((1, 2))
    order = np.nonzero(ok)[0][np.argsort(-z[tri[ok]].mean(1))]
    ids = -np.ones((H, W), np.int32)
    for i in order:
        p = q[tri[i]]
        if np.abs(p).max() > 1e5:
            continue
        cv2.fillConvexPoly(ids, np.round(p * 4).astype(np.int32), int(i), lineType=cv2.LINE_8, shift=2)
    yy, xx = np.nonzero(ids >= 0)
    t = ids[yy, xx]
    A = q[tri[t]]                                       # destination triangle
    B = uv_src[tri[t]]
    v0, v1 = A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]
    p = np.c_[xx, yy] - A[:, 0]
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    b1 = (p[:, 0] * v1[:, 1] - p[:, 1] * v1[:, 0]) / den
    b2 = (v0[:, 0] * p[:, 1] - v0[:, 1] * p[:, 0]) / den
    src_xy = B[:, 0] + b1[:, None] * (B[:, 1] - B[:, 0]) + b2[:, None] * (B[:, 2] - B[:, 0])
    mapx = np.full((H, W), -1, np.float32)
    mapy = np.full((H, W), -1, np.float32)
    mapx[yy, xx], mapy[yy, xx] = src_xy[:, 0], src_xy[:, 1]
    out = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(bg, bg, bg))
    out[ids < 0] = bg
    return out, ids


def render_points(cam, shape, P, col, bg=245, size=2):
    H, W = shape
    q, z = vs.project(cam, P)
    ok = (z > 0.5) & (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < W) & (q[:, 1] < H)
    out = np.full((H, W, 3), bg, np.uint8)
    zb = np.full((H, W), np.inf)
    o = np.nonzero(ok)[0][np.argsort(-z[ok])]
    qi = q[o].astype(int)
    for dx in range(size):
        for dy in range(size):
            x, y = np.clip(qi[:, 0] + dx, 0, W - 1), np.clip(qi[:, 1] + dy, 0, H - 1)
            out[y, x] = col[o]
    return out


def attach_by_flow(moving, mflow, sizes, pairs, rel=0.25, tol=1.0, max_size_ratio=1.5, rounds=2):
    """Regions touching a moving region join it when their mean flows match (within max(tol, rel |flow|) px) and
    they are not much larger than it. (moving, attached)"""
    moving = moving.copy()
    attached = np.zeros(len(moving), bool)
    sp_ = np.linalg.norm(mflow, axis=1)
    for _ in range(rounds):
        a, b = pairs[:, 0], pairs[:, 1]
        one = moving[a] != moving[b]
        g = np.where(moving[a], a, b)[one]                  # the moving side
        h = np.where(moving[a], b, a)[one]
        ok = (np.linalg.norm(mflow[h] - mflow[g], axis=1) <= np.maximum(tol, rel * sp_[g])) & (sp_[g] > tol) \
            & (sizes[h] <= max_size_ratio * sizes[g])
        if not ok.any():
            break
        moving[h[ok]] = True
        attached[h[ok]] = True
    return moving, attached


def supervoxel_labels(seq, ks, r, horizon, P, uv, lv, grays):
    """Static / moving from the regions of the video segmentation (supervoxels.py, level SV_LEVEL): a region moves when
    its mean dense flow says so and the LK tracks inside it do not contradict; every vertex takes its region's label
    and its region's mean flow (the object split). Extra vertices along the outline of the moving regions, so that the
    mesh tears at the real edge."""
    seg = svx.VideoSegmentation([seq.frames[j] for j in ks], y0=horizon)
    H, W = seq.shape
    regs = seg.full_res(SV_LEVEL, r, (H, W))
    flow = seg.flow_full(r, (H, W))
    n = int(regs.max()) + 1
    ok = regs >= 0
    cnt = np.maximum(np.bincount(regs[ok], minlength=n), 1)
    mag = np.linalg.norm(flow, axis=2)
    mean_mag = np.bincount(regs[ok], mag[ok], minlength=n) / cnt
    mflow = np.stack([np.bincount(regs[ok], flow[..., c][ok], minlength=n) for c in (0, 1)], 1) / cnt[:, None]

    def region_of(Q):
        q = np.clip(Q.round().astype(int), 0, [W - 1, H - 1])
        return regs[q[:, 1], q[:, 0]]
    rv = region_of(P)
    okv = rv >= 0
    votes_m = np.bincount(rv[okv], (lv[okv] == 1).astype(float), minlength=n)
    votes_s = np.bincount(rv[okv], (lv[okv] == 0).astype(float), minlength=n)
    moving_reg = (mean_mag > SV_MOVE) & (votes_m >= votes_s)      # most tracked vertices inside really move
    attached = np.zeros(n, bool)
    if WHEELS_JOIN:
        # a rolling wheel's pixels move from 0 (contact) to 2v (top), the MEAN flow of its region is the car's v. At
        # SV_LEVEL a dark tyre on the dark wet road is merged into the road, so the finest level is used: a level-0
        # region touching a car whose mean flow matches the car's joins that car's region, whatever its colour
        fine = seg.full_res(0, r, (H, W))
        nf = int(fine.max()) + 1
        okf = fine >= 0
        cf = np.maximum(np.bincount(fine[okf], minlength=nf), 1)
        ff = np.stack([np.bincount(fine[okf], flow[..., c][okf], minlength=nf) for c in (0, 1)], 1) / cf[:, None]
        for _ in range(2):
            car = moving_reg[np.maximum(regs, 0)] & ok
            pairs = svx.adjacency_2d(np.where(ok, fine, -1) + 1) - 1
            pairs = pairs[(pairs >= 0).all(1)]
            inside = np.bincount(fine[car], minlength=nf) > 0.5 * cf             # fine regions already in a car
            # the car region of a fine region inside a car: the coarse label of its pixels (mode)
            coarse_of = np.full(nf, -1)
            fr, cr_ = fine[car], regs[car]
            o = np.lexsort((cr_, fr))
            fr, cr_ = fr[o], cr_[o]
            first = np.r_[True, fr[1:] != fr[:-1]]
            coarse_of[fr[first]] = cr_[first]
            a, b = pairs[:, 0], pairs[:, 1]
            one = inside[a] != inside[b]
            g = np.where(inside[a], a, b)[one]
            h = np.where(inside[a], b, a)[one]
            gc = coarse_of[g]
            v = np.linalg.norm(mflow[gc], axis=1)
            good = (gc >= 0) & (v > SV_MOVE) & (np.linalg.norm(ff[h] - mflow[gc], axis=1) <= np.maximum(1.0, 0.25 * v)) \
                & (cf[h] <= 0.5 * cnt[gc])
            if not good.any():
                break
            target = -np.ones(nf, int)                     # fine region -> car region it joins (one pass)
            target[h[good]] = gc[good]
            pix = target[np.maximum(fine, 0)]
            sel = (pix >= 0) & ok & ~car
            regs[sel] = pix[sel]
            attached[gc[good]] = True
    mask = (moving_reg[np.maximum(regs, 0)] & ok).astype(np.uint8)
    # outline vertices of every moving region, 2 px inside it
    extra = []
    for g in np.nonzero(moving_reg)[0]:
        m = cv2.erode(((regs == g) & ok).astype(np.uint8), np.ones((5, 5), np.uint8))
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            if len(c) >= 3 * OUTLINE_STEP:
                extra.append(c[::OUTLINE_STEP, 0])
    if extra:
        extra = np.vstack(extra).astype(np.float32)
        extra = extra[cKDTree(P).query(extra)[0] > 3]
        P = np.vstack([P, extra]).astype(np.float32)
        uv = np.concatenate([uv, track_all(grays, r, extra)])
    tri = Delaunay(P).simplices
    e = np.max([np.linalg.norm(P[tri[:, a]] - P[tri[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    tri = tri[e < 2.2 * GRID]
    rv = region_of(P)
    lv = np.where(rv < 0, -1, np.where(moving_reg[np.maximum(rv, 0)], 1, 0))
    vis = {"regs": regs, "moving": moving_reg, "attached": attached}
    return P, tri, uv, vertex_lab(seq.frames[ks[r]], P), lv, mflow[np.maximum(rv, 0)], vis


# --- one reference frame --------------------------------------------------------------------------------------
def run(seq, k, ground):
    ks = [j for j in range(max(0, k - WIN), min(len(seq.frames), k + WIN + 1)) if seq.frames[j] is not None]
    r = ks.index(k)
    cams = seq.cams
    grays = [cv2.cvtColor(seq.frames[j], cv2.COLOR_BGR2GRAY) for j in ks]
    K, R, t = cams[k]
    C = -np.linalg.solve(R, t)
    # horizon: the image row where the ground is FAR away (straight ahead of the camera)
    H, W = seq.shape
    rows = np.arange(0, H, 4)
    g = itm.ground_point(cams[k], np.c_[np.full(len(rows), W / 2), rows])
    dist = np.linalg.norm(g - C[:2], axis=1)
    infront = (np.c_[g - C[:2]] @ (np.linalg.solve(R, [0, 0, 1.0])[:2])) > 0
    horizon = int(rows[np.argmax(infront & (dist < FAR))]) if (infront & (dist < FAR)).any() else 0
    P, tri = scene_vertices(grays[r], horizon, getattr(seq, "hood", None))
    uv = track_all(grays, r, P)
    change = patch_change(grays, r, P)
    lab = vertex_lab(seq.frames[k], P)
    lv = classify(uv, r, change, tri, lab)
    region_flow, seg_vis = None, None
    body_of = None
    if SEG == "sv":            # labels from the hierarchical video segmentation instead of single vertices
        P, tri, uv, lab, lv, region_flow, seg_vis = supervoxel_labels(seq, ks, r, horizon, P, uv, lv, grays)
    elif SEG == "mask":        # background subtraction, split by flow, the body by colour
        P, tri, uv, lab, lv, body_of, seg_vis = mask_labels(seq, ks, r, horizon, P, uv, grays)
    tri_lab = lv[tri]
    static_tri = (tri_lab == 0).all(1)
    moving_tri = (tri_lab == 1).all(1)
    # static: flat, on the ground
    Xs = np.c_[itm.ground_point(cams[k], P), np.zeros(len(P))]
    Cr, Dr = tt.rays(cams[k], {"pos": np.zeros(2), "yaw": 0.0}, P)
    ahead = np.einsum("ni,ni->n", Xs - np.array(Cr), Dr) > 0
    near = np.linalg.norm(Xs[:, :2] - C[:2], axis=1) < FAR
    static_tri &= (ahead & near)[tri].all(1)
    split_flow = (region_flow if region_flow is not None else vertex_flow(uv, r)) if SPLIT else None
    if body_of is None:
        objs = objects(tri, moving_tri, flow=split_flow)
    else:                      # one object per body mask: triangles whose three vertices lie on that body
        tb = body_of[tri]
        objs = [np.nonzero((tb == g).all(1))[0] for g in range(int(body_of.max()) + 1)]
        ride_of = {}
        for g, o in enumerate(objs):
            if len(o) >= MIN_TRI:
                o = largest_part(tri, o)
                ride_of[id(o)] = seg_vis["rides"][g]
                objs[g] = o
        objs = [o for o in objs if id(o) in ride_of]
    sols, mirrored = [], []
    for idx in objs:
        idx0 = idx
        cy = contact_y(grays[r], P, tri, idx) if MIRROR else None
        cuv = tyre_contacts(grays, r, P, tri, idx) if TYRES else None
        if cuv is not None and MIRROR:
            cy = float(np.max(cuv[:, r, 1]))           # the mirror image begins where the tyres touch the road
        ride = ride_of[id(idx0)] if body_of is not None else None
        s = solve_object(cams, ks, r, uv, tri[idx], lab, contact_y=cy, contacts=cuv, ride=ride)
        if s is not None and len(s["contacts"]) and (s["contact_res"] > CONTACT_RES).any():
            # a contact that does not move with the car (a wrong rim, a reflection): dropped, solved again
            good_c = np.isin(np.arange(len(cuv)), np.nonzero(
                (~np.isnan(cuv[:, :, 0])).sum(1) >= 4)[0][s["contact_res"] <= CONTACT_RES])
            cuv = cuv[good_c] if good_c.any() else None
            cy = (float(np.max(cuv[:, r, 1])) if cuv is not None else contact_y(grays[r], P, tri, idx)) if MIRROR else None
            s = solve_object(cams, ks, r, uv, tri[idx], lab, contact_y=cy, contacts=cuv, ride=ride)
        for _ in range(PRUNE):
            if s is None:
                break
            # vertices that do not follow the object's rigid motion (background dragged along by the tracker at
            # the outline, other objects) or that ran into the height bound: torn off, the rest solved again
            lim = max(RES_MAX, RES_REL * np.median(s["vertex_res"]))     # near, fast objects drift more
            bad = s["verts"][(s["vertex_res"] > lim) | (s["X"][:, 2] > ZMAX - 0.05)]
            if not len(bad):
                break
            idx = idx[~np.isin(tri[idx], bad).any(1)]
            idx = largest_part(tri, idx)
            if len(idx) < MIN_TRI:
                s = None
                break
            cy = contact_y(grays[r], P, tri, idx) if MIRROR else None
            s = solve_object(cams, ks, r, uv, tri[idx], lab, contact_y=cy, contacts=cuv, ride=ride)
        if s is not None:
            body = ~np.isin(s["verts"], s["mirror"])
            s["flat"] = bool(np.percentile(s["X"][body, 2], 95) < FLAT_H) if body.any() else True
        if s is not None and (s["speed_mps"] < MIN_SPEED or len(s["verts"]) < MIN_VERTS or s["flat"]):
            # too slow, too little evidence, or flat (a moving shadow / reflection strip on the road): road surface
            static_tri[idx0] = (ahead & near)[tri[idx0]].all(1)      # does not really move: flat like the road
            s = None
        if s is not None:
            s["tri_idx"] = idx
            if FILL:
                fill_holes(s, P, tri, idx0, cams[k])
            sols.append(s)
    for below in mirrored:                    # the reflection is seen on the road surface: flat like the ground
        static_tri[below] = (ahead & near)[tri[below]].all(1)
    return {"k": k, "ks": ks, "P": P, "tri": tri, "uv": uv, "lv": lv, "static_tri": static_tri, "seg": seg_vis,
            "moving_tri": moving_tri, "Xs": Xs, "objects": sols, "horizon": horizon, "C": C}


def assemble(res):
    """One vertex array for the whole scene: static vertices on the ground, object vertices from their solve."""
    P, tri = res["P"], res["tri"]
    X = res["Xs"].copy()
    keep = res["static_tri"].copy()
    for s in res["objects"]:
        X[s["verts"]] = s["X"]
        keep[s["tri_idx"]] = True
        if len(s.get("mirror", [])):           # the mirror image is seen ON the road surface
            X[s["mirror"]] = res["Xs"][s["mirror"]]
    return X, tri[keep]


def evaluate(res, Lq, Ld):
    P, C = res["P"], res["C"]
    X, _ = assemble(res)
    err = depth_errors(X, P, C, Lq, Ld)
    flat = depth_errors(res["Xs"], P, C, Lq, Ld)
    st = np.unique(res["tri"][res["static_tri"]])
    out = {"k": res["k"], "vertices": int(len(P)), "triangles": int(len(res["tri"])),
           "static_triangles": int(res["static_tri"].sum()), "moving_triangles": int(res["moving_tri"].sum()),
           "objects": len(res["objects"])}
    e = err[st]
    e = e[np.isfinite(e)]
    out["static: |depth err| median m"] = float(np.median(np.abs(e)))
    out["static: share within 0.5 m"] = float(np.mean(np.abs(e) < 0.5))
    out["static: share within 0.5 m, rel. err < 2 %"] = float(np.mean(np.abs(e) / np.maximum(
        np.linalg.norm(res["Xs"][st][np.isfinite(err[st])] - C, axis=1), 1) < 0.02))
    per = []
    filled = np.concatenate([s.get("filled", np.zeros(0, int)) for s in res["objects"]]) if res["objects"] else []
    fe = err[np.asarray(filled, int)]
    fe = fe[np.isfinite(fe)]
    out["filled vertices"] = int(len(filled))
    out["filled: |depth err| median m"] = float(np.median(np.abs(fe))) if len(fe) else float("nan")
    for s in res["objects"]:
        v = s["verts"][~np.isin(s["verts"], s.get("filled", []))]
        a, b = err[v], flat[v]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 5:
            continue
        rng = float(np.median(np.linalg.norm(s["X"] - C, axis=1)))
        per.append({"vertices": int(len(v)), "checked": int(ok.sum()), "range_m": rng,
                    "speed_mps": s["speed_mps"], "reprojection_px": s["reprojection_px"],
                    "ours |depth err| median m": float(np.median(np.abs(a[ok]))),
                    "ours depth err median m (signed)": float(np.median(a[ok])),
                    "flat (IPM) |depth err| median m": float(np.median(np.abs(b[ok]))),
                    "height p95 m": float(np.percentile(s["X"][:, 2], 95))})
    out["moving objects"] = per
    if per:
        solved = [s["verts"][~np.isin(s["verts"], s.get("filled", []))] for s in res["objects"]]
        allv = np.concatenate([err[v] for v in solved])
        allf = np.concatenate([flat[v] for v in solved])
        ok = np.isfinite(allv) & np.isfinite(allf)
        out["moving: ours |depth err| median m"] = float(np.median(np.abs(allv[ok])))
        out["moving: flat |depth err| median m"] = float(np.median(np.abs(allf[ok])))
    return out, err


# --- figures --------------------------------------------------------------------------------------------------
OBJ_COLS = np.array([[42, 120, 214], [235, 104, 52], [46, 158, 107], [138, 92, 194], [201, 162, 39], [214, 69, 122],
                     [56, 178, 201], [122, 75, 42]]) / 255


def overview_figure(path, seq, res, err, Ld, Lq, ev):
    k, P, tri = res["k"], res["P"], res["tri"]
    rgb = cv2.cvtColor(seq.frames[k], cv2.COLOR_BGR2RGB)
    H, W = seq.shape
    fig, axes = plt.subplots(2, 2, figsize=(19, 11.2))
    ax = axes[0, 0]
    ov = rgb.astype(float) / 255
    cls = np.zeros((H, W), np.int32) - 1
    for i in np.nonzero(res["static_tri"])[0]:
        cv2.fillConvexPoly(cls, P[tri[i]].round().astype(np.int32), 0)
    for o, s in enumerate(res["objects"]):
        for i in s["tri_idx"]:
            cv2.fillConvexPoly(cls, P[tri[i]].round().astype(np.int32), 1 + o)
    ov[cls == 0] = 0.75 * ov[cls == 0] + 0.25 * np.array([0.6, 0.6, 0.6])
    for o in range(len(res["objects"])):
        m = cls == 1 + o
        ov[m] = 0.4 * ov[m] + 0.6 * OBJ_COLS[o % len(OBJ_COLS)]
    torn = cls < 0
    ov[torn] = 0.35 * ov[torn] + 0.65
    ax.imshow(ov)
    ax.triplot(P[:, 0], P[:, 1], tri[res["static_tri"]][::3], color="white", lw=0.15, alpha=0.5)
    ax.set_title(f"frame {k}: {len(P)} vertices, {len(tri)} triangles - static (flat, grey tint) "
                 f"{ev['static_triangles']}, moving {ev['moving_triangles']} in {len(res['objects'])} objects (colours), "
                 "white = torn / not used", fontsize=9, loc="left", color=INK)
    X, tri_used = assemble(res)
    dist = np.linalg.norm(X - res["C"], axis=1)
    dimg = np.full((H, W), np.nan)
    order = np.argsort(-dist[tri_used].mean(1))
    for i in order:
        cv2.fillConvexPoly(dimg, P[tri_used[i]].round().astype(np.int32), float(dist[tri_used[i]].mean()))
    vmax = 90
    axes[0, 1].imshow(dimg, cmap="viridis_r", vmin=8, vmax=vmax)
    axes[0, 1].set_title("our depth (distance from the camera, m): flat ground + triangulated moving objects",
                         fontsize=9, loc="left", color=INK)
    order = np.argsort(-Ld)
    axes[1, 0].imshow(np.full((H, W), np.nan), cmap="viridis_r", vmin=8, vmax=vmax)
    sc = axes[1, 0].scatter(Lq[order, 0], Lq[order, 1], c=Ld[order], s=0.6, cmap="viridis_r", vmin=8, vmax=vmax)
    axes[1, 0].set_facecolor("#f5f5f3")
    axes[1, 0].set_title("LiDAR depth of the same frame (evaluation only)", fontsize=9, loc="left", color=INK)
    fig.colorbar(sc, ax=axes[1, 0], fraction=0.025, label="m")
    ok = np.isfinite(err)
    axes[1, 1].imshow(rgb, alpha=0.35)
    s2 = axes[1, 1].scatter(P[ok, 0], P[ok, 1], c=err[ok], s=3, cmap="RdBu_r", vmin=-3, vmax=3)
    fig.colorbar(s2, ax=axes[1, 1], fraction=0.025, label="ours - LiDAR (m)")
    mo = ev.get("moving: ours |depth err| median m", float("nan"))
    mf = ev.get("moving: flat |depth err| median m", float("nan"))
    axes[1, 1].set_title(f"depth error at the vertices: static {ev['static: |depth err| median m']:.2f} m median "
                         f"({100 * ev['static: share within 0.5 m']:.0f} % within 0.5 m);  moving {mo:.2f} m "
                         f"(flat would give {mf:.2f} m)", fontsize=9, loc="left", color=INK)
    for a in axes.ravel():
        a.set_xlim(0, W)
        a.set_ylim(H, res["horizon"] - 20)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70, facecolor="white")
    plt.close(fig)


def view_cameras(seq, res):
    """Three cameras: the original one, high behind it, low beside the nearest moving object."""
    k = res["k"]
    K = seq.cams[k][0].copy()
    K[:2] *= VIEW_SCALE
    C = res["C"]
    X, _ = assemble(res)
    objs = sorted(res["objects"], key=lambda s: np.linalg.norm(np.median(s["X"], 0)[:2] - C[:2]))
    target = np.median(np.vstack([s["X"] for s in objs]), 0) if objs else np.median(X, 0)
    target[2] = 0.0
    back = (C - target)[:2] / np.linalg.norm((C - target)[:2])
    cams = {"original camera": (seq.cams[k][0] * np.r_[VIEW_SCALE, VIEW_SCALE, 1][:, None], seq.cams[k][1], seq.cams[k][2]),
            "higher, behind the camera": look_at(K, np.r_[C[:2] + 12 * back, C[2] + 22], target)}
    if objs:
        near = np.median(objs[0]["X"], 0)
        near[2] = 0.8
        side = np.cross(np.r_[back, 0], [0, 0, 1.0])
        cams["low, beside the nearest moving object"] = look_at(K, near + 13 * side + np.r_[4 * back, 2.2], near)
    return cams


def views_figure(path, seq, res, ground):
    """Novel views of the reconstruction next to the LiDAR of the same frame coloured from the image."""
    k, P = res["k"], res["P"]
    X, tri_used = assemble(res)
    far = np.linalg.norm(X[tri_used].mean(1)[:, :2] - res["C"][:2], axis=1) < VIEW_RANGE
    tri_used = tri_used[far]
    src = seq.frames[k]
    L, Lq, _ = lidar_depth(seq, k, ground, seq.cams[k])
    Lc = src[Lq[:, 1].astype(int), Lq[:, 0].astype(int)]
    near = np.linalg.norm(L[:, :2] - res["C"][:2], axis=1) < VIEW_RANGE
    L, Lc = L[near], Lc[near]
    shape = (int(seq.shape[0] * VIEW_SCALE), int(seq.shape[1] * VIEW_SCALE))
    cams = view_cameras(seq, res)
    fig, axes = plt.subplots(len(cams), 2, figsize=(17, 4.9 * len(cams)))
    for row, (name, cam) in enumerate(cams.items()):
        img, _ = render(cam, shape, X, tri_used, P, src)
        pc = render_points(cam, shape, L, Lc, size=3)
        axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, 1].imshow(cv2.cvtColor(pc, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"ours (one image + its video, no masks, no labels) - {name}", fontsize=9.5, loc="left",
                               color=INK)
        axes[row, 1].set_title(f"check: LiDAR of the same frame, coloured from the image - {name}", fontsize=9.5,
                               loc="left", color=INK)
        for a in axes[row]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70, facecolor="white")
    plt.close(fig)


def scene_camera(seq, k, C):
    """The fixed novel viewpoint of the video: above, behind and to the side of the real camera."""
    H, W = seq.shape
    K = seq.cams[k][0].copy()
    K[:2] *= VIEW_SCALE
    target = np.r_[itm.ground_point(seq.cams[k], np.array([[W / 2, 0.62 * H]]))[0], 0.0]
    back = (C - target)[:2] / np.linalg.norm((C - target)[:2])
    side = np.cross(np.r_[back, 0], [0, 0, 1.0])[:2]
    return look_at(K, np.r_[C[:2] + 15 * back + 18 * side, C[2] + 16], target)


def label_tag(img, text, h=26):
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (255, 255, 255), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 31, 31), 1, cv2.LINE_AA)


def video_frame(path, seq, res, ground, ev):
    """2 x 2 panel: video segmentation | triangulation | the whole surface from a new viewpoint | LiDAR check."""
    k, P, tri = res["k"], res["P"], res["tri"]
    H, W = seq.shape
    shape = (int(H * VIEW_SCALE), int(W * VIEW_SCALE))
    src = seq.frames[k]
    # segmentation: the regions of the video segmentation (faint), the moving objects found (strong)
    seg = src.astype(float) / 255
    if res.get("seg"):
        regs = res["seg"]["regs"]
        pal = np.random.default_rng(5).uniform(0.2, 0.95, (int(regs.max()) + 2, 3))
        okr = regs >= 0
        seg[okr] = 0.55 * seg[okr] + 0.45 * pal[regs[okr]]
    cls = -np.ones((H, W), np.int32)
    for o, s in enumerate(res["objects"]):
        for i in s["tri_idx"]:
            cv2.fillConvexPoly(cls, P[tri[i]].round().astype(np.int32), o)
    seg = (seg * 255).astype(np.uint8)
    for o in range(len(res["objects"])):
        m = (cls == o).astype(np.uint8)
        col = tuple(int(255 * c) for c in OBJ_COLS[o % len(OBJ_COLS)][::-1])
        seg[m > 0] = (0.35 * seg[m > 0] + 0.65 * np.array(col)).astype(np.uint8)
        cv2.drawContours(seg, cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, (255, 255, 255), 2)
    # triangulation: static triangles (flat on the road) grey, moving objects in their colours, torn ones not drawn
    trimg = (0.55 * src).astype(np.uint8)
    for t in tri[res["static_tri"]]:
        cv2.polylines(trimg, [P[t].round().astype(np.int32)], True, (150, 150, 150), 1, cv2.LINE_AA)
    for o, s in enumerate(res["objects"]):
        col = tuple(int(255 * c) for c in OBJ_COLS[o % len(OBJ_COLS)][::-1])
        for i in s["tri_idx"]:
            cv2.polylines(trimg, [P[tri[i]].round().astype(np.int32)], True, col, 1, cv2.LINE_AA)
        for q in s.get("contact_uv", np.zeros((0, 1, 2)))[:, res["ks"].index(k)] if len(s.get("contact_uv", [])) else []:
            if np.isfinite(q).all():
                cv2.circle(trimg, (int(q[0]), int(q[1])), 7, (0, 255, 0), 2, cv2.LINE_AA)
    X, tri_used = assemble(res)
    far = np.linalg.norm(X[tri_used].mean(1)[:, :2] - res["C"][:2], axis=1) < VIEW_RANGE
    tri_used = tri_used[far]
    cam = scene_camera(seq, k, res["C"])
    surf, _ = render(cam, shape, X, tri_used, P, src)
    L, Lq, _ = lidar_depth(seq, k, ground, seq.cams[k])
    keep = np.linalg.norm(L[:, :2] - res["C"][:2], axis=1) < VIEW_RANGE
    lid = render_points(cam, shape, L[keep], src[Lq[keep, 1].astype(int), Lq[keep, 0].astype(int)], size=3)
    tiles = [cv2.resize(seg, (shape[1], shape[0]), interpolation=cv2.INTER_AREA),
             cv2.resize(trimg, (shape[1], shape[0]), interpolation=cv2.INTER_AREA), surf, lid]
    mo = ev.get("moving: ours |depth err| median m", float("nan"))
    names = [f"frame {k}: video segmentation, moving objects (white outline)",
             "triangulation: grey = static, laid flat; colour = moving; green = tyre contacts",
             f"whole surface from a new viewpoint (moving: depth err {mo:.1f} m vs LiDAR)",
             "check: LiDAR of the same frame, same viewpoint"]
    for t, nm in zip(tiles, names):
        label_tag(t, nm)
    cv2.imwrite(path, np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def triangle_colours(img, uv, tri):
    """Mean BGR of the image pixels inside every triangle (a triangle without pixels: mean of its vertices)."""
    idx = np.zeros(img.shape[:2], np.int32)
    for i, t in enumerate(tri):
        cv2.fillConvexPoly(idx, np.round(uv[t]).astype(np.int32), int(i + 1))
    sel = idx > 0
    cnt = np.bincount(idx[sel] - 1, minlength=len(tri)).astype(float)
    out = np.stack([np.bincount(idx[sel] - 1, img[..., c][sel].astype(float), minlength=len(tri)) for c in range(3)], 1)
    out /= np.maximum(cnt, 1)[:, None]
    q = np.clip(np.round(uv).astype(int), 0, [img.shape[1] - 1, img.shape[0] - 1])
    vert = img[q[:, 1], q[:, 0]].astype(float)
    out[cnt == 0] = vert[tri[cnt == 0]].mean(1)
    return out.astype(np.float32)


def dump_objects(path, seq, res):
    """Per window: every solved object (world points at the reference frame, pixels, colours, triangles, motion) for
    the fusion over the video (car_fusion.py)."""
    import pickle
    k, P = res["k"], res["P"]
    out = []
    for s in res["objects"]:
        v = s["verts"]
        keep = ~np.isin(v, s["mirror"])                      # the mirror image is not part of the car
        loc = -np.ones(len(res["P"]), int)
        loc[v] = np.arange(len(v))
        T = loc[res["tri"][s["tri_idx"]]]
        M, F = s["M"], len(res["ks"])
        r = res["ks"].index(k)
        a, b = max(0, r - 2), min(F - 1, r + 2)
        vel = (M[b, 1:] - M[a, 1:]) / max(res["ks"][b] - res["ks"][a], 1)   # centre displacement per time slot (m)
        q = np.clip(P[v].round().astype(int), 0, [seq.shape[1] - 1, seq.shape[0] - 1])
        out.append({"k": k, "X": s["X"], "uv": P[v], "bgr": seq.frames[k][q[:, 1], q[:, 0]], "tri": T,
                    "tri_bgr": triangle_colours(seq.frames[k], P[v], T),
                    "body": keep, "filled": np.isin(v, s.get("filled", [])), "bottom": np.isin(v, s["bottom"]),
                    "vel": vel, "speed_mps": s["speed_mps"], "camera": res["C"],
                    "reprojection_px": s["reprojection_px"]})
    with open(path, "wb") as f:
        pickle.dump(out, f)


def export_scene(dirname, seq, res, ground):
    """Textured mesh of the whole scene + the LiDAR of the frame for the three.js viewer (docs/scene_viewer)."""
    k, P = res["k"], res["P"]
    X, tri_used = assemble(res)
    near = np.linalg.norm(X[tri_used].mean(1)[:, :2] - res["C"][:2], axis=1) < VIEW_RANGE
    tri_used = tri_used[near]
    group = np.zeros(len(res["tri"]), int)
    for o, s in enumerate(res["objects"]):
        group[s["tri_idx"]] = 1 + o
    tid = {tuple(t): i for i, t in enumerate(res["tri"])}
    g = np.array([group[tid[tuple(t)]] for t in tri_used])
    used, inv = np.unique(tri_used, return_inverse=True)
    H, W = seq.shape
    L, Lq, _ = lidar_depth(seq, k, ground, seq.cams[k])
    keep = np.linalg.norm(L[:, :2] - res["C"][:2], axis=1) < VIEW_RANGE
    L, Lq = L[keep], Lq[keep]
    Lc = seq.frames[k][Lq[:, 1].astype(int), Lq[:, 0].astype(int)][:, ::-1]
    origin = np.r_[np.median(X[used, :2], axis=0), 0.0]            # scene centred for the viewer
    objs = []
    for o, s in enumerate(res["objects"]):
        solved = ~np.isin(s["verts"], s.get("filled", []))
        objs.append({"id": o + 1, "range_m": round(float(np.median(np.linalg.norm(s["X"][solved] - res["C"], axis=1))), 1),
                     "speed_mps": round(s["speed_mps"], 1), "vertices": int(solved.sum())})
    look = np.r_[itm.ground_point(seq.cams[k], np.array([[W / 2, H / 2]]))[0], 0.0]
    fov = float(np.degrees(2 * np.arctan(H / 2 / seq.cams[k][0][1, 1])))
    data = {"frame": k, "image": f"frame_{k}.jpg", "camera": np.round(res["C"] - origin, 3).tolist(),
            "look": np.round(look - origin, 3).tolist(), "fov": round(fov, 2),
            "positions": np.round(X[used] - origin, 3).ravel().tolist(),
            "uvs": np.round(np.c_[P[used, 0] / W, 1 - P[used, 1] / H], 5).ravel().tolist(),
            "index": inv.reshape(-1, 3).ravel().tolist(), "group": g.tolist(), "objects": objs,
            "lidar": np.round(L - origin, 2).ravel().tolist(), "lidar_rgb": Lc.astype(int).ravel().tolist()}
    os.makedirs(dirname, exist_ok=True)
    json.dump(data, open(os.path.join(dirname, f"scene_{k}.json"), "w"), separators=(",", ":"))
    cv2.imwrite(os.path.join(dirname, f"frame_{k}.jpg"), seq.frames[k], [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(frames):
    seq, ground = es.open_sequence()
    report = []
    for k in frames:
        if seq.frames[k] is None:                    # no image in this time slot
            continue
        res = run(seq, k, ground)
        _, Lq, Ld = lidar_depth(seq, k, ground, seq.cams[k])
        ev, err = evaluate(res, Lq, Ld)
        report.append(ev)
        print(json.dumps({a: b for a, b in ev.items() if a != "moving objects"}), flush=True)
        for o in ev["moving objects"]:
            print("   ", {a: round(b, 2) if isinstance(b, float) else b for a, b in o.items()}, flush=True)
        if VIDEO:
            os.makedirs(os.path.join(vd.OUT_DIR, f"whole_scene_video{SUFFIX}"), exist_ok=True)
            video_frame(os.path.join(vd.OUT_DIR, f"whole_scene_video{SUFFIX}", f"{k:04d}.jpg"), seq, res, ground, ev)
        if DUMP:
            os.makedirs(os.path.join(vd.OUT_DIR, DUMP_DIR), exist_ok=True)
            dump_objects(os.path.join(vd.OUT_DIR, DUMP_DIR, f"{k:04d}.pkl"), seq, res)
        if EXPORT:
            export_scene(os.path.join(vd.OUT_DIR, "scene_viewer"), seq, res, ground)
        if FIGURES:
            overview_figure(os.path.join(vd.OUT_DIR, f"whole_scene_{k}.png"), seq, res, err, Ld, Lq, ev)
            views_figure(os.path.join(vd.OUT_DIR, f"whole_scene_{k}_views.png"), seq, res, ground)
    json.dump(report, open(os.path.join(vd.OUT_DIR, f"whole_scene{SUFFIX}.json"), "w"), indent=1)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [70, 150])
