"""Pedestrians and cyclists from the fixed roadside camera, without learning: the silhouette path.

A person or a cyclist is thin, upright and partly non-rigid (legs, pedalling): the car pipeline's rigid triangulation
finds few points on it (on intersection A it caught the nearest cyclist in 6 windows, the far ones never). The
silhouette, however, is clear against the video's median background in every frame, and its bottom touches the ground:
  1. silhouettes : the foreground against the median background, tracked frame by frame (the connected component that
                   overlaps the prediction most), cast shadow removed; a frame where it merges with a car is skipped;
  2. path        : the bottom of the silhouette (feet, tyres) is on the ground, its IPM point per frame; smoothed over
                   time (median, then Savitzky-Golay); heading = the path's tangent;
  3. height      : the top of the silhouette on the vertical through the ground point, median over the frames;
  4. shape       : visual hull in the object's own frame (origin on the path, x along the heading, z up): a voxel stays
                   when it projects inside the silhouette in >= VOTE of the carving frames, it and its mirror image
                   (left-right symmetry), at most W_MAX wide with elliptic horizontal sections (a side view barely
                   constrains the depth along the ray: the prior keeps the body round and thin at its ends);
                   surface by marching cubes, coloured with the original pixels of the best view of each point;
  5. check       : carving uses the even frames; on the odd frames (never used) the model's silhouette is compared with
                   the observed one (IoU) - a label-free check of the shape and the path.
Labels are used only to pick the object (its 2D box in one frame, like a click) and to compare the size; with
TRACK=label the labelled 2D boxes replace the tracker (given tracking, to separate the shape from the detection).

    python vru_shape.py TRACK_ID ...        results in docs/vru/<id>.json, .npz and docs/vru/<id>.png
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import label as cc_label
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from skimage.measure import marching_cubes

import ipm_track_models as itm
import v2x_seq_classical as vc
import v2x_seq_demo as vd
import v2x_seq_shape as vs

OUT = os.environ.get("VRU_OUT", os.path.join(vd.OUT_DIR, "vru"))   # one folder per sequence: track ids repeat
VOTE = float(os.environ.get("VOTE", "0.75"))      # share of the carving frames whose silhouette must hold a voxel
TRACK = os.environ.get("TRACK", "own")             # "label": the labelled 2D boxes as the tracker (given tracking)
VOX_PX = float(os.environ.get("VOX_PX", "1.0"))   # voxel size in pixels at the object's range (1.2 - 6 cm)
VOX_MIN = float(os.environ.get("VOX_MIN", "0.012"))   # m: finer only costs time; a pixel is the limit anyway
VOX = 0.04                                          # m, set per object
BG_STEP = 4                                         # every 4th frame for the median background
MAX_SKIP = 15                                       # frames in a row without a silhouette (a car in front) before the track ends
REFINE = int(os.environ.get("REFINE", "0"))         # rounds of aligning the frames to the model (off: it drifts across the ray)
BLURS = (0.7, None)        # reflection test variants per object (blurred mirror image required / mirror line only);
BLUR = 0.7                 # the one in use: the variant whose model explains the held-out frames best is kept
W_MAX = float(os.environ.get("W_MAX", "0.6"))     # m: width prior (the depth along a side view is barely observed)
ELLIPSE = os.environ.get("ELLIPSE", "1") == "1"    # elliptic horizontal sections (thin at the ends, round, not a slab)


# --- data ---------------------------------------------------------------------------------------------------------
def load():
    fids, _, cams, _, camera_xy = vd.load_sequence()
    labels = {}
    for k, f in enumerate(fids):
        for o in json.load(open(os.path.join(vd.DATA, "label", "camera", f"{f}.json"))):
            labels.setdefault(o["track_id"], {})[k] = o
    return fids, cams, labels, camera_xy


def frame(fids, k):
    return cv2.imread(os.path.join(vd.DATA, "image", f"{fids[k]}.jpg"))


def background(fids, k, half=40, block=20):
    """Median background around frame k (frames k +- half, every BG_STEP-th): the wet road dries and the light
    changes over the video, so one median for all of it leaves false foreground. Cached per block of frames."""
    c = min(len(fids) - 1, (k // block) * block + block // 2)
    name = "_".join(os.path.normpath(vd.DATA).split(os.sep)[-2:])
    path = os.path.join(OUT, "bg", name, f"{c:04d}.png")
    if os.path.exists(path):
        return cv2.imread(path)
    imgs = [frame(fids, j) for j in range(max(0, c - half), min(len(fids), c + half + 1), BG_STEP)]
    imgs = [im for im in imgs if im is not None]
    bg = np.empty_like(imgs[0])
    for y in range(0, bg.shape[0], 120):
        bg[y:y + 120] = np.median(np.stack([im[y:y + 120] for im in imgs]), axis=0).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bg)
    return bg


# --- geometry -----------------------------------------------------------------------------------------------------
def ray(cam, u, v):
    K, R, t = cam
    return -R.T @ t, R.T @ np.linalg.inv(K) @ np.array([u, v, 1.0])


def ground_point(cam, u, v):
    """Where the viewing ray through pixel (u, v) meets the ground z = 0 (None if it does not)."""
    C, d = ray(cam, u, v)
    if abs(d[2]) < 1e-12 or -C[2] / d[2] <= 0:
        return None
    return C - C[2] / d[2] * d


def height_at(cam, u, v, foot):
    """Height of the ray through (u, v) where it passes the vertical line through the ground point foot."""
    C, d = ray(cam, u, v)
    s = ((foot[:2] - C[:2]) @ d[:2]) / (d[:2] @ d[:2])
    return float((C + s * d)[2])


# --- 1. silhouettes -----------------------------------------------------------------------------------------------
def foreground(img, bg, h, t=28.0):
    """Foreground of a crop, scaled to an object h px tall: colour distance to the background above t, opened (only
    when h >= 60 px: a far cyclist's frame is one pixel wide) and closed with a kernel that grows with h. (A lower,
    noise-adaptive threshold with hysteresis was tried: at 75-90 m it joins the person to the passing cars.)"""
    d = np.linalg.norm(img.astype(np.int16) - bg.astype(np.int16), axis=2)
    m = (d > t).astype(np.uint8)
    if h >= 60:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    c = max(3, 2 * int(round(h / 36)) + 1)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((c, c), np.uint8)), t


def shadow(img, bg, low=0.3):
    """Cast shadow (the HSV test of itm.shadow_free: darker than the background, similar hue and saturation), only in
    the lowest `low` of the rows: a shadow lies on the ground, while dark clothes higher up pass the same test."""
    hf = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hb = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.int16)
    ratio = (hf[..., 2] + 1) / (hb[..., 2] + 1)
    dh = np.abs(hf[..., 0] - hb[..., 0])
    dh = np.minimum(dh, 180 - dh)
    sh = (ratio > 0.35) & (ratio < 0.92) & (np.abs(hf[..., 1] - hb[..., 1]) < 40) & (dh < 12)
    sh[:int((1 - low) * len(sh))] = False
    return sh


def mirror_row(gray, dist, mask, x0, y0, min_score=0.45, faint=0.75, blur="auto"):
    """The ground contact row of a silhouette that includes its reflection on a wet road: the first row from the top
    (in its lower half) about which the image is mirror-symmetric (normalised correlation of the rows above with the
    flipped rows below, in the silhouette's columns) AND below which the silhouette is faint (mean colour distance to
    the background at most `faint` of that above) and blurred (mean gradient at most `blur` of that above): a wheel, a
    scooter or a person's legs are mirror-symmetric too, but as strong and as sharp below the line as above it.
    None when no row qualifies (a dry road)."""
    blur = BLUR if blur == "auto" else blur
    ys, xs = np.nonzero(mask)
    top, bot = ys.min(), ys.max()
    h = bot - top + 1
    D = max(3, int(0.07 * h))              # the reflection is short and blurred: a short window
    cols = slice(xs.min(), xs.max() + 1)
    G = gray[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]].astype(float)[:, cols]
    Dm = (dist * mask)[:, cols]
    M = mask[:, cols]
    if min(G.shape) < 2:                   # a sliver: no line to test
        return None
    gy, gx = np.gradient(G)
    Gm = np.hypot(gx, gy)                  # sharpness: a reflection is blurred, the object above it is not
    for r in range(top + h // 2, bot - 2):
        if r - D - 1 < 0 or r + D + 1 > len(G) or M[r:r + D].sum() < 3:
            continue
        sc = []
        for rr in (r - 1, r, r + 1):
            a, b = G[rr - D:rr], G[rr:rr + D][::-1]
            a, b = a - a.mean(), b - b.mean()
            sc.append((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
        if sc[1] < min_score or sc[1] < sc[0] or sc[1] < sc[2]:
            continue
        above = Dm[r - D:r].sum() / max(M[r - D:r].sum(), 1)
        below = Dm[r:r + D].sum() / max(M[r:r + D].sum(), 1)
        sharp_above = (Gm[r - D:r] * M[r - D:r]).sum() / max(M[r - D:r].sum(), 1)
        sharp_below = (Gm[r:r + D] * M[r:r + D]).sum() / max(M[r:r + D].sum(), 1)
        if (faint is None or below <= faint * above) and (blur is None or sharp_below <= blur * sharp_above):
            return r
    return None


def silhouette(img, bg, pred, clip=False):
    """The foreground inside the predicted box (x0, y0, x1, y1): the components lying mostly in it, shadow removed:
    (full-frame mask, box) or None when there is none, only a fragment, or it is merged with something much larger."""
    x0, y0, x1, y1 = pred
    h, w = y1 - y0, x1 - x0
    m = max(20, int(0.6 * h))
    X0, Y0 = max(0, int(x0 - m)), max(0, int(y0 - m))
    X1, Y1 = min(img.shape[1], int(x1 + m)), min(img.shape[0], int(y1 + m))
    if X1 - X0 < 8 or Y1 - Y0 < 8 or w < 2 or h < 2:
        return None
    fg, _ = foreground(img[Y0:Y1, X0:X1], bg[Y0:Y1, X0:X1], h)
    min_px = max(10, int(0.01 * h * h))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    # every component mostly inside the (slightly widened) prediction: a rider and the bike may come apart
    below = 0.0 if clip else 0.1      # a given box bounds the bottom (else the wet-road reflection enters it)
    px0, py0, px1, py1 = x0 - X0 - 0.05 * w, y0 - Y0 - 0.1 * h, x1 - X0 + 0.05 * w, y1 - Y0 + below * h
    inside = np.zeros(lab.shape, bool)
    inside[max(0, int(py0)):max(0, int(py1)), max(0, int(px0)):max(0, int(px1))] = True
    share = np.bincount(lab[inside], minlength=n) / np.maximum(st[:, 4], 1)
    keep = [i for i in range(1, n) if st[i, 4] >= min_px // 3 and share[i] >= 0.5]
    if clip:          # the box is given: the foreground inside it, whatever it touches outside (a companion, a car)
        comp = (fg > 0) & inside
    elif not keep:
        return None
    else:
        comp = np.isin(lab, keep)
    if not comp.any():
        return None
    ys, xs = np.nonzero(comp)
    by, bx, bh, bw = ys.min(), xs.min(), ys.max() + 1 - ys.min(), xs.max() + 1 - xs.min()
    if bh > 2.5 * h or bw > 2.5 * w:                       # merged with a car: skip the frame
        return None
    comp = comp[by:by + bh, bx:bx + bw]
    # the wet-road reflection under the tyres / feet: cut at the mirror line
    dist = np.linalg.norm(img[Y0 + by:Y0 + by + bh, X0 + bx:X0 + bx + bw].astype(np.int16)
                          - bg[Y0 + by:Y0 + by + bh, X0 + bx:X0 + bx + bw].astype(np.int16), axis=2)
    # own tracking: the component reaches down into the reflection, the first mirror line from the top is the contact
    # (on this very wet road the reflection is nearly as strong as the object: no faintness test); a given box already
    # bounds the bottom, and inside it the first mirror line would be a wheel's axle
    r = None if clip else mirror_row(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dist, comp, X0 + bx, Y0 + by, faint=None)
    if r is not None:
        comp = comp.copy()
        comp[r:] = False
    blob = vc.make_blob(comp.astype(np.uint8), X0 + bx, Y0 + by, min_area=min_px)
    if blob is None:
        return None
    gx0, gy0, gx1, gy1 = blob["box"]
    bx, by, bw, bh = gx0 - X0, gy0 - Y0, gx1 - gx0, gy1 - gy0
    if bh > 1.6 * h or bh < 0.4 * h:                          # reflection cut: still merged, or only a fragment
        return None
    sil = (blob["m"] > 0) & ~shadow(img[gy0:gy1, gx0:gx1], bg[gy0:gy1, gx0:gx1])
    if h >= 60:
        sil = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(sil.astype(np.uint8))
    if n2 < 2:
        return None
    main = 1 + np.argmax(st2[1:, 4])
    main_bottom = st2[main, 1] + st2[main, 3]
    # pieces of >= 10 % of the largest one, not lying entirely below it (road marks, reflection remnants)
    sil = np.isin(lab2, [i for i in range(1, n2) if st2[i, 4] >= 0.1 * st2[main, 4] and st2[i, 1] < main_bottom - 2])
    ys, xs = np.nonzero(sil)
    if len(xs) < min_px:
        return None
    full = np.zeros(img.shape[:2], bool)
    full[Y0 + by + ys, X0 + bx + xs] = True
    return full, (X0 + bx + xs.min(), Y0 + by + ys.min(), X0 + bx + xs.max() + 1, Y0 + by + ys.max() + 1)


def track(fids, k0, box0):
    """Silhouettes forward and backward from the seed frame k0: {k: (mask, box)}."""
    out = {}
    for step in (1, -1):
        boxes = [np.array(box0, float)]
        k, skip = k0, 0
        while 0 <= k < len(fids) and skip < MAX_SKIP:
            B = np.array(boxes[-5:])
            cen = (B[:, :2] + B[:, 2:]) / 2
            vel = np.median(np.diff(cen, axis=0), axis=0) if len(cen) > 1 else np.zeros(2)
            size = np.median(B[:, 2:] - B[:, :2], axis=0)
            c = cen[-1] + vel * (1 + skip)
            pred = np.r_[c - size / 2, c + size / 2]
            img = frame(fids, k)
            r = silhouette(img, background(fids, k), pred) if img is not None else None
            if r is None:
                skip += 1
            else:
                out[k] = r
                boxes.append(np.array(r[1], float))
                skip = 0
            if img is None or pred[2] < 0 or pred[0] > img.shape[1] or pred[3] < 0 or pred[1] > img.shape[0]:
                break
            k += step
    return dict(sorted(out.items()))


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy)


# --- 2./3. path and height ----------------------------------------------------------------------------------------
def contact(mask, box):
    """Bottom-centre of the silhouette: the median column of its lowest tenth, the lowest row."""
    x0, y0, x1, y1 = box
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    low = ys >= ys.max() - max(1, 0.1 * (y1 - y0))
    return x0 + float(np.median(xs[low])) + 0.5, y0 + ys.max() + 1.0


def top(mask, box):
    x0, y0, x1, y1 = box
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    hi = ys <= ys.min() + 1
    return x0 + float(np.mean(xs[hi])) + 0.5, y0 + ys.min()


def path(cams, sils):
    """Smoothed ground path {k: (centre xy, heading)} and the per-frame height of the silhouette top."""
    ks = np.array(sorted(sils))
    P = np.array([ground_point(cams[k], *contact(*sils[k]))[:2] for k in ks])
    win = min(len(ks) - (1 - len(ks) % 2), 21)
    Ps = median_filter(P, size=(5, 1), mode="nearest")
    if win >= 5:
        Ps = savgol_filter(Ps, win, 2, axis=0, mode="interp")
        D = savgol_filter(Ps, win, 2, deriv=1, axis=0, mode="interp")
    else:
        D = np.gradient(Ps, axis=0)
    if np.linalg.norm(Ps[-1] - Ps[0]) > 1.0:              # slow parts keep the overall direction of travel
        main = (Ps[-1] - Ps[0]) / np.linalg.norm(Ps[-1] - Ps[0])
        slow = np.linalg.norm(D, axis=1) < 0.05
        D[slow] = main
    yaw = np.arctan2(D[:, 1], D[:, 0])
    hts = [height_at(cams[k], *top(*sils[k]), np.r_[p, 0]) for k, p in zip(ks, Ps)]
    return {int(k): (p, a) for k, p, a in zip(ks, Ps, yaw)}, np.array(hts), P


# --- 4. shape -----------------------------------------------------------------------------------------------------
def axes(length=3.0, width=1.6, height=2.3):
    return (np.arange(-length / 2, length / 2, VOX) + VOX / 2, np.arange(-width / 2, width / 2, VOX) + VOX / 2,
            np.arange(0, height, VOX) + VOX / 2)


def elliptic_section(occ, xs, ys, half_width):
    """Width prior with a rounded cross-section: in every height slice the occupied length (seen well from the side)
    is the major axis of an ellipse whose half-width is half_width, so a body is widest in the middle and thin at its
    ends (a bicycle at its wheels, a person front and back) instead of a slab; at least one voxel wide."""
    out = occ.copy()
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    for iz in range(occ.shape[2]):
        cols = occ[:, :, iz].any(axis=1)
        if not cols.any():
            continue
        lo, hi = xs[cols].min(), xs[cols].max()
        c, a = (lo + hi) / 2, (hi - lo) / 2 + dx / 2
        allowed = np.maximum(half_width * np.sqrt(np.clip(1 - ((xs - c) / a) ** 2, 0, None)), dy / 2)
        out[:, :, iz] &= np.abs(ys)[None, :] <= allowed[:, None]
    return out


def carve(cams, sils, poses, ks):
    views = [(cams[k], poses[k][0], poses[k][1], sils[k][0]) for k in ks]
    occ = vs.carve_votes(axes(), views, symmetric=True) >= VOTE
    occ &= (np.abs(axes()[1]) <= W_MAX / 2)[None, :, None]
    if ELLIPSE:
        occ = elliptic_section(occ, axes()[0], axes()[1], W_MAX / 2)
    lab, n = cc_label(occ)
    if n > 1:
        occ = lab == 1 + np.argmax(np.bincount(lab.ravel())[1:])
    return occ


def surface(occ):
    """Marching-cubes surface of the voxel hull in the object frame: (V, F, vertex normals)."""
    ax = axes()
    pad = np.pad(occ.astype(float), 1)
    V, F, N, _ = marching_cubes(pad, 0.5)
    V = (V - 1) * VOX + [ax[0][0], ax[1][0], ax[2][0]]
    return V, F[:, ::-1].copy(), N


def colours(fids, cams, sils, poses, ks, V, N, k_best=5, inset=1):
    """Vertex colours from the original pixels: of the k_best frames that see a vertex most directly and largest -
    facing the camera, not hidden behind another part of the model (z-buffer), `inset` px inside the silhouette - the
    medoid pixel (an outlier of one view, from a part the model does not quite cover, is outvoted); bilinear sampling.
    Symmetric: a vertex and its mirror image take the better seen of the two. Never seen (the front and back of a
    body seen only from the side): the nearest seen vertex's."""
    S = np.full((2, len(V), k_best), -np.inf)
    Ck = np.zeros((2, len(V), k_best, 3))
    ker = np.ones((2 * inset + 1, 2 * inset + 1), np.uint8)
    for k in ks:
        img = frame(fids, k)
        sil = cv2.erode(sils[k][0].astype(np.uint8), ker) > 0 if inset else sils[k][0]
        c, yaw = poses[k]
        cen = -cams[k][1].T @ cams[k][2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        for m, sign in enumerate((1, -1)):
            Vm, Nm = V * [1, sign, 1], N * [1, sign, 1]
            W = vs.world(Vm, c, yaw)
            Nw = np.c_[cy * Nm[:, 0] - sy * Nm[:, 1], sy * Nm[:, 0] + cy * Nm[:, 1], Nm[:, 2]]
            view = cen - W
            dist = np.linalg.norm(view, axis=1)
            cos = (Nw * view).sum(1) / dist
            q, z = vs.project(cams[k], W)
            r = max(1, int(round(VOX * cams[k][0][0, 0] / np.median(z))))
            ok = (cos > 0.1) & vs.visible(q, z, img.shape[:2], r, 1.5 * VOX)
            u, v = np.round(q).astype(int).T
            ok[ok] &= sil[v[ok], u[ok]]
            score = cos / dist                                 # direct and near: the most pixels on the surface
            if ok.any():
                colour = np.zeros((len(V), 3))
                colour[ok] = vs.bilinear(img, q[ok])            # the original pixels
                vs.topk_update(S[m], Ck[m], score, colour, ok)
    (c0, s0, b0), (c1, s1, b1) = vs.topk_medoid(S[0], Ck[0]), vs.topk_medoid(S[1], Ck[1])
    out = np.where((b0 >= b1)[:, None], c0, c1)       # the better seen of the vertex and its mirror image
    seen = s0 | s1
    if seen.any() and not seen.all():          # never seen (front and back of a side view): the nearest seen pixel
        from scipy.spatial import cKDTree
        out[~seen] = out[seen][cKDTree(V[seen]).query(V[~seen])[1]]
    return out


def model_mask(cam, pose, P, shape):
    """The voxel model's silhouette in a frame: voxel centres projected and splatted to the voxel size."""
    q, z = vs.project(cam, vs.world(P, *pose))
    m = np.zeros(shape, np.uint8)
    u, v = q.round().astype(int).T
    ok = (z > 0) & (u >= 0) & (v >= 0) & (u < shape[1]) & (v < shape[0])
    m[v[ok], u[ok]] = 1
    r = max(1, int(round(VOX * cam[0][0, 0] / np.median(z) / 2)))
    return cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0


def refine(cams, sils, poses, occ, ks, reach=6):
    """Per frame, the image shift (du, dv) within +-reach px that best aligns the model's silhouette with the observed
    one, moved to the ground (where the centre's ground pixel shifted by (du, dv) meets it) and kept only along the
    direction of travel: across it, the depth along the ray, the ground contact decides."""
    X, Y, Z = np.meshgrid(*axes(), indexing="ij")
    P = np.c_[X[occ], Y[occ], Z[occ]]
    out = dict(poses)
    for k in ks:
        mask, (x0, y0, x1, y1) = sils[k]
        mm = model_mask(cams[k], poses[k], P, mask.shape)
        Y0, Y1, X0, X1 = max(0, y0 - reach - 20), y1 + reach + 20, max(0, x0 - reach - 20), x1 + reach + 20
        a, b = mm[Y0:Y1, X0:X1], mask[Y0:Y1, X0:X1]
        best, shift = -1, (0, 0)
        for dv in range(-reach, reach + 1):
            for du in range(-reach, reach + 1):
                s_ = np.roll(np.roll(a, dv, 0), du, 1)
                iou = (s_ & b).sum() / max((s_ | b).sum(), 1)
                if iou > best:
                    best, shift = iou, (du, dv)
        q, _ = vs.project(cams[k], np.r_[poses[k][0], 0.0][None])
        g = ground_point(cams[k], q[0, 0] + shift[0], q[0, 1] + shift[1])
        if g is not None:                  # only along the direction of travel: the depth stays with the contact,
            hd = np.array([np.cos(poses[k][1]), np.sin(poses[k][1])])   # else a bigger model drifts away
            out[k] = (poses[k][0] + ((g[:2] - poses[k][0]) @ hd) * hd, poses[k][1])
    return out


# --- 5. check -----------------------------------------------------------------------------------------------------
def holdout_iou(cams, sils, poses, occ, ks):
    """IoU of the model's silhouette with the observed one on frames not used for carving (their poses may be aligned
    to the model, a shift of a few pixels; the shape itself never saw them)."""
    X, Y, Z = np.meshgrid(*axes(), indexing="ij")
    P = np.c_[X[occ], Y[occ], Z[occ]]
    out = []
    for k in ks:
        if k not in poses:
            continue
        mask, (x0, y0, x1, y1) = sils[k]
        m = model_mask(cams[k], poses[k], P, mask.shape)
        X0, Y0 = max(0, x0 - 30), max(0, y0 - 30)
        a, b = m[Y0:y1 + 30, X0:x1 + 30], mask[Y0:y1 + 30, X0:x1 + 30]
        out.append((a & b).sum() / max((a | b).sum(), 1))
    return float(np.mean(out)) if out else float("nan")


# --- figure -------------------------------------------------------------------------------------------------------
def look_at(K, eye, target):
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    R = np.stack([r, np.cross(f, r), f])
    return K, R, -R @ eye


def render(img, cam, V, F, C, light=(0.3, -0.5, 0.8), ambient=0.75):
    """Painter's rendering with the vertices' original pixel colours, averaged per face, and a mild Lambert shading
    (ambient share `ambient`) so that the form stays readable without darkening the video colours."""
    q, z = vs.project(cam, V)
    F = F[(z[F] > 0.1).all(1)]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    lt = np.array(light) / np.linalg.norm(light)
    col = np.clip(C[F].mean(1) * (ambient + (1 - ambient) * np.abs(n @ lt))[:, None], 0, 255)
    for i in np.argsort(-z[F].mean(1)):
        pts = np.round(q[F[i]] * 4).astype(np.int32)
        cv2.fillConvexPoly(img, pts, tuple(float(c) for c in col[i]), lineType=cv2.LINE_AA, shift=2)
    return img


def figure(path_png, fids, cams, sils, poses, V, F, C, row, occ):
    # the frame where the model's silhouette agrees best with the observed one (a free view, not one behind a car)
    X, Y, Z = np.meshgrid(*axes(), indexing="ij")
    P = np.c_[X[occ], Y[occ], Z[occ]]

    def agree(j):
        m, (x0, y0, x1, y1) = sils[j]
        a = model_mask(cams[j], poses[j], P, m.shape)[max(0, y0 - 20):y1 + 20, max(0, x0 - 20):x1 + 20]
        b = m[max(0, y0 - 20):y1 + 20, max(0, x0 - 20):x1 + 20]
        return (a & b).sum() / max((a | b).sum(), 1) * np.sqrt(y1 - y0)
    k = max(sils, key=agree)
    img = frame(fids, k)
    x0, y0, x1, y1 = sils[k][1]
    h = y1 - y0
    X0, Y0, X1, Y1 = max(0, x0 - h // 2), max(0, y0 - h // 3), min(img.shape[1], x1 + h // 2), min(img.shape[0], y1 + h // 3)
    crop = img[Y0:Y1, X0:X1].copy()
    over = img.copy()
    render(over, cams[k], vs.world(V, *poses[k]), F, C)
    over = cv2.addWeighted(over, 0.75, img, 0.25, 0)[Y0:Y1, X0:X1]
    tiles = [crop, over]
    K = np.array([[700.0, 0, 200], [0, 700.0, 200], [0, 0, 1]])
    for eye in ([3.2, 3.6, 2.4], [0.01, 5.0, 1.2], [5.0, 0.01, 1.2]):
        t = np.full((400, 400, 3), 255, np.uint8)
        tiles.append(render(t, look_at(K, np.array(eye), np.array([0.0, 0.0, 0.9])), V, F, C))
    size = 300
    out = []
    for t in tiles:
        s = size / max(t.shape[:2])
        t = cv2.resize(t, (int(t.shape[1] * s), int(t.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        pad = np.full((size, size, 3), 255, np.uint8)
        pad[(size - t.shape[0]) // 2:(size - t.shape[0]) // 2 + t.shape[0],
            (size - t.shape[1]) // 2:(size - t.shape[1]) // 2 + t.shape[1]] = t
        out.append(pad)
    strip = np.full((30, size * len(out), 3), 255, np.uint8)
    txt = (f"{row['type']} {row['track']}: {row['range_m']:.0f} m, {row['frames']} frames; "
           f"H {row['height']:.2f}/{row['label_height']:.2f} m, L {row['length']:.2f}/{row['label_length']:.2f}, "
           f"W {row['width']:.2f}/{row['label_width']:.2f}; hold-out IoU {row['holdout_iou']:.2f}")
    cv2.putText(strip, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(path_png, np.vstack([strip, np.hstack(out)]))


# --- main ---------------------------------------------------------------------------------------------------------
def reconstruct(tid, fids, cams, labels, camera_xy):
    """The model of one object with the current reflection test (BLUR): dict(row, occ, V, F, C, sils, poses, ks, raw)
    or None."""
    global VOX
    lab = labels[tid]
    # the seed: the largest labelled box where the object stands free and its silhouette fills the box; selection only
    box = lambda k: [float(lab[k]["2d_box"][a]) for a in ("xmin", "ymin", "xmax", "ymax")]
    seed, box0 = None, None
    for k in sorted(lab, key=lambda k: -(lab[k]["2d_box"]["ymax"] - lab[k]["2d_box"]["ymin"]))[:40]:
        img = frame(fids, k)
        r = silhouette(img, background(fids, k), box(k)) if img is not None else None
        # a clean start: the silhouette fills the labelled box (no reflection, no neighbour merged)
        if r is not None and lab[k].get("truncated_state", 0) == 0 and box_iou(r[1], box(k)) >= 0.7:
            seed, box0 = k, list(r[1])
            break
    if seed is None:
        print(tid, "no free view of the object")
        return None
    # voxels of VOX_PX pixels at the object's range: finer near the camera, never finer than the image allows
    g = ground_point(cams[seed], *contact(*r))
    rng0 = float(np.linalg.norm(g[:2] - camera_xy)) if g is not None else 50.0
    VOX = float(np.clip(VOX_PX * rng0 / cams[seed][0][0, 0], VOX_MIN, 0.06))
    if TRACK == "label":      # given tracking: the labelled 2D box per frame replaces detection and tracking only
        sils = {}
        for k in sorted(lab):
            img = frame(fids, k)
            r = silhouette(img, background(fids, k), box(k), clip=True) if img is not None else None
            if r is not None and box_iou(r[1], box(k)) >= 0.5:   # partly hidden (a car in front): skipped
                sils[k] = r
    else:
        sils = track(fids, seed, box0)
    if len(sils) < 10:
        print(tid, "too few silhouettes:", len(sils))
        return None
    poses, hts, raw = path(cams, sils)
    ks = sorted(sils)
    carve_ks, check_ks = ks[0::2], ks[1::2]
    occ = carve(cams, sils, poses, carve_ks)
    for _ in range(REFINE):                    # align every frame to the model, carve again
        poses = refine(cams, sils, poses, occ, ks)
        occ = carve(cams, sils, poses, carve_ks)
    ax = axes()
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    o = lambda A: A[occ]
    L = float(np.ptp(o(X)) + VOX) if occ.any() else 0.0
    W = float(np.ptp(o(Y)) + VOX) if occ.any() else 0.0
    H = float(o(Z).max() + VOX / 2) if occ.any() else 0.0
    if occ.sum() < 20:
        print(tid, "the hull is empty: the silhouettes do not agree on a shape")
        return None
    V, F, N = surface(occ)
    C = colours(fids, cams, sils, poses, ks, V, N)          # colour from every frame (the shape never saw the odd ones)
    d = {k: lab[k] for k in ks if k in lab}
    lp = np.array([[o_["3d_location"]["x"], o_["3d_location"]["y"]] for o_ in d.values()])
    pp = np.array([poses[k][0] for k in d])
    rng = float(np.median(np.linalg.norm(pp - camera_xy, axis=1)))
    err = pp - lp
    rays = (lp - camera_xy) / np.linalg.norm(lp - camera_xy, axis=1, keepdims=True)
    o0 = lab[seed]
    row = {"track": tid, "type": o0["type"], "frames": len(ks), "first": ks[0], "last": ks[-1], "range_m": rng,
           "box_px": float(box0[3] - box0[1]), "height_top": float(np.median(hts)), "height": H, "length": L, "width": W,
           "label_length": float(o0["3d_dimensions"]["l"]), "label_width": float(o0["3d_dimensions"]["w"]),
           "label_height": float(o0["3d_dimensions"]["h"]),
           "path along-ray err m": float(np.median(np.abs((err * rays).sum(1)))),
           "path across-ray err m": float(np.median(np.abs(err[:, 0] * -rays[:, 1] + err[:, 1] * rays[:, 0]))),
           "holdout_iou": holdout_iou(cams, sils, poses, occ, check_ks), "voxels": int(occ.sum()), "vox_m": VOX}
    row["tracking"] = TRACK
    row["reflection_test"] = "mirror line, blurred below" if BLUR is not None else "mirror line"
    return dict(row=row, occ=occ, V=V, F=F, C=C, sils=sils, poses=poses, ks=ks, raw=raw)


def run(tid, fids, cams, labels, camera_xy):
    """Both reflection tests (a wet road mirrors sharply, a dry one not at all, legs are symmetric too): the model whose
    silhouettes match the held-out frames best is kept - a choice without labels. Writes <tid>.json/.npz/.png."""
    global BLUR, VOX
    best, vox = None, VOX
    for BLUR in BLURS:
        res = reconstruct(tid, fids, cams, labels, camera_xy)
        fit = lambda r: -1.0 if not np.isfinite(r["row"]["holdout_iou"]) else r["row"]["holdout_iou"]   # NaN: no check frames
        if res is not None and (best is None or fit(res) > fit(best)):
            best, vox = res, VOX
    if best is None:
        return None
    VOX = vox
    row, ks, poses = best["row"], best["ks"], best["poses"]
    os.makedirs(OUT, exist_ok=True)
    json.dump(row, open(os.path.join(OUT, f"{tid}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"{tid}.npz"), occ=best["occ"], V=best["V"], F=best["F"], C=best["C"],
                        ks=ks, poses=np.array([[*poses[k][0], poses[k][1]] for k in ks]), raw_path=best["raw"])
    figure(os.path.join(OUT, f"{tid}.png"), fids, cams, best["sils"], poses, best["V"], best["F"], best["C"], row,
           best["occ"])
    return row


def main(ids):
    os.makedirs(OUT, exist_ok=True)
    fids, cams, labels, camera_xy = load()
    for tid in ids:
        row = run(tid, fids, cams, labels, camera_xy)
        if row:
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["003971", "003977", "003999"])
