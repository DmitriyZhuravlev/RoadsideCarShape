"""
Layer 0 for the cars at intersection A: the tyres' ground contacts, found classically, fix the depth of the placement.

The tracker's placement comes from the silhouette's bottom edge; at 60-80 m one pixel there is ~0.27 m of ground, and
the bottom edge mixes bumpers, shadow and reflection remnants, so the depth along the viewing ray scatters by ~1.4 m.
A tyre is a dark blob whose lower edge against the road is sharp (its wet-road reflection is blurred):
  1. wheels   : the rims, as Hough circles in the lower band of the car (radius scaled to the car), up to two;
  2. contact  : in a wheel's columns, between the rim centre and RIM_REACH rim radii below, the lowest strong
                dark-to-bright edge going down (the tyre ends on the road; its wet-road reflection is blurred),
                refined to sub-pixel by a parabola;
  3. centre   : in the car frame the contacts are at (+-WHEELBASE/2, +-TRACK/2, 0) (a generic prior); with the
                tracker's heading every contact implies a car centre for every wheel; the assignment nearest to the
                tracker wins. Only the component ALONG the viewing ray is taken from the wheels (that is where the
                tracker is wrong); across the ray the tracker is kept (0.3-0.7 m already). Smoothed along the track.

python wheels.py  ->  along-ray error by distance, tracker vs tracker + wheels (labels only to score).
"""
import math
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

import v2x_seq_classical as vc
import v2x_seq_shape as vs

WHEELBASE, TRACK = 2.7, 1.6
WHEEL_MIN = 0.07
DARK_FRAC = 0.45
EDGE_FRAC = 0.5
SIDE_VIEW = 0.5          # |sin| of the angle between heading and viewing ray needed to use the rims
RIM_REACH = 2.6
CONTACT_MODE = "first"   # "first": the tyre's own lower edge; "lowest": the end of the shadow under it          # rim radii below the rim centre searched for the tyre's contact


def subpixel_peak(profile, i):
    p = np.asarray(profile, float)
    if i == 0 or i == len(p) - 1:
        return float(i)
    den = p[i - 1] - 2 * p[i] + p[i + 1]
    if abs(den) < 1e-12:
        return float(i)
    return i + float(np.clip(0.5 * (p[i - 1] - p[i + 1]) / den, -0.5, 0.5))


def ellipse_bottom(ellipse):
    """Lowest image point (largest v) of an OpenCV ellipse ((cx, cy), (w, h), angle in degrees)."""
    (cx, cy), (ew, eh), ang = ellipse
    t = np.linspace(0, 2 * np.pi, 720)
    a, b, r = ew / 2, eh / 2, math.radians(ang)
    x = cx + a * np.cos(t) * math.cos(r) - b * np.sin(t) * math.sin(r)
    y = cy + a * np.cos(t) * math.sin(r) + b * np.sin(t) * math.cos(r)
    i = int(np.argmax(y))
    return float(x[i]), float(y[i])


def wheel_contacts(gray, mask):
    """[(u, v)] sub-pixel ground contacts of the visible tyres of one car.
    Wheels are found by their rims (Hough circles in the lower band, radius scaled to the car); in each wheel's
    columns, between the rim centre and RIM_REACH rim radii below it, the contact is the lowest strong
    dark-to-bright edge going down (the tyre and its shadow end on the road; the wet-road reflection is blurred),
    refined to sub-pixel."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 300:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    h, w = y1 - y0, x1 - x0
    size = max(w, h)
    b0, b1 = int(y1 - 0.5 * h), min(gray.shape[0], int(y1 + 0.15 * h))
    band = cv2.GaussianBlur(gray[b0:b1, x0:x1], (5, 5), 1.2)
    circles = cv2.HoughCircles(band, cv2.HOUGH_GRADIENT, dp=1.2, minDist=0.25 * w, param1=80, param2=16,
                               minRadius=max(3, int(0.03 * size)), maxRadius=max(6, int(0.11 * size)))
    if circles is None:
        return []
    m = cv2.dilate(mask[b0:b1, x0:x1].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    gy = cv2.Sobel(band.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    out = []
    for cx, cy, r in circles[0]:
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= iy < m.shape[0] and 0 <= ix < m.shape[1]) or not m[iy, ix]:
            continue
        if b0 + cy < y1 - 0.45 * h:                                # rims sit in the lower part of the car
            continue
        a, bb = max(0, int(cx - 0.6 * r)), min(band.shape[1], int(cx + 0.6 * r) + 1)
        lo, hi = int(cy + 0.8 * r), min(band.shape[0], int(cy + RIM_REACH * r) + 1)
        if hi - lo < 4:
            continue
        prof = np.convolve(gy[lo:hi, a:bb].mean(1), np.ones(3) / 3, "same")
        if prof.max() <= 0:
            continue
        strong = np.flatnonzero(prof >= EDGE_FRAC * prof.max())
        if CONTACT_MODE == "first":
            # the first strong dark-to-bright edge below the tyre's inner part: where the tyre itself ends
            first = strong[strong >= int(0.3 * r)]
            i = int(first.min()) if len(first) else int(strong.max())
        else:
            i = int(strong.max())                                   # the lowest: where the shadow under it ends
        while i + 1 < len(prof) and prof[i + 1] > prof[i]:
            i += 1
        out.append((x0 + float(cx), b0 + lo + subpixel_peak(prof, i), float(r)))
    out = sorted(out, key=lambda c: -c[2])[:2]
    return [(u, v) for u, v, _ in out]


def ground(cam, uv):
    K, R, t = cam
    H = K @ np.c_[R[:, 0], R[:, 1], t]
    g = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(H).T
    return g[:, :2] / g[:, 2:3]


def implied_centre(cam, contacts, centre, yaw):
    """Car centre from the tyre contacts. The visible tyres are on the side facing the camera (that fixes the
    lateral sign: in a side view the lateral axis IS the viewing ray); with two contacts their order along the
    heading tells rear from front, with one the nearer of the two to the tracker's centre is taken."""
    if not contacts:
        return None
    G = ground(cam, np.array(contacts, float))
    K, R, t = cam
    foot = -np.linalg.solve(R, t)[:2]
    head = np.array([math.cos(yaw), math.sin(yaw)])
    lat = np.array([-head[1], head[0]])
    near = 1.0 if lat @ (foot - centre) > 0 else -1.0            # the camera-facing side
    lat_off = near * TRACK / 2 * lat
    if len(G) >= 2:
        order = np.argsort(G @ head)
        rear, front = G[order[0]], G[order[-1]]
        if np.linalg.norm(front - rear) < 0.5 * WHEELBASE:          # not two different axles
            G = G[:1]
        else:
            return (rear + front) / 2 - lat_off
    g = G[0]
    options = [g - lat_off - sx * WHEELBASE / 2 * head for sx in (-1, 1)]
    return min(options, key=lambda c: np.linalg.norm(c - centre))


def refine_depth(seq, obs, sigma=2.0, max_shift=3.0):
    """Replace the along-ray component of every pose by the tyres' (smoothed); keep the tracker across the ray."""
    foot = -np.linalg.solve(seq.cams[obs[0]["k"]][1], seq.cams[obs[0]["k"]][2])[:2]
    shift = np.full(len(obs), np.nan)
    for i, o in enumerate(obs):
        u = (np.asarray(o["pos"]) - foot) / np.linalg.norm(np.asarray(o["pos"]) - foot)
        if abs(math.cos(o["yaw"]) * u[1] - math.sin(o["yaw"]) * u[0]) < SIDE_VIEW:
            continue                                               # seen from front or rear: the rims are hidden
        gray = cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2GRAY)
        c = implied_centre(seq.cams[o["k"]], wheel_contacts(gray, o["mask"]), np.asarray(o["pos"]), o["yaw"])
        if c is None:
            continue
        u = (np.asarray(o["pos"]) - foot) / np.linalg.norm(np.asarray(o["pos"]) - foot)
        d = float((c - np.asarray(o["pos"])) @ u)
        if abs(d) <= max_shift:
            shift[i] = d
    ok = ~np.isnan(shift)
    out = [dict(o) for o in obs]
    if ok.sum() < 3:
        return out, 0.0
    idx = np.arange(len(obs))
    filled = np.interp(idx, idx[ok], shift[ok])
    sm = gaussian_filter1d(filled, sigma, mode="nearest")
    for o, t in zip(out, sm):
        u = (np.asarray(o["pos"]) - foot) / np.linalg.norm(np.asarray(o["pos"]) - foot)
        o["pos"] = np.asarray(o["pos"]) + t * u
    return out, float(ok.mean())


# --- diagnostic: tracker vs tracker + wheels, scored on the labels -------------------------------------------
def main():
    sys.path.insert(0, ".")
    import colour_flow_split as cfs
    import ipm_track_models2 as it2
    import ipm_track_models as it
    seq = vs.Sequence()
    Fs = cfs.flows(seq.frames)
    it2.SPLIT = [cfs.detect(f, seq.bg, F) for f, F in zip(seq.frames, Fs)]
    det = [[it2.full_mask(seq, b, f) for b in d] for d, f in zip(it2.SPLIT, seq.frames)]
    del Fs
    tracks = it2.tracker_observations(seq, det)
    foot = -np.linalg.solve(seq.cams[0][1], seq.cams[0][2])[:2]
    lab = {}
    for tid, q in seq.tracks.items():
        for o in q:
            lab.setdefault(o["k"], []).append(o)
    rows, seen = [], []
    for obs in tracks:
        new, frac = refine_depth(seq, obs)
        seen.append(frac)
        for o0, o1 in zip(obs, new):
            if o0["k"] not in lab:
                continue
            o = min(lab[o0["k"]], key=lambda p: np.linalg.norm(p["pos"] - o0["pos"]))
            if np.linalg.norm(o["pos"] - o0["pos"]) > 5:
                continue
            r = o["pos"] - foot
            d = np.linalg.norm(r)
            u = r / d
            rows.append((d, (o0["pos"] - o["pos"]) @ u, (o1["pos"] - o["pos"]) @ u,
                         np.linalg.norm(o0["pos"] - o["pos"]), np.linalg.norm(o1["pos"] - o["pos"])))
    R = np.array(rows)
    print(f"{len(tracks)} tracks, tyres found in {100 * np.mean(seen):.0f}% of their frames; {len(R)} car-frames scored")
    for lo, hi in ((0, 30), (30, 45), (45, 60), (60, 80), (80, 200)):
        s = (R[:, 0] >= lo) & (R[:, 0] < hi)
        if s.sum():
            print(f"{lo:3d}-{hi:3d} m (n={s.sum():4d}): along-ray error |median| {np.median(np.abs(R[s, 1])):.2f} -> "
                  f"{np.median(np.abs(R[s, 2])):.2f} m (bias {np.median(R[s, 1]):+.2f} -> {np.median(R[s, 2]):+.2f}); "
                  f"position {np.median(R[s, 3]):.2f} -> {np.median(R[s, 4]):.2f} m")
    print(f"all: position {np.median(R[:, 3]):.2f} -> {np.median(R[:, 4]):.2f} m; along-ray |median| "
          f"{np.median(np.abs(R[:, 1])):.2f} -> {np.median(np.abs(R[:, 2])):.2f} m")


if __name__ == "__main__":
    main()
