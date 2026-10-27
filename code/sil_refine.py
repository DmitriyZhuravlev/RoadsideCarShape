"""
Silhouette refinement of a car's track (no labels): the placement from the whole shape standing on the ground.

The tracker's closed-form placement carries a roughly constant offset per car (generic size prior, silhouettes a few
pixels too wide). A carved hull must stand on the ground (z >= 0): placed too far, its ground contact projects higher
than the silhouette's bottom, placed sideways its outline slides off the mask. So, per track, ROUNDS times:
  1. carve the silhouette hull with the current poses (loose consensus first, stricter later);
  2. one common correction of the whole track - along / across the car and of the heading - that minimises the
     two-way truncated chamfer between the projected hull surface and the masks of all frames (grid search);
  3. a small correction per frame (the same cost, one frame), then a smoothing of the trajectory.
"""
import math
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

import v2x_seq_shape as vs

# the cuboid as the first bound: a generic car (v2x_seq_demo.CAR_SIZE, a prior, not a label) with a 15% margin, centred
# on the tracker's closed-form placement; BOX=loose gives the old 6 x 3 m search box
HULL_BOX = (6.0, 3.0) if os.environ.get("BOX", "prior") == "loose" else (4.6 * 1.15, 1.85 * 1.15)

ROUNDS = 3
VOTES = (0.7, 0.8, 0.9)
TRUNC = 15.0
PAD = 120


class FrameTerm:
    """The chamfer cost of one frame, on a crop around its mask."""

    def __init__(self, cam, mask, box):
        H, W = mask.shape
        x0, y0, x1, y1 = box
        self.X0, self.Y0 = max(0, x0 - PAD), max(0, y0 - PAD)
        self.X1, self.Y1 = min(W, x1 + PAD), min(H, y1 + PAD)
        crop = mask[self.Y0:self.Y1, self.X0:self.X1].astype(np.uint8)
        self.dt_out = cv2.distanceTransform(1 - crop, cv2.DIST_L2, 3)
        cnt = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        self.contour = np.vstack([c[:, 0] for c in cnt]) if cnt else np.zeros((0, 2), int)
        self.contour = self.contour[::2]
        self.cam = cam
        self.shape = crop.shape

    def cost(self, P, pos, yaw):
        uv = np.round(vs.project(self.cam, vs.world(P, pos, yaw))[0] - [self.X0, self.Y0]).astype(int)
        h, w = self.shape
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if not ok.any() or len(self.contour) == 0:
            return 2 * TRUNC
        a = np.full(len(uv), TRUNC)
        a[ok] = np.minimum(self.dt_out[uv[ok, 1], uv[ok, 0]], TRUNC)
        splat = np.zeros((h, w), np.uint8)
        splat[uv[ok, 1], uv[ok, 0]] = 1
        splat = cv2.morphologyEx(cv2.dilate(splat, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        dist = cv2.distanceTransform(1 - splat, cv2.DIST_L2, 3)
        b = np.minimum(dist[self.contour[:, 1], self.contour[:, 0]], TRUNC)
        return float(a.mean() + b.mean())


def shifted(o, d):
    """Pose o moved by d = (along, across, dyaw) in its own frame."""
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    return np.asarray(o["pos"]) + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]]), o["yaw"] + d[2]


def surface(seq, obs, axes, vote, step=2):
    views = [(seq.cams[o["k"]], o["pos"], o["yaw"],
              cv2.erode(o["mask"].astype(np.uint8), np.ones((5, 5), np.uint8))) for o in obs[::step]]
    hull = vs.height_field(vs.carve_votes(axes, views) >= vote)
    S = vs.surface_points(hull, axes)
    return hull, S[np.random.default_rng(0).permutation(len(S))[:2500]] if len(S) else S


def refine(seq, obs, verbose=False):
    """A common shift in the car's own frame is a gauge freedom (the hull carved in that frame just moves the other
    way, the silhouettes do not change), so only what the silhouettes can observe is searched:
      - one common shift of every pose along its own viewing ray (in the world): a car placed too far does not reach
        the silhouette's bottom while standing on the ground;
      - small per-frame corrections (inconsistencies between frames), then a smooth trajectory."""
    obs = [dict(o) for o in obs]
    axes = vs.grid(*HULL_BOX, 1.6)
    terms = [FrameTerm(seq.cams[o["k"]], o["mask"], o["box"]) for o in obs]
    sub = list(range(0, len(obs), max(1, len(obs) // 12)))
    foot = [-np.linalg.solve(seq.cams[o["k"]][1], seq.cams[o["k"]][2])[:2] for o in obs]
    log = []
    for rnd in range(ROUNDS):
        hull, S = surface(seq, obs, axes, VOTES[rnd])
        if len(S) < 50:
            break
        ray = [(np.asarray(o["pos"]) - f) / np.linalg.norm(np.asarray(o["pos"]) - f) for o, f in zip(obs, foot)]

        def total(t):
            return float(np.mean([terms[i].cost(S, np.asarray(obs[i]["pos"]) + t * ray[i], obs[i]["yaw"]) for i in sub]))
        span = 3.0 if rnd == 0 else 1.0
        cand = np.arange(-span, span + 1e-9, 0.2)
        costs = [total(t) for t in cand]
        t0 = cand[int(np.argmin(costs))]
        fine = np.arange(t0 - 0.2, t0 + 0.2 + 1e-9, 0.05)
        fc = [total(t) for t in fine]
        t_best = fine[int(np.argmin(fc))]
        for o, r in zip(obs, ray):
            o["pos"] = np.asarray(o["pos"]) + t_best * r
        per = np.zeros((len(obs), 3))
        for i, (o, term) in enumerate(zip(obs, terms)):
            b0, c0 = np.zeros(3), term.cost(S, o["pos"], o["yaw"])
            for da in (-0.3, -0.15, 0, 0.15, 0.3):
                for dc in (-0.2, -0.1, 0, 0.1, 0.2):
                    for dy in np.radians((-2, 0, 2)):
                        c = term.cost(S, *shifted(o, (da, dc, dy)))
                        if c < c0:
                            b0, c0 = np.array([da, dc, dy]), c
            per[i] = b0
        per -= per.mean(0)                                   # the mean is gauge: keep only frame-to-frame changes
        per = np.stack([gaussian_filter1d(per[:, a], 2.0, mode="nearest") for a in range(3)], 1)
        for o, d in zip(obs, per):
            o["pos"], o["yaw"] = shifted(o, d)
        log.append({"round": rnd, "depth_shift_m": float(t_best), "cost": float(min(fc))})
        if verbose:
            print(f"   round {rnd}: depth shift {t_best:+.2f} m along the viewing rays, chamfer {min(fc):.2f} px", flush=True)
    return obs, log
