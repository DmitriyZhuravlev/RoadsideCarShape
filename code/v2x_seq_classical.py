"""
Fully classical end-to-end pipeline for moving vehicles (no neural networks) on
V2X-Seq-SPD, evaluated step by step against the 3D labels.

  detection : median background over the sequence, colour difference, morphology, blobs
  step 1    : Kalman filter on the ground plane (x, y, vx, vy); measurement covariance from
              the pixel-error model (depth error ~ d^2 along the viewing ray, ~ d across it);
              Mahalanobis gating, Hungarian association, coasting, confirmation after 3 hits
  step 2    : splitting of merged blobs: by the predicted cuboids of the tracks they cover
              (nearer car first) or, without tracks, by two-cluster Farneback optical flow
  step 3    : ground-contact line from horizontal-edge energy, which cuts the wet-road
              reflection below the car
  step 4    : stopped cars kept alive by normalised cross-correlation with their last image
  lifting   : lifting.vp_bounding_quadrangle -> the motion picks the closed-form candidate
              (face_candidates) and its front, solve_length gives centre and length
  offline   : Rauch-Tung-Striebel smoothing of every track

Writes docs/v2x_seq_classical_steps.png and docs/v2x_seq_classical_examples.png.
Usage: V2X_SEQ_DIR=... python v2x_seq_classical.py   (about 5 minutes)
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import lifting
import v2x_seq_demo as vd

L, W, H = vd.CAR_SIZE
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", vd.INK, vd.MUTED
STEPS = [("1 Kalman tracker", (1,)), ("+ 2 blob splitting", (1, 2)), ("+ 3 contact line", (1, 2, 3)),
         ("+ 4 stopped cars", (1, 2, 3, 4))]


def yaw_error(a, b, period=np.pi):
    d = (a - b) % period
    return np.degrees(min(d, period - d))


# --- detection: blobs are crops (box in image coordinates, m = crop mask) ---------------------
def foreground(frame, bg, threshold=28):
    d = np.linalg.norm(frame.astype(np.int16) - bg.astype(np.int16), axis=2)
    m = cv2.morphologyEx((d > threshold).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def make_blob(m, ox, oy, min_area=100):
    ys, xs = np.nonzero(m)
    if len(xs) < min_area:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = np.ascontiguousarray(m[y0:y1, x0:x1])
    contour = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    return {"box": (ox + x0, oy + y0, ox + x1, oy + y1), "m": m, "area": len(xs),
            "contour": contour[:, 0].astype(float) + [ox + x0, oy + y0]}


def blobs(fg, min_area=250):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area >= min_area:
            b = make_blob((labels[y:y + h, x:x + w] == k).astype(np.uint8), x, y)
            if b:
                out.append(b)
    return out


def contact_row(gray, b, fraction=0.35, search=0.6):
    """Lowest row in the lower part of the blob whose horizontal-edge energy is >= fraction of the maximum:
    the car's underside and tyres are sharp, its reflection on a wet road is blurred."""
    x0, y0, x1, y1 = b["box"]
    gy = np.abs(cv2.Sobel(gray[y0:y1, x0:x1].astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)) * b["m"]
    profile = np.convolve(gy.sum(1) / np.maximum(b["m"].sum(1), 1), np.ones(3) / 3, "same")
    lo = int((1 - search) * (y1 - y0))
    strong = np.where(profile[lo:] >= fraction * profile.max())[0]
    return y1 if len(strong) == 0 else y0 + lo + strong.max() + 1


def cut_reflection(b, gray):
    row = contact_row(gray, b)
    if row >= b["box"][3] - 1:
        return b
    m = b["m"].copy()
    m[row - b["box"][1]:] = 0
    return make_blob(m, *b["box"][:2]) or b


def flow_split(b, flow, min_fraction=0.3, min_separation=1.5):
    """Split a blob whose optical flow has two clearly different modes."""
    if b["area"] < 3000:
        return [b]
    ys, xs = np.nonzero(b["m"])
    x0, y0 = b["box"][:2]
    vectors = flow[ys + y0, xs + x0].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    _, labels, centres = cv2.kmeans(vectors, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    if np.linalg.norm(centres[0] - centres[1]) < min_separation or min(np.mean(labels == 0), np.mean(labels == 1)) < min_fraction:
        return [b]
    parts = []
    for c in (0, 1):
        m = np.zeros_like(b["m"])
        m[ys[labels == c], xs[labels == c]] = 1
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)))
        parts += [p for k in range(1, n) if stats[k, 4] >= 400 for p in [make_blob((lab == k).astype(np.uint8), x0, y0)] if p]
    return parts or [b]


def overlap(a, b):
    x0, y0 = max(a["box"][0], b["box"][0]), max(a["box"][1], b["box"][1])
    x1, y1 = min(a["box"][2], b["box"][2]), min(a["box"][3], b["box"][3])
    if x1 <= x0 or y1 <= y0:
        return 0
    A = a["m"][y0 - a["box"][1]:y1 - a["box"][1], x0 - a["box"][0]:x1 - a["box"][0]]
    B = b["m"][y0 - b["box"][1]:y1 - b["box"][1], x0 - b["box"][0]:x1 - b["box"][0]]
    return int((A & B).sum())


def place(c, box):
    m = np.zeros((box[3] - box[1], box[2] - box[0]), np.uint8)
    x0, y0, x1, y1 = c["box"]
    X0, Y0, X1, Y1 = max(x0, box[0]), max(y0, box[1]), min(x1, box[2]), min(y1, box[3])
    if X1 > X0 and Y1 > Y0:
        m[Y0 - box[1]:Y1 - box[1], X0 - box[0]:X1 - box[0]] = c["m"][Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    return m


# --- geometry --------------------------------------------------------------------------------
class Geometry:
    def __init__(self, cam, shape, snap=25.0):
        self.K, self.R, self.t = cam
        self.M = lifting.plane_homography(self.K, self.R, self.t, 0.0)
        self.vz = lifting.vertical_vanishing_point(self.K, self.R)
        centre = -np.linalg.inv(self.R) @ self.t
        self.foot, self.cam_h, self.shape, self.snap = centre[:2], centre[2], shape, snap

    def project(self, X):
        x = self.K @ (self.R @ np.asarray(X, float).T + self.t[:, None])
        return (x[:2] / x[2]).T

    def hull(self, center, yaw):
        fp = lifting.compute_rectangle_points(center, yaw, W, L)[0]
        uv = self.project(np.vstack([np.c_[fp, np.zeros(4)], np.c_[fp, np.full(4, H)]]))
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int) + 1
        X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(self.shape[1], x1), min(self.shape[0], y1)
        if X1 <= X0 or Y1 <= Y0:
            return None
        m = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull((uv - [X0, Y0]).round().astype(np.int32)), 1)
        return {"box": (X0, Y0, X1, Y1), "m": m, "area": int(m.sum())}

    def iou(self, b, center, yaw):
        h = self.hull(center, yaw)
        if h is None:
            return 0.0
        inter = overlap(h, b)
        return inter / max(h["area"] + b["area"] - inter, 1)

    def lift(self, b, heading=None):
        q = lifting.vp_bounding_quadrangle(b["contour"], self.M, self.vz)
        if q is None:
            return None
        cands = lifting.face_candidates(q, L, W)
        if heading is not None:
            # the motion picks the closed-form candidate and its front; the silhouette gives the precise angle
            yaw = heading
            near = [c for c in cands if yaw_error(c[0], heading) <= self.snap]
            if near:
                y0 = min(near, key=lambda c: yaw_error(c[0], heading))[0]
                yaw = y0 if np.cos(y0 - heading) > 0 else y0 + np.pi
            solved = lifting.solve_length(q, yaw, W)
            if solved is not None and 0.6 * L < solved[1] < 1.6 * L:
                return {"center": solved[0], "yaw": yaw % (2 * np.pi)}
        if not cands:
            return None
        if heading is not None:
            best = min(cands, key=lambda c: yaw_error(c[0], heading))
            yaw = best[0] if np.cos(best[0] - heading) > 0 else best[0] + np.pi
        else:
            best = max(cands, key=lambda c: self.iou(b, c[1], c[0]))
            yaw = best[0]
        return {"center": best[1], "yaw": yaw % (2 * np.pi)}

    def measurement_cov(self, center, pixel_std=4.0):
        """Along the viewing ray d^2 * px / (f * h_cam) (+ size slack), across it d * px / f."""
        r = center - self.foot
        d = max(np.linalg.norm(r), 1.0)
        u = r / d
        f = self.K[1, 1]
        s_r, s_l = 0.35 + d * d * pixel_std / (f * self.cam_h), 0.25 + d * pixel_std / f
        V = np.c_[u, [-u[1], u[0]]]
        return V @ np.diag([s_r ** 2, s_l ** 2]) @ V.T


def split_by_tracks(b, owners, hulls, geo):
    """Divide a blob between the tracks whose predicted cuboids it covers, nearer car first."""
    owners = sorted(owners, key=lambda tr: np.linalg.norm(tr.x[:2] - geo.foot))
    dist = np.stack([cv2.distanceTransform((1 - place(hulls[tr.id], b["box"])).astype(np.uint8), cv2.DIST_L2, 3)
                     for tr in owners])
    labels, parts = np.argmin(dist, axis=0), []
    for i, tr in enumerate(owners):
        p = make_blob(((labels == i) & (b["m"] > 0)).astype(np.uint8), *b["box"][:2])
        if p is not None and p["area"] > 0.6 * min(hulls[tr.id]["area"], b["area"]):
            p["owner"] = tr.id
            parts.append(p)
    return parts


# --- tracking --------------------------------------------------------------------------------
def rts_smooth(log):
    """Rauch-Tung-Striebel smoother. log: [(frame, x_pred, P_pred, x_filt, P_filt, F)] -> {frame: state}."""
    if not log:
        return {}
    xs = [step[3].copy() for step in log]
    Ps = [step[4].copy() for step in log]
    for i in range(len(log) - 2, -1, -1):
        F_next, P_pred_next = log[i + 1][5], log[i + 1][2]
        C = log[i][4] @ F_next.T @ np.linalg.pinv(P_pred_next)
        xs[i] = log[i][3] + C @ (xs[i + 1] - log[i + 1][1])
        Ps[i] = log[i][4] + C @ (Ps[i + 1] - P_pred_next) @ C.T
    return {step[0]: x for step, x in zip(log, xs)}


class Track:
    def __init__(self, tid, z, Rm, yaw, k):
        self.id, self.yaw, self.hits, self.miss, self.template = tid, yaw, 1, 0, None
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.zeros((4, 4))
        self.P[:2, :2], self.P[2, 2], self.P[3, 3] = Rm, 25.0, 25.0
        self.measured = {k}
        self.log = [[k, self.x.copy(), self.P.copy(), self.x.copy(), self.P.copy(), np.eye(4)]]

    @property
    def confirmed(self):
        return self.hits >= 3

    def predict(self, dt, k, acc=2.5):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x, self.P = F @ self.x, F @ self.P @ F.T + G @ G.T * acc ** 2
        self.log.append([k, self.x.copy(), self.P.copy(), self.x.copy(), self.P.copy(), F])

    def heading(self, min_speed=1.0):
        if np.hypot(*self.x[2:]) >= min_speed:
            return np.arctan2(self.x[3], self.x[2])
        return self.yaw if self.hits > 3 else None

    def update(self, z, Rm):
        Hm = np.eye(2, 4)
        S = Hm @ self.P @ Hm.T + Rm
        gain = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + gain @ (z - Hm @ self.x)
        self.P = (np.eye(4) - gain @ Hm) @ self.P
        self.hits, self.miss = self.hits + 1, 0
        self.log[-1][3], self.log[-1][4] = self.x.copy(), self.P.copy()
        self.measured.add(self.log[-1][0])

    def mahalanobis(self, z, Rm):
        d = z - self.x[:2]
        return float(d @ np.linalg.solve(self.P[:2, :2] + Rm, d))


def run(frames, times, bg, cam, steps, gate=13.8, max_miss=15, debug=(), detector=None):
    """Online pipeline. Returns per-frame confirmed objects, all confirmed tracks, and debug blobs.
    detector(k, frame) -> blobs replaces the built-in background-subtraction blobs when given."""
    cv2.setRNGSeed(0)  # k-means in flow_split: reproducible runs
    shape = frames[0].shape[:2]
    geo, tracks, finished, out, debug_blobs, prev_small, next_id = Geometry(cam, shape), [], [], [], {}, None, 0
    for k, frame in enumerate(frames):
        dt = times[k] - times[k - 1] if k else 0.1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        for tr in tracks:
            tr.predict(dt, k)
        dets = blobs(foreground(frame, bg)) if detector is None else detector(k, frame)
        if 3 in steps:
            dets = [cut_reflection(d, gray) for d in dets]
        if 2 in steps and prev_small is not None:
            flow = cv2.resize(cv2.calcOpticalFlowFarneback(prev_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0),
                              (shape[1], shape[0])) * 2
            live = [tr for tr in tracks if tr.confirmed and tr.heading() is not None]
            hulls = {tr.id: geo.hull(tr.x[:2], tr.heading()) for tr in live}
            live = [tr for tr in live if hulls[tr.id] is not None]
            split = []
            for d in dets:
                owners = [tr for tr in live if overlap(hulls[tr.id], d) > 0.25 * min(hulls[tr.id]["area"], d["area"])]
                split += split_by_tracks(d, owners, hulls, geo) if len(owners) >= 2 else flow_split(d, flow)
            dets = split
        if k in debug:
            debug_blobs[k] = dets
        # association: every (track, detection) pair is lifted with the track's own heading
        cost, meas = np.full((len(tracks), len(dets)), 1e6), {}
        for i, tr in enumerate(tracks):
            heading = tr.heading()
            p = geo.project([[*tr.x[:2], 0]])[0]
            for j, d in enumerate(dets):
                bx = d["box"]
                if d.get("owner", tr.id) != tr.id or not (bx[0] - 150 < p[0] < bx[2] + 150 and bx[1] - 150 < p[1] < bx[3] + 150):
                    continue
                e = geo.lift(d, heading)
                if e is not None:
                    Rm = geo.measurement_cov(e["center"])
                    m2 = tr.mahalanobis(e["center"], Rm)
                    if m2 < gate:
                        cost[i, j], meas[(i, j)] = m2, (e, Rm)
        used_t, used_d = set(), set()
        if tracks and dets:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e6:
                    e, Rm = meas[(i, j)]
                    tracks[i].update(e["center"], Rm)
                    tracks[i].yaw = e["yaw"]
                    x0, y0, x1, y1 = dets[j]["box"]
                    tracks[i].template = (frame[y0:y1, x0:x1].copy(), dets[j]["box"])
                    used_t.add(i)
                    used_d.add(j)
        for i, tr in enumerate(tracks):
            if i in used_t:
                continue
            tr.miss += 1
            if 4 in steps and tr.confirmed and np.hypot(*tr.x[2:]) < 1.0 and tr.template is not None:
                tpl, (x0, y0, x1, y1) = tr.template
                win = frame[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8]
                if win.shape[0] >= tpl.shape[0] and win.shape[1] >= tpl.shape[1] and \
                        cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED).max() > 0.7:
                    tr.miss, tr.x[2:] = 0, 0.0
        for j, d in enumerate(dets):
            if j not in used_d and "owner" not in d:
                e = geo.lift(d, None)
                if e is not None:
                    tracks.append(Track(next_id, e["center"], geo.measurement_cov(e["center"]), e["yaw"], k))
                    next_id += 1
        finished += [tr for tr in tracks if tr.confirmed and tr.miss > max_miss]
        tracks = [tr for tr in tracks if tr.miss <= (max_miss if tr.confirmed else 2)]
        objects = []
        for tr in tracks:
            if tr.confirmed:
                heading = tr.heading()
                objects.append((tr.id, {"center": tr.x[:2].copy(), "yaw": heading if heading is not None else tr.yaw,
                                        "speed": np.hypot(*tr.x[2:])}))
        out.append(objects)
        prev_small = small
    return out, finished + [tr for tr in tracks if tr.confirmed], debug_blobs


def smoothed_output(tracks, n_frames):
    """Offline result: smoothed states of every confirmed track between its first and last measurement."""
    out = [[] for _ in range(n_frames)]
    for tr in tracks:
        lo, hi = min(tr.measured), max(tr.measured)
        for k, x in rts_smooth(tr.log).items():
            if lo <= k <= hi:
                speed = np.hypot(*x[2:])
                out[k].append((tr.id, {"center": x[:2], "yaw": np.arctan2(x[3], x[2]) if speed > 0.5 else tr.yaw,
                                       "speed": speed}))
    return out


# --- evaluation ------------------------------------------------------------------------------
def moving_ground_truth(times, tracks, camera_xy):
    rows = []
    for tid, seq in tracks.items():
        ks = np.array([s["k"] for s in seq])
        P = np.array([s["pos"] for s in seq])
        for s in seq:
            w = np.abs(ks - s["k"]) <= vd.WINDOW
            if w.sum() < 4 or s["truncated"]:
                continue
            v = vd.fit_velocity(times[ks[w]], P[w])
            if np.linalg.norm(v) >= vd.MIN_SPEED:
                rows.append({"k": s["k"], "pos": s["pos"], "yaw": s["yaw"], "heading": np.arctan2(v[1], v[0]),
                             "speed": np.linalg.norm(v), "dist": np.linalg.norm(s["pos"] - camera_xy), "track": tid})
    return rows


def evaluate(out, gt, radius=3.0):
    found = []
    for g in gt:
        cands = [(np.linalg.norm(o["center"] - g["pos"]), o) for _, o in out[g["k"]]]
        d, o = min(cands, key=lambda c: c[0]) if cands else (np.inf, None)
        found.append(None if d > radius else (d, yaw_error(o["yaw"], g["yaw"]), yaw_error(o["yaw"], g["heading"], 2 * np.pi),
                                              abs(o["speed"] - g["speed"]), g["dist"]))
    ok = np.array([f for f in found if f])
    return {"coverage": len(ok) / len(gt), "position": float(np.median(ok[:, 0])), "orientation": float(np.mean(ok[:, 1] < 10)),
            "heading": float(np.mean(ok[:, 2] < 20)), "speed": float(np.median(ok[:, 3])), "rows": ok}


def reference(times, cams, tracks, camera_xy):
    """The same metrics with the labelled 2D boxes as detections (v2x_seq_demo, heading+length)."""
    rows, _ = vd.evaluate(times, cams, tracks, camera_xy)
    r = rows["heading+length"]
    return {"coverage": len(r) / len(moving_ground_truth(times, tracks, camera_xy)), "position": float(np.median(r[:, 0])),
            "orientation": float(np.mean(r[:, 1] < 10)), "heading": float(np.mean(r[:, 2] < 20)), "speed": float(np.median(r[:, 3]))}


# --- figures ---------------------------------------------------------------------------------
def steps_figure(path, results, ref):
    names = ["1", "1+2", "1–3", "1–4", "1–3\nsmoothed", "labelled\nboxes"]
    vals = [r for _, r in results] + [ref]
    colors = ["#c3c2b7"] * (len(results) - 1) + [BLUE, "#9d9c95"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.9))
    panels = [("coverage", "Moving cars found (%)", 100, "{:.0f}"), ("orientation", "Orientation < 10° (%)", 100, "{:.0f}"),
              ("position", "Median position error (m)", 1, "{:.2f}"), ("speed", "Median speed error (m/s)", 1, "{:.2f}")]
    for ax, (key, title, scale, fmt) in zip(axes, panels):
        v = [r[key] * scale for r in vals]
        bars = ax.bar(range(len(v)), v, 0.7, color=colors, zorder=2)
        for b, x in zip(bars, v):
            ax.annotate(fmt.format(x), (b.get_x() + b.get_width() / 2, x), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(v)), names, fontsize=8.5)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
        if scale == 100:
            ax.set_ylim(0, 105)
        vd.style(ax)
    fig.suptitle("V2X-Seq-SPD, moving cars and vans: fully classical pipeline, step by step\n"
                 "1 Kalman tracker · 2 blob splitting · 3 reflection cut at the contact line · 4 stopped cars · "
                 "blue = final (1–3 + offline smoothing) · right = the method with labelled 2D boxes",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)


def examples_figure(path, fids, times, cams, tracks, camera_xy, out, final_tracks, debug_blobs):
    frames_k = sorted(debug_blobs)
    fig = plt.figure(figsize=(19, 2.75 * len(frames_k) + 0.6))
    gs = fig.add_gridspec(len(frames_k), 3, width_ratios=[3.2, 3.2, 1.25])
    ax_bev = fig.add_subplot(gs[:, 2])
    by_frame = {}
    for tid, seq in tracks.items():
        for s in seq:
            by_frame.setdefault(s["k"], []).append(s)
    geo = Geometry(cams[0], (1080, 1920))
    for row, k in enumerate(frames_k):
        img = cv2.cvtColor(cv2.imread(os.path.join(vd.DATA, "image", f"{fids[k]}.jpg")), cv2.COLOR_BGR2RGB)
        ax_det, ax_cub = fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1])
        ax_det.imshow(img)
        for b in debug_blobs[k]:
            ax_det.plot(*np.vstack([b["contour"], b["contour"][:1]]).T, color="#ff3df5", linewidth=1.0)
        ax_det.set_title(f"t = {times[k] - times[0]:.1f} s: classical foreground blobs after contact cut and splitting",
                         color=INK, fontsize=9.5, loc="left")
        ax_cub.imshow(img)
        for s in by_frame.get(k, []):
            l, w, h = s["dims"]
            fp = lifting.compute_rectangle_points(s["pos"], s["yaw"], w, l)[0]
            uv = geo.project(np.vstack([np.c_[fp, np.full(4, s["z"] - h / 2)], np.c_[fp, np.full(4, s["z"] + h / 2)]]))
            for a, b in vd.EDGES:
                ax_cub.plot(*uv[[a, b]].T, color="white", linewidth=0.8, linestyle=(0, (3, 2)))
        for _, o in out[k]:
            fp = lifting.compute_rectangle_points(o["center"], o["yaw"], W, L)[0]
            uv = geo.project(np.vstack([np.c_[fp, np.zeros(4)], np.c_[fp, np.full(4, H)]]))
            for a, b in vd.EDGES:
                ax_cub.plot(*uv[[a, b]].T, color=BLUE, linewidth=1.5)
            arrow = geo.project([[*o["center"], 0], [*(o["center"] + 3 * np.array([np.cos(o["yaw"]), np.sin(o["yaw"])])), 0]])
            ax_cub.annotate("", xy=arrow[1], xytext=arrow[0], arrowprops=dict(arrowstyle="-|>", color="#ffd400", lw=1.4))
        ax_cub.set_title("cuboids from the classical pipeline (blue, smoothed), labels (white dashed), heading (yellow)",
                         color=INK, fontsize=9.5, loc="left")
        for ax in (ax_det, ax_cub):
            ax.set_xlim(0, 1920)
            ax.set_ylim(780, 180)
            ax.axis("off")
    for tid, seq in tracks.items():
        P = np.array([s["pos"] for s in seq])
        if np.linalg.norm(P[-1] - P[0]) >= 5:
            ax_bev.plot(P[:, 1], P[:, 0], color="#c3c2b7", linewidth=3, solid_capstyle="round", zorder=1)
    for tr in final_tracks:
        lo, hi = min(tr.measured), max(tr.measured)
        X = np.array([x[:2] for k, x in sorted(rts_smooth(tr.log).items()) if lo <= k <= hi])
        if len(X) > 5 and np.linalg.norm(X[-1] - X[0]) > 5:
            ax_bev.plot(X[:, 1], X[:, 0], color=BLUE, linewidth=1.0, zorder=2)
    ax_bev.plot(camera_xy[1], camera_xy[0], marker="^", color=INK, markersize=9)
    ax_bev.invert_xaxis()
    ax_bev.set_aspect("equal")
    ax_bev.set_ylim(0, 120)
    ax_bev.set_xlabel("lateral (m)", color=INK)
    ax_bev.set_ylabel("forward (m)", color=INK)
    ax_bev.set_title("25 s of trajectories\ngrey = labels, blue = classical pipeline", color=INK, fontsize=10, loc="left")
    vd.style(ax_bev)
    fig.tight_layout()
    fig.savefig(path, dpi=90, facecolor="white")
    plt.close(fig)


def draw_cuboid(img, uv, color, thickness=2, dashed=False, dash=8, gap=6):
    """Draw the 12 edges of a projected cuboid (rows 0-3 bottom, 4-7 top) on img in place."""
    for a, b in vd.EDGES:
        p, q = np.asarray(uv[a], float), np.asarray(uv[b], float)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
            continue
        if not dashed:
            cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), color, thickness, cv2.LINE_AA)
            continue
        length = np.linalg.norm(q - p)
        for s in np.arange(0, length, dash + gap):
            e = min(s + dash, length)
            cv2.line(img, tuple(np.round(p + (q - p) * s / length).astype(int)),
                     tuple(np.round(p + (q - p) * e / length).astype(int)), color, thickness, cv2.LINE_AA)
    return img


def detection_images(out_dir, fids, times, cams, tracks, camera_xy, out, frames_k, radius=3.0):
    """One image per frame: estimates that match a labelled car/van (blue), another labelled road user
    (orange), or nothing (red, false detection); moving cars/vans the pipeline missed (yellow dashed)."""
    os.makedirs(out_dir, exist_ok=True)
    geo = Geometry(cams[0], (1080, 1920))
    moving = {(g["track"], g["k"]) for g in moving_ground_truth(times, tracks, camera_xy)}
    colors = {"car": (214, 120, 42), "other": (52, 104, 235), "false": (40, 40, 220), "missed": (0, 215, 255)}
    tiles, totals = [], {"car": 0, "other": 0, "false": 0, "missed": 0, "moving": 0}
    for k in frames_k:
        img = cv2.imread(os.path.join(vd.DATA, "image", f"{fids[k]}.jpg"))
        labels = json.load(open(os.path.join(vd.DATA, "label", "camera", f"{fids[k]}.json")))
        lab = [(o, np.array([o["3d_location"]["x"], o["3d_location"]["y"]])) for o in labels]
        counts = {"car": 0, "other": 0, "false": 0, "missed": 0, "moving": 0}
        for _, o in out[k]:
            d = [(np.linalg.norm(p - o["center"]), lo) for lo, p in lab]
            dist, lo = min(d, key=lambda z: z[0]) if d else (np.inf, None)
            kind = "false" if dist > radius else ("car" if lo["type"] in ("Car", "Van") else "other")
            counts[kind] += 1
            fp = lifting.compute_rectangle_points(o["center"], o["yaw"], W, L)[0]
            draw_cuboid(img, geo.project(np.vstack([np.c_[fp, np.zeros(4)], np.c_[fp, np.full(4, H)]])), colors[kind], 2)
            a = geo.project([[*o["center"], 0], [*(o["center"] + 3 * np.array([np.cos(o["yaw"]), np.sin(o["yaw"])])), 0]])
            cv2.arrowedLine(img, tuple(np.round(a[0]).astype(int)), tuple(np.round(a[1]).astype(int)), (0, 230, 255), 2,
                            cv2.LINE_AA, tipLength=0.3)
        for tid, seq in tracks.items():
            s = next((s for s in seq if s["k"] == k), None)
            if s is None or (tid, k) not in moving:
                continue
            counts["moving"] += 1
            if min([np.linalg.norm(o["center"] - s["pos"]) for _, o in out[k]] or [np.inf]) > radius:
                counts["missed"] += 1
                l, w, h = s["dims"]
                fp = lifting.compute_rectangle_points(s["pos"], s["yaw"], w, l)[0]
                draw_cuboid(img, geo.project(np.vstack([np.c_[fp, np.full(4, s["z"] - h / 2)],
                                                        np.c_[fp, np.full(4, s["z"] + h / 2)]])), colors["missed"], 2, dashed=True)
        for key in totals:
            totals[key] += counts[key]
        crop = img[150:800].copy()
        lines = [f"t = {times[k] - times[0]:4.1f} s   moving cars/vans found: {counts['moving'] - counts['missed']}/{counts['moving']}",
                 f"blue: car/van ({counts['car']})   orange: other road user ({counts['other']})   "
                 f"red: false detection ({counts['false']})   yellow dashed: missed moving car"]
        cv2.rectangle(crop, (0, 0), (1180, 78), (255, 255, 255), -1)
        for i, text in enumerate(lines):
            cv2.putText(crop, text, (12, 32 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 31, 31), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"frame_{fids[k]}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        tiles.append(cv2.resize(crop, None, fx=0.5, fy=0.5))
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)])
    cv2.imwrite(os.path.join(out_dir, "overview.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return totals


def main_detections(frames_k=(30, 70, 110, 150, 190, 230)):
    """Only the final online pipeline (steps 1-3) and the per-frame detection images."""
    fids, times, cams, tracks, camera_xy = vd.load_sequence()
    frames = [cv2.imread(os.path.join(vd.DATA, "image", f"{f}.jpg")) for f in fids]
    bg = np.median(np.stack(frames[::2]), axis=0).astype(np.uint8)
    out, _, _ = run(frames, times, bg, cams[0], (1, 2, 3))
    totals = detection_images(os.path.join(vd.OUT_DIR, "v2x_seq_detections"), fids, times, cams, tracks, camera_xy,
                              out, frames_k)
    print("over the shown frames:", totals)


def main():
    os.makedirs(vd.OUT_DIR, exist_ok=True)
    fids, times, cams, tracks, camera_xy = vd.load_sequence()
    frames = [cv2.imread(os.path.join(vd.DATA, "image", f"{f}.jpg")) for f in fids]
    bg = np.median(np.stack(frames[::2]), axis=0).astype(np.uint8)
    gt = moving_ground_truth(times, tracks, camera_xy)
    results = []
    for name, steps in STEPS:
        out, _, _ = run(frames, times, bg, cams[0], steps)
        results.append((name, evaluate(out, gt)))
    examples = (60, 150, 230)
    out, final_tracks, debug_blobs = run(frames, times, bg, cams[0], (1, 2, 3), debug=examples)
    smooth = smoothed_output(final_tracks, len(frames))
    results.append(("1+2+3 + offline smoothing", evaluate(smooth, gt)))
    ref = reference(times, cams, tracks, camera_xy)
    for name, r in results + [("labelled 2D boxes (reference)", ref)]:
        print(f"{name:32s} found {r['coverage']:4.0%} | position {r['position']:.2f} m | orientation<10 {r['orientation']:4.0%} | "
              f"heading<20 {r['heading']:4.0%} | speed error {r['speed']:.2f} m/s")
    rows = results[-1][1]["rows"]
    for lo, hi in ((0, 40), (40, 60), (60, 80), (80, 200)):
        s = (rows[:, 4] >= lo) & (rows[:, 4] < hi)
        n_gt = sum(lo <= g["dist"] < hi for g in gt)
        print(f"   {lo}-{hi} m: found {s.sum() / max(n_gt, 1):.0%} of {n_gt}, position {np.median(rows[s, 0]) if s.any() else np.nan:.2f} m")
    steps_figure(os.path.join(vd.OUT_DIR, "v2x_seq_classical_steps.png"), results, ref)
    examples_figure(os.path.join(vd.OUT_DIR, "v2x_seq_classical_examples.png"), fids, times, cams, tracks, camera_xy,
                    smooth, final_tracks, debug_blobs)


if __name__ == "__main__":
    import sys
    main_detections() if "--detections" in sys.argv else main()
