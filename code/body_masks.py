"""
Car body masks from colour clustering (V2X-Seq, intersection A).

Where: the labelled 2D box of each car, padded (moving and stopped cars alike; later replaceable by a detector).
The masks themselves come from the image only:
  1. clustering   : k-means (K colours) on Lab of the pixels in the window;
  2. road         : what the road looks like right there - for a moving car, pixels that differ from the median
                    background are "not road"; for a stopped car (the background contains it) the colours of a thin
                    ring just outside the window are road (k-means, RING_K colours);
  3. body colour  : the largest cluster inside the window that is not road - a body is connected and of one colour;
  4. track        : the per-frame body colours of a car are combined over its whole track (median Lab, robust
                    spread), and every frame is segmented with that one colour: pixels within the colour distance,
                    the connected component under the window's centre;
  5. two masks    : body = that connected region of the body colour; car = body + everything inside the body's
                    convex hull that is not road (windows, tyres, lights - they sit on the body's edge, not enclosed),
                    holes filled.
Checked (evaluation only) against the car's LiDAR points projected into the image (recall) and the projected labelled
3D box (precision, IoU), next to the earlier masks: background subtraction in the box (v2x_seq_shape.Sequence.mask).

Writes docs/body_masks.png, docs/body_masks_stats.png and docs/body_masks.json.
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import car_flow_planes as cfp
import mask_diag as md
import mix3d as mx
import v2x_seq_demo as vd
import v2x_seq_shape as vs

K = 5
RING_K = 3
PAD = 12
RING = 10
ROAD_DE = 14.0                 # Lab distance: farther than this from every road colour = not road
FG_DIFF = 28.0                 # colour difference to the median background that counts as "moved"
MOVING_FRAC = 0.35             # a window with this share of moved pixels is a moving car
COLOUR_K = 2.5                 # body = pixels within COLOUR_K robust spreads of the track's body colour
MIN_SPREAD = 6.0
INK, MUTED = "#1f1f1e", "#8a8984"


def fill_holes(mask):
    """Fill every background region enclosed by the mask."""
    img = np.pad(mask, 1).astype(np.uint8)
    cv2.floodFill(img, np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8), (0, 0), 2)
    return (img != 2)[1:-1, 1:-1]


def window(o, shape, pad=PAD):
    H, W = shape
    x0, y0, x1, y1 = np.array(o["box"]).round().astype(int)
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def kmeans(X, k, seed=0):
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, lab, cen = cv2.kmeans(X.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return lab.ravel(), cen


class BodyMasks:
    def __init__(self, seq):
        self.seq = seq
        self.lab = {}

    def frame_lab(self, k):
        if k not in self.lab:
            self.lab[k] = cv2.cvtColor(cv2.GaussianBlur(self.seq.frames[k], (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        return self.lab[k]

    def body_colour(self, o):
        """(Lab centre, robust spread, clusters image, moving?) of the body in one frame, or None."""
        k = o["k"]
        X0, Y0, X1, Y1 = window(o, self.seq.shape)
        lab = self.frame_lab(k)[Y0:Y1, X0:X1]
        h, w = lab.shape[:2]
        if h < 8 or w < 8:
            return None
        lab_px = lab.reshape(-1, 3)
        labels, cen = kmeans(lab_px, K)
        diff = np.linalg.norm(self.seq.frames[k][Y0:Y1, X0:X1].astype(np.int16)
                              - self.seq.bg[Y0:Y1, X0:X1].astype(np.int16), axis=2).ravel()
        moving = np.mean(diff > FG_DIFF) >= MOVING_FRAC
        if moving:
            notroad = diff > FG_DIFF
        else:                                                   # stopped: road = the ring just outside the window
            H, W = self.seq.shape
            R0, R1 = max(0, Y0 - RING), min(H, Y1 + RING)
            C0, C1 = max(0, X0 - RING), min(W, X1 + RING)
            big = self.frame_lab(k)[R0:R1, C0:C1]
            ring = np.ones(big.shape[:2], bool)
            ring[Y0 - R0:Y1 - R0, X0 - C0:X1 - C0] = False
            _, road = kmeans(big[ring], RING_K, seed=1)
            notroad = np.min(np.linalg.norm(lab_px[:, None] - road[None], axis=2), axis=1) > ROAD_DE
        # the largest cluster whose pixels are mostly not road, weighted towards the window's centre
        yy, xx = np.indices((h, w))
        centre = (np.abs(yy - h / 2) < 0.35 * h) & (np.abs(xx - w / 2) < 0.35 * w)
        best, score = None, 0.0
        for c in range(K):
            sel = labels == c
            if not sel.any():
                continue
            s = (sel & notroad).sum() * (0.5 + np.mean(centre.ravel()[sel]))
            if np.mean(notroad[sel]) > 0.5 and s > score:
                best, score = c, s
        if best is None:
            return None
        px = lab_px[labels == best]
        med = np.median(px, axis=0)
        spread = max(MIN_SPREAD, 1.4826 * np.median(np.linalg.norm(px - med, axis=1)))
        return med, spread, labels.reshape(h, w), moving

    def track_colour(self, obs):
        cols = [self.body_colour(o) for o in obs]
        cols = [c for c in cols if c is not None]
        if not cols:
            return None
        meds = np.array([c[0] for c in cols])
        med = np.median(meds, axis=0)
        spread = float(np.median([c[1] for c in cols]))
        return med, spread

    def masks(self, o, colour):
        """(body, car) full-frame bool masks of one car in one frame."""
        med, spread = colour
        X0, Y0, X1, Y1 = window(o, self.seq.shape)
        lab = self.frame_lab(o["k"])[Y0:Y1, X0:X1]
        close = np.linalg.norm(lab - med, axis=2) <= COLOUR_K * spread
        close = cv2.morphologyEx(close.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, cc, st, ce = cv2.connectedComponentsWithStats(close)
        H, W = self.seq.shape
        body = np.zeros((H, W), bool)
        car = np.zeros((H, W), bool)
        if n <= 1:
            return body, car
        h, w = close.shape
        d = np.hypot(ce[1:, 0] - w / 2, ce[1:, 1] - h / 2) / max(h, w)
        j = 1 + int(np.argmax(st[1:, 4] * np.exp(-3 * d)))        # big and central
        b = cc == j
        b = cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        body[Y0:Y1, X0:X1] = b
        # car = body + everything inside the body's convex hull that is not road (windows, tyres, lights)
        hull = np.zeros_like(close)
        pts = np.argwhere(b)[:, ::-1].astype(np.int32)
        if len(pts) >= 3:
            cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
        notroad = self.not_road(o).reshape(h, w)
        c = b | ((hull > 0) & notroad)
        c = cv2.morphologyEx(c.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        car[Y0:Y1, X0:X1] = fill_holes(c)
        return body, car

    def not_road(self, o):
        """Per pixel of the window: differs from the background (moving car) or from the ring colours (stopped car)."""
        k = o["k"]
        X0, Y0, X1, Y1 = window(o, self.seq.shape)
        lab_px = self.frame_lab(k)[Y0:Y1, X0:X1].reshape(-1, 3)
        diff = np.linalg.norm(self.seq.frames[k][Y0:Y1, X0:X1].astype(np.int16)
                              - self.seq.bg[Y0:Y1, X0:X1].astype(np.int16), axis=2).ravel()
        if np.mean(diff > FG_DIFF) >= MOVING_FRAC:
            return diff > FG_DIFF
        H, W = self.seq.shape
        R0, R1 = max(0, Y0 - RING), min(H, Y1 + RING)
        C0, C1 = max(0, X0 - RING), min(W, X1 + RING)
        big = self.frame_lab(k)[R0:R1, C0:C1]
        ring = np.ones(big.shape[:2], bool)
        ring[Y0 - R0:Y1 - R0, X0 - C0:X1 - C0] = False
        _, road = kmeans(big[ring], RING_K, seed=1)
        return np.min(np.linalg.norm(lab_px[:, None] - road[None], axis=2), axis=1) > ROAD_DE


def evaluate(seq, o, mask, ground, cache):
    """LiDAR recall (projected car points inside the mask) and precision / IoU against the projected 3D box."""
    key = (o["k"], tuple(np.round(o["pos"], 2)))
    if key not in cache:
        L = mx.to_ground_frame(cfp.lidar_on_car(seq, o, ground), o)
        uv = vs.project(seq.cams[o["k"]], vs.world(L, o["pos"], o["yaw"]))[0].round().astype(int) if len(L) else np.zeros((0, 2), int)
        box = vs.label_box_silhouette(o, seq.cams[o["k"]], seq.shape) > 0
        cache[key] = (uv, box)
    uv, box = cache[key]
    H, W = seq.shape
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    recall = float(np.mean(mask[uv[ok, 1], uv[ok, 0]])) if ok.sum() >= 20 else np.nan
    boxd = cv2.dilate(box.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    precision = float((mask & boxd).sum() / max(mask.sum(), 1)) if mask.any() else np.nan
    return {"lidar_recall": recall, "precision": precision, "box_iou": vs.iou(mask, box)}


def speed_of(seq, q, o):
    P = {p["k"]: np.asarray(p["pos"]) for p in q}
    k = o["k"]
    if k - 2 in P and k + 2 in P:
        return float(np.linalg.norm(P[k + 2] - P[k - 2]) / (seq.times[k + 2] - seq.times[k - 2]))
    return np.nan


def main():
    seq = vs.Sequence()
    seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    ground = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))[2]
    bm = BodyMasks(seq)
    cache, rows, examples = {}, [], []
    for tid, q in seq.tracks.items():
        obs = [o for o in q if not o.get("truncated", False) and (o["box"][3] - o["box"][1]) >= 25]
        if len(obs) < 3:
            continue
        colour = bm.track_colour(obs[::3])
        if colour is None:
            continue
        for o in obs[::4]:
            body, car = bm.masks(o, colour)
            old = seq.mask(o) > 0
            r = {"tid": tid, "k": o["k"], "speed": speed_of(seq, q, o), "height_px": float(o["box"][3] - o["box"][1])}
            for name, m in (("body", body), ("car", car), ("background subtraction", old)):
                for kk, v in evaluate(seq, o, m, ground, cache).items():
                    r[f"{name}: {kk}"] = v
            rows.append(r)
            if len(examples) < 40 and o is obs[len(obs) // 2]:
                examples.append((tid, o, body, car, old, colour))
        bm.lab.clear()
    json.dump(rows, open(os.path.join(vd.OUT_DIR, "body_masks.json"), "w"), indent=1)
    R = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k != "tid"}
    print(f"{len(rows)} car-frames, {len(set(r['tid'] for r in rows))} cars")
    for grp, sel in (("moving", R["speed"] >= 0.5), ("stopped", R["speed"] < 0.5), ("all", np.ones(len(rows), bool))):
        print(f"{grp} (n={sel.sum()}):")
        for name in ("background subtraction", "body", "car"):
            print(f"   {name:24s} LiDAR points inside {np.nanmedian(R[f'{name}: lidar_recall'][sel]):.2f} | "
                  f"inside the 3D box {np.nanmedian(R[f'{name}: precision'][sel]):.2f} | "
                  f"IoU with the 3D box {np.nanmedian(R[f'{name}: box_iou'][sel]):.2f}", flush=True)
    examples_figure(os.path.join(vd.OUT_DIR, "body_masks.png"), seq, bm, examples)
    stats_figure(os.path.join(vd.OUT_DIR, "body_masks_stats.png"), R)


def examples_figure(path, seq, bm, examples, n=10):
    pick = examples[:: max(1, len(examples) // n)][:n]
    cols = ("frame", "colour clusters (k-means Lab)", "body colour (the chosen cluster)", "body mask",
            "car mask (body + non-road inside its convex hull)", "earlier: background subtraction")
    fig, axes = plt.subplots(len(pick), len(cols), figsize=(3.1 * len(cols), 2.6 * len(pick)))
    for r, (tid, o, body, car, old, colour) in enumerate(pick):
        X0, Y0, X1, Y1 = window(o, seq.shape, pad=PAD + 8)
        rgb = cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2RGB)[Y0:Y1, X0:X1]
        bc = bm.body_colour(o)
        Xw0, Yw0, Xw1, Yw1 = window(o, seq.shape)
        clusters = -np.ones(rgb.shape[:2], int)
        if bc is not None:
            clusters[Yw0 - Y0:Yw1 - Y0, Xw0 - X0:Xw1 - X0] = bc[2]
        lab = bm.frame_lab(o["k"])[Y0:Y1, X0:X1]
        dist = np.linalg.norm(lab - colour[0], axis=2) / (COLOUR_K * colour[1])
        panels = (rgb, md.label_overlay(rgb, clusters, alpha=0.75, seed=3), None,
                  None, None, None)
        for c in range(len(cols)):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(rgb)
            elif c == 1:
                ax.imshow(panels[1])
            elif c == 2:
                ax.imshow(np.clip(1 - dist, 0, 1), cmap="magma")
            else:
                m = (body, car, old)[c - 3][Y0:Y1, X0:X1]
                ov = rgb.astype(float) / 255
                ov[m] = 0.45 * ov[m] + 0.55 * np.array([0.16, 0.47, 0.84])
                ax.imshow(ov)
                ax.contour(m, levels=[0.5], colors="#eb6834", linewidths=0.8)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=8.5, loc="left", color=INK)
        sp = "stopped" if bc is not None and not bc[3] else "moving"
        axes[r, 0].set_title((cols[0] + "\n" if r == 0 else "") + f"car {tid}, frame {o['k']} ({sp})", fontsize=8,
                             loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor="white")
    plt.close(fig)


def stats_figure(path, R):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
    names = ("background subtraction", "body", "car")
    for ax, (key, title) in zip(axes, (("lidar_recall", "share of the car's LiDAR points inside the mask"),
                                       ("precision", "share of the mask inside the projected 3D box"),
                                       ("box_iou", "IoU with the projected 3D box"))):
        for g, (grp, sel) in enumerate((("moving", R["speed"] >= 0.5), ("stopped", R["speed"] < 0.5))):
            vals = [np.nanmedian(R[f"{n}: {key}"][sel]) for n in names]
            x = np.arange(3) + (g - 0.5) * 0.38
            ax.bar(x, vals, 0.36, color=("#2a78d6", "#eb6834")[g], label=grp)
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(range(3), names, fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.legend(fontsize=8, frameon=False)
        vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
