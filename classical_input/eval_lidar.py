"""Our classical cuboids against the cars' accumulated LiDAR, scored as in the article (Sect. 5), next to the labelled
boxes. Written from the article's description of the protocol, not from its code:

  - per labelled car, the LiDAR points inside its box (+0.1 m) are collected over the frames of the clip in the box
    frame, mirrored about the box's vertical centre plane along its heading, and turned into a height field of 5 cm
    columns (the highest point per column, heights above the fitted road plane);
  - "closed": the gaps between scan lines filled by a grey-scale closing with a radius of 15 cm;
  - a box is a height field of its footprint at its height; both are centred on their footprints (a box's footprint
    centre is its centre, the reference's the middle of its occupied columns) and compared in their own frames, on a
    grid that covers both: solid IoU = sum min / sum max.

Our cuboid of a car is the median of the cuboids of our objects matched to it (nearest object within 3 m, frames
where it is matched); a car is scored when it was matched in at least MIN_FRAMES frames and its reference holds at
least MIN_POINTS points.

    python eval_lidar.py [DATA_DIR] [FIRST] [N]      # the frames of out/ (default: the 60-frame run)
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import grey_closing

import camera
import run_example as rx

CELL, CLOSE_R, MARGIN = 0.05, 0.15, 0.1
MIN_FRAMES, MIN_POINTS = 3, 100
GATE = float(os.environ.get("GATE", "3.0"))   # m; 6 on DAIR-V2X-C, whose labels lag the image (clock offset)
CARS = ("Car", "Van", "Truck", "Bus")


def box_field(l, w, h, X, Y):
    xs, ys = np.arange(-X + CELL / 2, X, CELL), np.arange(-Y + CELL / 2, Y, CELL)
    return np.where((np.abs(ys)[:, None] <= w / 2) & (np.abs(xs)[None, :] <= l / 2), h, 0.0)


def solid_iou(H, L):
    d = np.maximum(H, L).sum()
    return float(np.minimum(H, L).sum() / d) if d > 0 else 0.0


def footprint_iou(H, L):
    a, b = H > 0, L > 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def height_err(H, L):
    """Median |H - L| over the columns occupied in both (m)."""
    both = (H > 0) & (L > 0)
    return float(np.median(np.abs(H[both] - L[both]))) if both.any() else float("nan")


def close_gaps(L):
    r = int(round(CLOSE_R / CELL))
    i, j = np.mgrid[:2 * r + 1, :2 * r + 1]
    return grey_closing(L, footprint=(i - r) ** 2 + (j - r) ** 2 <= r * r)


def reference(points, X, Y):
    """Height field of the points (box frame: x, y, height) on the grid of half-size X, Y, centred on its occupied
    columns; None if too few points."""
    if len(points) < MIN_POINTS:
        return None
    P = np.r_[points, points * [1, -1, 1]]                       # mirrored about the centre plane
    ix = np.floor((P[:, 0] + X) / CELL).astype(int)
    iy = np.floor((P[:, 1] + Y) / CELL).astype(int)
    nx, ny = int(round(2 * X / CELL)), int(round(2 * Y / CELL))
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    F = np.zeros((ny, nx))
    np.maximum.at(F, (iy[ok], ix[ok]), P[ok, 2])
    cols = np.nonzero(F.any(0))[0]
    shift = int(round(nx / 2 - 0.5 * (cols.min() + cols.max() + 1)))  # the middle of the occupied columns to the centre
    return np.roll(F, shift, axis=1)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else rx.ROOT
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    fids = sorted(f[:-4] for f in os.listdir(os.path.join(root, "image")))[first:first + n]
    cam = camera.Camera(root, fids[0])
    pts, labs, ours = {}, {}, {}
    for k, f in enumerate(fids):
        ls = [l for l in rx.labels(cam, f) if l["type"] in CARS]
        labs[k] = ls
        X = cam.to_g(camera.read_pcd_xyz(os.path.join(root, "velodyne", f + ".pcd")))
        for l in ls:                                             # the car's points in its box frame
            c, yaw, (L, W, H) = l["centre"], l["yaw"], l["lwh"]
            d = X[:, :2] - c
            x = d @ [np.cos(yaw), np.sin(yaw)]
            y = d @ [-np.sin(yaw), np.cos(yaw)]
            sel = (np.abs(x) <= L / 2 + MARGIN) & (np.abs(y) <= W / 2 + MARGIN) & (X[:, 2] > 0.1) & \
                  (X[:, 2] < H + 0.3)
            pts.setdefault(l["track"], []).append(np.c_[x[sel], y[sel], X[sel, 2]])
        for o in json.load(open(os.path.join(rx.OUT, "cuboids", f + ".json"))):
            if not ls:
                continue
            c = np.array(o["centre"][:2])
            dist, l = min(((np.linalg.norm(l["centre"] - c), l) for l in ls), key=lambda t: t[0])
            if dist <= GATE:
                ours.setdefault(l["track"], []).append(o["lwh"])
        print(f, len(ls), "labelled cars", flush=True)
    rows = []
    for tr, sizes in ours.items():
        if len(sizes) < MIN_FRAMES:
            continue
        lab = next(l for k in labs for l in labs[k] if l["track"] == tr)
        lo = np.median(sizes, axis=0)
        X = max(lo[0], lab["lwh"][0]) / 2 + 0.3
        Y = max(lo[1], lab["lwh"][1]) / 2 + 0.3
        ref = reference(np.vstack(pts[tr]), X, Y)
        if ref is None:
            continue
        refc = close_gaps(ref)
        Bo, Bl = box_field(*lo, X, Y), box_field(*lab["lwh"], X, Y)
        rows.append({"track": tr, "type": lab["type"], "frames": len(sizes), "points": int(len(np.vstack(pts[tr]))),
                     "range_m": float(np.linalg.norm(lab["centre"])), "ours_lwh": lo.round(2).tolist(),
                     "label_lwh": np.round(lab["lwh"], 2).tolist(),
                     "ours raw": solid_iou(Bo, ref), "box raw": solid_iou(Bl, ref),
                     "ours closed": solid_iou(Bo, refc), "box closed": solid_iou(Bl, refc),
                     "ours footprint": footprint_iou(Bo, refc), "box footprint": footprint_iou(Bl, refc),
                     "ours height err": height_err(Bo, refc), "box height err": height_err(Bl, refc),
                     "size err": np.abs(lo - np.array(lab["lwh"])).round(3).tolist()})
    json.dump(rows, open(os.path.join(rx.OUT, "lidar_eval.json"), "w"), indent=1)
    keys = ("ours raw", "box raw", "ours closed", "box closed")
    print(f"{len(rows)} cars:", {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys})
    d = np.array([r["ours closed"] - r["box closed"] for r in rows])
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    print(f"paired (closed) ours - box: {d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}]")


if __name__ == "__main__":
    main()
