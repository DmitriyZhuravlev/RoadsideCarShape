"""
Moving vehicles from a fixed roadside camera: the lifting method on V2X-Seq-SPD
(infrastructure side, example sequence 0058, 251 frames at 10 Hz, camera ~8.5 m
above an intersection).

Per labelled car/van track (2D boxes from the labels, i.e. a perfect 2D detector):
  ground plane      : RANSAC plane through the infrastructure LiDAR point cloud
  IPM quadrangle    : lifting.vp_bounding_quadrangle on that plane
  heading           : direction of motion of the box's ground point over +-0.5 s
  footprint, three variants
    3 lines         : closed-form candidates (face_candidates), the one closest to the heading
    sides only      : centre from the two side lines with heading and class size known
    heading+length  : centre and length from the three lines with the heading known
                      (lifting.solve_length, the construction of the moving-object paper)
  speed             : from the lifted centres over +-0.5 s
Compared with the 3D labels of the cars that are moving (> 1.5 m/s) and not truncated.

Writes docs/v2x_seq_examples.png and docs/v2x_seq_errors.png.
Usage: V2X_SEQ_DIR=<.../V2X-Seq-SPD-Example/infrastructure-side> python v2x_seq_demo.py
"""
import glob
import json
import os
import struct

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lifting

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("V2X_SEQ_DIR", os.path.join(REPO, "..", "..", "dataset", "spd", "V2X-Seq-SPD-Example",
                                                  "infrastructure-side"))
OUT_DIR = os.path.join(REPO, "docs")
CAR_SIZE = (4.6, 1.85, 1.5)      # length, width, height prior (m)
WINDOW = 5                       # frames on each side for heading and speed (+-0.5 s)
MIN_SPEED = 1.5                  # m/s: moving cars only
VARIANTS = ("3 lines", "sides only", "heading+length")
COLORS = {"3 lines": "#eb6834", "sides only": "#8a8984", "heading+length": "#2a78d6"}
INK, MUTED, GRID = "#1f1f1e", "#8a8984", "#ecebe6"
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


# --- point cloud and ground plane -------------------------------------------------------------
def lzf_decompress(data: bytes, out_len: int) -> bytes:
    """Decompress LZF data (PCD 'binary_compressed')."""
    out = bytearray()
    i = 0
    while i < len(data):
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            out.extend(data[i:i + ctrl + 1])
            i += ctrl + 1
        else:
            length = ctrl >> 5
            ref = len(out) - ((ctrl & 0x1f) << 8) - 1
            if length == 7:
                length += data[i]
                i += 1
            ref -= data[i]
            i += 1
            for k in range(length + 2):  # byte by byte: source and destination may overlap
                out.append(out[ref + k])
    if len(out) != out_len:
        raise ValueError(f"LZF: expected {out_len} bytes, got {len(out)}")
    return bytes(out)


def read_pcd(path):
    """x, y, z of a PCD v0.7 file with DATA binary_compressed (field-major; fields may differ in size)."""
    raw = open(path, "rb").read()
    tag = b"DATA binary_compressed\n"
    start = raw.index(tag) + len(tag)
    lines = raw[:start].decode().splitlines()
    n = int(next(line for line in lines if line.startswith("POINTS")).split()[1])
    fields = next(line for line in lines if line.startswith("FIELDS")).split()[1:]
    size = next((line for line in lines if line.startswith("SIZE")), None)
    sizes = [int(v) for v in size.split()[1:]] if size else [4] * len(fields)
    offset = np.r_[0, np.cumsum(np.array(sizes) * n)]
    csize, usize = struct.unpack("<II", raw[start:start + 8])
    buf = lzf_decompress(raw[start + 8:start + 8 + csize], usize)
    return np.stack([np.frombuffer(buf, np.float32, count=n, offset=int(offset[fields.index(a)])) for a in "xyz"], 1)


def fit_ground_plane(points, iterations=3000, tolerance=0.08, seed=0):
    """RANSAC for the near-horizontal plane z = a x + b y + c with most support."""
    rng = np.random.default_rng(seed)
    cand = points[(points[:, 2] < 0) & (points[:, 2] > -4) & (np.hypot(points[:, 0], points[:, 1]) < 90)]
    best = None
    for _ in range(iterations):
        a, b, c = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6 or abs(n[2]) / np.linalg.norm(n) < 0.95:
            continue
        inliers = np.abs((cand - a) @ (n / np.linalg.norm(n))) < tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers
    q = cand[best]
    coef, *_ = np.linalg.lstsq(np.c_[q[:, 0], q[:, 1], np.ones(len(q))], q[:, 2], rcond=None)
    return coef


# --- sequence --------------------------------------------------------------------------------
def load_sequence():
    info = sorted(json.load(open(os.path.join(DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    fids = [d["frame_id"] for d in info]
    times = np.array([int(d["image_timestamp"]) for d in info]) / 1e6
    plane = fit_ground_plane(read_pcd(os.path.join(DATA, info[0]["pointcloud_path"])))
    print(f"ground plane from the point cloud: z = {plane[0]:+.4f} x {plane[1]:+.4f} y {plane[2]:+.3f} "
          f"(tilt {np.degrees(np.arctan(np.hypot(*plane[:2]))):.2f} deg)")
    ground = plane[2]
    cams = []
    for f in fids:
        K = np.array(json.load(open(os.path.join(DATA, "calib", "camera_intrinsic", f"{f}.json")))["cam_K"]).reshape(3, 3)
        ex = json.load(open(os.path.join(DATA, "calib", "virtuallidar_to_camera", f"{f}.json")))
        R, t = np.array(ex["rotation"]), np.array(ex["translation"]).ravel()
        cams.append((K, R, t + ground * R[:, 2]))  # world = LiDAR frame shifted so that the ground is Z = 0
    tracks = {}
    for k, f in enumerate(fids):
        for o in json.load(open(os.path.join(DATA, "label", "camera", f"{f}.json"))):
            if o["type"] not in ("Car", "Van"):
                continue
            tracks.setdefault(o["track_id"], []).append({
                "k": k, "pos": np.array([o["3d_location"]["x"], o["3d_location"]["y"]]), "yaw": o["rotation"],
                "dims": (o["3d_dimensions"]["l"], o["3d_dimensions"]["w"], o["3d_dimensions"]["h"]),
                "z": o["3d_location"]["z"] - ground, "truncated": o["truncated_state"],
                "box": [o["2d_box"][x] for x in ("xmin", "ymin", "xmax", "ymax")]})
    camera_xy = (-np.linalg.inv(cams[0][1]) @ cams[0][2])[:2]
    return fids, times, cams, tracks, camera_xy


# --- lifting ---------------------------------------------------------------------------------
def solve_sides(q, yaw, l, w):
    """Centre from the side lines A-R and B-T only, with yaw and size known."""
    A, R, T, B = q
    mid = (A + B) / 2
    inside = mid + 0.3 * ((R + T) / 2 - mid) / np.linalg.norm((R + T) / 2 - mid)
    u, rows, rhs = np.array([np.cos(yaw), np.sin(yaw)]), [], []
    v = np.array([-u[1], u[0]])
    for p1, p2 in ((A, R), (B, T)):
        n = np.array([-(p2 - p1)[1], (p2 - p1)[0]])
        n /= np.linalg.norm(n)
        d = -n @ p1
        if n @ inside + d < 0:
            n, d = -n, -d
        rows.append(n)
        rhs.append(-d + 0.5 * l * abs(n @ u) + 0.5 * w * abs(n @ v))
    try:
        return np.linalg.solve(np.array(rows), np.array(rhs))
    except np.linalg.LinAlgError:
        return None


def yaw_error(a, b, period=np.pi):
    d = (a - b) % period
    return np.degrees(min(d, period - d))


def fit_velocity(t, p):
    return np.array([np.polyfit(t, p[:, 0], 1)[0], np.polyfit(t, p[:, 1], 1)[0]])


def lift_track(seq, times, cams, variant):
    """Per frame of one track: (index in seq, centre, yaw, footprint) or None."""
    l, w, _ = CAR_SIZE
    quads, refs = [], []
    for s in seq:
        K, R, t = cams[s["k"]]
        x0, y0, x1, y1 = s["box"]
        q = lifting.vp_bounding_quadrangle(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                                           lifting.plane_homography(K, R, t, 0.0), lifting.vertical_vanishing_point(K, R))
        quads.append(q)
        refs.append(None if q is None else (q[0] + q[3]) / 2)  # midpoint of the bottom side on the ground
    ks = np.array([s["k"] for s in seq])
    out = []
    for i, s in enumerate(seq):
        near = [j for j in np.where(np.abs(ks - s["k"]) <= WINDOW)[0] if refs[j] is not None]
        if len(near) < 4 or quads[i] is None:
            out.append(None)
            continue
        vel = fit_velocity(times[ks[near]], np.array([refs[j] for j in near]))
        heading, q = np.arctan2(vel[1], vel[0]), quads[i]
        center, yaw, length = None, heading, l
        if variant == "sides only":
            center = solve_sides(q, heading, l, w)
        if variant == "heading+length":
            solved = lifting.solve_length(q, heading, w)
            if solved is not None and 0.6 * l < solved[1] < 1.6 * l:
                center, length = solved
        if variant == "3 lines" or (variant == "heading+length" and center is None):
            cands = lifting.face_candidates(q, l, w)
            if cands:
                best = min(cands, key=lambda c: yaw_error(c[0], heading))
                center = best[1]
                yaw = best[0] if np.cos(best[0] - heading) > 0 else best[0] + np.pi
        if center is None:
            out.append(None)
            continue
        out.append((i, center, yaw, lifting.compute_rectangle_points(center, yaw, w, length)[0]))
    return out


def evaluate(times, cams, tracks, camera_xy):
    rows = {v: [] for v in VARIANTS}
    estimates = {v: {} for v in VARIANTS}
    for tid, seq in tracks.items():
        ks = np.array([s["k"] for s in seq])
        P = np.array([s["pos"] for s in seq])
        for variant in VARIANTS:
            est = [e for e in lift_track(seq, times, cams, variant) if e is not None]
            estimates[variant][tid] = est
            for i, center, yaw, _ in est:
                s = seq[i]
                w = np.abs(ks - s["k"]) <= WINDOW
                gt_vel = fit_velocity(times[ks[w]], P[w])
                near = [e for e in est if abs(seq[e[0]]["k"] - s["k"]) <= WINDOW]
                if np.linalg.norm(gt_vel) < MIN_SPEED or s["truncated"] != 0 or len(near) < 4:
                    continue
                speed = np.linalg.norm(fit_velocity(times[[seq[e[0]]["k"] for e in near]], np.array([e[1] for e in near])))
                rows[variant].append((float(np.linalg.norm(center - s["pos"])), yaw_error(yaw, s["yaw"]),
                                      yaw_error(yaw, np.arctan2(gt_vel[1], gt_vel[0]), 2 * np.pi),
                                      abs(speed - np.linalg.norm(gt_vel)), np.linalg.norm(gt_vel),
                                      float(np.linalg.norm(s["pos"] - camera_xy))))
    return {v: np.array(r) for v, r in rows.items()}, estimates


# --- figures ---------------------------------------------------------------------------------
def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def project(cam, X):
    K, R, t = cam
    x = K @ (R @ np.asarray(X, float).T + t[:, None])
    return (x[:2] / x[2]).T


def examples_figure(path, fids, times, cams, tracks, estimates, camera_xy, frames=(60, 150, 230)):
    variant = "heading+length"
    fig = plt.figure(figsize=(17, 4.2 * len(frames)))
    gs = fig.add_gridspec(len(frames), 2, width_ratios=[2.4, 1])
    ax_bev = fig.add_subplot(gs[:, 1])
    for row, k in enumerate(frames):
        ax = fig.add_subplot(gs[row, 0])
        img = cv2.imread(os.path.join(DATA, "image", f"{fids[k]}.jpg"))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        n_moving = 0
        for tid, seq in tracks.items():
            est = {seq[e[0]]["k"]: e for e in estimates[variant][tid]}
            s = next((s for s in seq if s["k"] == k), None)
            if s is None or k not in est:
                continue
            ks = np.array([x["k"] for x in seq])
            w = np.abs(ks - k) <= WINDOW
            if np.linalg.norm(fit_velocity(times[ks[w]], np.array([x["pos"] for x in seq])[w])) < MIN_SPEED:
                continue
            n_moving += 1
            l, wd, h = s["dims"]
            gt_fp = lifting.compute_rectangle_points(s["pos"], s["yaw"], wd, l)[0]
            gt = project(cams[k], np.vstack([np.c_[gt_fp, np.full(4, s["z"] - h / 2)], np.c_[gt_fp, np.full(4, s["z"] + h / 2)]]))
            for a, b in EDGES:
                ax.plot(*gt[[a, b]].T, color="white", linewidth=0.9, linestyle=(0, (3, 2)))
            _, center, yaw, fp = est[k]
            uv = project(cams[k], np.vstack([np.c_[fp, np.zeros(4)], np.c_[fp, np.full(4, CAR_SIZE[2])]]))
            for a, b in EDGES:
                ax.plot(*uv[[a, b]].T, color=COLORS[variant], linewidth=1.6)
            arrow = project(cams[k], [[*center, 0], [*(center + 3.0 * np.array([np.cos(yaw), np.sin(yaw)])), 0]])
            ax.annotate("", xy=arrow[1], xytext=arrow[0], arrowprops=dict(arrowstyle="-|>", color="#ffd400", lw=1.6))
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(780, 180)
        ax.axis("off")
        ax.set_title(f"frame {fids[k]} (t = {times[k] - times[0]:.1f} s): {n_moving} moving cars/vans; "
                     "blue = estimate, white dashed = label, yellow = estimated heading",
                     color=INK, fontsize=10, loc="left")
    for tid, seq in tracks.items():
        P = np.array([s["pos"] for s in seq])
        if np.linalg.norm(P[-1] - P[0]) < 5:
            continue
        ax_bev.plot(P[:, 1], P[:, 0], color="#c3c2b7", linewidth=3, solid_capstyle="round", zorder=1)
        E = np.array([e[1] for e in estimates[variant][tid]])
        if len(E):
            ax_bev.plot(E[:, 1], E[:, 0], ".", color=COLORS[variant], markersize=2.5, zorder=2)
    ax_bev.plot(camera_xy[1], camera_xy[0], marker="^", color=INK, markersize=9)
    ax_bev.annotate("camera", (camera_xy[1], camera_xy[0]), xytext=(6, -12), textcoords="offset points", fontsize=9)
    ax_bev.invert_xaxis()
    ax_bev.set_aspect("equal")
    ax_bev.set_xlabel("lateral (m)", color=INK)
    ax_bev.set_ylabel("forward (m)", color=INK)
    ax_bev.set_title("Trajectories of moving cars, 25 s\ngrey = labels, blue dots = estimates", color=INK,
                     fontsize=10, loc="left")
    style(ax_bev)
    fig.tight_layout()
    fig.savefig(path, dpi=95, facecolor="white")
    plt.close(fig)


def errors_figure(path, rows):
    bins = [(0, 40), (40, 60), (60, 80), (80, 200)]
    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    xs = np.arange(len(bins))
    for k, v in enumerate(VARIANTS):
        r = rows[v]
        med = [np.median(r[(r[:, 5] >= lo) & (r[:, 5] < hi), 0]) for lo, hi in bins]
        bars = ax_p.bar(xs + (k - 1) * 0.27, med, 0.25, color=COLORS[v], label=v, zorder=2)
        for b, m in zip(bars, med):
            ax_p.annotate(f"{m:.1f}", (b.get_x() + b.get_width() / 2, m), xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=8, color=INK)
    n = [int(((rows[VARIANTS[0]][:, 5] >= lo) & (rows[VARIANTS[0]][:, 5] < hi)).sum()) for lo, hi in bins]
    ax_p.set_xticks(xs, [f"{lo}–{hi} m\n(n={c})" if hi < 200 else f"> {lo} m\n(n={c})" for (lo, hi), c in zip(bins, n)])
    ax_p.set_ylabel("median position error (m)", color=INK)
    ax_p.set_title("Position error by distance from the camera", color=INK, fontsize=10.5, loc="left")
    ax_p.legend(loc="upper left", fontsize=9, frameon=False)
    style(ax_p)

    labels = ["orientation\n< 10°", "heading incl.\nfront/back < 20°", "speed error\n< 1 m/s", "position\n< 1 m"]
    for k, v in enumerate(VARIANTS):
        r = rows[v]
        shares = [np.mean(r[:, 1] < 10), np.mean(r[:, 2] < 20), np.mean(r[:, 3] < 1), np.mean(r[:, 0] < 1)]
        bars = ax_s.bar(np.arange(4) + (k - 1) * 0.27, np.array(shares) * 100, 0.25, color=COLORS[v], zorder=2)
        for b, m in zip(bars, shares):
            ax_s.annotate(f"{m * 100:.0f}", (b.get_x() + b.get_width() / 2, m * 100), xytext=(0, 3),
                          textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax_s.set_xticks(np.arange(4), labels)
    ax_s.set_ylim(0, 105)
    ax_s.set_ylabel("share of car-frames (%)", color=INK)
    ax_s.set_title("Accuracy shares", color=INK, fontsize=10.5, loc="left")
    style(ax_s)
    fig.suptitle(f"V2X-Seq-SPD, moving cars and vans ({len(rows[VARIANTS[0]])} car-frames, labelled 2D boxes, "
                 "heading from motion)", color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fids, times, cams, tracks, camera_xy = load_sequence()
    rows, estimates = evaluate(times, cams, tracks, camera_xy)
    for v, r in rows.items():
        print(f"{v:15s}: n={len(r)}, position median {np.median(r[:, 0]):.2f} m, orientation median {np.median(r[:, 1]):.1f} deg "
              f"(<10: {np.mean(r[:, 1] < 10):.0%}), heading<20 {np.mean(r[:, 2] < 20):.0%}, "
              f"speed error median {np.median(r[:, 3]):.2f} m/s of {np.median(r[:, 4]):.1f}")
    errors_figure(os.path.join(OUT_DIR, "v2x_seq_errors.png"), rows)
    examples_figure(os.path.join(OUT_DIR, "v2x_seq_examples.png"), fids, times, cams, tracks, estimates, camera_xy)


if __name__ == "__main__":
    main()
