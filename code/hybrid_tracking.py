"""
Hybrid: initialise once, then let the closed form choose the orientation.

For a 2D box and known footprint dimensions, the inscribed-rectangle solution (Efficient3D: angle = theta +- psi)
leaves at most 2 orientations per assignment of length/width, 4 with the swap (lifting.face_candidates), each modulo
180 deg, and every candidate comes with its position. So:
  1. initialise every object once - here from the labels of its first frame (dimensions and heading); in practice any
     detector or the first seconds of motion;
  2. on every later frame: 2D box -> IPM quadrangle (vertical-VP tangent sides) -> the <= 4 candidates -> the one
     closest to the previous heading (and, optionally, whose projected cuboid best fits the motion mask); its
     position comes with it. If no rectangle fits the quadrangle, keep the heading and solve the position from the
     two side lines. "hybrid": the chosen heading, position from the three lines with a free length
     (lifting.solve_length), because the fixed-size inscribed rectangle is ill-conditioned far away.
Compared with the label-free heading from the motion of the box's ground point (v2x_seq_demo "heading+length") and
with the best candidate (closest to the labelled orientation; upper bound) on both roadside cameras.

Writes docs/hybrid_tracking.png and docs/hybrid_tracking_examples.png.
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lifting
import v2x_coop_eval as ce
import v2x_seq_classical as vc
import v2x_seq_demo as vd

METHODS = ("motion heading (label-free)", "previous angle", "previous angle + mask", "hybrid: angle + 3-line position",
           "best candidate (upper bound)")
COLORS = ("#8a8984", "#2a78d6", "#1d4f8f", "#eb6834", "#c3c2b7")
INK, MUTED = "#1f1f1e", "#8a8984"


def nearest_candidate(candidates, prev_yaw, mask_scores=None, angle_sigma_deg=10.0, mask_weight=1.0):
    """Pick the inscribed-rectangle candidate that continues the previous orientation.
    candidates: (yaw in [0, pi), center, corners); cost = (d / sigma)^2 + mask_weight * 4 * (1 - mask score), d = angle
    to prev_yaw modulo pi. Returns (index, heading, cost) with heading = yaw or yaw + pi, whichever is closer to
    prev_yaw on the full circle (wrapped to (-pi, pi]); None if there are no candidates."""
    if not candidates:
        return None
    yaw = np.array([c[0] for c in candidates])
    d = np.abs((yaw - prev_yaw + np.pi / 2) % np.pi - np.pi / 2)
    cost = (d / np.radians(angle_sigma_deg)) ** 2
    if mask_scores is not None:
        cost = cost + mask_weight * 4 * (1 - np.asarray(mask_scores))
    i = int(np.argmin(cost))
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    heading = min((yaw[i], yaw[i] + np.pi), key=lambda h: abs(wrap(h - prev_yaw)))
    return i, float(wrap(heading)), float(cost[i])


def quadrangle(cam, box):
    K, R, t = cam
    x0, y0, x1, y1 = box
    return lifting.vp_bounding_quadrangle(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                                          lifting.plane_homography(K, R, t, 0.0), lifting.vertical_vanishing_point(K, R))


def mask_iou(cam, fp, h, fg, box, pad=12):
    """IoU of the projected cuboid with the motion mask, inside the (padded) 2D box."""
    x0, y0, x1, y1 = (int(v) for v in np.round(box))
    H, W = fg.shape
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    uv = vd.project(cam, np.vstack([np.c_[fp, np.zeros(4)], np.c_[fp, np.full(4, h)]])) - [x0, y0]
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(uv.round().astype(np.int32)), 1)
    f = fg[y0:y1, x0:x1] > 0
    return float((f & (m > 0)).sum() / max((f | (m > 0)).sum(), 1))


def run_camera(name, loader):
    images, times, cams, tracks, camera_xy = loader()
    frames = [cv2.imread(p) for p in images]
    bg = np.median(np.stack(frames[::2]), axis=0).astype(np.uint8)
    fg_cache = {}

    def fg(k):
        if k not in fg_cache:
            fg_cache[k] = vc.foreground(frames[k], bg)
        return fg_cache[k]
    rows, examples = [], {}
    for tid, seq in tracks.items():
        if len(seq) < 5:
            continue
        motion = {seq[i]["k"]: (c, y) for i, c, y, _ in
                  [e for e in vd.lift_track(seq, times, cams, "heading+length") if e is not None]}
        first = seq[0]
        l, w, h = first["dims"]
        prev = {"previous angle": first["yaw"], "previous angle + mask": first["yaw"]}
        for s in seq[1:]:
            k, cam = s["k"], cams[s["k"]]
            q = quadrangle(cam, s["box"])
            cands = lifting.face_candidates(q, l, w) if q is not None else []
            est = {}
            for m in ("previous angle", "previous angle + mask"):
                scores = [mask_iou(cam, c[2], h, fg(k), s["box"]) for c in cands] if (m.endswith("mask") and cands) else None
                pick = nearest_candidate(cands, prev[m], scores)
                if pick is not None:
                    i, heading, _ = pick
                    est[m] = (cands[i][1], heading, cands[i][2])
                elif q is not None:
                    c = vd.solve_sides(q, prev[m], l, w)
                    if c is not None:
                        est[m] = (c, prev[m], lifting.compute_rectangle_points(c, prev[m], w, l)[0])
                if m in est:
                    prev[m] = est[m][1]
            if "previous angle + mask" in est and q is not None:
                c, y, fp = est["previous angle + mask"]
                solved = lifting.solve_length(q, y, w)
                if solved is not None and 0.6 * l < solved[1] < 1.6 * l:
                    c, length = solved
                    fp = lifting.compute_rectangle_points(c, y, w, length)[0]
                est["hybrid: angle + 3-line position"] = (c, y, fp)
            if cands:
                b = min(cands, key=lambda c: vd.yaw_error(c[0], s["yaw"]))
                est["best candidate (upper bound)"] = (b[1], b[0], b[2])
            if k in motion:
                c, y = motion[k]
                est["motion heading (label-free)"] = (c, y, lifting.compute_rectangle_points(c, y, w, l)[0])
            if s["truncated"]:
                continue
            gt = lifting.compute_rectangle_points(s["pos"], s["yaw"], s["dims"][1], s["dims"][0])[0]
            row = {"dist": float(np.linalg.norm(s["pos"] - camera_xy)), "k": k}
            for m, (c, y, fp) in est.items():
                row[m] = (float(np.linalg.norm(c - s["pos"])), vd.yaw_error(y, s["yaw"]), bev_iou(fp, gt))
            rows.append(row)
            examples.setdefault(k, []).append((s, cands, est))
    return rows, examples, (images, cams, camera_xy)


def bev_iou(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    return float(inter / max(cv2.contourArea(a) + cv2.contourArea(b) - inter, 1e-9))


def summary(rows):
    out = {}
    for m in METHODS:
        r = np.array([x[m] for x in rows if m in x])
        common = np.array([x[m] for x in rows if all(mm in x for mm in METHODS)])
        out[m] = {"n": len(r), "position": float(np.median(r[:, 0])), "orient10": float(np.mean(r[:, 1] < 10)),
                  "orient_med": float(np.median(r[:, 1])), "bev_iou": float(np.median(r[:, 2])),
                  "lost30": float(np.mean(r[:, 1] > 30)),
                  "position_common": float(np.median(common[:, 0])) if len(common) else np.nan}
    return out


def figure(path, summaries):
    names = list(summaries)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    panels = (("position", "median position error (m)", 1, "{:.2f}"), ("orient10", "orientation within 10° (%)", 100, "{:.0f}"),
              ("bev_iou", "median bird's-eye-view IoU", 1, "{:.2f}"))
    width = 0.16
    for ax, (key, title, scale, fmt) in zip(axes, panels):
        for j, (m, color) in enumerate(zip(METHODS, COLORS)):
            v = [summaries[n][m][key] * scale for n in names]
            bars = ax.bar(np.arange(len(names)) + (j - 2) * width, v, width * 0.92, color=color, zorder=2, label=m)
            for b, x in zip(bars, v):
                ax.annotate(fmt.format(x), (b.get_x() + b.get_width() / 2, x), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(names)), names, fontsize=9)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        vd.style(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Initialise once (labels of the first frame), then the closest of the <= 4 closed-form orientations",
                 fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def examples_figure(path, picks):
    fig = plt.figure(figsize=(18, 4.6 * len(picks)))
    for r, (name, (images, cams, camera_xy), k, items) in enumerate(picks):
        img = cv2.cvtColor(cv2.imread(images[k]), cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(len(picks), 3, 3 * r + 1, )
        ax2 = fig.add_subplot(len(picks), 3, (3 * r + 2, 3 * r + 3))
        ax.imshow(img)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.axis("off")
        for s, cands, est in items:
            gt = lifting.compute_rectangle_points(s["pos"], s["yaw"], s["dims"][1], s["dims"][0])[0]
            ax2.fill(*gt.T, facecolor="none", edgecolor=MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
            for c in cands:
                ax2.fill(*c[2].T, facecolor="none", edgecolor="#c3c2b7", linewidth=0.8)
            if "previous angle + mask" in est:
                c, y, fp = est["previous angle + mask"]
                ax2.fill(*fp.T, facecolor="none", edgecolor="#1d4f8f", linewidth=1.8)
                ax2.annotate("", xy=c + 2.5 * np.array([np.cos(y), np.sin(y)]), xytext=c,
                             arrowprops=dict(arrowstyle="->", color="#1d4f8f", linewidth=1.2))
                uv = vd.project(cams[k], np.vstack([np.c_[fp, np.zeros(4)], np.c_[fp, np.full(4, s["dims"][2])]]))
                for a, b in vd.EDGES:
                    ax.plot(*uv[[a, b]].T, color="#2a78d6", linewidth=1.4)
        ax2.plot(*camera_xy, "k^")
        ax2.set_aspect("equal")
        pts = np.array([s["pos"] for s, _, _ in items] + [camera_xy])
        lo, hi = pts.min(0) - 4, pts.max(0) + 4
        ax2.set_xlim(lo[0], hi[0])
        ax2.set_ylim(lo[1], hi[1])
        ax2.tick_params(labelsize=8, colors=MUTED)
        ax.set_title(f"{name}, frame {k}: chosen cuboids", fontsize=9.5, loc="left", color=INK)
        ax2.set_title("bird's-eye view (m): grey = the <= 4 closed-form candidates, blue = chosen (arrow: heading), "
                      "dashed = label", fontsize=9.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def main():
    summaries, picks = {}, []
    for name, loader in (("A: yizhuang09", ce.load_spd), ("B: yizhuang08 (new)", ce.load_coop)):
        rows, examples, ctx = run_camera(name, loader)
        summaries[name] = summary(rows)
        for m, s in summaries[name].items():
            print(f"{name:22s} {m:30s} n={s['n']:5d} position {s['position']:.2f} m (common frames {s['position_common']:.2f}),"
                  f" orientation median {s['orient_med']:.1f} deg, <10 {s['orient10']:.0%}, lost >30 {s['lost30']:.0%},"
                  f" BEV IoU {s['bev_iou']:.2f}")
        ks = sorted(examples, key=lambda k: -len(examples[k]))
        picks.append((name, ctx, ks[0], examples[ks[0]]))
    figure(os.path.join(vd.OUT_DIR, "hybrid_tracking.png"), summaries)
    examples_figure(os.path.join(vd.OUT_DIR, "hybrid_tracking_examples.png"), picks)


if __name__ == "__main__":
    main()
