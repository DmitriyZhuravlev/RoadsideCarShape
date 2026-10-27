"""
Why the masks are wrong: the label-free mask of a car stage by stage, and statistics over all labelled car-frames.

Stages (colour_flow_split.detect + ipm_track_models2.full_mask):
  1 colour difference to the median background        3 optical flow inside the blob (direction = hue, speed = value)
  2 threshold + morphology, wet-road reflection cut     4 colour regions: k-means on Lab colour + position
  5 final: velocity clusters voted per colour region, queue split, HSV shadow test
Statistics: how far the final mask of the matched detection extends beyond the labelled 2D box (bottom, top, sides),
how much of the box it misses, by distance (labels only to check).

Writes docs/mask_diag_examples.png, docs/mask_diag_stats.png and docs/mask_diag.json.
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import colour_flow_split as cfs
import ipm_track_models2 as it2
import v2x_seq_classical as vc
import v2x_seq_demo as vd
import v2x_seq_shape as vs

INK, MUTED = "#1f1f1e", "#8a8984"
EXAMPLES = (("004012", 168), ("004008", 158), ("003994", 97), ("003991", 112), ("004005", 150), ("003988", 70))


def label_overlay(img_rgb, labels, alpha=0.55, seed=0):
    """Regions blended with random colours, boundaries in white; labels -1 = none."""
    out = img_rgb.astype(np.float32).copy()
    K = int(labels.max()) + 1
    if K > 0:
        cols = np.random.default_rng(seed).integers(40, 256, (K, 3)).astype(np.float32)
        on = labels >= 0
        out[on] = alpha * cols[labels[on]] + (1 - alpha) * out[on]
    b = np.zeros(labels.shape, bool)
    dx = labels[:, 1:] != labels[:, :-1]
    dy = labels[1:] != labels[:-1]
    b[:, :-1] |= dx & ((labels[:, 1:] >= 0) | (labels[:, :-1] >= 0))
    b[:-1] |= dy & ((labels[1:] >= 0) | (labels[:-1] >= 0))
    out[b] = 255
    return out.clip(0, 255).astype(np.uint8)


def flow_rgb(F, mask):
    mag, ang = cv2.cartToPolar(F[..., 0], F[..., 1])
    hsv = np.zeros(F.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max(np.percentile(mag[mask], 95), 1e-3) * 255, 0, 255).astype(np.uint8) * mask
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def stages(seq, k, F_half, box, pad=60):
    """All stages for the blob that overlaps the labelled box `box` most, cropped around it."""
    frame, bg = seq.frames[k], seq.bg
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = np.linalg.norm(frame.astype(np.int16) - bg.astype(np.int16), axis=2)
    fg = vc.foreground(frame, bg)
    blobs = vc.blobs(fg, min_area=cfs.MIN_AREA)
    if not blobs:
        return None
    b = max(blobs, key=lambda d: cfs.box_iou(d["box"], box))
    cut = vc.cut_reflection(b, gray)
    F = cfs.full_res(F_half, frame.shape)
    lab_img = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    x0, y0 = cut["box"][:2]
    ys, xs = np.nonzero(cut["m"])
    Y, X = ys + y0, xs + x0
    reg = cfs.colour_regions(lab_img, Y, X, len(xs))
    vel, n_vel = cfs.motions(F_half[Y // 2, X // 2].astype(np.float32))
    parts = cfs.split_blob(cut, lab_img, F_half)
    finals = [it2.full_mask(seq, p, frame)["mask"] for p in parts]
    # crop window
    bx = np.array(box, float)
    X0, Y0 = int(max(0, min(b["box"][0], bx[0]) - pad)), int(max(0, min(b["box"][1], bx[1]) - pad))
    X1, Y1 = int(min(frame.shape[1], max(b["box"][2], bx[2]) + pad)), int(min(frame.shape[0], max(b["box"][3], bx[3]) + pad))
    H, W = frame.shape[:2]
    full = lambda m, o: np.pad(m, ((o[1], H - o[1] - m.shape[0]), (o[0], W - o[0] - m.shape[1])))
    blob_full = full(b["m"], b["box"][:2]) > 0
    cut_full = full(cut["m"], cut["box"][:2]) > 0
    reg_img = -np.ones((H, W), int)
    reg_img[Y, X] = reg
    vel_img = -np.ones((H, W), int)
    vel_img[Y, X] = vel
    part_img = -np.ones((H, W), int)
    for i, m in enumerate(finals):
        part_img[m & (part_img < 0)] = i
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    c = (slice(Y0, Y1), slice(X0, X1))
    return {"rgb": rgb[c], "diff": diff[c], "blob": blob_full[c], "cut": cut_full[c],
            "cut_row": cut["box"][3] - Y0 if cut["box"][3] < b["box"][3] else None,
            "flow": flow_rgb(F, cut_full)[c], "regions": label_overlay(rgb[c], reg_img[c], seed=1),
            "velocity": label_overlay(rgb[c], vel_img[c], alpha=0.6, seed=5), "n_vel": n_vel,
            "parts": label_overlay(rgb[c], part_img[c], alpha=0.5, seed=3), "box": bx - [X0, Y0, X0, Y0],
            "n_regions": int(reg.max() + 1), "n_parts": len(finals)}


def examples_figure(path, seq, Fs):
    rows = []
    for tid, k in EXAMPLES:
        o = next((p for p in seq.tracks[tid] if p["k"] == k), None)
        if o is None:
            continue
        s = stages(seq, k, Fs[k], o["box"])
        if s is not None:
            rows.append((tid, k, s))
    cols = ("frame", "colour difference to the background", "blob after threshold (reflection cut: red line)",
            "optical flow in the blob", "colour clusters (k-means Lab + position)", "velocity clusters",
            "final mask: velocity voted per colour region + shadow test")
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.3 * len(cols), 2.9 * len(rows)))
    for r, (tid, k, s) in enumerate(rows):
        x0, y0, x1, y1 = s["box"]
        imgs = (s["rgb"], s["diff"], s["blob"], s["flow"], s["regions"], s["velocity"], s["parts"])
        for c, im in enumerate(imgs):
            ax = axes[r, c]
            if c == 1:
                ax.imshow(np.clip(im, 0, 120), cmap="magma")
            elif c == 2:
                ax.imshow(s["rgb"])
                ov = np.zeros(im.shape + (4,))
                ov[im] = (0.92, 0.41, 0.2, 0.55)
                ov[s["cut"]] = (0.16, 0.47, 0.84, 0.55)
                ax.imshow(ov)
                if s["cut_row"] is not None:
                    ax.axhline(s["cut_row"], color="red", linewidth=1)
            else:
                ax.imshow(im)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="white", linestyle="--", linewidth=0.8))
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=8, loc="left", color=INK)
        axes[r, 0].set_title(f"car {tid}, frame {k}" if r else f"{cols[0]}\ncar {tid}, frame {k}", fontsize=8,
                             loc="left", color=INK)
        axes[r, 4].set_xlabel(f"{s['n_regions']} regions", fontsize=7)
        axes[r, 5].set_title(("" if r else cols[5] + "\n") + f"{s['n_vel']} motion(s)", fontsize=8, loc="left", color=INK)
        axes[r, 6].set_title(("" if r else cols[6] + "\n") + f"{s['n_parts']} object(s)", fontsize=8, loc="left",
                             color=INK)
    fig.suptitle("orange = blob, blue = kept after the reflection cut; white dashed = labelled 2D box (check only)",
                 fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor="white")
    plt.close(fig)


def statistics(seq, Fs):
    """Matched final mask vs labelled box, per moving labelled car-frame."""
    foot = -np.linalg.solve(seq.cams[0][1], seq.cams[0][2])[:2]
    rows = []
    for k in range(0, len(seq.frames), 3):
        labs = [o for q in seq.tracks.values() for o in q if o["k"] == k and seq.usable(o)]
        if not labs:
            continue
        dets = cfs.detect(seq.frames[k], seq.bg, Fs[k])
        masks = [it2.full_mask(seq, d, seq.frames[k]) for d in dets]
        for o in labs:
            if not masks:
                continue
            best = max(masks, key=lambda m: cfs.box_iou(m["box"], o["box"]))
            iou = cfs.box_iou(best["box"], o["box"])
            x0, y0, x1, y1 = o["box"]
            bx = best["box"]
            h = y1 - y0
            ys, xs = np.nonzero(best["mask"])
            inside = best["mask"][int(y0):int(y1), int(x0):int(x1)]
            rows.append({"dist": float(np.linalg.norm(o["pos"] - foot)), "iou": iou,
                         "below_px": float(bx[3] - y1), "above_px": float(y0 - bx[1]),
                         "sides_px": float(max(0, x0 - bx[0]) + max(0, bx[2] - x1)), "h_px": float(h),
                         "box_covered": float(inside.mean()) if inside.size else 0.0,
                         "outside_box": float(1 - inside.sum() / max(len(xs), 1))})
    return rows


def stats_figure(path, rows):
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    bands = ((0, 40), (40, 60), (60, 80), (80, 200))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    keys = (("below_px", "mask bottom below the box bottom (px)"), ("above_px", "mask top above the box top (px)"),
            ("outside_box", "share of mask pixels outside the box"), ("box_covered", "share of the box covered by mask"))
    for ax, (key, title) in zip(axes, keys):
        data = [R[key][(R["dist"] >= a) & (R["dist"] < b)] for a, b in bands]
        ax.boxplot(data, showfliers=False, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#2a78d6", alpha=0.6), medianprops=dict(color=INK))
        ax.set_xticks(range(1, 5), [f"{a}-{b} m\n(n={len(d)})" for (a, b), d in zip(bands, data)], fontsize=8)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def main():
    seq = vs.Sequence()
    Fs = cfs.flows(seq.frames)
    examples_figure(os.path.join(vd.OUT_DIR, "mask_diag_examples.png"), seq, Fs)
    rows = statistics(seq, Fs)
    stats_figure(os.path.join(vd.OUT_DIR, "mask_diag_stats.png"), rows)
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out = {}
    for a, b in ((0, 40), (40, 60), (60, 80), (80, 200)):
        s = (R["dist"] >= a) & (R["dist"] < b)
        if not s.any():
            continue
        out[f"{a}-{b} m"] = {k: float(np.median(R[k][s])) for k in ("iou", "below_px", "above_px", "sides_px",
                                                                     "box_covered", "outside_box", "h_px")}
        out[f"{a}-{b} m"]["n"] = int(s.sum())
        out[f"{a}-{b} m"]["merged_share"] = float(np.mean(R["outside_box"][s] > 0.4))
        out[f"{a}-{b} m"]["bottom_over_5px_share"] = float(np.mean(R["below_px"][s] > 5))
        print(f"{a:3d}-{b:3d} m (n={s.sum():4d}): box IoU {out[f'{a}-{b} m']['iou']:.2f} | bottom {out[f'{a}-{b} m']['below_px']:+.0f} px "
              f"(> 5 px in {100 * out[f'{a}-{b} m']['bottom_over_5px_share']:.0f}%) | top {out[f'{a}-{b} m']['above_px']:+.0f} px | "
              f"sides +{out[f'{a}-{b} m']['sides_px']:.0f} px | box covered {100 * out[f'{a}-{b} m']['box_covered']:.0f}% | "
              f"mask outside box {100 * out[f'{a}-{b} m']['outside_box']:.0f}% (merged-like {100 * out[f'{a}-{b} m']['merged_share']:.0f}%) | "
              f"car height {out[f'{a}-{b} m']['h_px']:.0f} px", flush=True)
    json.dump(out, open(os.path.join(vd.OUT_DIR, "mask_diag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
