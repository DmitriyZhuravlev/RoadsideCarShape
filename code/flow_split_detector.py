"""
Classical car detector with merged cars split by dense optical-flow clusters.

Base detector (v2x_seq_classical): median background -> motion mask -> blobs -> contact-line cut of the reflection.
Merged blobs: cars that touch in the image share one blob. Their optical flow still differs: cars in other lanes or at
other speeds move differently, and in a queue the farther car has a smaller image motion that points more towards the
vanishing point. Split (per blob, Farneback flow between frame k and k+1 at half resolution):
  features per pixel: flow normalised by the blob's median flow magnitude (so the rule does not depend on distance),
  plus the pixel position scaled by the blob size times SPATIAL (keeps the parts compact);
  k-means for k = 2..4; a split is kept if every part holds >= MIN_PART of the blob and the closest pair of clusters
  differs in normalised flow by >= MIN_SEPARATION; the largest such k wins. Each part -> its connected regions -> boxes,
  then the contact-line cut again.
Compared with no split and with the previous 2-cluster split (v2x_seq_classical.flow_split), against the labelled 2D
boxes of moving, untruncated cars: found (IoU >= 0.5), merged (the best detection also covers another car).

Writes docs/flow_split_examples.png and docs/flow_split_scores.png.
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v2x_coop_eval as ce
import v2x_seq_classical as vc
import v2x_seq_demo as vd

SPATIAL = 0.6            # weight of the (size-normalised) pixel position against the normalised flow
MIN_PART = 0.12          # smallest part, as a share of the blob
MIN_SEPARATION = 0.25    # smallest difference of normalised flow between two parts
MIN_AREA = 600           # blobs smaller than this are not split
INK, MUTED = "#1f1f1e", "#8a8984"
MODES = ("no split", "2-cluster flow split (before)", "flow clusters + position (new)")
COLORS = ("#c3c2b7", "#8a8984", "#2a78d6")


def boxes_iou_matrix(A, B):
    """(n, m) IoU of boxes (x0, y0, x1, y1); zeros when either list is empty."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    A, B = np.asarray(A, float), np.asarray(B, float)
    x0 = np.maximum(A[:, None, 0], B[None, :, 0])
    y0 = np.maximum(A[:, None, 1], B[None, :, 1])
    x1 = np.minimum(A[:, None, 2], B[None, :, 2])
    y1 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    union = area(A)[:, None] + area(B)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def half_res_flow(prev_gray, gray):
    small = lambda g: cv2.resize(g, None, fx=0.5, fy=0.5)
    f = cv2.calcOpticalFlowFarneback(small(prev_gray), small(gray), None, 0.5, 4, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (gray.shape[1], gray.shape[0])) * 2.0


def split_by_flow_clusters(b, flow, seed=0):
    """Parts of a blob with different (normalised) optical flow; [b] if no split is convincing.
    Returns (parts, labels_image or None) - the label image (blob-sized, -1 outside) for illustration."""
    if b["area"] < MIN_AREA:
        return [b], None
    ys, xs = np.nonzero(b["m"])
    x0, y0 = b["box"][:2]
    F = flow[ys + y0, xs + x0].astype(np.float32)
    scale = max(float(np.median(np.linalg.norm(F, axis=1))), 0.5)
    h, w = b["m"].shape
    feats = np.c_[F / scale, SPATIAL * xs / max(w, h), SPATIAL * ys / max(w, h)].astype(np.float32)
    sample = np.random.default_rng(seed).permutation(len(feats))[:4000]
    best = None
    for k in (2, 3, 4):
        cv2.setRNGSeed(seed)
        _, lab, centres = cv2.kmeans(feats[sample], k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.05),
                                     3, cv2.KMEANS_PP_CENTERS)
        share = np.bincount(lab.ravel(), minlength=k) / len(sample)
        sep = min(np.linalg.norm(centres[i, :2] - centres[j, :2]) for i in range(k) for j in range(i + 1, k))
        if share.min() >= MIN_PART and sep >= MIN_SEPARATION:
            best = centres
    if best is None:
        return [b], None
    labels = np.argmin(((feats[:, None, :] - best[None]) ** 2).sum(2), axis=1)
    img = np.full((h, w), -1, int)
    img[ys, xs] = labels
    parts = []
    for c in range(len(best)):
        m = (img == c).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        for j in range(1, n):
            if stats[j, 4] >= max(150, 0.5 * MIN_PART * b["area"]):
                p = vc.make_blob((lab == j).astype(np.uint8), x0, y0)
                if p:
                    parts.append(p)
    return (parts, img) if len(parts) >= 2 else ([b], None)


def detect(frame, gray, prev_gray, bg, mode):
    """Boxes of one frame for a mode in MODES (the flow needs the previous frame)."""
    blobs = vc.blobs(vc.foreground(frame, bg))
    flow = half_res_flow(prev_gray, gray) if (prev_gray is not None and mode != MODES[0]) else None
    out, info = [], []
    for b in blobs:
        if flow is None:
            parts, img = [b], None
        elif mode == MODES[1]:
            parts, img = vc.flow_split(b, flow), None
        else:
            parts, img = split_by_flow_clusters(b, flow)
        if img is not None:
            info.append((b, img))
        out += [vc.cut_reflection(p, gray)["box"] for p in parts]
    return out, info


def labelled_boxes(tracks, k, moving_only=True):
    out = []
    for q in tracks.values():
        ks = [s["k"] for s in q]
        for i, s in enumerate(q):
            if s["k"] != k or s["truncated"]:
                continue
            if moving_only:
                near = [j for j in range(len(q)) if abs(ks[j] - k) <= 3]
                if len(near) < 2 or np.linalg.norm(q[near[-1]]["pos"] - q[near[0]]["pos"]) < 0.5:
                    continue
            out.append(s["box"])
    return out


def score(dets, labels, match=0.5, touch=0.1):
    M = boxes_iou_matrix(labels, dets)
    found = int((M.max(1) >= match).sum()) if M.size else 0
    merged = 0
    for i in range(len(labels)):
        if M.shape[1] == 0:
            break
        j = int(np.argmax(M[i]))
        if M[i, j] > 0 and np.sum(M[:, j] >= touch) >= 2:
            merged += 1
    return len(labels), found, merged, len(dets)


def evaluate(name, loader):
    images, times, cams, tracks, camera_xy = loader()
    frames = [cv2.imread(p) for p in images]
    bg = np.median(np.stack(frames[::2]), axis=0).astype(np.uint8)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    totals = {m: np.zeros(4, int) for m in MODES}
    examples = []
    for k in range(1, len(frames)):
        labels = labelled_boxes(tracks, k)
        for m in MODES:
            dets, info = detect(frames[k], grays[k], grays[k - 1], bg, m)
            totals[m] += score(dets, labels)
            if m == MODES[2] and info and len(examples) < 6 and k % 7 == 0:
                examples.append((name, images[k], info, dets, labels))
    return totals, examples


def examples_figure(path, examples):
    picks = examples[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    palette = np.array([[42, 120, 214], [235, 104, 52], [46, 158, 107], [138, 92, 194]])
    for ax, (name, path_img, info, dets, labels) in zip(axes.ravel(), picks):
        img = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
        b, lab = max(info, key=lambda x: x[0]["area"])
        x0, y0, x1, y1 = b["box"]
        crop = img[y0:y1, x0:x1].copy()
        over = crop.copy()
        for c in range(lab.max() + 1):
            over[lab == c] = palette[c % 4]
        crop = (0.55 * crop + 0.45 * over).astype(np.uint8)
        pad = 30
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        X1, Y1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
        canvas = img[Y0:Y1, X0:X1].copy()
        canvas[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = crop
        ax.imshow(canvas)
        for bx in labels:
            if bx[2] > X0 and bx[0] < X1 and bx[3] > Y0 and bx[1] < Y1:
                ax.add_patch(plt.Rectangle((bx[0] - X0, bx[1] - Y0), bx[2] - bx[0], bx[3] - bx[1], fill=False,
                                           edgecolor="white", linestyle=(0, (3, 2)), linewidth=1.2))
        for bx in dets:
            if bx[2] > X0 and bx[0] < X1 and bx[3] > Y0 and bx[1] < Y1:
                ax.add_patch(plt.Rectangle((bx[0] - X0, bx[1] - Y0), bx[2] - bx[0], bx[3] - bx[1], fill=False,
                                           edgecolor="#f2c94c", linewidth=1.6))
        ax.set_title(f"{name}: merged blob split into {lab.max() + 1} flow clusters", fontsize=9.5, loc="left",
                     color=INK)
        ax.axis("off")
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle("Colours = flow clusters inside one motion blob; yellow = resulting detections; dashed = labelled cars",
                 fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=95, facecolor="white")
    plt.close(fig)


def scores_figure(path, results):
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = (("found (IoU ≥ 0.5) %", lambda t: 100 * t[1] / max(t[0], 1)),
              ("merged with another car %", lambda t: 100 * t[2] / max(t[0], 1)),
              ("detections per labelled moving car", lambda t: t[3] / max(t[0], 1)))
    for ax, (title, f) in zip(axes, panels):
        for j, (m, color) in enumerate(zip(MODES, COLORS)):
            v = [f(results[n][m]) for n in names]
            bars = ax.bar(np.arange(len(names)) + (j - 1) * 0.27, v, 0.25, color=color, zorder=2, label=m)
            for b, x in zip(bars, v):
                ax.annotate(f"{x:.0f}" if "%" in title else f"{x:.2f}", (b.get_x() + b.get_width() / 2, x),
                            xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(names)), names)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        vd.style(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    results, examples = {}, []
    for name, loader in (("A: yizhuang09", ce.load_spd), ("B: yizhuang08", ce.load_coop)):
        totals, ex = evaluate(name, loader)
        results[name] = totals
        examples += ex
        for m, t in totals.items():
            print(f"{name:14s} {m:32s} labelled moving cars {t[0]}, found {t[1] / max(t[0], 1):.0%}, merged "
                  f"{t[2] / max(t[0], 1):.0%}, detections {t[3]}", flush=True)
    scores_figure(os.path.join(vd.OUT_DIR, "flow_split_scores.png"), results)
    if examples:
        examples_figure(os.path.join(vd.OUT_DIR, "flow_split_examples.png"), examples)


if __name__ == "__main__":
    main()
