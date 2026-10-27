"""
Separating touching cars by velocity and colour (no labels; V2X-Seq intersection A).

Every foreground blob (median background, wet-road reflection cut) is split as follows:
  1. velocity : dense Farneback flow inside the blob, k-means with k = 1..3 motions; a new motion is accepted only if
                it lowers the within-cluster spread by SPREAD_GAIN and lies >= MIN_SEP px/frame from the others and
                holds >= MIN_FRAC of the blob - a rigid car is one coherent motion, whatever its colours;
  2. colour   : compact colour regions (k-means on Lab colour + position, ~ one region per REGION_PX pixels); each
                region takes the motion most of its pixels have, so the cuts follow colour edges;
  3. queues   : cars at the same speed: a motion group is split again where its regions form two side-by-side groups
                of clearly different body colour (Delta E >= DE_SPLIT), each a large part of the group.
Compared with plain blobs and the flow-only split of v2x_seq_classical (2 clusters) on the labelled 2D boxes
(evaluation only): merged cars, fragmented cars, box IoU.

Writes docs/colour_flow_split.png and docs/colour_flow_split.json.
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v2x_seq_classical as vc
import v2x_seq_demo as vd
import v2x_seq_shape as vs

SPREAD_GAIN = 0.45
MIN_SEP = 1.2              # px/frame at full resolution
MIN_FRAC = 0.15
REGION_PX = 700
DE_SPLIT = 28.0
MIN_AREA = 600
INK, MUTED = "#1f1f1e", "#8a8984"


def vote_regions(region, pixel_label, n_labels):
    """Every pixel gets the majority label of its region."""
    if len(region) == 0:
        return np.zeros(0, int)
    counts = np.zeros((region.max() + 1, n_labels), int)
    np.add.at(counts, (region, pixel_label), 1)
    return counts.argmax(1)[region]


def flows(frames):
    """Farneback flow k-1 -> k at half resolution, in full-resolution px/frame, float16 (frame 0: -(0 -> 1))."""
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5) for f in frames]
    out = [None] * len(frames)
    for k in range(1, len(frames)):
        out[k] = (cv2.calcOpticalFlowFarneback(small[k - 1], small[k], None, 0.5, 3, 15, 3, 5, 1.2, 0) * 2).astype(np.float16)
    out[0] = (-cv2.calcOpticalFlowFarneback(small[1], small[0], None, 0.5, 3, 15, 3, 5, 1.2, 0) * 2).astype(np.float16)
    return out


def full_res(F, shape):
    return cv2.resize(F.astype(np.float32), (shape[1], shape[0]))


def motions(V, rng_seed=0):
    """k-means on the flow vectors with k = 1..3, the smallest k that explains the motion."""
    cv2.setRNGSeed(rng_seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.05)
    best_lab, best_c = np.zeros(len(V), int), V.mean(0, keepdims=True)
    spread = ((V - best_c) ** 2).sum()
    for k in (2, 3):
        comp, lab, cen = cv2.kmeans(V.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lab = lab.ravel()
        fr = np.bincount(lab, minlength=k) / len(lab)
        sep = min(np.linalg.norm(cen[a] - cen[b]) for a in range(k) for b in range(a + 1, k))
        if comp < (1 - SPREAD_GAIN) * spread and sep >= MIN_SEP and fr.min() >= MIN_FRAC:
            best_lab, best_c, spread = lab, cen, comp
        else:
            break
    return best_lab, len(best_c)


def colour_regions(lab_img, ys, xs, n_pix):
    k = int(np.clip(n_pix / REGION_PX, 2, 60))
    s = np.sqrt(n_pix / k)                                   # region size: position weighted like ~10 Lab units
    feats = np.c_[lab_img[ys, xs], 10 * xs / s, 10 * ys / s].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, reg, _ = cv2.kmeans(feats, k, None, crit, 1, cv2.KMEANS_PP_CENTERS)
    return reg.ravel()


def split_queue(lab_img, ys, xs, reg):
    """Two side-by-side groups of regions with different body colour -> two cars."""
    R = reg.max() + 1
    cnt = np.bincount(reg, minlength=R)
    if len(xs) < 4000 or R < 4:
        return None
    col = np.stack([np.bincount(reg, lab_img[ys, xs][:, c], minlength=R) for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    cx = np.bincount(reg, xs, minlength=R) / np.maximum(cnt, 1)
    big = cnt > 0.04 * len(xs)
    if big.sum() < 4:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, g, cen = cv2.kmeans(col[big].astype(np.float32), 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    g = g.ravel()
    if np.linalg.norm(cen[0] - cen[1]) < DE_SPLIT:
        return None
    ids = np.flatnonzero(big)
    area = np.array([cnt[ids[g == j]].sum() for j in (0, 1)])
    if area.min() < 0.3 * len(xs):
        return None
    x_mean = [np.average(cx[ids[g == j]], weights=cnt[ids[g == j]]) for j in (0, 1)]
    x_spread = np.sqrt(np.average((cx[ids] - np.average(cx[ids], weights=cnt[ids])) ** 2, weights=cnt[ids]))
    if abs(x_mean[0] - x_mean[1]) < 1.2 * x_spread:              # interleaved colours: windows / body, not two cars
        return None
    # every region (also the small ones) joins the side whose mean x it is closer to
    return (np.abs(cx - x_mean[1]) < np.abs(cx - x_mean[0])).astype(int)[reg]


def split_blob(b, frame_lab, F):
    x0, y0 = b["box"][:2]
    ys, xs = np.nonzero(b["m"])
    if len(xs) < 2 * MIN_AREA:
        return [b]
    Y, X = ys + y0, xs + x0
    lab, n = motions(F[Y // 2, X // 2].astype(np.float32))
    reg = colour_regions(frame_lab, Y, X, len(xs))
    lab = vote_regions(reg, lab, n) if n > 1 else lab
    parts = []
    for c in range(n):
        sel = lab == c
        sub = None
        if sel.sum() > 4000:
            sub = split_queue(frame_lab, Y[sel], X[sel], np.unique(reg[sel], return_inverse=True)[1].ravel())
        groups = [sel] if sub is None else [np.flatnonzero(sel)[sub == j] for j in (0, 1)]
        for gsel in groups:
            m = np.zeros_like(b["m"])
            m[ys[gsel], xs[gsel]] = 1
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            nn, cc, st, _ = cv2.connectedComponentsWithStats(m)
            if nn > 1:
                m = (cc == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
            p = vc.make_blob(m, x0, y0, min_area=MIN_AREA)
            if p:
                parts.append(p)
    return parts or [b]


def detect(frame, bg, F):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab_img = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    out = []
    for b in vc.blobs(vc.foreground(frame, bg), min_area=MIN_AREA):
        b = vc.cut_reflection(b, gray)
        out += split_blob(b, lab_img, F)
    return out


def detect_flow_only(frame, bg, F):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Ff = full_res(F, frame.shape)
    out = []
    for b in vc.blobs(vc.foreground(frame, bg), min_area=MIN_AREA):
        out += vc.flow_split(vc.cut_reflection(b, gray), Ff)
    return out


def detect_plain(frame, bg, F):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [vc.cut_reflection(b, gray) for b in vc.blobs(vc.foreground(frame, bg), min_area=MIN_AREA)]


# --- evaluation on the labelled 2D boxes (evaluation only) -------------------------------------------------
def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def score(seq, dets_per_frame):
    """Per labelled moving car-frame: is it covered by a detection of its own (IoU >= 0.5), merged into a
    detection that also covers another car's centre, or split into several detections?"""
    own, merged, split, ious = 0, 0, 0, []
    n = 0
    for tid, q in seq.tracks.items():
        P = np.array([o["pos"] for o in q])
        if len(q) < 5 or np.linalg.norm(P[-1] - P[0]) < 5:
            continue
        for o in q:
            if o["truncated"] if "truncated" in o else False:
                continue
            dets = dets_per_frame[o["k"]]
            if not dets:
                continue
            n += 1
            lb = o["box"]
            iou = [box_iou(d["box"], lb) for d in dets]
            j = int(np.argmax(iou))
            ious.append(iou[j])
            if iou[j] >= 0.5:
                own += 1
            others = [p for t2, q2 in seq.tracks.items() if t2 != tid for p in q2 if p["k"] == o["k"]]
            d = dets[j]["box"]
            if any(d[0] <= (p["box"][0] + p["box"][2]) / 2 <= d[2] and d[1] <= (p["box"][1] + p["box"][3]) / 2 <= d[3]
                   for p in others) and iou[j] < 0.5:
                merged += 1
            inside = [dd for dd in dets if box_iou(dd["box"], lb) > 0.1 and
                      (dd["box"][2] - dd["box"][0]) * (dd["box"][3] - dd["box"][1]) < 0.6 * (lb[2] - lb[0]) * (lb[3] - lb[1])]
            if len(inside) >= 2:
                split += 1
    return {"car_frames": n, "own_detection": own / n, "merged": merged / n, "fragmented": split / n,
            "median_iou": float(np.median(ious))}


def figure(path, seq, Fs, ks):
    fig, axes = plt.subplots(len(ks), 3, figsize=(20, 4.6 * len(ks)))
    cmap = plt.cm.tab10
    for r, k in enumerate(ks):
        for c, (name, fn) in enumerate((("plain blobs", detect_plain), ("flow only (2 clusters)", detect_flow_only),
                                        ("velocity + colour", detect))):
            ax = axes[r, c]
            img = cv2.cvtColor(seq.frames[k], cv2.COLOR_BGR2RGB).copy()
            dets = fn(seq.frames[k], seq.bg, Fs[k])
            over = img.copy()
            for i, d in enumerate(dets):
                x0, y0, x1, y1 = d["box"]
                col = (np.array(cmap(i % 10)[:3]) * 255).astype(np.uint8)
                region = over[y0:y1, x0:x1]
                region[d["m"] > 0] = col
            img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
            ax.imshow(img)
            for i, d in enumerate(dets):
                x0, y0, x1, y1 = d["box"]
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=cmap(i % 10), linewidth=1.2))
            for o in [p for q in seq.tracks.values() for p in q if p["k"] == k]:
                x0, y0, x1, y1 = o["box"]
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="white", linewidth=0.8,
                                           linestyle="--"))
            ax.set_xlim(300, 1920)
            ax.set_ylim(1080, 250)
            ax.axis("off")
            ax.set_title(f"frame {k}: {name}, {len(dets)} objects (white dashed = labels, check only)", fontsize=9.5,
                         loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=70, facecolor="white")
    plt.close(fig)


def main():
    seq = vs.Sequence()
    Fs = flows(seq.frames)
    res = {}
    for name, fn in (("plain blobs", detect_plain), ("flow only (2 clusters)", detect_flow_only),
                     ("velocity + colour", detect)):
        dets = [fn(f, seq.bg, F) for f, F in zip(seq.frames, Fs)]
        res[name] = score(seq, dets)
        r = res[name]
        print(f"{name:24s}: own detection {r['own_detection']:.0%}, merged {r['merged']:.0%}, fragmented "
              f"{r['fragmented']:.0%}, median box IoU {r['median_iou']:.2f} ({r['car_frames']} car-frames)", flush=True)
    json.dump(res, open(os.path.join(vd.OUT_DIR, "colour_flow_split.json"), "w"), indent=1)
    figure(os.path.join(vd.OUT_DIR, "colour_flow_split.png"), seq, Fs, (60, 150, 200))


if __name__ == "__main__":
    main()
