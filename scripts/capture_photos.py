"""Run the fusion and save, per car, a photo from the video: the window where the car is largest and most side-on,
cropped around its body points (+35 %). docs/photos<OUT_SUFFIX>/track_<n>.jpg (label-free: the video only)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
os.chdir(os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
import cv2, numpy as np
import whole_scene_cars as w
grab = {}
orig_colours = w.best_view_colours
def colours(seq, f, *a, **k):
    grab["seq"] = seq
    return orig_colours(seq, f, *a, **k)
w.best_view_colours = colours
orig_export = w.export
def export(dirname, models):
    orig_export(dirname, models)
    seq = grab.get("seq")
    if seq is None:
        return
    out = os.path.join(w.vd.OUT_DIR, "photos" + w.OUT_SUFFIX); os.makedirs(out, exist_ok=True)
    H, W = seq.shape
    for n, (f, row) in models.items():
        cand = []
        for k, d in f["track"]:
            uv = d["uv"][d["body"]]
            if not len(uv) or seq.frames[k] is None:
                continue
            x0, y0 = uv.min(0); x1, y1 = uv.max(0)
            if x0 < 5 or y0 < 5 or x1 > W - 5 or y1 > H - 5:          # cut by the image border
                continue
            v = d["c"] - d["camera"][:2]
            side = abs(np.sin(np.arctan2(d["vel"][1], d["vel"][0]) - np.arctan2(v[1], v[0])))
            cand.append(((x1 - x0) * d["range"], (x1 - x0) * (y1 - y0) * (0.3 + side), k, x0, y0, x1, y1))
        if not cand:
            continue
        med = np.median([c[0] for c in cand])                          # about the body's length in metres x const
        ok = [c for c in cand if abs(c[0] / med - 1) <= 0.25] or cand   # merged with a neighbour: much wider
        best = max(ok, key=lambda c: c[1])[1:]
        _, k, x0, y0, x1, y1 = best
        # the body points miss the roof and windows: the box grows upward by half the body's width
        bx0, by0, bx1, by1 = int(x0 - 4), int(y0 - 0.6 * (y1 - y0)), int(x1 + 4), int(y1 + 6)   # roof above the body
        mx, my = 0.18 * (bx1 - bx0), 0.18 * (by1 - by0)
        X0, Y0, X1, Y1 = int(max(0, bx0 - mx)), int(max(0, by0 - my)), int(min(W, bx1 + mx)), int(min(H, by1 + my))
        img = seq.frames[k].copy()
        cv2.rectangle(img, (bx0, max(0, by0)), (bx1, by1), (40, 200, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out, f"track_{n}.jpg"), img[Y0:Y1, X0:X1])
w.export = export
w.main()
