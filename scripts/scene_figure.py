"""Overview figure: one frame of intersection A, and the same frame with every car model of the video drawn at its
pose in that frame. Needs figures/models_A.pkl (scripts/capture_models.py)."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cars_figure as cf  # noqa: E402  (also switches to the 3dseg folder)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import v2x_seq_shape as vs  # noqa: E402
import whole_scene_cars as w  # noqa: E402


def main():
    models = pickle.load(open(os.path.join(cf.FIG, "models_A.pkl"), "rb"))
    seq = vs.Sequence()
    good = {n: m for n, m in models.items() if m["vcol"] is not None and m["row"].get("fused closed: iou", 0) >= 0.55
            and m["row"]["range_m"] >= 40}                 # the near tracks are the weak case (Sect. 6)
    def fitting(k):                                          # models drawn in frame k: they fit the car's silhouette
        out = []
        for n, m in good.items():
            uv = next((u for j, u in m["windows"] if j == k), None)
            if k in m["poses"] and uv is not None and len(uv) and cf.image_fit(seq, m, k, uv)[0] >= 0.7:
                out.append(n)
        return out

    frames = sorted({k for m in good.values() for k, _ in m["windows"]})
    if len(sys.argv) > 1:
        k = int(sys.argv[1])
    else:
        k = max(frames, key=lambda j: (len(fitting(j)), -j))
    shown = fitting(k)
    img = seq.frames[k].copy()
    order = sorted(shown, key=lambda n: -np.linalg.norm(
        w.to_world({"poses": good[n]["poses"]}, k, good[n]["V"]).mean(0)[:2] - seq.camera_xy))
    for n in order:                                          # far cars first
        m = good[n]
        cf.render_mesh(img, seq.cams[k], w.to_world({"poses": m["poses"]}, k, m["V"]), m["F"], m["vcol"], alpha=0.8)
    boxes = []
    for n in shown:
        q, _ = vs.project(seq.cams[k], w.to_world({"poses": good[n]["poses"]}, k, good[n]["V"]))
        boxes.append(np.r_[q.min(0), q.max(0)])
    b = np.array(boxes)
    x0, y0 = np.maximum(b[:, :2].min(0) - [140, 110], 0).astype(int)
    x1, y1 = np.minimum(b[:, 2:].max(0) + [140, 110], [seq.shape[1], seq.shape[0]]).astype(int)
    out = np.vstack([seq.frames[k][y0:y1, x0:x1], np.full((10, x1 - x0, 3), 255, np.uint8), img[y0:y1, x0:x1]])
    cv2.imwrite(os.path.join(cf.FIG, "scene_models.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("frame", k, "models", len(shown), [good[n]["row"]["label"] for n in shown])


if __name__ == "__main__":
    main()
