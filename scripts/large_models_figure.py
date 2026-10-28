"""figures/large_models.png: the best car models of the larger evaluation (DAIR-V2X-C clips, V2X-Seq-SPD sequences),
N per dataset, best solid IoU against the closed LiDAR first, each from the same new viewpoint with its original pixels.

Inputs: dataset/<run>/chosen.json and <clip>/cars.json (run_subset.py), code/3dseg/docs/cars_viewer_<tag>_<clip>/meshes.json.
    python scripts/large_models_figure.py [N]
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cars_figure as cf  # noqa: E402

DATA = os.path.expanduser("~/Documents/MyArticles/dataset")
DOCS = os.path.expanduser("~/Documents/MyArticles/code/3dseg/docs")
RUNS = [("DAIR-V2X-C", "dair_c_runs", "dairc"), ("V2X-Seq-SPD", "spd_runs", "spd")]
EYE = np.array([7.8, 8.4, 4.6])                    # front-left, from above (as in the roadside view)


def models(run, tag):
    out = []
    for c in json.load(open(os.path.join(DATA, run, "chosen.json"))):
        cars = os.path.join(DATA, run, c["clip"], "cars.json")
        mesh = os.path.join(DOCS, f"cars_viewer_{tag}_{c['clip']}", "meshes.json")
        if not (os.path.exists(cars) and os.path.exists(mesh)):
            continue
        meshes = json.load(open(mesh))
        for r in json.load(open(cars)):
            m = meshes.get(f"track {r['track']}")
            if m is not None and "fused closed: iou" in r:
                out.append((r, c, m))
    return sorted(out, key=lambda t: -t[0]["fused closed: iou"])


def tile(r, c, m, name, size=(400, 300)):
    V, F = np.array(m["v"]).reshape(-1, 3), np.array(m["f"]).reshape(-1, 3)
    C = np.array(m["c"]).reshape(-1, 3)[:, ::-1] * 255
    w, h = size
    K = np.array([[1.75 * w, 0, w / 2], [0, 1.75 * w, h / 2], [0, 0, 1]])
    img = cf.render_mesh(np.full((h, w, 3), 255, np.uint8), cf.look_at(K, EYE, np.array([0.0, 0.0, 0.7])), V, F, C)
    title = (f"{name} {c['intersection'].replace('yizhuang', 'yz')} {r['body_type']}: IoU {r['fused closed: iou']:.2f},"
             f" L {r['length']:.1f}/{r['label_length']:.1f}")
    return cf.panel(img, title, size=size)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rows = []
    for name, run, tag in RUNS:
        ms = models(run, tag)[:n]
        if ms:
            rows.append(np.hstack([tile(r, c, m, name.split("-")[0] if name.startswith("DAIR") else "SPD")
                                   for r, c, m in ms]))
    w = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    cv2.imwrite(os.path.join(cf.FIG, "large_models.png"), np.vstack(rows))
    print("large_models.png:", [len(models(run, tag)) for _, run, tag in RUNS], "models")


if __name__ == "__main__":
    main()
