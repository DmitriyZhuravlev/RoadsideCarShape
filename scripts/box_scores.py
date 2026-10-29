"""Reference boxes scored exactly like the models (for Table 1): the labelled box, and a box of our own estimated
length, width and height, each as a height field against the car's accumulated LiDAR, raw and gap-closed.

    V2X_SEQ_DIR=<sequence> python scripts/box_scores.py RESULTS_JSON      (e.g. docs/whole_scene_cars_abl_A5.json)

Prints the means over the cars of RESULTS_JSON that have a LiDAR score: IoU raw, IoU closed, scanned IoU closed
(the closed IoU over the cells the scanner actually returned; None when no car has one), footprint IoU closed.
"""
import json
import os
import sys

import numpy as np

CODE = os.path.expanduser("~/Documents/MyArticles/code/3dseg")
os.chdir(CODE)
sys.path.insert(0, CODE)
import ego_sequence as es  # noqa: E402
import lidar_masks as lm  # noqa: E402
import whole_scene_cars as w  # noqa: E402


def box(xs, ys, length, width, height):
    return np.where((np.abs(xs)[:, None] < length / 2) & (np.abs(ys)[None] < width / 2), height, 0.0)


def main():
    rows = [r for r in json.load(open(sys.argv[1])) if "fused: iou" in r]
    seq, ground = es.open_sequence()
    shapes = lm.LidarShapes(seq, ground)
    xs = np.arange(-5, 5, 0.1) + 0.05
    ys = np.arange(-1.5, 1.5, 0.1) + 0.05
    out = {"label": [], "own": []}
    for r in rows:
        Lt, lx, ly, n = w.lidar_heights(shapes, r["label"])
        Lc = w.close_lidar_holes(Lt, lx[1] - lx[0])
        for name, (L, W, H) in (("label", (r["label_length"], r["label_width"], r["label_height"])),
                                ("own", (r["length"], r["width"], r["height"]))):
            b = box(xs, ys, L, W, H)
            c = w.shape_scores(b, xs, ys, Lc, lx, ly, Lt)
            raw = w.shape_scores(b, xs, ys, Lt, lx, ly)
            out[name].append((raw["iou"], c["iou"], c.get("scanned_iou"), c["footprint_iou"]))
    res = {k: [round(float(np.mean([t[i] for t in v if t[i] is not None])), 3)
               if any(t[i] is not None for t in v) else None for i in range(4)] for k, v in out.items()}
    print(json.dumps({"cars": len(rows), **res}))


if __name__ == "__main__":
    main()
