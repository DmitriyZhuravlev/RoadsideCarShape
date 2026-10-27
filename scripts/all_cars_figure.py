"""Supplementary figures from the v7 models of intersection A (figures/models_A.pkl):
  all_cars.png  - every car with a LiDAR check, from the same 3/4 view, sorted by IoU;
  turntable.png - two cars from 8 directions around them (the far side is the mirror image of the seen one).
No video frames are needed."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cars_figure as cf  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

KN = np.array([[700.0, 0, 200], [0, 700.0, 150], [0, 0, 1]])
TARGET = np.array([0.0, 0.0, 0.7])


def ring_eyes(n, radius, height, start_deg=0.0):
    """Camera positions on a horizontal circle around the origin, looking at a car standing at the origin."""
    angles = np.radians(np.linspace(start_deg, start_deg + 360, n, endpoint=False))
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.full(n, height)))


def view(m, eye, size=(400, 300)):
    """The model from `eye`, image centre in the middle of the tile, focal length scaled with the tile width."""
    w, h = size
    K = np.array([[1.75 * w, 0, w / 2], [0, 1.75 * w, h / 2], [0, 0, 1]])
    img = np.full((h, w, 3), 255, np.uint8)
    return cf.render_mesh(img, cf.look_at(K, eye, TARGET), m["V"], m["F"], m["vcol"])


def main():
    models = pickle.load(open(os.path.join(cf.FIG, "models_A.pkl"), "rb"))
    good = sorted([m for m in models.values() if m["vcol"] is not None and "fused closed: iou" in m["row"]],
                  key=lambda m: -m["row"]["fused closed: iou"])
    eye = np.array([6.5, 7.0, 3.8])                                   # front-left, from above
    tiles = []
    for m in good:
        r = m["row"]
        t = f"{r['label']}: IoU {r['fused closed: iou']:.2f}, L {r['length']:.1f}/{r['label_length']:.1f} m"
        tiles.append(cf.panel(view(m, eye), t, size=(400, 300)))
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(os.path.join(cf.FIG, "all_cars.png"), grid)
    print("all cars:", len(good))
    labels = sys.argv[1:] or ["004010", "003890"]
    rows = []
    for lab in labels:
        m = next(m for m in good if m["row"]["label"] == lab)
        eyes = ring_eyes(8, 9.5, 3.2, start_deg=22.5)
        rows.append(np.hstack([cf.panel(view(m, e, size=(320, 240)),
                                        f"{lab}, {np.degrees(np.arctan2(e[1], e[0])):.0f} deg", size=(320, 240))
                               for e in eyes]))
    cv2.imwrite(os.path.join(cf.FIG, "turntable.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
