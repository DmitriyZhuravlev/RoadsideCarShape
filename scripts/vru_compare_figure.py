"""Pedestrians and cyclists for the article, on the precise input: frame | SAM 2.1 mask | LiDAR box | the model as the
camera saw it | turned 90 deg (the row images of vru_lidar.py). Pedestrians: carve with symmetry only; cyclists: carve
bounded by the surface triangulated over the pass (tri_vru.py, OCC_OVERRIDE)."""
import os, re
import cv2, numpy as np
D = os.path.expanduser("~/Documents/MyArticles/code/3dseg/docs/vru_precise")
ROWS = [("seq_0077_000118/carve/vru_000118.jpg", "seq_0077_000118/carve.log"),
        ("seq_0059_003926/carve/vru_003926.jpg", "seq_0059_003926/carve.log"),
        ("seq_0059_003897/tri/vru_003897.jpg", "seq_0059_003897/final.log")]
# (cyclist 003765 of seq_0063 left out: its mask takes in the courier box of the scooter beside it)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "vru_compare.png")
rows = []
for img, log in ROWS:
    im = cv2.imread(os.path.join(D, img))[:, :5 * 320].copy()     # six panels of 320 px: drop the view from behind
    im[282:, :960] = 255                                            # the old caption over the first three panels
    txt = open(os.path.join(D, log)).read().splitlines()
    last = [l for l in txt if "IoU vs its LiDAR" in l][-1]
    iou = float(re.search(r"IoU vs its LiDAR ([\d.]+)", last).group(1))
    L, W, H = map(float, re.search(r"size L ([\d.]+) W ([\d.]+) H ([\d.]+)", last).groups())
    l, w, h = map(float, re.search(r"LiDAR box L ([\d.]+) W ([\d.]+) H ([\d.]+)", next(t for t in txt if "LiDAR box L" in t)).groups())
    band = np.full((28, im.shape[1], 3), 255, np.uint8)
    cv2.putText(band, f"IoU vs its LiDAR {iou:.2f}; model {L:.2f} x {W:.2f} x {H:.2f} m, LiDAR box {l:.2f} x {w:.2f} x {h:.2f} m",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    rows += [im, band, np.full((8, im.shape[1], 3), 255, np.uint8)]
cv2.imwrite(OUT, np.vstack(rows[:-1])); print(OUT)
