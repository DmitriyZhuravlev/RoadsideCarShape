import os, sys, pickle, numpy as np, cv2
os.chdir(os.path.expanduser("~/Documents/MyArticles/code/3dseg")); sys.path.insert(0, "."); sys.argv = sys.argv[:1] + ["x"] + sys.argv[1:]
from vru_shape import look_at, render
out = sys.argv[2]; rows = []
for lab, p in zip(sys.argv[3::2], sys.argv[4::2]):
    d = pickle.load(open(p, "rb")); V, F, C = d["V"], d["F"], d["col"]
    V = V - np.r_[(V[:, :2].max(0) + V[:, :2].min(0)) / 2, 0]
    TW, TH = 300, 300; K = np.array([[2.2 * TW, 0, TW / 2], [0, 2.2 * TW, TH / 2], [0, 0, 1]]); dist = 4.2
    tiles = []
    for az in range(0, 360, 45):
        a = np.radians(az - 90)
        eye = np.array([dist * np.cos(a), dist * np.sin(a), 1.4])
        img = render(np.full((TH, TW, 3), (238, 234, 228), np.uint8), look_at(K, eye, np.array([0, 0, 0.8])), V, F, C.astype(np.uint8))
        cv2.putText(img, f"{az} deg", (6, TH - 8), 0, 0.45, (90, 90, 90), 1, cv2.LINE_AA); tiles.append(img)
    row = np.hstack(tiles); cv2.putText(row, lab, (8, 22), 0, 0.6, (30, 30, 30), 2, cv2.LINE_AA); rows.append(row)
cv2.imwrite(out, np.vstack(rows)); print("->", out)
