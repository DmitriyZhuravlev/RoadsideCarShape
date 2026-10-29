"""Pedestrians and cyclists for the article: video (object framed) | own tracking | given tracking (labelled 2D boxes)
| given tracking from the side; plain Poisson surfaces."""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/Documents/MyArticles/RoadsideCarShape/scripts"))
sys.path.insert(0, os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
import cv2, numpy as np
import cars_figure as cf, spsr
D = os.path.expanduser("~/Documents/MyArticles/code/3dseg/docs")
S = os.environ.get("VRU_PHOTOS", os.path.dirname(os.path.abspath(__file__)))
PICK = [("seq_0036", "002251"), ("seq_0077", "000118"), ("seq_0059", "003921"), ("seq_0063", "003765")]
W, H = 260, 300
def render(npz, eye, title):
    z = np.load(npz); V, F, C = spsr.smooth_mesh(z["V"], z["F"].astype(int), z["C"].astype(float), h=0.02, alpha=0.0)
    K = np.array([[1.6 * W, 0, W / 2], [0, 1.6 * W, H / 2], [0, 0, 1]])
    return cf.panel(cf.render_mesh(np.full((H, W, 3), 255, np.uint8), cf.look_at(K, eye, np.array([0.0, 0.0, 0.85])), V, F, C), title, size=(W, H))
def photo(p, title):
    img = cv2.imread(p); sc = min(W / img.shape[1], (H - 26) / img.shape[0])
    img = cv2.resize(img, (max(1, int(img.shape[1] * sc)), max(1, int(img.shape[0] * sc))), interpolation=cv2.INTER_CUBIC)
    can = np.full((H - 26, W, 3), 255, np.uint8); y0, x0 = (H - 26 - img.shape[0]) // 2, (W - img.shape[1]) // 2
    can[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return cf.panel(can, title, size=(W, H))
e1, e2 = np.array([3.6, 3.0, 1.8]), np.array([0.0, 4.6, 1.2])
rows = []
for seq, tid in PICK:
    own, lab = (f"{D}/{d}/{seq}/{tid}" for d in ("vru_new", "vru_lab"))
    ro, rl = json.load(open(own + ".json")), json.load(open(lab + ".json"))
    rows.append(np.hstack([photo(os.path.join(S, f"vrulab_{seq}_{tid}.jpg"), f"{rl['type']}: video"),
                           render(own + ".npz", e1, f"own tracking, IoU {ro['holdout_iou']:.2f}"),
                           render(lab + ".npz", e1, f"given tracking, IoU {rl['holdout_iou']:.2f}"),
                           render(lab + ".npz", e2, f"H {rl['height']:.2f} / {rl['label_height']:.2f} m")]))
out = os.path.join(D, "vru_compare.png"); cv2.imwrite(out, np.vstack(rows)); print(out)
