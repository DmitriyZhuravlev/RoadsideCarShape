"""The chosen illustrations: video photo | model front-left | model rear-left, one car per row."""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.expanduser("~/Documents/MyArticles/RoadsideCarShape/scripts"))
import cars_figure as cf
D = os.path.expanduser("~/Documents/MyArticles/code/3dseg/docs")
SUFFIX = os.environ.get("SUFFIX", "_sub2")   # the model runs with the colours sampled on the subdivided surface
# (dataset, run of the model on the precise annotation scored against the labels, label, run whose video photos frame
# the car: the photo is the video, the same for any annotation)
PICK = [("DAIR-V2X-C", "ann_P_clip_52_1_all", "000126", "all_clip_52_1"),
        ("V2X-Seq-SPD", "ann_P_seq_0002_all", "001510", "all_seq_0002"),
        ("DAIR-V2X-C", "ann_P_clip_52_1_all", "000163", "all_clip_52_1"),
        ("V2X-Seq-SPD", "ann_P_seq_0002_all", "001502", "all_seq_0002"),
        ("DAIR-V2X-C", "ann_P_clip_52_1_all", "000067", "all_clip_52_1"),
        ("V2X-Seq-SPD", "ann_P_seq_0082_all", "000164", "all_seq_0082"),
        ("DAIR-V2X-C", "ann_P_clip_105_0_all", "000096", "all_clip_105_0"),
        ("V2X-Seq-SPD", "ann_P_seq_0054_all", "002324", "all_seq_0054")]
W, H = 400, 270
def tile(m, eye, title):
    V, F = np.array(m["v"]).reshape(-1, 3), np.array(m["f"]).reshape(-1, 3)
    C = np.array(m["c"]).reshape(-1, 3)[:, ::-1] * 255
    K = np.array([[1.55 * W, 0, W / 2], [0, 1.55 * W, H / 2], [0, 0, 1]])
    return cf.panel(cf.render_mesh(np.full((H, W, 3), 255, np.uint8), cf.look_at(K, eye, np.array([0.0, 0.0, 0.7])), V, F, C), title, size=(W, H))
def photo(ph, title):
    img = cv2.imread(ph); sc = min(W / img.shape[1], (H - 26) / img.shape[0])
    img = cv2.resize(img, (max(1, int(img.shape[1] * sc)), max(1, int(img.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    can = np.full((H - 26, W, 3), 255, np.uint8); y0, x0 = (H - 26 - img.shape[0]) // 2, (W - img.shape[1]) // 2
    can[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return cf.panel(can, title, size=(W, H))
rows = []
def row_of(run, lab):
    return max((x for x in json.load(open(f"{D}/whole_scene_cars_{run}.json")) if x.get("label") == lab and "fused closed: iou" in x),
               key=lambda x: x["fused closed: iou"])       # a label can be split over two track fragments
for name, run, lab, prun in PICK:
    r = row_of(run + SUFFIX, lab)
    m = json.load(open(f"{D}/cars_viewer_{run}{SUFFIX}/meshes.json"))[f"track {r['track']}"]
    rows.append(np.hstack([photo(f"{D}/photos_{prun}/track_{row_of(prun, lab)['track']}.jpg", f"{name}: video"),
                           tile(m, np.array([7.8, 8.4, 4.6]), f"model, IoU {r['fused closed: iou']:.2f}"),
                           tile(m, np.array([-7.5, 8.6, 3.8]), f"length {r['length']:.1f} m (label {r['label_length']:.1f} m)")]))
two = [np.hstack([rows[i], np.full((rows[i].shape[0], 16, 3), 255, np.uint8), rows[i + 1]]) for i in range(0, len(rows), 2)]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "best_models.png"); cv2.imwrite(out, np.vstack(two)); print(out)
