"""best_models.png from the classical input: video photo | model front-left | model rear-left, one car per row.
The cars are those of best_models_figure.py wherever the classical tracks contain the label; otherwise (seq_0002) the best
classical cars of the clip. Photo = the frame of the track with the largest visible share, cropped around its cuboid.
Writes figures/best_models.png (OUT_NAME env to change)."""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.expanduser("~/Documents/MyArticles/RoadsideCarShape/scripts"))
import cars_figure as cf
D = os.path.expanduser("~/Documents/MyArticles/code/3dseg/docs")
CLIPS = os.path.expanduser("~/Documents/MyArticles/improve_runs/clips")
SUFFIX = "_fb"
# (dataset, classical run name, label, clip folder in improve_runs/clips)
PICK = [("DAIR-V2X-C", "c52", "000126", "c52"), ("V2X-Seq-SPD", "s0002", "001486", "s0002"),
        ("DAIR-V2X-C", "c52", "000163", "c52"), ("V2X-Seq-SPD", "s0002", "001528", "s0002"),
        ("DAIR-V2X-C", "c114", "000078", "c114"), ("V2X-Seq-SPD", "s0002", "001498", "s0002"),
        ("DAIR-V2X-C", "c105", "000096", "c105"), ("V2X-Seq-SPD", "s0054", "002324", "s0054")]
W, H = 400, 270


def tile(m, eye, title):
    V, F = np.array(m["v"]).reshape(-1, 3), np.array(m["f"]).reshape(-1, 3)
    C = np.array(m["c"]).reshape(-1, 3)[:, ::-1] * 255
    K = np.array([[1.55 * W, 0, W / 2], [0, 1.55 * W, H / 2], [0, 0, 1]])
    return cf.panel(cf.render_mesh(np.full((H, W, 3), 255, np.uint8), cf.look_at(K, eye, np.array([0.0, 0.0, 0.7])), V, F, C), title, size=(W, H))


def photo(name, clip, track, title, margin=1.5):
    """Frame of the classical track with the largest visible share, cropped around the cuboid (nothing drawn)."""
    cub = f"{D}/precise_ann_{name}/cuboids"
    best = None
    for fn in sorted(os.listdir(cub)):
        if not fn.endswith(".json"):
            continue
        for e in json.load(open(os.path.join(cub, fn))):
            if str(e["track"]) == track and (best is None or e.get("visible", 0) > best[1].get("visible", 0)):
                best = (fn[:-5], e)
    img = cv2.imread(f"{CLIPS}/{clip}/image/{best[0]}.jpg")
    uv = np.array(best[1]["corners_uv"])
    (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
    ph = H - 26
    cw = max(margin * (x1 - x0), margin * (y1 - y0) * W / ph)
    ch = cw * ph / W
    X0 = int(min(max(0, (x0 + x1) / 2 - cw / 2), max(0, img.shape[1] - cw)))
    Y0 = int(min(max(0, (y0 + y1) / 2 - ch / 2), max(0, img.shape[0] - ch)))
    crop = cv2.resize(img[Y0:Y0 + int(ch), X0:X0 + int(cw)], (W, ph), interpolation=cv2.INTER_CUBIC)
    return cf.panel(crop, title, size=(W, H))


def row_of(name, lab):
    rs = [x for x in json.load(open(f"{D}/whole_scene_cars_ann_{name}{SUFFIX}.json")) if x.get("label") == lab and "fused closed: iou" in x]
    return max(rs, key=lambda x: x["fused closed: iou"]) if rs else None


rows = []
for ds, name, lab, clip in PICK:
    r = row_of(name, lab)
    if r is None:
        print("MISSING", name, lab)
        continue
    m = json.load(open(f"{D}/cars_viewer_ann_{name}{SUFFIX}/meshes.json"))[f"track {r['track']}"]
    rows.append(np.hstack([photo(name, clip, r["ann_track"], f"{ds}: video"),
                           tile(m, np.array([7.8, 8.4, 4.6]), f"model, IoU {r['fused closed: iou']:.2f}"),
                           tile(m, np.array([-7.5, 8.6, 3.8]), f"length {r['length']:.1f} m (label {r['label_length']:.1f} m)")]))
    print(name, lab, r["ann_track"], round(r["fused closed: iou"], 2), flush=True)
two = [np.hstack([rows[i], np.full((rows[i].shape[0], 16, 3), 255, np.uint8), rows[i + 1]]) for i in range(0, len(rows) - 1, 2)]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", os.environ.get("OUT_NAME", "best_models.png"))
cv2.imwrite(out, np.vstack(two)); print(out)
