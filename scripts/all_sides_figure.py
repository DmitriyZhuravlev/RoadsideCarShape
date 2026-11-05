"""One car model from all sides next to the video: the video frame with the car framed, the model from eight directions
every 45 deg and from above (colours sampled on the subdivided surface, MESH_SUBDIV=2). Writes figures/all_sides.png.
Usage: python scripts/all_sides_figure.py [run label photo_run]   (default: DAIR-V2X-C clip 114_0, car 000046)"""
import os, sys, json
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cars_figure as cf
D = os.path.expanduser("~/Documents/MyArticles/code/3dseg/docs")
RUN, LAB, PRUN = sys.argv[1:4] if len(sys.argv) > 3 else ("ann_P_clip_114_0_all_sub2", "000046", "all_clip_114_0")
W, H = 480, 270


def best(run):
    """The scored row of the label (a label can be split over two track fragments: the better one)."""
    return max((x for x in json.load(open(f"{D}/whole_scene_cars_{run}.json")) if x.get("label") == LAB and "fused closed: iou" in x),
               key=lambda x: x["fused closed: iou"])


r = best(RUN)
m = json.load(open(f"{D}/cars_viewer_{RUN}/meshes.json"))[f"track {r['track']}"]
V, F = np.array(m["v"]).reshape(-1, 3), np.array(m["f"]).reshape(-1, 3)
C = np.array(m["c"]).reshape(-1, 3)[:, ::-1] * 255
K = np.array([[1.9 * W, 0, W / 2], [0, 1.9 * W, H / 2 + 4], [0, 0, 1]])
target = np.array([0.0, 0.0, 0.7])                          # car frame: x to the front, y to the left, z up
VIEWS = [(n, np.array([11.5 * np.cos(np.radians(a)), 11.5 * np.sin(np.radians(a)), 3.5])) for a, n in
         [(0, "front"), (45, "front left"), (90, "left"), (135, "rear left"), (180, "rear"), (225, "rear right"),
          (270, "right"), (315, "front right")]] + [("top", np.array([0.0, -1.0, 14.0]))]   # top: the car lies across
tiles = [cf.panel(cf.render_mesh(np.full((H, W, 3), 255, np.uint8), cf.look_at(K, e, target), V, F, C), n, size=(W, H))
         for n, e in VIEWS]
img = cv2.imread(f"{D}/photos_{PRUN}/track_{best(PRUN)['track']}.jpg")
sc = min(W / img.shape[1], (H - 26) / img.shape[0])
img = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
can = np.full((H - 26, W, 3), 255, np.uint8)
y0, x0 = (H - 26 - img.shape[0]) // 2, (W - img.shape[1]) // 2
can[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
tiles.insert(0, cf.panel(can, "video", size=(W, H)))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "all_sides.png")
cv2.imwrite(out, np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:])]))
print(out, {k: r[k] for k in ("track", "windows", "n_side", "range_m", "length", "label_length", "width", "height",
                              "fused closed: iou", "fused closed: height_err_cm", "heading_err_deg")})
