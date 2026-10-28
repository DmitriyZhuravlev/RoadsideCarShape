"""Pool the DAIR-V2X-C clip results (run_subset.py) and render the car models.

    python pool.py            labelled-box scores per clip (if missing), pooled table -> pooled.json
    python pool.py --render   figures/models_grid.png (best cars, 3/4 view) and figures/models_lidar.png
    python pool.py --boxes CLIP   (internal) score the labelled boxes of one clip against its LiDAR, raw and closed

The labelled box is scored exactly like the models: a box of the label's length, width and height as a height field,
against the car's accumulated LiDAR (raw, and with the scan gaps closed), the better of the two headings.
"""
import json
import os
import subprocess
import sys

import numpy as np

from run_subset import HERE, SPD, TAG   # noqa: E402  (RUNS, SPD, TAG from the environment, DAIR-V2X-C by default)
CODE = os.path.expanduser("~/Documents/MyArticles/code/3dseg")
SCRIPTS = os.path.expanduser("~/Documents/MyArticles/RoadsideCarShape/scripts")
JOBS = 3


def box_scores(clip):
    os.chdir(CODE)
    sys.path.insert(0, CODE)
    import ego_sequence as es
    import lidar_masks as lm
    import whole_scene_cars as w
    seq, ground = es.open_sequence()
    shapes = lm.LidarShapes(seq, ground)
    xs = np.arange(-4, 4, 0.1) + 0.05
    ys = np.arange(-1.5, 1.5, 0.1) + 0.05
    out = {}
    for r in json.load(open(os.path.join(HERE, clip, "cars.json"))):
        if "fused: iou" not in r:
            continue
        Lt, lx, ly, n = w.lidar_heights(shapes, r["label"])
        box = np.where((np.abs(xs)[:, None] < r["label_length"] / 2) & (np.abs(ys)[None] < r["label_width"] / 2),
                       r["label_height"], 0.0)
        raw = w.shape_scores(box, xs, ys, Lt, lx, ly)
        closed = w.shape_scores(box, xs, ys, w.close_lidar_holes(Lt, lx[1] - lx[0]), lx, ly)
        out[str(r["track"])] = {"box: iou": raw["iou"], "box closed: iou": closed["iou"]}
    json.dump(out, open(os.path.join(HERE, clip, "boxes.json"), "w"), indent=1)


def load_rows():
    rows = []
    for c in json.load(open(os.path.join(HERE, "chosen.json"))):
        cars = os.path.join(HERE, c["clip"], "cars.json")
        if not os.path.exists(cars):
            continue
        boxes_path = os.path.join(HERE, c["clip"], "boxes.json")
        boxes = json.load(open(boxes_path)) if os.path.exists(boxes_path) else {}
        for r in json.load(open(cars)):
            r.update(boxes.get(str(r["track"]), {}))
            r["clip"], r["intersection"] = c["clip"], c["intersection"]
            rows.append(r)
    return rows


def pool(rows):
    """Per group ("all" and each intersection): evaluated cars (with both our and the box's closed IoU), mean IoU of
    ours / the labelled box against the raw and the closed LiDAR, median position error along / across the ray."""
    groups = {"all": rows}
    for s in sorted({r["intersection"] for r in rows}):
        groups[s] = [r for r in rows if r["intersection"] == s]
    mean = lambda g, k: float(np.mean([r[k] for r in g if k in r])) if any(k in r for r in g) else float("nan")
    med = lambda g, k: float(np.median([r[k] for r in g if k in r])) if any(k in r for r in g) else float("nan")
    out = {}
    for name, g in groups.items():
        g = [r for r in g if "fused closed: iou" in r and "box closed: iou" in r]
        out[name] = {"cars": len(g), "ours raw": mean(g, "fused: iou"), "ours closed": mean(g, "fused closed: iou"),
                     "box raw": mean(g, "box: iou"), "box closed": mean(g, "box closed: iou"),
                     "along m": med(g, "pose: along-ray err m"), "across m": med(g, "pose: across-ray err m")}
    return out


def score_boxes():
    todo = [c["clip"] for c in json.load(open(os.path.join(HERE, "chosen.json")))
            if os.path.exists(os.path.join(HERE, c["clip"], "cars.json"))
            and not os.path.exists(os.path.join(HERE, c["clip"], "boxes.json"))]
    procs = []
    for clip in todo:
        running = [p for p in procs if p.poll() is None]
        if len(running) >= JOBS:
            running[0].wait()
        env = dict(os.environ, V2X_SEQ_DIR=os.path.join(SPD, clip))
        log = open(os.path.join(HERE, clip, "boxes.log"), "w")
        procs.append(subprocess.Popen([sys.executable, __file__, "--boxes", clip], env=env, stdout=log,
                                      stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()


def models():
    """(row, V, F, BGR 0-255 colours, LiDAR points) of every evaluated car with a mesh, best closed IoU first."""
    out = []
    for r in load_rows():
        path = os.path.join(CODE, "docs", f"cars_viewer_{TAG}_{r['clip']}", "meshes.json")
        if "fused closed: iou" not in r or not os.path.exists(path):
            continue
        m = json.load(open(path)).get(f"track {r['track']}")
        if m is None:
            continue
        out.append((r, np.array(m["v"]).reshape(-1, 3), np.array(m["f"]).reshape(-1, 3),
                    np.array(m["c"]).reshape(-1, 3)[:, ::-1] * 255, np.array(m.get("lidar_top", [])).reshape(-1, 3)))
    return sorted(out, key=lambda t: -t[0]["fused closed: iou"])


def render(n_grid=20, n_lidar=6):
    sys.path.insert(0, SCRIPTS)
    import cars_figure as cf
    import cv2
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    target = np.array([0.0, 0.0, 0.7])

    def view(V, F, C, eye, size=(400, 300), lidar=None):
        w_, h_ = size
        K = np.array([[1.75 * w_, 0, w_ / 2], [0, 1.75 * w_, h_ / 2], [0, 0, 1]])
        cam = cf.look_at(K, eye, target)
        img = cf.render_mesh(np.full((h_, w_, 3), 255, np.uint8), cam, V, F, C)
        if lidar is not None and len(lidar):
            Kc, R, t = cam
            X = lidar @ R.T + t
            uv = (X @ Kc.T)[:, :2] / X[:, 2:3]
            for u, v in uv[X[:, 2] > 0.1].round().astype(int):
                cv2.circle(img, (u, v), 1, (40, 40, 220), -1, cv2.LINE_AA)
        return img

    ms = models()
    eye = np.array([7.8, 8.4, 4.6])                    # front-left, from above, far enough for 6 m models
    tiles = [cf.panel(view(V, F, C, eye), f"{r['intersection'][-2:]} {r['body_type']}: IoU {r['fused closed: iou']:.2f}, "
                      f"L {r['length']:.1f}/{r['label_length']:.1f}", size=(400, 300))
             for r, V, F, C, _ in ms[:n_grid]]
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(os.path.join(HERE, "figures", "models_grid.png"), grid)
    rows = []
    for r, V, F, C, L in ms[:n_lidar]:
        name = f"{r['clip']} track {r['track']}"
        rows.append(np.hstack([cf.panel(view(V, F, C, eye), f"{name}: model", size=(400, 300)),
                               cf.panel(view(V, F, C, -eye * [1, 1, -1]), "the other side", size=(400, 300)),
                               cf.panel(view(V, F, C, eye, lidar=L), f"LiDAR (red), IoU {r['fused closed: iou']:.2f}",
                                        size=(400, 300))]))
    cv2.imwrite(os.path.join(HERE, "figures", "models_lidar.png"), np.vstack(rows))
    print(len(ms), "models;", min(n_grid, len(ms)), "in the grid")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--boxes":
        box_scores(sys.argv[2])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--render":
        render()
        return
    score_boxes()
    res = pool(load_rows())
    json.dump(res, open(os.path.join(HERE, "pooled.json"), "w"), indent=1)
    print(f"{'':12s} {'cars':>4s} {'ours raw':>8s} {'box raw':>8s} {'ours cl.':>8s} {'box cl.':>8s} {'along':>6s} {'across':>6s}")
    for name, s in res.items():
        print(f"{name:12s} {s['cars']:4d} {s['ours raw']:8.3f} {s['box raw']:8.3f} {s['ours closed']:8.3f} "
              f"{s['box closed']:8.3f} {s['along m']:6.2f} {s['across m']:6.2f}")


if __name__ == "__main__":
    main()
