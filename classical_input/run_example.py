"""Run the optical-flow clustering on a short clip and show the result.

    python run_example.py [DATA_DIR] [FIRST] [N]     # defaults: V2X-Seq-SPD example (intersection A), frames 20..44

DATA_DIR is an infrastructure side in the V2X-Seq-SPD layout (image/, velodyne/, calib/, label/). One LiDAR sweep
fits the road plane once (calibration); the labels are read only by evaluate() and for the grey reference boxes of
the figures. Writes to out/:
  masks/<fid>.png     uint16 instance mask (0 background, i + 1 the i-th object of cuboids/<fid>.json)
  cuboids/<fid>.json  per object: track, type, centre (road frame and virtual-LiDAR frame), yaw, lwh, box2d, error
  background.png      the mean of the whole video (half resolution)
  steps.png           the steps on one frame; frames.png  masks and cuboids on several frames
  metrics.json        the comparison with the labels
"""
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import camera
import ofc

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Documents/MyArticles/dataset/spd/V2X-Seq-SPD-Example/infrastructure-side")
FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N = int(sys.argv[3]) if len(sys.argv) > 3 else 25
SHOW = 10              # the frame of the clip shown step by step
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
COLOURS = plt.get_cmap("tab10").colors


def colour(i):
    return tuple(int(255 * c) for c in COLOURS[i % 10][::-1])


def draw_cuboid(img, cam, o, col, th=2):
    uv = cam.project(ofc.corners(o["centre"], o["yaw"], *o["lwh"]))
    for a, b in EDGES:
        if np.isfinite(uv[[a, b]]).all():
            cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), col, th, cv2.LINE_AA)


def overlay(img, cam, objs):
    out = img.copy()
    for o in objs:
        m = cv2.resize(o["mask"].astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
        out[m] = (0.5 * out[m] + 0.5 * np.array(colour(o["track"]))).astype(np.uint8)
    for o in objs:
        draw_cuboid(out, cam, o, colour(o["track"]))
        x0, y0 = int(o["box"][0]), int(o["box"][1])
        cv2.putText(out, f'{o["track"]} {o["track_type"]}', (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, f'{o["track"]} {o["track_type"]}', (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    colour(o["track"]), 2, cv2.LINE_AA)
    return out


def labels(cam, fid):
    """Evaluation only: the labelled objects of a frame in the road frame, with their 2D boxes."""
    L = json.load(open(os.path.join(ROOT, "label/virtuallidar", fid + ".json")))
    B = {o["track_id"]: o for o in json.load(open(os.path.join(ROOT, "label/camera", fid + ".json")))}
    out = []
    for o in L:
        loc = o["3d_location"]
        c = cam.to_g(np.array([loc["x"], loc["y"], loc["z"]]))
        d = cam.to_g(np.array([np.cos(o["rotation"]), np.sin(o["rotation"]), 0.0]) + cam.o)
        b = B.get(o["track_id"], {}).get("2d_box")
        out.append({"track": o["track_id"], "type": o["type"], "centre": c[:2], "yaw": float(np.arctan2(d[1], d[0])),
                    "lwh": (o["3d_dimensions"]["l"], o["3d_dimensions"]["w"], o["3d_dimensions"]["h"]),
                    "box": None if b is None else (b["xmin"], b["ymin"], b["xmax"], b["ymax"]),
                    "truncated": o.get("truncated_state", 0), "occluded": o.get("occluded_state", 0)})
    return out


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def evaluate(cam, fids, results, gate=3.0, max_range=80.0, min_speed=1.0, dt=0.1):
    """Evaluation only: the labelled objects that move in the frame (>= min_speed m/s between the neighbouring
    frames; a stopped object has no flow), lie within max_range m and are neither truncated nor without a 2D box,
    against our objects, matched per frame by footprint centre (Hungarian, at most `gate` m)."""
    labs = [labels(cam, f) for f in fids]
    pos = [{l["track"]: l["centre"] for l in ls} for ls in labs]

    def speed(k, t):
        a, b = pos[max(k - 1, 0)].get(t), pos[min(k + 1, len(pos) - 1)].get(t)
        steps = min(k + 1, len(pos) - 1) - max(k - 1, 0)
        return np.linalg.norm(b - a) / (steps * dt) if a is not None and b is not None and steps else 0.0

    rows, n_lab, n_det = [], 0, 0
    for k, (ls, objs) in enumerate(zip(labs, results)):
        ls = [l for l in ls if speed(k, l["track"]) >= min_speed and np.linalg.norm(l["centre"]) <= max_range
              and l["box"] is not None and l["truncated"] == 0]
        objs = [o for o in objs if np.linalg.norm(o["centre"]) <= max_range]
        n_lab, n_det = n_lab + len(ls), n_det + len(objs)
        if not ls or not objs:
            continue
        D = np.array([[np.linalg.norm(o["centre"] - l["centre"]) for l in ls] for o in objs])
        for i, j in zip(*linear_sum_assignment(D)):
            if D[i, j] <= gate:
                o, l = objs[i], ls[j]
                dyaw = abs((o["yaw"] - l["yaw"] + np.pi / 2) % np.pi - np.pi / 2)
                rows.append({"label": l["type"], "ours": o["track_type"], "centre_m": D[i, j],
                             "yaw_deg": np.degrees(dyaw), "box_iou": iou(o["box"], l["box"]),
                             "dl": o["lwh"][0] - l["lwh"][0], "dw": o["lwh"][1] - l["lwh"][1],
                             "dh": o["lwh"][2] - l["lwh"][2], "range_m": float(np.linalg.norm(l["centre"]))})
    med = lambda k, f=abs: float(np.median([f(r[k]) for r in rows])) if rows else float("nan")
    same = {"Car": "car", "Van": "car", "Truck": "large", "Bus": "large", "Cyclist": "two-wheeler",
            "Motorcyclist": "two-wheeler", "Tricyclist": "two-wheeler", "Pedestrian": "pedestrian"}
    return {"frames": len(fids), "labelled moving objects (object-frames)": n_lab, "our objects": n_det,
            "matched": len(rows), "recall": len(rows) / max(n_lab, 1), "precision": len(rows) / max(n_det, 1),
            "median centre error m": med("centre_m"), "median yaw error deg (mod 180)": med("yaw_deg"),
            "median 2D box IoU": med("box_iou", lambda x: x), "median |dl| m": med("dl"), "median |dw| m": med("dw"),
            "median |dh| m": med("dh"),
            "class agreement (car / large / two-wheeler / pedestrian)":
                float(np.mean([same.get(r["label"]) == same.get(r["ours"]) for r in rows])) if rows else float("nan"),
            "matched range m (min, median, max)": [float(np.min([r["range_m"] for r in rows])),
                                                   med("range_m", lambda x: x),
                                                   float(np.max([r["range_m"] for r in rows]))] if rows else None}, labs


def save(fid, objs, cam, shape):
    os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "cuboids"), exist_ok=True)
    m = np.zeros(shape, np.uint16)
    for i, o in enumerate(objs):
        m[cv2.resize(o["mask"].astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0] = i + 1
    cv2.imwrite(os.path.join(OUT, "masks", fid + ".png"), m)
    rec = [{"track": o["track"], "type": o["track_type"], "frame_type": o["type"],
            "centre": [*map(float, o["centre"]), o["lwh"][2] / 2],
            "centre_virtuallidar": cam.to_v([*o["centre"], o["lwh"][2] / 2]).tolist(), "yaw": o["yaw"],
            "lwh": list(map(float, o["lwh"])), "box2d": list(map(float, o["box"])), "size_deviation": o["error"],
            "tangent_rms_m": float(o["tangent_rms"]) if np.isfinite(o["tangent_rms"]) else None, "from_track": "predicted" if o.get("predicted") else
            "split" if o.get("split_by_track") else "detected",
            "segments": o["segments"]} for o in objs]
    json.dump(rec, open(os.path.join(OUT, "cuboids", fid + ".json"), "w"), indent=1)


def figures(cam, fids, frames, results, debug, labs, show, bg):
    k = show
    img = frames[k]
    dbg = debug[k]
    full = lambda x: cv2.resize(x, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    pal = (np.array(plt.get_cmap("tab20").colors) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(ofc.flow_hsv(dbg["flow"]), cv2.COLOR_HSV2RGB)
    fl = pal[dbg["flow_labels"] % 20]
    fl[np.linalg.norm(dbg["colours"], axis=1)[dbg["flow_labels"]] < ofc.F_MIN] = 235
    fg = np.where(dbg["foreground"][..., None], cv2.cvtColor(ofc.half(img), cv2.COLOR_BGR2RGB), 235).astype(np.uint8)
    segs = np.full(dbg["flow_labels"].shape + (3,), 235, np.uint8)
    for i, m in enumerate(dbg["segments"]):
        segs[m] = pal[i % 20]
    panels = [("(a) video frame", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
              ("(b) mean of the whole video (background)", cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)),
              (f"(c) foreground: differs from the mean by > {ofc.FG_DIFF}", fg),
              (f"(d) colour segmentation of the frame (k-means, k = {ofc.K_COLOUR}, Lab)",
               pal[dbg["colour_labels"] % 20]),
              ("(e) optical flow in HSV (hue: direction, value: magnitude)", hsv),
              (f"(f) flow segmentation (k-means, k = {ofc.K}), grey: static", fl),
              (f"(g) segments: foreground, inside one colour and one flow segment ({len(dbg['segments'])})", segs),
              ("(h) merged segments and lifted cuboids", cv2.cvtColor(overlay(img, cam, results[k]),
                                                                       cv2.COLOR_BGR2RGB))]
    fig, ax = plt.subplots(5, 2, figsize=(16, 22.5))
    for a, (t, p) in zip(ax.ravel(), panels):
        a.imshow(full(p) if p.shape[:2] != img.shape[:2] else p)
        a.set_title(t, fontsize=11, loc="left")
        a.axis("off")
    bev(ax[4, 0], results[k], labs[k])
    ax[4, 1].axis("off")
    fig.suptitle(f"Background, colour and flow segmentation, lifting: frame {fids[k]}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "steps.png"), dpi=80)
    plt.close(fig)

    pick = np.linspace(1, len(frames) - 1, 4).round().astype(int)
    fig, ax = plt.subplots(2, 2, figsize=(16, 9.4))
    for a, i in zip(ax.ravel(), pick):
        a.imshow(cv2.cvtColor(overlay(frames[i], cam, results[i]), cv2.COLOR_BGR2RGB))
        a.set_title(f"frame {fids[i]}: {len(results[i])} objects", fontsize=11, loc="left")
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "frames.png"), dpi=80)
    plt.close(fig)


def bev(a, objs, labs):
    for l in labs:
        f = ofc.corners(l["centre"], l["yaw"], *l["lwh"])[:4, :2]
        a.fill(f[:, 1], f[:, 0], color="0.8", ec="0.5", lw=0.8)
    for o in objs:
        f = ofc.corners(o["centre"], o["yaw"], *o["lwh"])[:4, :2]
        a.fill(f[:, 1], f[:, 0], fc="none", ec=COLOURS[o["track"] % 10], lw=2)
        a.text(o["centre"][1], o["centre"][0] + 3, str(o["track"]), color=COLOURS[o["track"] % 10], fontsize=9,
               ha="center")
    a.plot(0, 0, "k^")
    a.set_xlim(40, -40)
    a.set_ylim(0, 120)
    a.set_aspect("equal")
    a.set_xlabel("y (m)")
    a.set_ylabel("x (m), road frame")
    a.set_title("(i) road plane: ours (coloured) and the labelled boxes (grey, evaluation only)", fontsize=11,
                loc="left")


def main():
    fids = sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, "image")))
    cam = camera.Camera(ROOT, fids[FIRST])
    print(f"road plane: camera {cam.height:.2f} m above the road")
    clip = fids[FIRST:FIRST + N]
    cache = os.path.join(OUT, "background.png")
    if os.path.exists(cache):
        bg = cv2.imread(cache)
    else:
        bg = ofc.mean_background([os.path.join(ROOT, "image", f + ".jpg") for f in fids])
        os.makedirs(OUT, exist_ok=True)
        cv2.imwrite(cache, bg)
    print(f"mean background of {len(fids)} frames")
    prev = cv2.imread(os.path.join(ROOT, "image", fids[FIRST - 1] + ".jpg"))
    frames, results, debug = [], [], []
    for fid in clip:
        img = cv2.imread(os.path.join(ROOT, "image", fid + ".jpg"))
        objs, dbg = ofc.detect(cam, prev, img, bg)
        frames.append(img)
        results.append(objs)
        debug.append(dbg if len(frames) == SHOW + 1 else None)     # the masks of every segment are large
        prev = img
        print(fid, len(objs), "objects:", ", ".join(f'{o["type"]} e={o["error"]:.2f}' for o in objs))
    ofc.link(results)
    ofc.refine_tracks(cam, results)
    ofc.temporal_split(cam, results, tuple(int(round(s * ofc.SCALE)) for s in frames[0].shape[:2]))
    for fid, objs in zip(clip, results):
        save(fid, objs, cam, frames[0].shape[:2])
    metrics, labs = evaluate(cam, clip, results)
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    print(json.dumps(metrics, indent=1))
    figures(cam, clip, frames, results, debug, labs, show=SHOW, bg=bg)


if __name__ == "__main__":
    main()
