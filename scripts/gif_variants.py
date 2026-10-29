"""Variants of animations for the README (to choose from): rotating models of several datasets, a 2 x 2 grid of
rotating cars, a cyclist / pedestrian, and the video of intersection A with the model drawn onto the moving car.

    python scripts/gif_variants.py OUT_DIR
"""
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cars_figure as cf  # noqa: E402  (switches to the 3dseg folder)
import turntable_gif as tg  # noqa: E402
import v2x_seq_shape as vs  # noqa: E402
import whole_scene_cars as w  # noqa: E402

DOCS = os.path.expanduser("~/Documents/MyArticles/code/3dseg/docs")
FIG = cf.FIG


def from_meshes(path, track):
    m = json.load(open(path))[f"track {track}"]
    return (np.array(m["v"]).reshape(-1, 3), np.array(m["f"]).reshape(-1, 3),
            np.array(m["c"]).reshape(-1, 3)[:, ::-1] * 255, m["row"])


def turn(V, F, C, n=48, size=(560, 360), dist=9.5, height=4.6, target_z=0.75, f=1.25, shadow=(2.6, 1.2)):
    """Frames of a camera circling the model (BGR colours 0-255), with a soft ground shadow."""
    V = V - [(V[:, 0].max() + V[:, 0].min()) / 2, (V[:, 1].max() + V[:, 1].min()) / 2, 0]
    w_, h = size
    K = np.array([[f * w_, 0, w_ / 2], [0, f * w_, h / 2 + 10], [0, 0, 1]])
    g = np.array([[x, y, 0.0] for x in np.linspace(-shadow[0], shadow[0], 27) for y in np.linspace(-shadow[1], shadow[1], 13)])
    out = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        cam = cf.look_at(K, np.array([dist * np.cos(a), dist * np.sin(a), height]), np.array([0.0, 0.0, target_z]))
        img = np.full((h, w_, 3), 255, np.uint8)
        q, _ = vs.project(cam, g)
        sh = img.copy()
        cv2.fillConvexPoly(sh, cv2.convexHull(np.round(q).astype(np.int32)), (225, 225, 225), cv2.LINE_AA)
        img = cv2.addWeighted(sh, 0.7, img, 0.3, 0)
        out.append(cf.render_mesh(img, cam, V, F, C))
    return out


def label(frames, text):
    for fr in frames:
        cv2.putText(fr, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    return frames


def grid_frames(sequences, cols=2, gap=6):
    """Frame i = the i-th frames of all sequences in a grid (row-major, `cols` columns, white gaps)."""
    H, W = sequences[0][0].shape[:2]
    rows = (len(sequences) + cols - 1) // cols
    out = []
    for i in range(len(sequences[0])):
        g = np.full((rows * H + (rows - 1) * gap, cols * W + (cols - 1) * gap, 3), 255, np.uint8)
        for j, s in enumerate(sequences):
            r, c = divmod(j, cols)
            g[r * (H + gap):r * (H + gap) + H, c * (W + gap):c * (W + gap) + W] = s[i]
        out.append(g)
    return out


def video_overlay(models, seq, lab, size=(480, 300)):
    """Intersection A: the car's own window frames, the crop following it; left the video, right our model on it."""
    m = next(m for m in models.values() if m["row"].get("label") == lab)
    out = []
    for k in sorted(m["poses"]):
        cam = seq.cams[k]
        Vw = w.to_world({"poses": m["poses"]}, k, m["V"])
        q, z = vs.project(cam, Vw)
        if (z <= 0).any():
            continue
        cx, cy = q.mean(0)
        s = 1.6 * max(np.ptp(q[:, 0]), np.ptp(q[:, 1]) * size[0] / size[1])
        hw, hh = s / 2, s / 2 * size[1] / size[0]
        X0, Y0, X1, Y1 = int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)
        if X0 < 0 or Y0 < 0 or X1 > seq.shape[1] or Y1 > seq.shape[0]:
            continue
        frame = seq.frames[k]
        a = cv2.resize(frame[Y0:Y1, X0:X1], size, interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(cf.render_mesh(frame.copy(), cam, Vw, m["F"], m["vcol"], alpha=0.85)[Y0:Y1, X0:X1], size,
                       interpolation=cv2.INTER_CUBIC)
        cv2.putText(a, "video", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(b, "our 3D model", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        out.append(np.hstack([a, np.full((size[1], 6, 3), 255, np.uint8), b]))
    return out


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    spd = os.path.join(DOCS, os.environ.get("SPD_MESHES", "cars_viewer_spd_seq_0002"), "meshes.json")
    dair = os.path.join(DOCS, os.environ.get("DAIR_MESHES", "cars_viewer_dairc_clip_21_0"), "meshes.json")

    def track_of(path, label):                    # track numbers change between runs; the label does not
        return next(int(k.split()[1]) for k, m in json.load(open(path)).items() if m["row"].get("label") == label)
    models = pickle.load(open(os.path.join(FIG, "models_A.pkl"), "rb"))
    a_by_label = {m["row"].get("label"): m for m in models.values()}
    made = {}
    # single cars
    cars = {"G1_spd_red_van": (spd, track_of(spd, "001528")), "G2_spd_hatchback": (spd, track_of(spd, "001570")),
            "G3_spd_sedan": (spd, track_of(spd, "001493")), "G4_dair_white_sedan": (dair, track_of(dair, "000023"))}
    for name, (p, t) in cars.items():
        V, F, C, r = from_meshes(p, t)
        made[name] = (turn(V, F, C), f"{r['body_type']}, IoU {r['fused closed: iou']:.2f}, L {r['length']:.1f}/{r['label_length']:.1f} m")
    for name, lab in (("G5_A_white_SUV", "003890"), ("G6_A_van", "004010")):
        m = a_by_label[lab]
        r = m["row"]
        made[name] = (turn(m["V"], m["F"], m["vcol"]), f"{r['body_type']}, IoU {r['fused closed: iou']:.2f}, L {r['length']:.1f}/{r['label_length']:.1f} m")
    for name, (fr, cap) in made.items():
        print(name, tg.save_gif(fr, os.path.join(out, name + ".gif")) // 1000, "kB;", cap)
    # 2 x 2 grid of four cars, labelled
    four = [label([f.copy() for f in made[n][0]], made[n][1]) for n in
            ("G1_spd_red_van", "G4_dair_white_sedan", "G5_A_white_SUV", "G3_spd_sedan")]
    small = [[cv2.resize(f, (420, 270), interpolation=cv2.INTER_AREA) for f in s] for s in four]
    print("G7_grid_4_cars", tg.save_gif(grid_frames(small), os.path.join(out, "G7_grid_4_cars.gif")) // 1000, "kB")
    # a cyclist and a pedestrian (vru_shape.py models)
    for name, path in (("G8_cyclist", "vru_spd/seq_0093/000911.npz"), ("G9_pedestrian", "vru_spd/seq_0059/003926.npz")):
        d = np.load(os.path.join(DOCS, path))
        fr = turn(d["V"], d["F"], d["C"], size=(360, 420), dist=5.2, height=2.6, target_z=0.9, f=1.6, shadow=(0.9, 0.4))
        print(name, tg.save_gif(fr, os.path.join(out, name + ".gif")) // 1000, "kB")
    # the video with the model on the moving car (intersection A)
    seq = vs.Sequence()
    for name, lab in (("G10_video_overlay_SUV", "003890"), ("G11_video_overlay_van", "004010")):
        fr = video_overlay(models, seq, lab)
        print(name, len(fr), "frames", tg.save_gif(fr, os.path.join(out, name + ".gif"), fps=6) // 1000, "kB")


if __name__ == "__main__":
    main()
