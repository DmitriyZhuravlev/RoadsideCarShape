"""
The proposed method on a second roadside camera, unchanged (no parameter touched).

  camera A: V2X-Seq-SPD example, intersection yizhuang09, camera 1 (251 frames, 25 s) - where it was developed;
  camera B: DAIR-V2X-C cooperative example, intersection yizhuang08, camera 7 (84 frames, 8 s) - new.

Two detection sources, the same geometry (lifting.vp_bounding_quadrangle -> heading from the motion of the
box's ground point -> lifting.solve_length, v2x_seq_demo.lift_track "heading+length"):
  1. labelled 2D boxes (the geometry alone);
  2. the classical pipeline (median background, motion blobs, contact line, Kalman tracking; v2x_seq_classical).
Camera B's labels have no track ids: labelled objects are linked frame to frame by their position (used for
the motion of labelled boxes and for scoring only).

Calibration matters at the 0.1 deg level: the tilt that puts the labelled boxes' ground contact on the 2D box
bottoms is also reported (upper bound, uses the labels; the lane-marking vanishing point was 0.5 deg off).

Writes docs/coop_examples.png and docs/coop_errors.png.
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v2x_seq_classical as vc
import v2x_seq_demo as vd

REPO = os.path.dirname(os.path.abspath(__file__))
COOP = os.environ.get("V2X_COOP_DIR", os.path.join(REPO, "..", "..", "dataset", "coop",
                                                    "example-cooperative-vehicle-infrastructure", "infrastructure-side"))
EST, ALT, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#8a8984"


def associate_tracks(frames_objects, max_dist=2.5, max_gap=3):
    """Greedy frame-to-frame association of labelled objects into tracks (labels without track ids).

    frames_objects: list over frames k = 0..n-1; each item is a list of dicts that have key "pos" (np.ndarray (2,), metres).
    Objects are linked to the track whose last position (seen at most max_gap frames before) is nearest, if that
    distance is <= max_dist * (frames since last seen); each track gets at most one object per frame; the pairs are
    taken greedily in order of increasing distance; unmatched objects start new tracks.
    Each object dict gets a key "k" = its frame index. Returns dict track_id (int, from 0) -> list of the object dicts
    in frame order.
    """
    active, tracks = {}, {}
    for k, objects in enumerate(frames_objects):
        pairs = sorted((float(np.linalg.norm(o["pos"] - pos)), tid, i)
                       for i, o in enumerate(objects) for tid, (last, pos) in active.items()
                       if k - last <= max_gap and np.linalg.norm(o["pos"] - pos) <= max_dist * (k - last))
        used_t, used_o = set(), set()
        for _, tid, i in pairs:
            if tid in used_t or i in used_o:
                continue
            used_t.add(tid)
            used_o.add(i)
            objects[i]["k"] = k
            tracks[tid].append(objects[i])
            active[tid] = (k, objects[i]["pos"])
        for i, o in enumerate(objects):
            if i not in used_o:
                o["k"] = k
                tid = len(tracks)
                tracks[tid] = [o]
                active[tid] = (k, o["pos"])
    return tracks


def tilt_cameras(cams, delta):
    """Rotate every camera about its own optical centre by `delta` radians about the camera x axis (tilt).
    cams: list of (K, R, t) with x ~ K (R X + t). Returns new (K, Rx R, Rx t) tuples (centre unchanged)."""
    Rx = np.array([[1, 0, 0], [0, np.cos(delta), -np.sin(delta)], [0, np.sin(delta), np.cos(delta)]])
    return [(K, Rx @ R, Rx @ t) for K, R, t in cams]


def label_tilt(cams, tracks, deltas=np.radians(np.arange(-0.5, 0.501, 0.01))):
    """Tilt correction that puts the labelled boxes' ground contact on the labelled 2D box bottoms
    (an upper bound: it uses the labels; the lane-marking vanishing point is not precise enough)."""
    objs = [s for q in tracks.values() for s in q if not s["truncated"]]
    fps = [np.c_[vd.lifting.compute_rectangle_points(s["pos"], s["yaw"], s["dims"][1], s["dims"][0])[0], np.zeros(4)]
           for s in objs]
    cost = []
    for d in deltas:
        cd = tilt_cameras(cams, d)
        cost.append(np.median([abs(vd.project(cd[s["k"]], fp)[:, 1].max() - s["box"][3]) for s, fp in zip(objs, fps)]))
    return float(deltas[int(np.argmin(cost))])


def load_coop():
    """Same structure as v2x_seq_demo.load_sequence, from the cooperative example's data_info paths."""
    info = sorted(json.load(open(os.path.join(COOP, "data_info.json"))), key=lambda d: int(d["image_timestamp"]))
    times = np.array([int(d["image_timestamp"]) for d in info]) / 1e6
    plane = vd.fit_ground_plane(vd.read_pcd(os.path.join(COOP, info[0]["pointcloud_path"])))
    ground = plane[2]
    cams, frames_objects, images = [], [], []
    for d in info:
        K = np.array(json.load(open(os.path.join(COOP, d["calib_camera_intrinsic_path"])))["cam_K"]).reshape(3, 3)
        ex = json.load(open(os.path.join(COOP, d["calib_virtuallidar_to_camera_path"])))
        R, t = np.array(ex["rotation"]), np.array(ex["translation"]).ravel()
        cams.append((K, R, t + ground * R[:, 2]))
        objs = []
        for o in json.load(open(os.path.join(COOP, d["label_camera_std_path"]))):
            if o["type"] not in ("Car", "Van"):
                continue
            objs.append({"pos": np.array([float(o["3d_location"]["x"]), float(o["3d_location"]["y"])]),
                         "yaw": float(o["rotation"]),
                         "dims": tuple(float(o["3d_dimensions"][a]) for a in ("l", "w", "h")),
                         "z": float(o["3d_location"]["z"]) - ground, "truncated": int(o["truncated_state"]),
                         "box": [float(o["2d_box"][x]) for x in ("xmin", "ymin", "xmax", "ymax")]})
        frames_objects.append(objs)
        images.append(os.path.join(COOP, d["image_path"]))
    tracks = associate_tracks(frames_objects)
    camera_xy = (-np.linalg.inv(cams[0][1]) @ cams[0][2])[:2]
    print(f"camera B: ground z = {ground:+.3f} (tilt {np.degrees(np.arctan(np.hypot(*plane[:2]))):.2f} deg), "
          f"camera height {-ground + 0:.2f} m above LiDAR origin; {len(tracks)} labelled tracks")
    return images, times, cams, tracks, camera_xy


def load_spd():
    fids, times, cams, tracks, camera_xy = vd.load_sequence()
    return [os.path.join(vd.DATA, "image", f"{f}.jpg") for f in fids], times, cams, tracks, camera_xy


def summarise(name, rows):
    r = rows["heading+length"]
    return {"camera": name, "n": len(r), "position": float(np.median(r[:, 0])), "orientation10": float(np.mean(r[:, 1] < 10)),
            "heading20": float(np.mean(r[:, 2] < 20)), "speed": float(np.median(r[:, 3])),
            "bins": {f"{lo}-{hi}": (int(((r[:, 5] >= lo) & (r[:, 5] < hi)).sum()),
                                    float(np.median(r[(r[:, 5] >= lo) & (r[:, 5] < hi), 0])) if ((r[:, 5] >= lo) & (r[:, 5] < hi)).any() else None)
                     for lo, hi in ((0, 40), (40, 60), (60, 80), (80, 200))}}


def classical(images, times, cams, tracks, camera_xy):
    frames = [cv2.imread(p) for p in images]
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    _, final, _ = vc.run(frames, times, bg, cams[0], (1, 2, 3))
    smooth = vc.smoothed_output(final, len(frames))
    gt = vc.moving_ground_truth(times, tracks, camera_xy)
    return vc.evaluate(smooth, gt), smooth, frames


def cuboid(cam, fp, h=1.5):
    X = np.vstack([np.c_[fp, np.zeros(4)], np.c_[fp, np.full(4, h)]])
    return vd.project(cam, X)


def examples_figure(path, data, picks):
    fig = plt.figure(figsize=(18, 4.3 * len(picks)))
    for r, (name, (images, times, cams, tracks, camera_xy, estimates, smooth), k) in enumerate(picks):
        img = cv2.cvtColor(cv2.imread(images[k]), cv2.COLOR_BGR2RGB)
        ax_l = fig.add_subplot(len(picks), 3, 3 * r + 1)
        ax_c = fig.add_subplot(len(picks), 3, 3 * r + 2)
        ax_b = fig.add_subplot(len(picks), 3, 3 * r + 3)
        for ax in (ax_l, ax_c):
            ax.imshow(img)
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            ax.axis("off")
        cam = cams[k]
        gts = [s for seq in tracks.values() for s in seq if s["k"] == k]
        for s in gts:
            fp = vd.lifting.compute_rectangle_points(s["pos"], s["yaw"], s["dims"][1], s["dims"][0])[0]
            uv = cuboid(cam, fp, s["dims"][2])
            for ax in (ax_l, ax_c):
                for a, b in vd.EDGES:
                    ax.plot(*uv[[a, b]].T, color="white", linewidth=0.8, linestyle=(0, (3, 2)))
            ax_b.fill(*fp.T, facecolor="none", edgecolor=MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
        n_l = 0
        for tid, est in estimates.items():
            for i, center, yaw, fp in est:
                if tracks[tid][i]["k"] != k:
                    continue
                n_l += 1
                uv = cuboid(cam, fp)
                for a, b in vd.EDGES:
                    ax_l.plot(*uv[[a, b]].T, color=EST, linewidth=1.6)
                ax_b.fill(*fp.T, facecolor="none", edgecolor=EST, linewidth=1.6)
        n_c = 0
        for tid, o in smooth[k]:
            fp = vd.lifting.compute_rectangle_points(o["center"], o["yaw"], vd.CAR_SIZE[1], vd.CAR_SIZE[0])[0]
            uv = cuboid(cam, fp)
            n_c += 1
            for a, b in vd.EDGES:
                ax_c.plot(*uv[[a, b]].T, color=ALT, linewidth=1.6)
            ax_b.fill(*fp.T, facecolor="none", edgecolor=ALT, linewidth=1.3)
        ax_l.set_title(f"{name}, frame {k}: labelled 2D boxes -> IPM lifting ({n_l} moving cars)", fontsize=9.5,
                       loc="left", color=INK)
        ax_c.set_title(f"fully classical: motion -> tracking -> IPM lifting ({n_c} objects)", fontsize=9.5,
                       loc="left", color=INK)
        ax_b.plot(*camera_xy, "k^", markersize=7)
        ax_b.set_aspect("equal")
        ax_b.set_title("bird's-eye view (m): blue/orange = estimates, dashed = labels", fontsize=9.5, loc="left",
                       color=INK)
        ax_b.tick_params(labelsize=8, colors=MUTED)
        pts = np.array([s["pos"] for s in gts] + [camera_xy])
        lo, hi = pts.min(0) - 5, pts.max(0) + 5
        ax_b.set_xlim(lo[0], hi[0])
        ax_b.set_ylim(lo[1], hi[1])
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def errors_figure(path, summaries, classical_scores):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    names = [s["camera"] for s in summaries]
    panels = (("position", "median position error (m)", 1, "labelled 2D boxes"),
              ("orientation10", "orientation < 10° (%)", 100, "labelled 2D boxes"),
              ("speed", "median speed error (m/s)", 1, "labelled 2D boxes"))
    for ax, (key, label, scale, src) in zip(axes, panels):
        for j, (which, color, tag) in enumerate(((lambda s: s, EST, "as calibrated"),
                                                 (lambda s: s["tilted"], "#9fb9d8", "tilt corrected (labels)"))):
            v = [which(s)[key] * scale for s in summaries]
            bars = ax.bar(np.arange(len(v)) + (j - 0.5) * 0.36, v, 0.34, color=color, zorder=2, label=tag)
            for b, x in zip(bars, v):
                ax.annotate(f"{x:.2f}" if scale == 1 else f"{x:.0f}", (b.get_x() + b.get_width() / 2, x),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(names)), [f"{n}\n(tilt {s['tilt']:+.2f}°)" for n, s in zip(names, summaries)],
                      fontsize=8.5)
        if key == "position":
            ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"{label}\n({src})", fontsize=10, loc="left", color=INK)
        vd.style(ax)
    ax = axes[3]
    v = [100 * c["coverage"] for c in classical_scores]
    bars = ax.bar(range(len(v)), v, 0.6, color=[ALT, ALT], zorder=2)
    for b, x, c in zip(bars, v, classical_scores):
        ax.annotate(f"{x:.0f}%\n{c['position']:.2f} m", (b.get_x() + b.get_width() / 2, x), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(v)), names, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_title("moving cars found and position\n(fully classical)", fontsize=10, loc="left", color=INK)
    vd.style(ax)
    fig.suptitle("Same method, no parameter changed: camera A (developed on) vs camera B (new intersection)",
                 fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    data, summaries, classical_scores = {}, [], []
    for name, loader in (("A: yizhuang09", load_spd), ("B: yizhuang08 (new)", load_coop)):
        images, times, cams, tracks, camera_xy = loader()
        rows, estimates = vd.evaluate(times, cams, tracks, camera_xy)
        s = summarise(name, rows)
        delta = label_tilt(cams, tracks)
        s_tilt = summarise(name, vd.evaluate(times, tilt_cameras(cams, delta), tracks, camera_xy)[0])
        s["tilt"], s["tilted"] = float(np.degrees(delta)), s_tilt
        print(f"{name}: with the tilt corrected by {np.degrees(delta):+.2f} deg (from the labels): position "
              f"{s_tilt['position']:.2f} m, orientation<10 {s_tilt['orientation10']:.0%}, speed error {s_tilt['speed']:.2f} m/s"
              f" | by distance {s_tilt['bins']}")
        c, smooth, _ = classical(images, times, cams, tracks, camera_xy)
        summaries.append(s)
        classical_scores.append(c)
        data[name] = (images, times, cams, tracks, camera_xy, estimates["heading+length"], smooth)
        print(f"{name}: labelled boxes n={s['n']}, position {s['position']:.2f} m, orientation<10 {s['orientation10']:.0%}, "
              f"heading<20 {s['heading20']:.0%}, speed error {s['speed']:.2f} m/s | by distance {s['bins']}")
        print(f"{name}: fully classical: found {c['coverage']:.0%}, position {c['position']:.2f} m, "
              f"orientation<10 {c['orientation']:.0%}, heading<20 {c['heading']:.0%}, speed error {c['speed']:.2f} m/s")
    errors_figure(os.path.join(vd.OUT_DIR, "coop_errors.png"), summaries, classical_scores)
    a, b = list(data)
    picks = [(b, data[b], 20), (b, data[b], 45), (b, data[b], 70), (a, data[a], 150)]
    examples_figure(os.path.join(vd.OUT_DIR, "coop_examples.png"), data, picks)


if __name__ == "__main__":
    main()
