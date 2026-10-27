"""
3D car detection on the roadside cameras: initialise each car once from the dataset's label file, then follow it
with our own classical detector, and compare how the orientation is chosen.

Detector (no neural networks): median background -> motion mask -> blobs -> contact-line cut of the reflection
(v2x_seq_classical); each car's blob is the one that best overlaps the projection of its predicted cuboid, clipped
to that prediction (merged cars). From the blob's box: IPM quadrangle with sides through the vertical vanishing
point (lifting.vp_bounding_quadrangle). Three ways to orient the car, each run as its own tracker:
  motion        : heading = direction of motion of the car's own estimated positions (+-0.5 s); kept while it stands;
  article       : the <= 4 closed-form inscribed-rectangle orientations (Efficient3D, angle = theta +- psi, and the
                  length/width swap; lifting.face_candidates), the one closest to the previous heading;
  article+motion: the closed-form orientation closest to the direction of motion while the car moves, closest to the
                  previous heading while it stands.
Position in all three: the three-line solve with free length (lifting.solve_length, moving.tex); no blob -> the
car coasts on its predicted position.

Initialisation from the label file: size, orientation and position of a car's first labelled frame and the position
of its second (initial velocity). A standing car (slower than MIN_SPEED) is part of the median background and has no
motion blob: it keeps its place.

Calibration: as given, and with the camera tilt fitted to the initialisation labels only (the first labelled frame
of every car: its ground position must be the IPM of its 2D box's ground point). Label-free calibration from the
cars' 3D trajectories (parallel tracks, constant speed, constant length) was tried and does not resolve 0.1 deg on
these short clips (25 s and 8 s).

main() runs the assisted variant (run_assisted): each car's detector blob is found through its labelled 2D box, so
detection/association failures (merged blobs, cars waiting at the light inside the median background) do not hide
the orientation question; run() is the fully automatic tracker.

Writes docs/init_track_errors.png and docs/init_track_examples.png.
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lifting
import v2x_coop_eval as ce
import v2x_seq_classical as vc
import v2x_seq_demo as vd
from hybrid_tracking import nearest_candidate

METHODS = ("motion", "article", "article+motion")
COLORS = {"motion": "#eb6834", "article": "#2a78d6", "article+motion": "#1d4f8f"}
INK, MUTED = "#1f1f1e", "#8a8984"
MAX_COAST = 10
MIN_SPEED = 1.0          # m/s: below this the car is standing and keeps its heading
JUMP_GATE = 3.0          # m: a measured position farther than this (plus half the expected motion) from the prediction is rejected


def box_iou(a, b):
    """IoU of two axis-aligned boxes (x0, y0, x1, y1); 0.0 if they do not overlap."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x0 >= x1 or y0 >= y1:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def clip_to_prediction(blob_box, pred_box, grow=0.25):
    """Intersection of the blob's box with the predicted box enlarged by `grow` on every side; None if empty."""
    w, h = pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]
    x0, y0 = max(blob_box[0], pred_box[0] - grow * w), max(blob_box[1], pred_box[1] - grow * h)
    x1, y1 = min(blob_box[2], pred_box[2] + grow * w), min(blob_box[3], pred_box[3] + grow * h)
    return None if x0 >= x1 or y0 >= y1 else (float(x0), float(y0), float(x1), float(y1))


def quadrangle(cam, box):
    K, R, t = cam
    x0, y0, x1, y1 = box
    return lifting.vp_bounding_quadrangle(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                                          lifting.plane_homography(K, R, t, 0.0), lifting.vertical_vanishing_point(K, R))


def cuboid_box(cam, center, yaw, l, w, h):
    fp = lifting.compute_rectangle_points(center, yaw, w, l)[0]
    uv = vd.project(cam, np.vstack([np.c_[fp, np.zeros(4)], np.c_[fp, np.full(4, h)]]))
    return (*uv.min(0), *uv.max(0)), fp


def init_tilt(cams, tracks):
    """Tilt correction from the initialisation labels only (first non-truncated frame of every car)."""
    firsts = {tid: next((s for s in q if not s["truncated"]), None) for tid, q in tracks.items()}
    firsts = {tid: s for tid, s in firsts.items() if s is not None and not (s is tracks[tid][-1])}
    return ce.label_tilt(cams, {tid: [s] for tid, s in firsts.items()}), firsts


SPLIT = True             # split merged cars by optical-flow clusters (flow_split_detector)


def detections(frames, bg):
    """Per frame: list of (whole blob box, [boxes of its flow-cluster parts]) with the contact-line cut. The parts
    are used only where one blob is claimed by two or more cars (if SPLIT)."""
    import flow_split_detector as fsd
    out, prev = [], None
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        flow = fsd.half_res_flow(prev, gray) if (prev is not None and SPLIT) else None
        items = []
        for b in vc.blobs(vc.foreground(f, bg)):
            parts = fsd.split_by_flow_clusters(b, flow)[0] if flow is not None else [b]
            items.append((vc.cut_reflection(b, gray)["box"],
                          [vc.cut_reflection(p, gray)["box"] for p in parts] if len(parts) > 1 else []))
        out.append(items)
        prev = gray
    return out


def resolve(items, claims):
    """Candidate boxes of a frame: a blob claimed by >= 2 cars is replaced by its flow-cluster parts."""
    boxes = []
    for j, (whole, parts) in enumerate(items):
        boxes += parts if (claims.get(j, 0) >= 2 and parts) else [whole]
    return boxes


class Car:
    def __init__(self, s, method, s2=None):
        """Initialised from the label file: size, orientation and position of the first labelled frame, and the
        position of the second one (gives the initial velocity)."""
        self.l, self.w, self.h = s["dims"]
        self.method, self.yaw = method, float(s["yaw"])
        self.hist = [(s["k"], np.array(s["pos"], float))]       # (frame, centre) of measured frames
        if s2 is not None:
            self.hist.append((s2["k"], np.array(s2["pos"], float)))
        self.center, self.coast = self.hist[-1][1].copy(), 0

    def velocity(self, times, k, window=5):
        pts = [(kk, c) for kk, c in self.hist if k - kk <= window]
        if len(pts) < 2:
            pts = self.hist[-2:]
        if len(pts) < 2:
            return np.zeros(2)
        t = times[[kk for kk, _ in pts]]
        P = np.array([c for _, c in pts])
        return np.array([np.polyfit(t, P[:, i], 1)[0] for i in (0, 1)])

    def predict(self, times, k):
        """Constant velocity from the last measured position (not from the coasted one)."""
        last_k, last_c = self.hist[-1]
        return last_c + self.velocity(times, last_k) * (times[k] - times[last_k])

    def update(self, cam, box, times, k, predicted):
        q = quadrangle(cam, box)
        if q is None:
            return False
        v = self.velocity(times, k - 1)
        moving = np.linalg.norm(v) >= MIN_SPEED
        motion_dir = float(np.arctan2(v[1], v[0])) if moving else self.yaw
        if self.method == "motion":
            yaw = motion_dir
        else:
            ref = motion_dir if (self.method == "article+motion" and moving) else self.yaw
            pick = nearest_candidate(lifting.face_candidates(q, self.l, self.w), ref)
            yaw = pick[1] if pick is not None else ref
        solved = lifting.solve_length(q, yaw, self.w)
        if solved is not None and 0.6 * self.l < solved[1] < 1.6 * self.l:
            center = solved[0]
        else:
            center = vd.solve_sides(q, yaw, self.l, self.w)
            if center is None:
                return False
        dt = times[k] - times[self.hist[-1][0]]
        if np.linalg.norm(center - predicted) > JUMP_GATE + 0.5 * np.linalg.norm(v) * dt:
            return False                                   # implausible jump: coast instead
        self.center, self.yaw, self.coast = center, yaw, 0
        self.hist.append((k, center))
        return True


def run(name, loader, calibrated):
    images, times, cams, tracks, camera_xy = loader()
    delta, firsts = init_tilt(cams, tracks)
    if calibrated:
        cams = ce.tilt_cameras(cams, delta)
    frames = [cv2.imread(p) for p in images]
    bg = np.median(np.stack(frames[::2]), axis=0).astype(np.uint8)
    dets = detections(frames, bg)
    label = {(tid, s["k"]): s for tid, q in tracks.items() for s in q}
    rows, shown = [], {}
    for method in METHODS:
        cars = {}
        for k in range(len(frames)):
            for tid, s in firsts.items():
                q = tracks[tid]
                if s["k"] == k:
                    cars[tid] = Car(s, method, q[[x is s for x in q].index(True) + 1])
            # predicted boxes, then greedy one-to-one assignment of blobs by IoU
            preds = {}
            for tid, car in cars.items():
                if car.coast > MAX_COAST or k <= car.hist[min(1, len(car.hist) - 1)][0]:
                    continue
                c = car.predict(times, k)
                preds[tid] = (c, cuboid_box(cams[k], c, car.yaw, car.l, car.w, car.h)[0])
            def match_cost(p, b):        # bottom-centre distance in predicted-box widths; must overlap
                if box_iou(p, b) <= 0:
                    return np.inf
                return np.hypot((p[0] + p[2] - b[0] - b[2]) / 2, p[3] - b[3]) / max(p[2] - p[0], 8)
            claims = {}
            for tid, p in preds.items():
                costs = [match_cost(p[1], w) for w, _ in dets[k]]
                if costs and np.min(costs) <= 0.8:
                    claims[int(np.argmin(costs))] = claims.get(int(np.argmin(costs)), 0) + 1
            boxes_k = resolve(dets[k], claims)
            clipped = {(tid, j): clip_to_prediction(b, p[1]) for tid, p in preds.items() for j, b in enumerate(boxes_k)}
            pairs = sorted((match_cost(preds[tid][1], c), tid, j) for (tid, j), c in clipped.items() if c is not None)
            used_t, used_b = set(), set()
            for cost, tid, j in pairs:
                if cost > 0.8 or tid in used_t or j in used_b:
                    continue
                box = clipped[(tid, j)]
                if cars[tid].update(cams[k], box, times, k, preds[tid][0]):
                    used_t.add(tid)
                    used_b.add(j)
            for tid in preds:
                if tid not in used_t:      # no blob: coast; a standing car belongs to the background and stays put
                    car = cars[tid]
                    standing = np.linalg.norm(car.velocity(times, car.hist[-1][0])) < MIN_SPEED
                    car.center, car.coast = preds[tid][0], (0 if standing else car.coast + 1)
            for tid in preds:
                s = label.get((tid, k))
                car = cars[tid]
                if s is None or s["truncated"] or car.coast > MAX_COAST:
                    continue
                if k <= car.hist[min(1, len(car.hist) - 1)][0] and not (tid in used_t):
                    continue
                rows.append({"method": method, "k": k, "tid": tid, "measured": tid in used_t,
                             "dist": float(np.linalg.norm(s["pos"] - camera_xy)),
                             "pos": float(np.linalg.norm(car.center - s["pos"])),
                             "orient": vd.yaw_error(car.yaw, s["yaw"]),
                             "speed": float(np.linalg.norm(car.velocity(times, k)))})
                shown.setdefault((method, k), []).append((tid, car.center.copy(), car.yaw, car.l, car.w, car.h))
    total = sum(1 for q in tracks.values() for s in q if not s["truncated"]) - 2 * len(firsts)
    return {"rows": rows, "shown": shown, "ctx": (images, cams, tracks, camera_xy), "delta": float(np.degrees(delta)),
            "total": total}


def run_assisted(name, loader, calibrated):
    """Same detector boxes and the same three orientation rules, but each car's blob is found through its labelled 2D
    box (oracle association), so that detection/association failures do not hide the orientation question. After the
    initialisation nothing else is taken from the labels: heading history and positions are the method's own."""
    images, times, cams, tracks, camera_xy = loader()
    delta, firsts = init_tilt(cams, tracks)
    if calibrated:
        cams = ce.tilt_cameras(cams, delta)
    frames = [cv2.imread(p) for p in images]
    bg = np.median(np.stack(frames[::2]), axis=0).astype(np.uint8)
    dets = detections(frames, bg)
    rows, shown, claims_cache = [], {}, {}
    for method in METHODS:
        for tid, s0 in firsts.items():
            q = tracks[tid]
            i0 = [x is s0 for x in q].index(True)
            car = Car(s0, method, q[i0 + 1])
            for s in q[i0 + 2:]:
                k = s["k"]
                if k not in claims_cache:
                    others = [x["box"] for qq in tracks.values() for x in qq if x["k"] == k]
                    claims_cache[k] = resolve(dets[k], {j: sum(box_iou(w, o) > 0.1 for o in others)
                                                        for j, (w, _) in enumerate(dets[k])})
                cands = [(box_iou(b, s["box"]), b) for b in claims_cache[k]]
                cands = [c for c in cands if c[0] > 0.1]
                measured = False
                if cands:
                    box = clip_to_prediction(max(cands)[1], s["box"], grow=0.1)
                    pred = car.predict(times, k)
                    measured = box is not None and car.update(cams[k], box, times, k, pred)
                if not measured:
                    car.center = car.predict(times, k)
                if s["truncated"]:
                    continue
                rows.append({"method": method, "k": k, "tid": tid, "measured": measured,
                             "dist": float(np.linalg.norm(s["pos"] - camera_xy)),
                             "pos": float(np.linalg.norm(car.center - s["pos"])),
                             "orient": vd.yaw_error(car.yaw, s["yaw"]),
                             "speed": float(np.linalg.norm(car.velocity(times, k)))})
                shown.setdefault((method, k), []).append((tid, car.center.copy(), car.yaw, car.l, car.w, car.h))
    total = sum(1 for tid in firsts for s in tracks[tid] if not s["truncated"]) - 2 * len(firsts)
    return {"rows": rows, "total": total, "delta": float(np.degrees(delta)), "shown": shown,
            "ctx": (images, cams, tracks, camera_xy)}


def summarise(res):
    out = {}
    for m in METHODS:
        r = [x for x in res["rows"] if x["method"] == m]
        meas = [x for x in r if x["measured"]]
        moving = [x for x in r if x["speed"] >= MIN_SPEED]
        standing = [x for x in r if x["speed"] < MIN_SPEED]
        out[m] = {"followed": len(r) / max(res["total"], 1), "measured": len(meas) / max(res["total"], 1),
                  "position": float(np.median([x["pos"] for x in r])),
                  "orient10": float(np.mean([x["orient"] < 10 for x in r])),
                  "orient10_moving": float(np.mean([x["orient"] < 10 for x in moving])) if moving else np.nan,
                  "orient10_standing": float(np.mean([x["orient"] < 10 for x in standing])) if standing else np.nan,
                  "lost": float(np.mean([x["pos"] > 5 for x in r]))}
    return out


def errors_figure(path, table):
    names = list(table)
    keys = (("position", "median position error (m)", 1, "{:.2f}"),
            ("orient10_moving", "orientation < 10°, moving cars (%)", 100, "{:.0f}"),
            ("orient10_standing", "orientation < 10°, standing cars (%)", 100, "{:.0f}"),
            ("lost", "position error > 5 m (%)", 100, "{:.0f}"))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    width = 0.26
    for ax, (key, title, scale, fmt) in zip(axes, keys):
        for j, m in enumerate(METHODS):
            v = [table[n][m][key] * scale for n in names]
            bars = ax.bar(np.arange(len(names)) + (j - 1) * width, v, width * 0.92, color=COLORS[m], zorder=2, label=m)
            for b, x in zip(bars, v):
                if np.isfinite(x):
                    ax.annotate(fmt.format(x), (b.get_x() + b.get_width() / 2, x), xytext=(0, 2),
                                textcoords="offset points", ha="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(names)), names, fontsize=7.5)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        vd.style(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Initialised once from the label file, then boxes from our classical detector: direction of motion vs "
                 "the closed-form orientation (Efficient3D)", fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def examples_figure(path, picks):
    fig = plt.figure(figsize=(18, 4.4 * len(picks)))
    for r, (title, res, k) in enumerate(picks):
        images, cams, tracks, camera_xy = res["ctx"]
        img = cv2.cvtColor(cv2.imread(images[k]), cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(len(picks), 3, 3 * r + 1)
        axb = fig.add_subplot(len(picks), 3, (3 * r + 2, 3 * r + 3))
        ax.imshow(img)
        for q in tracks.values():
            for s in q:
                if s["k"] == k and not s["truncated"]:
                    fp = lifting.compute_rectangle_points(s["pos"], s["yaw"], s["dims"][1], s["dims"][0])[0]
                    axb.fill(*fp.T, facecolor="none", edgecolor=MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
        for m in ("motion", "article"):
            for tid, c, yaw, l, w, h in res["shown"].get((m, k), []):
                _, fp = cuboid_box(cams[k], c, yaw, l, w, h)
                axb.fill(*fp.T, facecolor="none", edgecolor=COLORS[m], linewidth=1.5)
                if m == "article":
                    uv = vd.project(cams[k], np.vstack([np.c_[fp, np.zeros(4)], np.c_[fp, np.full(4, h)]]))
                    for a, b in vd.EDGES:
                        ax.plot(*uv[[a, b]].T, color=COLORS[m], linewidth=1.3)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.axis("off")
        ax.set_title(f"{title}, frame {k}: cuboids with the article's orientation", fontsize=9.5, loc="left", color=INK)
        axb.plot(*camera_xy, "k^")
        pts = np.array([c for _, c, *_ in res["shown"].get(("article", k), [])] + [camera_xy])
        lo, hi = pts.min(0) - 5, pts.max(0) + 5
        axb.set_xlim(lo[0], hi[0])
        axb.set_ylim(lo[1], hi[1])
        axb.set_aspect("equal")
        axb.tick_params(labelsize=8, colors=MUTED)
        axb.set_title("bird's-eye view (m): orange = direction of motion only, blue = article's orientation, dashed = label",
                      fontsize=9.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def main():
    table, picks = {}, []
    for name, loader in (("A", ce.load_spd), ("B", ce.load_coop)):
        for calibrated in (False, True):
            res = run_assisted(name, loader, calibrated)
            label = f"{name}, {'tilt from init' if calibrated else 'as given'}"
            table[label] = summarise(res)
            for m, s in table[label].items():
                print(f"{label:22s} {m:15s} followed {s['followed']:.0%} (measured {s['measured']:.0%}), position "
                      f"{s['position']:.2f} m, orient<10 {s['orient10']:.0%} (moving {s['orient10_moving']:.0%}, "
                      f"standing {s['orient10_standing']:.0%}), lost {s['lost']:.0%}"
                      + (f" | tilt {res['delta']:+.2f} deg" if calibrated else ""), flush=True)
            if calibrated:
                ks = sorted({k for (m, k) in res["shown"] if m == "article"},
                            key=lambda k: -len(res["shown"][("article", k)]))
                picks.append((f"{name} (tilt {res['delta']:+.2f}°)", res, ks[len(ks) // 3]))
    errors_figure(os.path.join(vd.OUT_DIR, "init_track_errors.png"), table)
    examples_figure(os.path.join(vd.OUT_DIR, "init_track_examples.png"), picks)


if __name__ == "__main__":
    main()
