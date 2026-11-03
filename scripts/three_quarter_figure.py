"""Three-quarter figure: cars of intersection A seen half-turned, so that one window shows the side and the front or the
rear. Per car: (a) the body triangulation in the frame; (b) the same window's solved vertices from above, in the car's
frame, over the car's LiDAR footprint (evaluation only) - the vertices of the side and of the front/rear lie on two
faces at an angle, which a side profile stretched over the width could not produce; (c) the window's triangles from a
viewpoint the camera never had; (d) the model fused over the whole video from the same viewpoint.
Needs figures/models_A.pkl (scripts/capture_models.py) and the window dumps of the article's run.
Usage: DUMPS=docs/ba_objects/A python scripts/three_quarter_figure.py [label:frame ...]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DUMPS", "docs/ba_objects/A")
import cars_figure as cf  # noqa: E402  (also switches to the 3dseg folder)
import method_figure as mf  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pickle  # noqa: E402

import v2x_seq_shape as vs  # noqa: E402
import whole_scene_cars as w  # noqa: E402

PICK = ["004002:95", "003890:77", "004008:157"]
SIZE = (420, 280)


def view_angle(A, b, C):
    """Angle (deg, 0..90) between the car's heading and the ray from the camera centre C to the car's centre."""
    h = A.T @ np.array([1.0, 0.0])
    h /= np.linalg.norm(h)
    r = A.T @ (-np.asarray(b)) - C[:2]
    r /= np.linalg.norm(r)
    return float(np.degrees(np.arcsin(min(1.0, abs(h[0] * r[1] - h[1] * r[0])))))


def track_solved(o, A0, b0, cache=os.environ.get("BA_CACHE", "docs/ba_cache")):
    """The window's vertices as the track solve placed them (car_ba.py, cached by the article's run): the cached
    window whose vertex count matches, whose rotation is the model's of this frame and whose translation is nearest
    (the model is re-centred on its symmetry plane after the solve, a shift of some decimetres)."""
    import glob
    best = (1.0, None)
    for f in sorted(glob.glob(os.path.join(cache, os.path.basename(os.path.normpath(mf.DUMPS)) + "_*_e.pkl"))):
        data = pickle.load(open(f, "rb"))
        for X, pose in zip(data["X"], data["pose"]):
            A = w.rot(pose[0]).T
            if len(X) == len(o["X"]) and np.allclose(A, A0, atol=1e-5):
                dist = np.linalg.norm(-A @ pose[1:] - b0)
                best = min(best, (dist, X), key=lambda t: t[0])
    if best[1] is None:
        raise LookupError("no cached track solve for this window")
    return best[1]


def window_triangles(o, A0, b0):
    """The window's body triangles as the fusion uses them - track-solved vertices, without the outline (the tracked
    mask edge, no surface point), slivers and outliers - and their colours."""
    import car_ba
    on = o["body"][o["tri"]].all(1) & ~car_ba.outline(o)[o["tri"]].any(1)
    T = o["tri"][on]
    col = o["tri_bgr"][on].astype(float)
    X = o["X"] = track_solved(o, A0, b0)
    area = 0.5 * np.linalg.norm(np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), axis=1)
    T, col = T[area < 1.5], col[area < 1.5]
    Y = np.c_[X[:, :2] @ A0.T + b0, X[:, 2]]
    med = np.median(Y[o["body"], :2], axis=0)
    inl = np.all(np.abs(Y[:, :2] - med) < [3.2, 2.0], axis=1) & (Y[:, 2] > -0.3) & (Y[:, 2] < 3.0)
    keep = inl[T].all(1)
    return T[keep], col[keep], Y


def top_view(m, T, Y, Ccar, px=55.0):
    """Orthographic view from above in the car frame: the LiDAR footprint (grey), the window's triangles coloured by
    height (painted from low to high) and the direction the camera looks from (arrow)."""
    W_, H_ = SIZE
    img = np.full((H_, W_, 3), 255, np.uint8)
    to_px = lambda P: np.c_[W_ / 2 + P[:, 0] * px, H_ / 2 - P[:, 1] * px]
    L = cf.lidar_points(m)
    for p in to_px(L[:, :2]):
        cv2.rectangle(img, tuple(np.round(p - 1.4).astype(int)), tuple(np.round(p + 1.4).astype(int)),
                      (200, 200, 200), -1)
    z = Y[T].mean(1)[:, 2]
    zc = np.clip(z / 1.8, 0, 1)
    cmap = cv2.applyColorMap(np.round(zc * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS)[:, 0]
    for i in np.argsort(z):
        q = np.round(to_px(Y[T[i], :2]) * 4).astype(np.int32)
        c = tuple(int(v) for v in cmap[i])
        cv2.fillConvexPoly(img, q, c, cv2.LINE_AA, shift=2)
        cv2.polylines(img, [q], True, tuple(int(0.7 * v) for v in c), 1, cv2.LINE_AA, shift=2)
    d = -Ccar[:2] / np.linalg.norm(Ccar[:2])                     # viewing direction at the car, in the car frame
    tip = np.array([W_ / 2, H_ / 2]) - np.array([d[0], -d[1]]) * 0.33 * min(W_, H_)
    tail = tip - np.array([d[0], -d[1]]) * 55
    cv2.arrowedLine(img, tuple(np.round(tail).astype(int)), tuple(np.round(tip).astype(int)), (40, 40, 200), 2,
                    cv2.LINE_AA, tipLength=0.3)
    cv2.putText(img, "camera", tuple(np.round(tail + [-30, -8]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (40, 40, 200), 1, cv2.LINE_AA)
    return img


def main():
    models = pickle.load(open(os.path.join(cf.FIG, "models_A.pkl"), "rb"))
    by_label = {m["row"].get("label"): n for n, m in models.items() if "fused closed: iou" in m["row"]}
    seq = vs.Sequence()
    rows = []
    for spec in sys.argv[1:] or PICK:
        lab, k = spec.split(":")
        k = int(k)
        n = by_label[lab]
        m = models[n]
        uv = dict(m["windows"])[k]
        o = mf.window_object(k, uv)
        A0, b0 = m["poses"][k]
        T, tcol, Y = window_triangles(o, A0, b0)
        cam = seq.cams[k]
        C = -np.linalg.solve(cam[1], cam[2])
        Ccar = A0 @ C[:2] + b0
        ang = view_angle(A0, b0, C)
        # (a) the triangulation on the frame
        x0, y0 = o["uv"][o["body"]].min(0)
        x1, y1 = o["uv"][o["body"]].max(0)
        pad = 0.3 * max(x1 - x0, y1 - y0)
        X0, Y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        X1, Y1 = int(min(seq.shape[1], x1 + pad)), int(min(seq.shape[0], y1 + pad))
        s_ = min(SIZE[0] / (X1 - X0), SIZE[1] / (Y1 - Y0))
        crop = cv2.resize(seq.frames[k][Y0:Y1, X0:X1], None, fx=s_, fy=s_, interpolation=cv2.INTER_CUBIC)
        lines = crop.copy()
        for t in T:
            q = (o["uv"][t] - [X0, Y0]) * s_
            cv2.polylines(lines, [np.round(q * 4).astype(np.int32)], True, (0, 215, 255), 1, cv2.LINE_AA, shift=2)
        crop = cv2.addWeighted(lines, 0.7, crop, 0.3, 0)
        # (b) from above
        top = top_view(m, T, Y, Ccar)
        # (c), (d): a viewpoint on the seen end of the car, on the other side of its axis from the camera
        end = np.sign(Ccar[0]) or 1.0
        side = -np.sign(Ccar[1]) or 1.0
        az = np.arctan2(side * 0.75, end * 1.0)
        eye_car = np.array([[8.6 * np.cos(az), 8.6 * np.sin(az), 3.2]])
        f = {"poses": m["poses"]}
        Kn = np.array([[900.0, 0, SIZE[0] / 2], [0, 900.0, SIZE[1] / 2], [0, 0, 1]])
        camn = cf.look_at(Kn, w.to_world(f, k, eye_car)[0], w.to_world(f, k, np.array([[0.0, 0.0, 0.6]]))[0])
        bg = np.full((SIZE[1], SIZE[0], 3), 255, np.uint8)
        V3 = o["X"][T].reshape(-1, 3)
        F3 = np.arange(len(V3)).reshape(-1, 3)
        one = cf.render_mesh(bg.copy(), camn, V3, F3, np.repeat(tcol, 3, axis=0), edge=0.75)
        fused = cf.render_mesh(bg.copy(), camn, w.to_world(f, k, m["V"]), m["F"], m["vcol"])
        # how much of the car's width the window's own vertices span at its seen end (front or rear 0.8 m)
        Yb = Y[np.unique(T)]
        xe = np.percentile(Yb[:, 0], 97) if end > 0 else np.percentile(Yb[:, 0], 3)
        endv = Yb[np.abs(Yb[:, 0] - xe) < 0.8]
        span = np.percentile(endv[:, 1], 95) - np.percentile(endv[:, 1], 5) if len(endv) > 5 else 0.0
        r = m["row"]
        name = "front" if end > 0 else "rear"
        rows.append(np.hstack([cf.panel(crop, f"car {r['label']}, frame {k}: {ang:.0f} deg to the ray"),
                               cf.panel(top, f"one window from above, over LiDAR"),
                               cf.panel(one, f"one window, new view ({name})"),
                               cf.panel(fused, f"fused model, same view")]))
        print(lab, "frame", k, "angle", round(ang), "triangles", len(T), "end", name,
              "vertices at the end", len(endv), "span across %.2f m" % span, "label width", r.get("label_width"))
    cv2.imwrite(os.path.join(cf.FIG, "three_quarter.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
