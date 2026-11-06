"""Method figure: for a few well-reconstructed cars of intersection A - the body triangulation in one frame, the same
triangles in 3D after the window's solve, and the model fused over the whole video, both from the same new viewpoint.
Needs figures/models_A.pkl (scripts/capture_models.py). Usage: python scripts/method_figure.py [track ...]"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cars_figure as cf  # noqa: E402  (also switches to the 3dseg folder)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import v2x_seq_shape as vs  # noqa: E402
import whole_scene_cars as w  # noqa: E402

DUMPS = os.environ.get("DUMPS", "docs/whole_scene_objects_dense")


def window_object(k, uv):
    """The dumped object of window k whose body pixels are uv."""
    path = os.path.join(DUMPS, f"{k:04d}.pkl")
    for o in pickle.load(open(path, "rb")) if os.path.exists(path) else []:
        b = o["uv"][o["body"]]
        if len(b) == len(uv) and np.allclose(b, uv):
            return o
    return None


def main():
    models = pickle.load(open(os.path.join(cf.FIG, "models_A.pkl"), "rb"))
    by_label = {m["row"].get("label"): n for n, m in models.items()}
    tracks = [by_label[a] if a in by_label else int(a) for a in sys.argv[1:]] or [1, 3]
    seq = vs.Sequence()
    rows = []
    for n in tracks:
        m = models[n]
        cands = cf.candidate_windows(m, [o for j, o in models.items() if j != n], seq.shape)
        # the best-fitting window that has a dump (not every window of the run is in DUMPS)
        for (fit, _), k, uv in sorted(((cf.image_fit(seq, m, k, uv), k, uv) for k, uv in cands),
                                      key=lambda t: -t[0][0] * np.sqrt(t[0][1])):
            o = window_object(k, uv)
            if o is not None:
                break
        T = o["tri"][o["body"][o["tri"]].all(1)]
        A_ = 0.5 * np.linalg.norm(np.cross(o["X"][T[:, 1]] - o["X"][T[:, 0]], o["X"][T[:, 2]] - o["X"][T[:, 0]]), axis=1)
        T = T[A_ < 1.5]                                       # long slivers across the outline are no surface
        tcol = o["tri_bgr"][o["body"][o["tri"]].all(1)][A_ < 1.5].astype(float)
        # the window's outliers that the fusion drops too: farther than the car-size box from the body's median
        # (+-3.2 m along the heading, +-2 m across it; whole_scene_cars.track_in_car_frame)
        A0, b0 = m["poses"][k]
        Y = np.c_[o["X"][:, :2] @ A0.T + b0, o["X"][:, 2]]
        med = np.median(Y[o["body"], :2], axis=0)
        inl = np.all(np.abs(Y[:, :2] - med) < [3.2, 2.0], axis=1) & (Y[:, 2] > -0.3) & (Y[:, 2] < 3.0)
        keep = inl[T].all(1)
        T, tcol = T[keep], tcol[keep]
        # (a) the triangulation on the video frame: drawn thin and half-transparent after upscaling the crop
        x0, y0 = o["uv"][o["body"]].min(0)
        x1, y1 = o["uv"][o["body"]].max(0)
        pad = 0.3 * max(x1 - x0, y1 - y0)
        X0, Y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        X1, Y1 = int(min(seq.shape[1], x1 + pad)), int(min(seq.shape[0], y1 + pad))
        s_ = min(420 / (X1 - X0), 280 / (Y1 - Y0))
        crop = cv2.resize(seq.frames[k][Y0:Y1, X0:X1], None, fx=s_, fy=s_, interpolation=cv2.INTER_CUBIC)
        lines = crop.copy()
        for t in T:
            q = (o["uv"][t] - [X0, Y0]) * s_
            cv2.polylines(lines, [np.round(q * 4).astype(np.int32)], True, (0, 215, 255), 1, cv2.LINE_AA, shift=2)
        crop = cv2.addWeighted(lines, 0.7, crop, 0.3, 0)
        # the same new viewpoint for the window's triangles and the fused model (world coordinates of frame k)
        cam = seq.cams[k]
        C = -np.linalg.solve(cam[1], cam[2])
        A, b = m["poses"][k]
        cc = A @ C[:2] + b
        az = np.arctan2(cc[1], cc[0]) + np.radians(40)
        eye_car = np.array([[8.6 * np.cos(az), 8.6 * np.sin(az), 3.4]])
        target_car = np.array([[0.0, 0.0, 0.6]])
        f = {"poses": m["poses"]}
        Kn = np.array([[900.0, 0, 320], [0, 900.0, 210], [0, 0, 1]])
        camn = cf.look_at(Kn, w.to_world(f, k, eye_car)[0], w.to_world(f, k, target_car)[0])
        bg = np.full((420, 640, 3), 255, np.uint8)
        V3 = o["X"][T].reshape(-1, 3)
        F3 = np.arange(len(V3)).reshape(-1, 3)
        one = cf.render_mesh(bg.copy(), camn, V3, F3, np.repeat(tcol, 3, axis=0), edge=0.75)
        fused = cf.render_mesh(bg.copy(), camn, w.to_world(f, k, m["V"]), m["F"], m["vcol"])
        r = m["row"]
        rows.append(np.hstack([cf.panel(crop, f"car {r['label']}: body triangulation, frame {k}"),
                               cf.panel(one, f"one window ({len(T)} triangles), new view"),
                               cf.panel(fused, f"fused over {r['windows']} windows, same view")]))
        print(n, r["label"], "frame", k, "fit", round(fit, 2), "triangles", len(T))
    cv2.imwrite(os.path.join(cf.FIG, "method_steps.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
