"""Figure of the best-reconstructed, well-visible cars of intersection A: the car in the video; our model drawn back
into the same frame; the model from a new viewpoint; the model with the car's LiDAR (evaluation only).
Needs figures/models_A.pkl (scripts/capture_models.py). Usage: python scripts/cars_figure.py [track ...]"""
import os
import pickle
import sys

sys.path.insert(0, os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.chdir(os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
import cv2
import numpy as np

import v2x_seq_shape as vs
import whole_scene_cars as w

LIGHT = np.array([0.3, -0.5, 0.8])


def render_mesh(img, cam, V, F, vcol, light=LIGHT, alpha=1.0, edge=0.0):
    """Painter's rendering of a triangle mesh with per-vertex colours, Lambert-shaded, onto img; returns img."""
    K, R, t = cam
    Xc = V @ R.T + t
    z = Xc[:, 2]
    uv = (Xc @ K.T)[:, :2] / np.maximum(Xc @ K.T, 1e-9)[:, 2:3]
    F = F[(z[F] > 0.1).all(1)]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    lt = np.asarray(light, float) / np.linalg.norm(light)
    col = np.clip(vcol[F].mean(1) * (0.55 + 0.45 * np.abs(n @ lt))[:, None], 0, 255)
    layer, mask = img.copy(), np.zeros(img.shape[:2], np.uint8)
    for i in np.argsort(-z[F].mean(1)):
        pts = np.round(uv[F[i]] * 4).astype(np.int32)
        c = tuple(float(v) for v in col[i])
        cv2.fillConvexPoly(layer, pts, c, cv2.LINE_AA, shift=2)
        cv2.fillConvexPoly(mask, pts, 255, cv2.LINE_AA, shift=2)
        if edge > 0:
            cv2.polylines(layer, [pts], True, tuple(v * edge for v in c), 1, cv2.LINE_AA, shift=2)
    m = (mask > 0)[:, :, None]
    img[:] = np.where(m, alpha * layer + (1 - alpha) * img, img).astype(np.uint8)
    return img


def look_at(K, eye, target):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return K, R, -R @ eye


def candidate_windows(m, others, shape, margin=20):
    """Windows where the car's body is inside the image and not overlapped by another car's body."""
    H, W = shape
    out = []
    for k, uv in m["windows"]:
        if not len(uv):
            continue
        x0, y0 = uv.min(0)
        x1, y1 = uv.max(0)
        if x0 < margin or y0 < margin or x1 > W - margin or y1 > H - margin:
            continue
        clash = False
        for o in others:
            for k2, uv2 in o["windows"]:
                if k2 == k and len(uv2):
                    a0, b0 = uv2.min(0)
                    a1, b1 = uv2.max(0)
                    if min(x1, a1) + 15 > max(x0, a0) and min(y1, b1) + 15 > max(y0, b0):
                        clash = True
        if not clash:
            out.append((k, uv))
    return out


def image_fit(seq, m, k, uv):
    """IoU between our model projected into frame k and the car's full silhouette there (both as image masks)."""
    sil = w.full_silhouette(seq, k, {"uv": uv, "body": np.ones(len(uv), bool)})
    if sil is None:
        return 0.0, 0.0
    s, x0, y0 = sil
    full = np.zeros(seq.shape, bool)
    full[y0:y0 + s.shape[0], x0:x0 + s.shape[1]] = s
    q, _ = vs.project(seq.cams[k], w.to_world({"poses": m["poses"]}, k, m["V"]))
    mod = np.zeros(seq.shape, np.uint8)
    cv2.fillConvexPoly(mod, cv2.convexHull(np.round(q).astype(np.int32)), 1)
    mod = mod > 0
    return float((mod & full).sum() / max((mod | full).sum(), 1)), float(full.sum())


def lidar_points(m):
    """The car's LiDAR top surface placed on the model like in the evaluation (centred footprints, best heading)."""
    Lt, lx, ly = m["lidar"]
    ix, iy = np.nonzero(Lt > w.GROUND_H)
    P = np.c_[lx[ix], ly[iy], Lt[ix, iy]]
    P[:, :2] -= (P[:, :2].max(0) + P[:, :2].min(0)) / 2
    jx, jy = np.nonzero(m["H"] > w.GROUND_H)
    c = np.array([(m["xs"][jx].max() + m["xs"][jx].min()) / 2, (m["ys"][jy].max() + m["ys"][jy].min()) / 2])
    P[:, :2] += c
    if m["row"].get("flip", 1) < 0:
        P[:, 0] = 2 * c[0] - P[:, 0]
    return P


def panel(img, title, size=(420, 280), strip=26):
    h, w_ = img.shape[:2]
    s = min(size[0] / w_, size[1] / h)
    img = cv2.resize(img, (int(w_ * s), int(h * s)), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    out = np.full((size[1] + strip, size[0], 3), 255, np.uint8)
    y, x = strip + (size[1] - img.shape[0]) // 2, (size[0] - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.putText(out, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    out[:, -3:] = 255
    return out


def main():
    models = pickle.load(open(os.path.join(FIG, os.environ.get("MODELS_PKL_NAME", "models_A.pkl")), "rb"))
    good = [n for n, m in models.items() if "fused closed: iou" in m["row"] and m["vcol"] is not None]
    seq = vs.Sequence()
    pick = {}
    for n in good:
        m = models[n]
        cands = candidate_windows(m, [o for j, o in models.items() if j != n], seq.shape)
        scored = [(image_fit(seq, m, k, uv), k) for k, uv in cands]
        if scored:
            (fit, area), k = max(scored, key=lambda t: t[0][0] * np.sqrt(t[0][1]))
            pick[n] = (k, fit)
            print(f"track {n}: LiDAR IoU {m['row']['fused closed: iou']:.2f}, image fit {fit:.2f} in frame {k}")
    by_label = {models[n]["row"].get("label"): n for n in pick}
    tracks = [by_label[a] if a in by_label else int(a) for a in sys.argv[1:]] or \
        sorted(pick, key=lambda n: -models[n]["row"]["fused closed: iou"] * pick[n][1])[:4]
    rows = []
    for n in tracks:
        m = models[n]
        k = pick[n][0]
        f = {"poses": m["poses"]}
        Vw = w.to_world(f, k, m["V"])
        cam = seq.cams[k]
        q, _ = vs.project(cam, Vw)
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        pad = 0.35 * max(x1 - x0, y1 - y0)
        X0, Y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        X1, Y1 = int(min(seq.shape[1], x1 + pad)), int(min(seq.shape[0], y1 + pad))
        frame = seq.frames[k]
        crop = frame[Y0:Y1, X0:X1].copy()
        over = render_mesh(frame.copy(), cam, Vw, m["F"], m["vcol"], alpha=0.75)[Y0:Y1, X0:X1]
        # a new viewpoint: 40 deg around the car from the camera's side, lower and closer
        C = -np.linalg.solve(cam[1], cam[2])
        A, b = m["poses"][k]
        cc = A @ C[:2] + b                                   # camera position in the car frame
        az = np.arctan2(cc[1], cc[0]) + np.radians(40)
        eye = np.array([8.6 * np.cos(az), 8.6 * np.sin(az), 3.4])
        Kn = np.array([[900.0, 0, 320], [0, 900.0, 210], [0, 0, 1]])
        camn = look_at(Kn, eye, np.array([0.0, 0.0, 0.6]))
        bg = np.full((420, 640, 3), 255, np.uint8)
        novel = render_mesh(bg.copy(), camn, m["V"], m["F"], m["vcol"], edge=0.0)
        P = lidar_points(m)
        withl = render_mesh(bg.copy(), camn, m["V"], m["F"], np.full_like(m["vcol"], 205.0), alpha=0.85)
        qp, zp = vs.project(camn, P)
        for (u, v), z_ in zip(qp, zp):
            if z_ > 0:
                cv2.circle(withl, (int(u), int(v)), 1, (40, 40, 210), -1, cv2.LINE_AA)
        r = m["row"]
        cap = (f"car {r['label']}, frame {k}, {r['range_m']:.0f} m; IoU {r['fused closed: iou']:.2f}, "
               f"L {r['length']:.1f} / {r['label_length']:.1f} m")
        print(cap)
        rows.append(np.hstack([panel(crop, f"video, car {r['label']}"), panel(over, "our model in the same frame"),
                               panel(novel, "our model, new viewpoint"),
                               panel(withl, f"with its LiDAR (red), IoU {r['fused closed: iou']:.2f}")]))
    out = np.vstack(rows)
    cv2.imwrite(os.path.join(FIG, os.environ.get("OUT_NAME", "cars_best.png")), out)
    print(out.shape)


if __name__ == "__main__":
    main()
