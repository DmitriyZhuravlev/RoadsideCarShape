"""
A 3D world model from a fixed roadside video, with classical methods only (V2X-Seq, intersection A).

  ground plane   : z = 0 of the calibrated camera; its texture is the IPM orthophoto of the median background
                   (10 cm/px);
  road markings  : LSD segments of the median background, mapped to the ground by IPM, kept when painted (brighter
                   than the road beside them);
  right angles   : pairs of painted ground lines perpendicular in the IPM that meet (crosswalk stripes x stop lines,
                   lane lines x stop lines); a camera-tilt error bends these angles, so the tilt that makes them right
                   is a self-calibration (compared with the tilt fitted to the labels; at this intersection it fails
                   - slanted crosswalks and arrows - so the model uses the tilt from the initialisation labels);
  vertical poles : LSD segments pointing to the vertical vanishing point; foot = IPM of the lower end, height = where
                   the ray of the upper end meets the vertical through the foot;
  traffic cones and barrels : red regions of the background (HSV), foot by IPM, height like the poles;
  moving objects : median background + motion mask + optical-flow split + contact-line cut + Kalman tracking + the
                   closed-form lifting (Efficient3D candidates, chosen by the direction of motion; moving.tex three-line
                   length) - v2x_seq_classical.run - smoothed offline; cuboids and trajectories.
The model is rendered from the original camera and from new virtual cameras (the ground texture through a plane
homography, cuboids far-to-near with Lambert shading).

Writes docs/world3d_lines.png, docs/world3d_views.png, docs/world3d_map.png and docs/world3d_metrics.json.
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import init_and_track as it
import v2x_coop_eval as ce
import v2x_seq_classical as vc
import v2x_seq_demo as vd

RES = 0.1                               # m per orthophoto pixel
MAX_RANGE = 130.0                       # m: beyond this the IPM is too coarse to draw
CAR = (4.6, 1.85, 1.5)
INK, MUTED = "#1f1f1e", "#8a8984"
LIGHT = np.array([0.4, -0.3, 0.87]) / np.linalg.norm([0.4, -0.3, 0.87])


# --- camera helpers -------------------------------------------------------------------------------
def centre(cam):
    K, R, t = cam
    return -R.T @ t


def rays(cam, uv):
    K, R, t = cam
    return (R.T @ (np.linalg.inv(K) @ np.c_[uv, np.ones(len(uv))].T)).T


def to_ground(cam, uv):
    C, d = centre(cam), rays(cam, np.atleast_2d(uv))
    s = -C[2] / np.where(np.abs(d[:, 2]) > 1e-12, d[:, 2], -1e-12)
    return C[None, :2] + d[:, :2] * s[:, None], s > 0


def project(cam, X):
    K, R, t = cam
    x = K @ (R @ np.asarray(X, float).T + t[:, None])
    return (x[:2] / x[2]).T, x[2]


def height_on_vertical(C, ray, G):
    """Height of the point of the vertical line G + z e_z closest to the ray C + s ray (least squares)."""
    e = np.array([0.0, 0.0, 1.0])
    A = np.array([[ray @ ray, -(ray @ e)], [ray @ e, -1.0]])
    if abs(np.linalg.det(A)) < 1e-12:
        return np.nan
    return float(np.linalg.solve(A, np.array([ray @ (G - C), e @ (G - C)]))[1])


def look_at(eye, target, f=1000.0, size=(1280, 720)):
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    return K, R, -R @ np.asarray(eye, float)


# --- ground: orthophoto -------------------------------------------------------------------------------
class Ortho:
    def __init__(self, cam, image, lo, hi, res=RES):
        self.lo, self.hi, self.res = np.asarray(lo, float), np.asarray(hi, float), res
        self.size = (int((hi[0] - lo[0]) / res), int((hi[1] - lo[1]) / res))
        self.A = np.array([[res, 0, lo[0]], [0, -res, hi[1]], [0, 0, 1.0]])           # pixel -> ground (X, Y, 1)
        K, R, t = cam
        H = K @ np.c_[R[:, 0], R[:, 1], t] @ self.A
        self.image = cv2.warpPerspective(image, H, self.size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        # keep only ground that the camera sees in front of it and within range
        jj, ii = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]))
        X = lo[0] + (jj + 0.5) * res
        Y = hi[1] - (ii + 0.5) * res
        uv, depth = project(cam, np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)])
        h, w = image.shape[:2]
        ok = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ok &= np.hypot(X.ravel() - centre(cam)[0], Y.ravel() - centre(cam)[1]) < MAX_RANGE
        self.valid = ok.reshape(X.shape)
        self.image[~self.valid] = (205, 205, 200)

    def px(self, P):
        P = np.atleast_2d(P)
        return np.c_[(P[:, 0] - self.lo[0]) / self.res, (self.hi[1] - P[:, 1]) / self.res]


# --- lines: painted ground markings and right angles -------------------------------------------
def painted_segments(bg, cam, min_len_m=1.0, contrast=35):
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    s = cv2.createLineSegmentDetector().detect(gray)[0].reshape(-1, 4)
    K, R, t = cam
    horizon = (np.linalg.inv(K).T @ R[:, 2])                          # image of the ground's line at infinity
    horizon = horizon / np.linalg.norm(horizon[:2])                   # signed distance in pixels
    out = []
    road = float(np.median(gray[gray.shape[0] // 2:]))
    for x0, y0, x1, y1 in s:
        p = np.array([[x0, y0], [x1, y1]])
        if np.any(np.c_[p, np.ones(2)] @ horizon * np.sign(horizon @ [K[0, 2], gray.shape[0] - 1, 1]) <= 5):
            continue                                                   # above or too close to the horizon
        g, ok = to_ground(cam, p)
        if not ok.all() or np.linalg.norm(g[1] - g[0]) < min_len_m or np.hypot(*(g.mean(0) - centre(cam)[:2])) > 90:
            continue
        # painted: brighter than the road on the segment itself
        n = 12
        pts = p[0] + np.linspace(0, 1, n)[:, None] * (p[1] - p[0])
        v = gray[np.clip(pts[:, 1].astype(int), 0, gray.shape[0] - 1), np.clip(pts[:, 0].astype(int), 0, gray.shape[1] - 1)]
        if np.median(v) < road + contrast:
            continue
        out.append({"img": p, "g": g})
    return out


def right_angles(segs, tol_deg=6.0, reach=0.8):
    """Pairs of ground segments perpendicular within tol that meet (their lines cross within `reach` m of both)."""
    pairs = []
    for i in range(len(segs)):
        a = segs[i]["g"]
        da = (a[1] - a[0]) / np.linalg.norm(a[1] - a[0])
        for j in range(i + 1, len(segs)):
            b = segs[j]["g"]
            db = (b[1] - b[0]) / np.linalg.norm(b[1] - b[0])
            if abs(da @ db) > np.sin(np.radians(tol_deg)):
                continue
            M = np.c_[da, -db]
            if abs(np.linalg.det(M)) < 1e-6:
                continue
            u, v = np.linalg.solve(M, b[0] - a[0])
            X = a[0] + u * da
            la, lb = np.linalg.norm(a[1] - a[0]), np.linalg.norm(b[1] - b[0])
            if -reach <= u <= la + reach and -reach <= v <= lb + reach:
                pairs.append((i, j, X))
    return pairs


def orthogonality_tilt(cam, segs, pairs, deltas=np.radians(np.arange(-0.6, 0.601, 0.01))):
    """Camera tilt that makes the detected right angles right (median squared deviation from 90 deg)."""
    cost = []
    for d in deltas:
        c = ce.tilt_cameras([cam], d)[0]
        dev = []
        for i, j, _ in pairs:
            ga, _ = to_ground(c, segs[i]["img"])
            gb, _ = to_ground(c, segs[j]["img"])
            da, db = ga[1] - ga[0], gb[1] - gb[0]
            dev.append(np.degrees(np.arccos(np.clip(abs(da @ db) / (np.linalg.norm(da) * np.linalg.norm(db)), 0, 1))))
        cost.append(np.median((90 - np.array(dev)) ** 2) if dev else np.nan)
    cost = np.array(cost)
    return float(deltas[int(np.nanargmin(cost))]), deltas, cost


# --- static vertical structures -----------------------------------------------------------------------
def vertical_poles(bg, cam, ortho, min_px=30, max_angle=1.5):
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    s = cv2.createLineSegmentDetector().detect(gray)[0].reshape(-1, 4)
    K, R, t = cam
    vz = K @ R[:, 2]                                                    # vertical vanishing point (homogeneous)
    poles = []
    for x0, y0, x1, y1 in s:
        p = np.array([[x0, y0], [x1, y1]], float)
        if np.linalg.norm(p[1] - p[0]) < min_px:
            continue
        mid = p.mean(0)
        to_vp = vz[:2] - mid * vz[2]
        seg = p[1] - p[0]
        cosang = abs(seg @ to_vp) / (np.linalg.norm(seg) * np.linalg.norm(to_vp) + 1e-12)
        if cosang < np.cos(np.radians(max_angle)):
            continue
        bottom, top = (p[0], p[1]) if p[0, 1] > p[1, 1] else (p[1], p[0])
        g, ok = to_ground(cam, bottom[None])
        if not ok[0]:
            continue
        G = np.r_[g[0], 0.0]
        q = ortho.px(g)[0]
        if not (0 <= q[0] < ortho.size[0] and 0 <= q[1] < ortho.size[1]) or not ortho.valid[int(q[1]), int(q[0])]:
            continue
        z = height_on_vertical(centre(cam), rays(cam, top[None])[0], G)
        if 0.6 < z < 18:
            poles.append((g[0], z))
    merged = []                                                          # merge verticals of one pole
    for g, z in sorted(poles, key=lambda p: -p[1]):
        if all(np.linalg.norm(g - m[0]) > 0.8 for m in merged):
            merged.append((g, z))
    return merged


def red_objects(bg, cam, ortho):
    """Traffic cones and barrels: red regions of the background; foot by IPM, height from the top."""
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 16) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 100) & (hsv[..., 2] > 70)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(red)
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < 25 or area > 40000:
            continue
        g, ok = to_ground(cam, np.array([[x + w / 2, y + h]], float))
        if not ok[0]:
            continue
        q = ortho.px(g)[0]
        if not (0 <= q[0] < ortho.size[0] and 0 <= q[1] < ortho.size[1]) or not ortho.valid[int(q[1]), int(q[0])]:
            continue
        z = height_on_vertical(centre(cam), rays(cam, np.array([[x + w / 2, y]], float))[0], np.r_[g[0], 0.0])
        # footprint from the region's width at its distance
        d = np.linalg.norm(np.r_[g[0], 0] - centre(cam))
        width = float(np.clip(w * d / cam[0][0, 0], 0.25, 1.2))
        if 0.3 < z < 2.5:
            out.append((g[0], z, width))
    return out


# --- moving objects ---------------------------------------------------------------------------------
def moving_objects(frames, times, bg, cam):
    _, final, _ = vc.run(frames, times, bg, cam, (1, 2, 3))
    return vc.smoothed_output(final, len(frames))


# --- rendering ----------------------------------------------------------------------------------------
def box_faces(center, yaw, l, w, h, z0=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    u, v = np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    base = np.r_[center, z0]
    corners = [base + a * l / 2 * u + b * w / 2 * v for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    bot = np.array(corners)
    top = bot + [0, 0, h]
    faces = [(top, np.array([0, 0, 1.0]))]
    for i in range(4):
        j = (i + 1) % 4
        quad = np.array([bot[i], bot[j], top[j], top[i]])
        n = np.cross(bot[j] - bot[i], [0, 0, 1.0])
        n /= np.linalg.norm(n) + 1e-12
        faces.append((quad, n if n @ (quad.mean(0)[:2].tolist() + [0] - np.r_[center, 0]) >= 0 else -n))
    return faces


def render(view, ortho, objects, trails=(), size=(1280, 720)):
    """objects: list of (faces, rgb). trails: list of (ground points (n, 2), rgb). Returns an RGB image."""
    K, R, t = view
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:] = np.linspace(235, 200, size[1])[:, None, None].astype(np.uint8)            # sky-ish background
    H = K @ np.c_[R[:, 0], R[:, 1], t] @ ortho.A                                       # orthophoto pixel -> view
    jj, ii = np.meshgrid(np.arange(0, ortho.size[0], 8), np.arange(0, ortho.size[1], 8))
    depth = (H[2] @ np.stack([jj.ravel(), ii.ravel(), np.ones(jj.size)]))
    if (depth > 0).all():
        tex = cv2.warpPerspective(cv2.cvtColor(ortho.image, cv2.COLOR_BGR2RGB), H, size, flags=cv2.INTER_LINEAR)
        cover = cv2.warpPerspective(ortho.valid.astype(np.uint8) * 255, H, size, flags=cv2.INTER_NEAREST) > 0
        img[cover] = tex[cover]
    for P, color in trails:
        uv, d = project(view, np.c_[P, np.full(len(P), 0.05)])
        ok = d > 0
        if ok.sum() > 1:
            cv2.polylines(img, [uv[ok].round().astype(np.int32)], False, color, 2, cv2.LINE_AA)
    polys = []
    C = centre(view)
    for faces, color in objects:
        for quad, n in faces:
            uv, d = project(view, quad)
            if (d <= 0.5).any():
                continue
            if n @ (C - quad.mean(0)) <= 0:
                continue                                                               # back face
            shade = 0.45 + 0.55 * max(0.0, n @ LIGHT)
            polys.append((d.mean(), uv, tuple(int(c * shade) for c in color)))
    for _, uv, color in sorted(polys, key=lambda p: -p[0]):
        pts = uv.round().astype(np.int32)
        cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, tuple(int(c * 0.55) for c in color), 1, cv2.LINE_AA)
    return img


def track_color(tid):
    rng = np.random.default_rng(int(tid) * 7919 + 3)
    hsv = np.uint8([[[rng.integers(0, 180), 170, 230]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def visible_movers(movers, cam, shape):
    """Drop estimates of cars cut by the image border (their box bottom is not their ground contact) and beyond the
    drawable range."""
    C = centre(cam)
    out = []
    for objs in movers:
        keep = []
        for tid, o in objs:
            l, w, h = CAR
            faces = box_faces(o["center"], o["yaw"], l, w, h)
            uv, d = project(cam, np.vstack([q for q, _ in faces]))
            inside = (d > 0).all() and uv[:, 0].min() > 2 and uv[:, 1].max() < shape[0] - 2 and uv[:, 0].max() < shape[1] - 2
            if inside and np.hypot(*(o["center"] - C[:2])) < MAX_RANGE:
                keep.append((tid, o))
        out.append(keep)
    return out


def scene_objects(poles, cones, movers_k):
    objs = []
    for g, z in poles:
        objs.append((box_faces(g, 0.0, 0.25, 0.25, z), (110, 110, 118)))
    for g, z, w in cones:
        objs.append((box_faces(g, 0.0, w, w, z), (230, 90, 60)))
    for tid, o in movers_k:
        l, w, h = CAR
        objs.append((box_faces(o["center"], o["yaw"], l, w, h), track_color(tid)))
    return objs


# --- figures -----------------------------------------------------------------------------------------
def lines_figure(path, bg, ortho, segs, pairs, poles, cones, tilt_curve, tilt_label, tilt_lines):
    fig = plt.figure(figsize=(19, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    for s in segs:
        ax.plot(*s["img"].T, color="#2a78d6", linewidth=1.2)
    for i, j, _ in pairs:
        for k in (i, j):
            ax.plot(*segs[k]["img"].T, color="#eb6834", linewidth=2)
    ax.set_title(f"median background: {len(segs)} painted ground lines (blue), {len(pairs)} right-angle junctions in "
                 f"the IPM (orange)", fontsize=10, loc="left", color=INK)
    ax.axis("off")
    ax = fig.add_subplot(2, 2, (2, 4))
    ax.imshow(cv2.cvtColor(ortho.image, cv2.COLOR_BGR2RGB), extent=(ortho.lo[0], ortho.hi[0], ortho.lo[1], ortho.hi[1]))
    for s in segs:
        ax.plot(*s["g"].T, color="#2a78d6", linewidth=1.0)
    for i, j, X in pairs:
        ax.plot(*X, "o", color="#eb6834", markersize=4)
    if poles:
        P = np.array([g for g, _ in poles])
        ax.scatter(*P.T, s=14, c="#3a3a40", marker="s", label=f"{len(poles)} vertical poles")
    if cones:
        P = np.array([g for g, _, _ in cones])
        ax.scatter(*P.T, s=14, c="#e5533a", marker="^", label=f"{len(cones)} cones / barrels")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("IPM orthophoto (10 cm/px) with the ground lines, right angles and static objects", fontsize=10,
                 loc="left", color=INK)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax = fig.add_subplot(2, 2, 3)
    deltas, cost = tilt_curve
    ax.plot(np.degrees(deltas), cost, color="#2a78d6", linewidth=2)
    ax.axvline(tilt_lines, color="#2a78d6", linestyle=(0, (3, 2)), label=f"from right angles: {tilt_lines:+.2f}°")
    ax.axvline(tilt_label, color="#eb6834", linestyle=(0, (3, 2)), label=f"from the labels: {tilt_label:+.2f}°")
    ax.set_xlabel("camera tilt correction (°)", color=INK)
    ax.set_ylabel("median (90° − angle)² of the junctions", color=INK)
    ax.set_title("Self-calibration: the tilt that makes the painted right angles right", fontsize=10, loc="left",
                 color=INK)
    ax.legend(fontsize=9, frameon=False)
    vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=90, facecolor="white")
    plt.close(fig)


def views_figure(path, frames, cam, ortho, poles, cones, movers, ks, views):
    cols = 2 + len(views)
    fig, axes = plt.subplots(len(ks), cols, figsize=(5.2 * cols, 3.1 * len(ks)))
    tracks = {}
    for k, objs in enumerate(movers):
        for tid, o in objs:
            tracks.setdefault(tid, []).append((k, o["center"]))
    for r, k in enumerate(ks):
        trails = [(np.array([c for kk, c in v if kk <= k]), track_color(tid)) for tid, v in tracks.items()
                  if sum(kk <= k for kk, _ in v) > 1 and any(kk == k for kk, _ in v)]
        objs = scene_objects(poles, cones, movers[k])
        axes[r, 0].imshow(cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB))
        axes[r, 0].set_title(f"video frame {k}", fontsize=9.5, loc="left", color=INK)
        K, Rc, tc = cam
        s = 1280 / frames[k].shape[1]
        Kc = np.diag([s, s, 1.0]) @ K
        axes[r, 1].imshow(render((Kc, Rc, tc), ortho, objs, trails, size=(1280, int(frames[k].shape[0] * s))))
        axes[r, 1].set_title("3D model, original camera", fontsize=9.5, loc="left", color=INK)
        for c, (name, view) in enumerate(views):
            axes[r, 2 + c].imshow(render(view, ortho, objs, trails))
            axes[r, 2 + c].set_title(f"3D model, {name}", fontsize=9.5, loc="left", color=INK)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor="white")
    plt.close(fig)


def map_figure(path, ortho, poles, cones, movers, tracks, camera_xy, metrics):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(cv2.cvtColor(ortho.image, cv2.COLOR_BGR2RGB), extent=(ortho.lo[0], ortho.hi[0], ortho.lo[1], ortho.hi[1]),
              alpha=0.85)
    for q in tracks.values():
        P = np.array([s["pos"] for s in q])
        ax.plot(*P.T, color="white", linewidth=1.0, linestyle=(0, (3, 2)))
    est = {}
    for k, objs in enumerate(movers):
        for tid, o in objs:
            est.setdefault(tid, []).append(o["center"])
    for tid, P in est.items():
        P = np.array(P)
        ax.plot(*P.T, color=np.array(track_color(tid)) / 255, linewidth=1.8)
    if poles:
        ax.scatter(*np.array([g for g, _ in poles]).T, s=12, c="#3a3a40", marker="s")
    if cones:
        ax.scatter(*np.array([g for g, _, _ in cones]).T, s=12, c="#e5533a", marker="^")
    ax.plot(*camera_xy, "k^", markersize=10)
    ax.set_xlim(ortho.lo[0], ortho.hi[0])
    ax.set_ylim(ortho.lo[1], ortho.hi[1])
    ax.set_title(f"World map from the video: {len(est)} tracked moving objects (colour), labelled trajectories (white "
                 f"dashed) | moving cars found {metrics['found']:.0%}, position {metrics['position_m']:.2f} m, "
                 f"heading < 20° {metrics['heading_under20']:.0%}, speed error {metrics['speed_err']:.2f} m/s",
                 fontsize=10, loc="left", color=INK)
    ax.tick_params(labelsize=8, colors=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=95, facecolor="white")
    plt.close(fig)


def main():
    images, times, cams, tracks, camera_xy = ce.load_spd()
    frames = [cv2.imread(p) for p in images]
    bg = np.median(np.stack(frames[::2]), axis=0).astype(np.uint8)
    tilt_label, _ = it.init_tilt(cams, tracks)
    cam0 = cams[0]
    P = np.array([s["pos"] for q in tracks.values() for s in q])
    P = P[np.hypot(*(P - camera_xy).T) < MAX_RANGE]
    lo, hi = P.min(0) - 8, P.max(0) + 8
    ortho0 = Ortho(cam0, bg, lo, hi)
    segs = painted_segments(bg, cam0)
    pairs = right_angles(segs)
    tilt_lines, deltas, cost = orthogonality_tilt(cam0, segs, pairs)
    print(f"{len(segs)} painted ground lines, {len(pairs)} right-angle junctions; tilt from right angles "
          f"{np.degrees(tilt_lines):+.2f} deg, from the initialisation labels {np.degrees(tilt_label):+.2f} deg", flush=True)
    # the tilt from the right angles is reported, but the model uses the tilt fitted to the initialisation labels:
    # at an intersection not every pair of painted lines that look perpendicular is perpendicular (slanted crosswalks,
    # turn arrows), and the right-angle tilt came out with the wrong sign
    cam = ce.tilt_cameras([cam0], tilt_label)[0]
    ortho = Ortho(cam, bg, lo, hi)
    segs = painted_segments(bg, cam)
    pairs = right_angles(segs)
    poles = vertical_poles(bg, cam, ortho)
    cones = red_objects(bg, cam, ortho)
    print(f"static: {len(poles)} vertical poles, {len(cones)} cones/barrels", flush=True)
    movers_all = moving_objects(frames, times, bg, cam)
    movers = visible_movers(movers_all, cam, frames[0].shape[:2])
    gt = vc.moving_ground_truth(times, {tid: q for tid, q in tracks.items()}, camera_xy)
    score = vc.evaluate(movers_all, gt)
    metrics = {"tilt_lines_deg": float(np.degrees(tilt_lines)), "tilt_labels_deg": float(np.degrees(tilt_label)),
               "painted_lines": len(segs), "right_angles": len(pairs), "poles": len(poles), "cones": len(cones),
               "found": score["coverage"], "position_m": score["position"], "orient_under10": score["orientation"],
               "heading_under20": score["heading"], "speed_err": score["speed"]}
    print(json.dumps(metrics, indent=1), flush=True)
    json.dump(metrics, open(os.path.join(vd.OUT_DIR, "world3d_metrics.json"), "w"), indent=1)
    lines_figure(os.path.join(vd.OUT_DIR, "world3d_lines.png"), bg, ortho, segs, pairs, poles, cones, (deltas, cost),
                 float(np.degrees(tilt_label)), float(np.degrees(tilt_lines)))
    mid = (lo + hi) / 2
    span = float(np.max(hi - lo))
    C0 = centre(cam)
    away = (mid - C0[:2]) / np.linalg.norm(mid - C0[:2])
    side = np.array([-away[1], away[0]])
    views = [("drone view", look_at(np.r_[C0[:2] - 0.25 * span * away + 0.35 * span * side, 0.55 * span],
                                    np.r_[mid, 0.0])),
             ("from the far side", look_at(np.r_[mid + 0.55 * span * away - 0.15 * span * side, 0.22 * span],
                                           np.r_[mid - 0.2 * span * away, 0.0]))]
    views_figure(os.path.join(vd.OUT_DIR, "world3d_views.png"), frames, cam, ortho, poles, cones, movers,
                 (60, 150, 230), views)
    map_figure(os.path.join(vd.OUT_DIR, "world3d_map.png"), ortho, poles, cones, movers, tracks, camera_xy, metrics)


if __name__ == "__main__":
    main()
