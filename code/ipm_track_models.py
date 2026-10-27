"""
3D models of the moving cars at intersection A from IPM and their tracks, with no annotation in the pipeline
(V2X-Seq). The dataset's labels and LiDAR are used only to pick which detected car to score and to score it.

  1. detection   : median-background foreground over the whole frame, shadow band stripped, connected blobs;
  2. tracking    : blobs linked frame to frame by box overlap;
  3. IPM track   : the blob's lowest (ground-contact) pixels mapped onto the ground plane z = 0 through the camera
                   (the ground homography); smoothed; heading = direction of travel;
  4. motion      : corners tracked inside the blob, object-level bundle adjustment of the per-frame pose
                   (x, y, heading) and the 3D points, started from the IPM track (ba_models);
  5. model       : the bike-style object frame is re-centred on the points; silhouette hull from the blobs (no label
                   box) -> cuboid -> triangulation of the contact points -> recursive subdivision pulled onto the hull
                   and, on top surfaces, to the tracked points -> colours from the video;
  6. check       : held-out frames (every 4th): the model moved by OUR trajectory against the car's LiDAR in world
                   coordinates, its silhouette against the blob; our track against the labelled positions/headings.

Writes docs/ipm_models_*.png and docs/ipm_models_metrics.json.
"""
import json
import math
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Delaunay, cKDTree

import ba_models as bm
import car_flow_planes as cfp
import mesh_refine as mr
import mix3d as mx
import tri_tracks as tt
import v2x_seq_classical as vc
import v2x_seq_demo as vd
import v2x_seq_shape as vs

# the cuboid as the first bound: a generic car (v2x_seq_demo.CAR_SIZE, a prior, not a label) with a 15% margin, centred
# on the tracker's closed-form placement; BOX=loose gives the old 6 x 3 m search box
HULL_BOX = (6.0, 3.0) if os.environ.get("BOX", "prior") == "loose" else (4.6 * 1.15, 1.85 * 1.15)
import video3d as v3

MIN_AREA = 600             # px
INK, MUTED = "#1f1f1e", "#8a8984"


# --- detection and tracking -------------------------------------------------------------------------------
def shadow_free(frame, bg, mask, v_lo=0.35, v_hi=0.92, s_tol=40, h_tol=12):
    """HSV shadow test (Cucchiara et al. 2003): darker than the background with similar hue and saturation."""
    hf = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
    hb = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.int16)
    ratio = (hf[..., 2] + 1) / (hb[..., 2] + 1)
    dh = np.abs(hf[..., 0] - hb[..., 0])
    dh = np.minimum(dh, 180 - dh)
    shadow = (ratio > v_lo) & (ratio < v_hi) & (np.abs(hf[..., 1] - hb[..., 1]) < s_tol) & (dh < h_tol)
    m = cv2.morphologyEx((mask & ~shadow).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return lab == 1 + np.argmax(st[1:, 4]) if n > 1 else m > 0


def blobs(frame, bg):
    """Foreground blobs with the wet-road reflection cut at the contact line (sharp underside vs blurred mirror
    image, v2x_seq_classical.cut_reflection) and the cast shadow removed."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = []
    for b in vc.blobs(vc.foreground(frame, bg), min_area=MIN_AREA):
        b = vc.cut_reflection(b, gray)
        x0, y0, x1, y1 = b["box"]
        pad = 4
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(frame.shape[1], x1 + pad), min(frame.shape[0], y1 + pad)
        m = np.zeros((Y1 - Y0, X1 - X0), bool)
        m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = b["m"] > 0
        m = shadow_free(frame[Y0:Y1, X0:X1], bg[Y0:Y1, X0:X1], m)
        full = np.zeros(frame.shape[:2], bool)
        full[Y0:Y1, X0:X1] = m
        ys, xs = np.nonzero(full)
        if len(xs) < MIN_AREA:
            continue
        out.append({"box": (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), "mask": full})
    return out


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_blobs(boxes_per_frame, min_iou=0.3, min_len=10):
    """Greedy frame-to-frame association by box IoU; tracks are lists of (t, j)."""
    done, active = [], []
    for t, boxes in enumerate(boxes_per_frame):
        prev = [tr for tr in active if tr[-1][0] == t - 1]
        pairs = sorted(((box_iou(boxes_per_frame[tr[-1][0]][tr[-1][1]], b), i, j)
                        for i, tr in enumerate(prev) for j, b in enumerate(boxes)), reverse=True)
        used_i, used_j, nxt = set(), set(), []
        for iou, i, j in pairs:
            if iou < min_iou:
                break
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            nxt.append(prev[i] + [(t, j)])
        done += [tr for i, tr in enumerate(prev) if i not in used_i]
        nxt += [[(t, j)] for j in range(len(boxes)) if j not in used_j]
        active = nxt
    done += active
    return [tr for tr in done if len(tr) >= min_len]


# --- IPM track ----------------------------------------------------------------------------------------------
def ground_point(cam, uv):
    """Pixels -> points on the ground plane z = 0 (the camera's ground homography; R is a general matrix here)."""
    K, R, t = cam
    H = K @ np.c_[R[:, 0], R[:, 1], t]
    g = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(H).T
    return g[:, :2] / g[:, 2:3]


def contact_pixel(mask):
    ys, xs = np.nonzero(mask)
    yb = ys.max()
    return np.array([xs[ys >= yb - 3].mean(), yb], float)


def ipm_track(seq, det, track, sigma=1.5):
    ks = np.array([t for t, _ in track])
    G = np.vstack([ground_point(seq.cams[k], contact_pixel(det[k][j]["mask"])[None]) for k, j in track])
    xy = np.stack([gaussian_filter1d(G[:, a], sigma, mode="nearest") for a in (0, 1)], 1)
    v = np.gradient(xy, axis=0)
    yaw = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    yaw = gaussian_filter1d(yaw, 2 * sigma, mode="nearest")
    obs = []
    for i, (k, j) in enumerate(track):
        obs.append({"k": int(k), "pos": xy[i].copy(), "yaw": float(yaw[i]), "mask": det[k][j]["mask"],
                    "box": det[k][j]["box"], "raw_ground": G[i]})
    return obs, float(np.median(np.linalg.norm(v, axis=1)))


# --- corners, bundle adjustment ------------------------------------------------------------------------------
def track_corners(seq, obs, r, P, span):
    n = len(P)
    uv = np.full((n, len(obs), 2), np.nan, np.float32)
    uv[:, r] = P
    for step in (1, -1):
        alive = np.ones(n, bool)
        cur = P.copy()
        i = r
        while 0 <= i + step < len(obs) and abs(i + step - r) <= span and alive.any():
            a, b = obs[i], obs[i + step]
            ga = cv2.cvtColor(seq.frames[a["k"]], cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(seq.frames[b["k"]], cv2.COLOR_BGR2GRAY)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, cur[:, None], None, **tt.LK)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, nxt, None, **tt.LK)
            nxt = nxt[:, 0]
            m = cv2.dilate(b["mask"].astype(np.uint8), np.ones((5, 5), np.uint8))
            q = np.clip(nxt.round().astype(int), 0, [seq.shape[1] - 1, seq.shape[0] - 1])
            fb = np.linalg.norm(back[:, 0] - cur, axis=1)
            alive &= (st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < tt.FB_TOL) & (m[q[:, 1], q[:, 0]] > 0)
            i += step
            uv[alive, i] = nxt[alive]
            cur = nxt
    return uv


def motion_and_points(seq, obs, sigma_pos=0.6, sigma_yaw_deg=4.0, recentre=True):
    U = []
    for r in range(0, len(obs), bm.REF_STEP):
        gray = cv2.cvtColor(seq.frames[obs[r]["k"]], cv2.COLOR_BGR2GRAY)
        P, _ = tt.reference_mesh(gray, obs[r]["mask"])
        u = track_corners(seq, obs, r, P, bm.SPAN)
        U.append(u[(~np.isnan(u[:, :, 0])).sum(1) >= 3])
    U = np.concatenate(U)
    X0, ang0 = bm.triangulate(seq, obs, U)
    ok = ~np.isnan(X0[:, 0]) & (np.abs(X0[:, 0]) < 6) & (np.abs(X0[:, 1]) < 4) & (X0[:, 2] > -0.5) & (X0[:, 2] < 3)
    U, X0 = U[ok], X0[ok]
    pi, fi = np.nonzero(~np.isnan(U[:, :, 0]))
    uv = U[pi, fi]
    bm.SIGMA_POS, bm.SIGMA_YAW = sigma_pos, math.radians(sigma_yaw_deg)   # IPM poses are rougher than labels
    d, X, res = bm.bundle_adjust(seq.cams, obs, X0, fi, pi, uv)
    good = res < 3 * bm.MAX_RES
    d, X, res = bm.bundle_adjust(seq.cams, obs, X, fi[good], pi[good], uv[good], d0=d)
    fi, pi, uv = fi[good], pi[good], uv[good]
    obs2 = bm.with_corrections(obs, d)
    worst = np.zeros(len(X))
    np.maximum.at(worst, pi, res)
    Ub = np.full_like(U, np.nan)
    Ub[pi, fi] = uv
    _, ang = bm.triangulate(seq, obs2, Ub)
    keep = (worst < bm.MAX_RES) & (np.bincount(pi, minlength=len(X)) >= 3) & (ang >= bm.MIN_ANGLE)
    X = X[keep]
    # re-centre the object frame on the points (origin = middle of the footprint, on the ground)
    c = (np.percentile(X[:, :2], 98, axis=0) + np.percentile(X[:, :2], 2, axis=0)) / 2 if recentre else np.zeros(2)
    out = []
    for o in obs2:
        q = dict(o)
        cy, sy = math.cos(o["yaw"]), math.sin(o["yaw"])
        q["pos"] = np.asarray(o["pos"]) + np.array([cy * c[0] - sy * c[1], sy * c[0] + cy * c[1]])
        out.append(q)
    X = X - np.r_[c, 0.0]
    return out, X, {"reprojection_px": float(np.median(res)), "points": int(len(X)),
                    "pose_change_cm": float(100 * np.median(np.linalg.norm(d[:, :2], axis=1))),
                    "yaw_change_deg": float(np.degrees(np.median(np.abs(d[:, 2]))))}


# --- model ---------------------------------------------------------------------------------------------------
def shaved(m, px=2):
    return cv2.erode(m.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0


def build_model(seq, obs, X, levels=3):
    train = [o for i, o in enumerate(obs) if i % 4]
    axes = vs.grid(*HULL_BOX, 1.6)
    views = [(seq.cams[o["k"]], o["pos"], o["yaw"], shaved(o["mask"]).astype(np.uint8)) for o in train]
    hull = vs.height_field(vs.carve_votes(axes, views) >= vs.VOTE)
    V0, F0, box, _ = mr.coarse_mesh(hull, axes)
    V, F = mr.refine(V0, F0, hull, axes, X, box, levels=levels)[-1]
    N = mr.vertex_normals(V, F)
    cols = []
    for o in train:
        C = mx.camera_in_object(seq.cams[o["k"]], o)
        vis = np.einsum("ij,ij->i", N, C - V) / np.linalg.norm(C - V, axis=1) > 0.2
        uv = vs.project(seq.cams[o["k"]], vs.world(V, o["pos"], o["yaw"]))[0]
        u, w = np.clip(uv.round().astype(int), 0, [seq.shape[1] - 1, seq.shape[0] - 1]).T
        c = seq.frames[o["k"]][w, u][:, ::-1].astype(float)
        c[~(vis & o["mask"][w, u])] = np.nan
        cols.append(c)
    with np.errstate(all="ignore"):
        cols = np.nanmedian(np.stack(cols), 0) / 255
    from car_models import mirror_fill
    return V, F, mirror_fill(V, cols), hull, axes


# --- evaluation (labels and LiDAR only here) -----------------------------------------------------------------
def match_label(seq, obs):
    """The labelled car whose 2D boxes overlap our blobs most (evaluation only)."""
    best, score = None, 0.0
    for tid, q in seq.tracks.items():
        by_k = {o["k"]: o for o in q}
        s = [box_iou(o["box"], by_k[o["k"]]["box"]) for o in obs if o["k"] in by_k]
        if s and np.mean(s) * len(s) > score:
            best, score = tid, np.mean(s) * len(s)
    return best


def evaluate(seq, obs, V, F, tid, ground):
    lab = {o["k"]: o for o in seq.tracks[tid]}
    test = [o for i, o in enumerate(obs) if i % 4 == 0 and o["k"] in lab]
    S = mr.sample_surface(V, F)
    acc, comp, pos_err, yaw_err, iou = [], [], [], [], []
    for o in test:
        lo = lab[o["k"]]
        Lw = vs.world(mx.to_ground_frame(cfp.lidar_on_car(seq, lo, ground), lo), lo["pos"], lo["yaw"])
        Sw = vs.world(S, o["pos"], o["yaw"])
        if len(Lw) >= 30:
            acc.append(np.median(cKDTree(Lw).query(Sw)[0]))
            comp.append(np.median(cKDTree(Sw).query(Lw)[0]))
        pos_err.append(np.linalg.norm(np.asarray(o["pos"]) - lo["pos"]))
        dy = (o["yaw"] - lo["yaw"] + np.pi / 2) % np.pi - np.pi / 2      # labels may point either way along the car
        yaw_err.append(abs(np.degrees(dy)))
        iou.append(vs.iou(mr.mesh_silhouette(V, F, seq.cams[o["k"]], o["pos"], o["yaw"], seq.shape), o["mask"]))
    # diagnostic only: one constant offset of the body frame fitted to the labels separates placement from shape
    A = np.vstack([[[math.cos(o["yaw"]), -math.sin(o["yaw"])], [math.sin(o["yaw"]), math.cos(o["yaw"])]] for o in test])
    b = np.concatenate([lab[o["k"]]["pos"] - np.asarray(o["pos"]) for o in test])
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    pos_al, acc_al, comp_al = [], [], []
    for o in test:
        lo = lab[o["k"]]
        R2 = np.array([[math.cos(o["yaw"]), -math.sin(o["yaw"])], [math.sin(o["yaw"]), math.cos(o["yaw"])]])
        pa = np.asarray(o["pos"]) + R2 @ c
        pos_al.append(np.linalg.norm(pa - lo["pos"]))
        Lw = vs.world(mx.to_ground_frame(cfp.lidar_on_car(seq, lo, ground), lo), lo["pos"], lo["yaw"])
        if len(Lw) >= 30:
            Sw = vs.world(S, pa, o["yaw"])
            acc_al.append(np.median(cKDTree(Lw).query(Sw)[0]))
            comp_al.append(np.median(cKDTree(Sw).query(Lw)[0]))
    # gauge-free placement: the model's own footprint centre and long axis in the world, versus the labelled box
    fp = V[V[:, 2] < 0.8][:, :2] if (V[:, 2] < 0.8).sum() > 20 else V[:, :2]
    cen = (np.percentile(fp, 95, axis=0) + np.percentile(fp, 5, axis=0)) / 2
    ev_ax = np.linalg.eigh(np.cov((fp - fp.mean(0)).T))[1][:, -1]
    ax_ang = math.atan2(ev_ax[1], ev_ax[0])
    cen_err, head_err = [], []
    for o in test:
        R2 = np.array([[math.cos(o["yaw"]), -math.sin(o["yaw"])], [math.sin(o["yaw"]), math.cos(o["yaw"])]])
        cen_err.append(np.linalg.norm(np.asarray(o["pos"]) + R2 @ cen - lab[o["k"]]["pos"]))
        d = (o["yaw"] + ax_ang - lab[o["k"]]["yaw"] + np.pi / 2) % np.pi - np.pi / 2
        head_err.append(abs(math.degrees(d)))
    return {"model_centre_error_cm": 100 * float(np.median(cen_err)), "model_heading_error_deg": float(np.median(head_err)),
            "model_to_lidar_cm": 100 * float(np.median(acc)), "lidar_to_model_cm": 100 * float(np.median(comp)),
            "body_offset_cm": 100 * float(np.linalg.norm(c)), "aligned_position_error_cm": 100 * float(np.median(pos_al)),
            "aligned_model_to_lidar_cm": 100 * float(np.median(acc_al)),
            "aligned_lidar_to_model_cm": 100 * float(np.median(comp_al)),
            "position_error_cm": 100 * float(np.median(pos_err)), "heading_error_deg": float(np.median(yaw_err)),
            "held_out_iou": float(np.mean(iou)), "held_out_frames": len(test)}


def ipm_only_eval(seq, obs, tid):
    lab = {o["k"]: o for o in seq.tracks[tid]}
    pe = [np.linalg.norm(np.asarray(o["pos"]) - lab[o["k"]]["pos"]) for o in obs if o["k"] in lab]
    ye = [abs(np.degrees((o["yaw"] - lab[o["k"]]["yaw"] + np.pi / 2) % np.pi - np.pi / 2)) for o in obs if o["k"] in lab]
    return 100 * float(np.median(pe)), float(np.median(ye))


# --- figure ----------------------------------------------------------------------------------------------------
def car_row(fig, gs, row, seq, tid, obs_ipm, obs, V, F, cols, X, ev):
    lab = {o["k"]: o for o in seq.tracks[tid]}
    o = obs[len(obs) // 2 // 4 * 4]
    ax = fig.add_subplot(gs[row, 0])
    img = cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    cnt = cv2.findContours(o["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    for c in cnt:
        ax.plot(*np.r_[c[:, 0], c[:1, 0]].T, color="#eb6834", linewidth=1)
    if "raw_ground" in o and not np.allclose(o["raw_ground"], obs_ipm[len(obs) // 2 // 4 * 4]["pos"]):
        ax.scatter(*contact_pixel(o["mask"]), s=30, c="#2a78d6", zorder=3)
    x0, y0, x1, y1 = o["box"]
    ax.set_xlim(x0 - 40, x1 + 40)
    ax.set_ylim(y1 + 30, y0 - 30)
    ax.axis("off")
    ax.set_title(f"car {tid}, frame {o['k']}: detected blob (orange)\n{len(obs)} frames tracked, no labels",
                 fontsize=8.5, loc="left", color=INK)
    ax = fig.add_subplot(gs[row, 1])
    G = np.array([q["raw_ground"] for q in obs_ipm])
    ax.scatter(*G.T, s=6, c="#c3c2b7", label="IPM placements")
    ax.plot(*np.array([q["pos"] for q in obs_ipm]).T, color="#eb6834", linewidth=1.2, label="IPM track (smoothed)")
    ax.plot(*np.array([q["pos"] for q in obs]).T, color="#2a78d6", linewidth=1.6, label="after bundle adjustment")
    L = np.array([lab[q["k"]]["pos"] for q in obs if q["k"] in lab])
    ax.plot(*L.T, "k--", linewidth=1, label="labels (check only)")
    ax.set_aspect("equal")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(f"bird's-eye track: position {ev['position_error_cm']:.0f} cm, heading {ev['heading_error_deg']:.1f} deg",
                 fontsize=8.5, loc="left", color=INK)
    ax.tick_params(labelsize=6)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    fc = np.clip(cols[F].mean(1) * (0.45 + 0.55 * np.abs(fn @ np.array([0.35, -0.45, 0.82])))[:, None], 0, 1)
    C = mx.camera_in_object(seq.cams[o["k"]], o)
    az = np.degrees(np.arctan2(C[1], C[0]))
    for c, (elev, azim) in enumerate(((20, az + 25), (3, -90))):
        ax = fig.add_subplot(gs[row, 2 + c], projection="3d")
        ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc, edgecolors="none"))
        ax.scatter(*X.T, s=0.6, c="#2a78d6", alpha=0.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(0, 3)
        ax.set_box_aspect((1, 1, 0.5))
        ax.view_init(elev, azim)
        ax.axis("off")
        if c == 0:
            ax.set_title(f"3D model (+ tracked points), no labels\nmodel->LiDAR {ev['model_to_lidar_cm']:.0f} cm, "
                         f"LiDAR->model {ev['lidar_to_model_cm']:.0f} cm (world, our track)", fontsize=8.5, loc="left",
                         color=INK)
    ax = fig.add_subplot(gs[row, 4])
    oh = [q for i, q in enumerate(obs) if i % 4 == 0][len(obs) // 8]
    Vw = vs.world(V, oh["pos"], oh["yaw"])
    uv, depth = vs.project(seq.cams[oh["k"]], Vw)
    out = seq.frames[oh["k"]].copy()
    for i in np.argsort(-depth[F].mean(1)):
        cv2.fillConvexPoly(out, uv[F[i]].round().astype(np.int32), (fc[i][::-1] * 255).tolist(), lineType=cv2.LINE_AA)
    out = cv2.addWeighted(out, 0.85, seq.frames[oh["k"]], 0.15, 0)
    ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    x0, y0, x1, y1 = oh["box"]
    ax.set_xlim(x0 - 40, x1 + 40)
    ax.set_ylim(y1 + 30, y0 - 30)
    ax.axis("off")
    ax.set_title(f"held-out frame {oh['k']}: model on our track, IoU {ev['held_out_iou']:.2f}", fontsize=8.5,
                 loc="left", color=INK)


def main(targets):
    seq = vs.Sequence()
    seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    ground = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))[2]
    det = [blobs(f, seq.bg) for f in seq.frames]
    tracks = link_blobs([[b["box"] for b in d] for d in det])
    print(f"{sum(len(d) for d in det)} blobs, {len(tracks)} tracks of >= 10 frames", flush=True)
    results, rows = {}, []
    for tr in tracks:
        obs_ipm, speed = ipm_track(seq, det, tr)
        if speed < 0.15:                                      # m/frame: parked or waiting cars give no parallax
            continue
        tid = match_label(seq, obs_ipm)
        if tid is None or (targets and tid not in targets):
            continue
        pos_ipm, yaw_ipm = ipm_only_eval(seq, obs_ipm, tid)
        try:
            obs, X, ba = motion_and_points(seq, obs_ipm)
            V, F, cols, hull, axes = build_model(seq, obs, X)
        except Exception as e:                                # noqa: BLE001 - report and go on
            print(f"track for car {tid}: failed ({e})", flush=True)
            continue
        ev = evaluate(seq, obs, V, F, tid, ground)
        ev.update({"ipm_position_error_cm": pos_ipm, "ipm_heading_error_deg": yaw_ipm, "frames": len(obs), **ba})
        results[tid] = ev
        rows.append((tid, obs_ipm, obs, V, F, cols, X, ev))
        print(f"car {tid}: {len(obs)} frames | IPM track: {pos_ipm:.0f} cm, {yaw_ipm:.1f} deg -> after BA "
              f"{ev['position_error_cm']:.0f} cm, {ev['heading_error_deg']:.1f} deg | {ba['points']} points, reprojection "
              f"{ba['reprojection_px']:.2f} px | model->LiDAR {ev['model_to_lidar_cm']:.0f} cm, LiDAR->model "
              f"{ev['lidar_to_model_cm']:.0f} cm, held-out IoU {ev['held_out_iou']:.2f} || offset {ev['body_offset_cm']:.0f} cm removed: "
              f"track {ev['aligned_position_error_cm']:.0f} cm, model->LiDAR {ev['aligned_model_to_lidar_cm']:.0f} cm, "
              f"LiDAR->model {ev['aligned_lidar_to_model_cm']:.0f} cm", flush=True)
    json.dump(results, open(os.path.join(vd.OUT_DIR, "ipm_models_metrics.json"), "w"), indent=1)
    for p in range(0, len(rows), 4):
        chunk = rows[p:p + 4]
        fig = plt.figure(figsize=(22, 4.3 * len(chunk)))
        gs = fig.add_gridspec(len(chunk), 5, width_ratios=(1, 1, 1.1, 1.1, 1))
        for r, args in enumerate(chunk):
            car_row(fig, gs, r, seq, *args)
        fig.tight_layout()
        fig.savefig(os.path.join(vd.OUT_DIR, f"ipm_models_{p // 4 + 1}.png"), dpi=80, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
