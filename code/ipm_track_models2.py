"""
3D models of the moving cars at intersection A without annotation - placement from the whole silhouette.

ipm_track_models.py placed each car by the IPM of ONE ground-contact pixel; at 25-65 m one pixel is 5-20 cm of
ground and the lowest pixel wanders (reflection remnants, bumper, neighbours), so the placement was off by metres.
Here the placement comes from the classical tracker (v2x_seq_classical.run, no labels, generic car-size prior):
  - every blob is lifted in closed form: the VP-tangent quadrangle of its whole silhouette on the IPM plane
    (lifting.vp_bounding_quadrangle), the face candidates, the motion picking the candidate and its front, and
    solve_length giving the centre (Efficient3D / moving.tex);
  - a Kalman filter on the ground plane with the pixel-error covariance, Hungarian association, splitting of merged
    blobs by the predicted cuboids, the wet-road reflection cut, stopped-car templates; Rauch-Tung-Striebel smoothing.
Then, per moving track:
  - mask = the blob (reflection cut, HSV shadow test) clipped to its own projected cuboid (separates neighbours);
  - corners tracked inside the mask, object-level bundle adjustment from the tracker's poses (ba_models);
  - silhouette hull -> cuboid -> triangulation of the contact points -> recursive subdivision pulled to the tracked
    points -> colours (ipm_track_models.build_model).
Labels and LiDAR only to name and score the cars (ipm_track_models.match_label / evaluate).

Writes docs/ipm2_models_*.png and docs/ipm2_models_metrics.json.
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

import colour_flow_split as cfs
import ipm_track_models as it
import sil_refine as sr
import wheels as wh
import tri_smooth as ts
import tri_tracks as tt
from scipy.spatial import cKDTree
import v2x_seq_classical as vc
import v2x_seq_demo as vd
import v2x_seq_shape as vs

SPLIT = None
SIL = os.environ.get("SIL", "0") == "1"                 # silhouette refinement of the placement (neutral on average)
WHEELS = os.environ.get("WHEELS", "1") == "1"           # tyre contacts fix the depth along the viewing ray
STITCH = os.environ.get("STITCH", "1") == "1"           # stitched triangulations instead of bundle-adjusted points
SUFFIX = ("_split" if os.environ.get("SPLIT", "1") == "1" else "") + ("_sil" if SIL else "") + ("_wheels" if WHEELS else "") + ("_stitch" if STITCH else "")
MIN_FRAMES = 12
MIN_SPEED = 1.5            # m/s: parked and waiting cars give no parallax


def full_mask(seq, b, frame):
    """A split blob as a full-frame mask, with the HSV shadow test."""
    x0, y0, x1, y1 = b["box"]
    pad = 4
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(frame.shape[1], x1 + pad), min(frame.shape[0], y1 + pad)
    m = np.zeros((Y1 - Y0, X1 - X0), bool)
    m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = b["m"] > 0
    m = it.shadow_free(frame[Y0:Y1, X0:X1], seq.bg[Y0:Y1, X0:X1], m)
    full = np.zeros(frame.shape[:2], bool)
    full[Y0:Y1, X0:X1] = m
    ys, xs = np.nonzero(full)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else b["box"]
    return {"box": box, "mask": full}


def stitched_points(seq, obs, every=6, span=8, lam=10.0):
    """Label-free stitched triangulations: in every `every`-th frame a Delaunay triangulation of corners inside the
    car's own mask, its vertices tracked +-span frames inside the masks, all depths solved together with the
    smoothness condition across shared edges (tri_smooth), on the given (tracker) motion - no bundle adjustment."""
    pts = []
    for r in range(span // 2, len(obs), every):
        o_r = obs[r]
        gray = cv2.cvtColor(seq.frames[o_r["k"]], cv2.COLOR_BGR2GRAY)
        try:
            P, tri = tt.reference_mesh(gray, o_r["mask"])
        except Exception:                                   # noqa: BLE001 - too few corners in this frame
            continue
        if len(tri) < 4:
            continue
        uv = it.track_corners(seq, obs, r, P, span)
        C = np.full(uv.shape[:2] + (3,), np.nan)
        D = np.full(uv.shape[:2] + (3,), np.nan)
        for j, o in enumerate(obs):
            ok = ~np.isnan(uv[:, j, 0])
            if ok.any():
                C[ok, j], D[ok, j] = tt.rays(seq.cams[o["k"]], o, uv[ok, j].astype(float))
        X0, ang = tt.intersect_rays(C, D)
        Cr, Dr = tt.rays(seq.cams[o_r["k"]], o_r, P.astype(float))
        Cr = np.array(Cr)
        Dr = Dr / np.linalg.norm(Dr, axis=1, keepdims=True)
        s0 = np.einsum("ni,ni->n", X0 - Cr, Dr)
        tracked = ~np.isnan(s0) & (ang >= 1.0)
        if tracked.sum() < 6:
            continue
        s0[~tracked] = np.nan
        _, j = cKDTree(P[tracked]).query(P[~tracked], k=min(4, tracked.sum()))
        s0[~tracked] = np.median(s0[tracked][np.atleast_2d(j)], axis=1) if (~tracked).any() else s0[~tracked]
        s = ts.stitched(seq, obs, P, tri, uv, s0, Cr, Dr, lam, ref=r)
        X = Cr + s[:, None] * Dr
        X = X[(np.abs(X[:, 0]) < 4) & (np.abs(X[:, 1]) < 2.5) & (X[:, 2] > -0.3) & (X[:, 2] < 2.6)]
        pts.append(X)
    return np.vstack(pts) if pts else np.zeros((0, 3))


def clip_to_hull(mask, hull_box, hull_m, dilate_px=15):
    """The mask restricted to a projected cuboid, dilated by dilate_px."""
    H, W = mask.shape
    x0, y0, x1, y1 = hull_box
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    full = np.zeros((H, W), np.uint8)
    if X1 > X0 and Y1 > Y0:
        full[Y0:Y1, X0:X1] = hull_m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    region = cv2.dilate(full, np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)) > 0
    return mask & region


def tracker_observations(seq, det):
    """Per confirmed track: per frame the RTS-smoothed centre and heading, and its clipped blob mask."""
    out_online, tracks, _ = vc.run(seq.frames, seq.times, seq.bg, seq.cams[0], (1, 2, 3, 4),
                                   detector=(lambda k, frame: SPLIT[k]) if SPLIT else None)
    geo = vc.Geometry(seq.cams[0], seq.shape)
    result = []
    for tr in tracks:
        lo, hi = min(tr.measured), max(tr.measured)
        states = {k: x for k, x in vc.rts_smooth(tr.log).items() if lo <= k <= hi}
        obs = []
        for k in sorted(states):
            x = states[k]
            speed = float(np.hypot(x[2], x[3]))                         # m/s (the filter state is metric)
            yaw = math.atan2(x[3], x[2]) if speed > 0.5 else tr.yaw
            h = geo.hull(x[:2], yaw)
            if h is None:
                continue
            best, score = None, 0
            for b in det[k]:
                inter = clip_to_hull(b["mask"], h["box"], h["m"], 0).sum()
                if inter > score:
                    best, score = b, inter
            if best is None or score < 0.3 * h["area"]:
                continue
            m = clip_to_hull(best["mask"], h["box"], h["m"])
            ys, xs = np.nonzero(m)
            if len(xs) < it.MIN_AREA:
                continue
            obs.append({"k": int(k), "pos": x[:2].copy(), "yaw": float(yaw), "speed": speed, "mask": m,
                        "box": (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), "raw_ground": x[:2].copy()})
        # keep the longest run of consecutive frames
        runs, cur = [], []
        for o in obs:
            if cur and o["k"] != cur[-1]["k"] + 1:
                runs.append(cur)
                cur = []
            cur.append(o)
        if cur:
            runs.append(cur)
        if runs:
            best = max(runs, key=len)
            if len(best) >= MIN_FRAMES and np.median([o["speed"] for o in best]) >= MIN_SPEED:
                yaw = np.unwrap([o["yaw"] for o in best])
                for o, y in zip(best, yaw):
                    o["yaw"] = float(y)
                result.append(best)
    return result


def summary_figure(path, results):
    """Label-free placement: one contact pixel (ipm_track_models) vs the whole silhouette + tracker (this file)."""
    v1 = json.load(open(os.path.join(vd.OUT_DIR, "ipm_models_metrics.json")))
    v2 = [r for r in results.values() if np.isfinite(r["lidar_to_model_cm"])]
    v1 = [r for r in v1.values() if np.isfinite(r["lidar_to_model_cm"])]
    keys = (("ipm_position_error_cm", "tracker_position_error_cm", "placement error (cm)"),
            ("ipm_heading_error_deg", "tracker_heading_error_deg", "heading error (deg)"),
            ("lidar_to_model_cm", "lidar_to_model_cm", "LiDAR -> model, world (cm)"),
            ("held_out_iou", "held_out_iou", "held-out silhouette IoU"))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for ax, (k1, k2, title) in zip(axes, keys):
        a = np.array([r[k1] for r in v1])
        b = np.array([r[k2] for r in v2])
        parts = ax.boxplot([a, b], widths=0.5, patch_artist=True, showfliers=False,
                           medianprops=dict(color="#1f1f1e"))
        for patch, c in zip(parts["boxes"], ("#c3c2b7", "#2a78d6")):
            patch.set_facecolor(c)
        for i, v in enumerate((a, b)):
            ax.annotate(f"median {np.median(v):.2f}" if "IoU" in title else f"median {np.median(v):.0f}",
                        (i + 1, np.median(v)), xytext=(28, 0), textcoords="offset points", fontsize=8, va="center")
        ax.set_xticks([1, 2], ["one contact pixel\n(IPM)", "whole silhouette\n(closed form + tracker)"], fontsize=8.5)
        ax.set_title(f"{title}: {len(a)} / {len(b)} tracks", fontsize=9.5, loc="left", color=it.INK)
        vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def main(targets):
    seq = vs.Sequence()
    seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    ground = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))[2]
    global SPLIT
    if os.environ.get("SPLIT", "1") == "1":                 # velocity + colour separation of touching cars
        Fs = cfs.flows(seq.frames)
        SPLIT = [cfs.detect(f, seq.bg, F) for f, F in zip(seq.frames, Fs)]
        det = [[full_mask(seq, b, f) for b in d] for d, f in zip(SPLIT, seq.frames)]
        del Fs
    else:
        det = [it.blobs(f, seq.bg) for f in seq.frames]
    tracks = tracker_observations(seq, det)
    print(f"{len(tracks)} moving tracks of >= {MIN_FRAMES} frames from the classical tracker", flush=True)
    results, rows, seen = {}, [], set()
    for obs_trk in tracks:
        tid = it.match_label(seq, obs_trk)
        if tid is None or (targets and tid not in targets):
            continue
        key = tid if tid not in seen else f"{tid}#{len([s for s in seen if str(s).startswith(tid)])}"
        seen.add(key)
        pos0, yaw0 = it.ipm_only_eval(seq, obs_trk, tid)
        pos_s, yaw_s = pos0, yaw0
        if SIL:
            obs_trk, _ = sr.refine(seq, obs_trk)
            pos_s, yaw_s = it.ipm_only_eval(seq, obs_trk, tid)
        if WHEELS:                                         # layer 0: depth from the tyres' ground contacts
            obs_trk, _ = wh.refine_depth(seq, obs_trk)
            pos_s, yaw_s = it.ipm_only_eval(seq, obs_trk, tid)
        try:
            if STITCH:                                      # no bundle adjustment: it biases the metric depth
                obs, X = obs_trk, stitched_points(seq, obs_trk)
                ba = {"reprojection_px": float("nan"), "points": int(len(X)), "pose_change_cm": 0.0, "yaw_change_deg": 0.0}
            else:
                obs, X, ba = it.motion_and_points(seq, obs_trk, sigma_pos=0.4, sigma_yaw_deg=3.0, recentre=False)
            V, F, cols, hull, axes = it.build_model(seq, obs, X)
        except Exception as e:                                          # noqa: BLE001 - report and go on
            print(f"car {key}: failed ({e})", flush=True)
            continue
        ev = it.evaluate(seq, obs, V, F, tid, ground)
        ev.update({"tracker_position_error_cm": pos0, "tracker_heading_error_deg": yaw0,
                   "silhouette_position_error_cm": pos_s, "silhouette_heading_error_deg": yaw_s, "frames": len(obs), **ba})
        results[key] = ev
        rows.append((tid, obs_trk, obs, V, F, cols, X, ev))
        print(f"car {key}: {len(obs)} frames | tracker {pos0:.0f} cm, {yaw0:.1f} deg -> silhouettes {pos_s:.0f} cm, "
              f"{yaw_s:.1f} deg -> after BA "
              f"{ev['position_error_cm']:.0f} cm, {ev['heading_error_deg']:.1f} deg | {ba['points']} points, "
              f"reprojection {ba['reprojection_px']:.2f} px | model->LiDAR {ev['model_to_lidar_cm']:.0f} cm, "
              f"LiDAR->model {ev['lidar_to_model_cm']:.0f} cm, IoU {ev['held_out_iou']:.2f} | model centre "
              f"{ev['model_centre_error_cm']:.0f} cm, axis {ev['model_heading_error_deg']:.1f} deg || offset "
              f"{ev['body_offset_cm']:.0f} cm: track {ev['aligned_position_error_cm']:.0f} cm, LiDAR->model "
              f"{ev['aligned_lidar_to_model_cm']:.0f} cm", flush=True)
    keys = ("model_centre_error_cm", "model_heading_error_deg", "tracker_position_error_cm", "tracker_heading_error_deg", "silhouette_position_error_cm",
            "silhouette_heading_error_deg", "position_error_cm", "heading_error_deg",
            "model_to_lidar_cm", "lidar_to_model_cm", "held_out_iou", "aligned_position_error_cm",
            "aligned_lidar_to_model_cm", "points")
    summary = {k: float(np.nanmedian([r[k] for r in results.values()])) for k in keys}
    summary["cars"] = len(results)
    print("median over cars:", json.dumps(summary, indent=1))
    json.dump({"summary": summary, "per_car": results}, open(os.path.join(vd.OUT_DIR, f"ipm2_models_metrics{SUFFIX}.json"), "w"),
              indent=1)
    rows = [r for r in rows if np.isfinite(r[-1]["lidar_to_model_cm"])]           # cars the LiDAR saw
    rows.sort(key=lambda r: r[-1]["lidar_to_model_cm"])
    summary_figure(os.path.join(vd.OUT_DIR, f"ipm2{SUFFIX}_summary.png"), results)
    for p in range(0, len(rows), 4):
        chunk = rows[p:p + 4]
        fig = plt.figure(figsize=(22, 4.3 * len(chunk)))
        gs = fig.add_gridspec(len(chunk), 5, width_ratios=(1, 1, 1.1, 1.1, 1))
        for r, args in enumerate(chunk):
            it.car_row(fig, gs, r, seq, *args)
        fig.tight_layout()
        fig.savefig(os.path.join(vd.OUT_DIR, f"ipm2{SUFFIX}_models_{p // 4 + 1}.png"), dpi=80, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
