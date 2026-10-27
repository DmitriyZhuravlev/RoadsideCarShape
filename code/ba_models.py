"""
Full 3D car models from one fixed roadside camera, with the car's own motion refined by bundle adjustment
(V2X-Seq, intersection A).

  1. tracks      : a Delaunay triangulation in every REF_STEP-th frame (corners inside the motion mask), every vertex
                   tracked +-SPAN frames with pyramidal Lucas-Kanade and a forward-backward check (tri_tracks);
  2. bundle adj. : unknowns = a correction (dx, dy, dyaw) of the car's pose in every frame (the car drives on the known
                   ground plane, which fixes the scale) and the 3D position of every track in the car frame; robust
                   (Huber) least squares on the reprojection errors, with a weak prior towards the smoothed labelled
                   cuboid poses; two rounds, outlier observations dropped in between;
  3. points      : the tracks that reproject within MAX_RES px and are seen under at least MIN_ANGLE deg;
  4. full model  : silhouette hull with the refined poses -> cuboid -> triangulation of the contact points ->
                   recursive subdivision pulled onto the hull and, on surfaces facing up, to the evidence of the car's
                   motion: the bundle-adjusted track points (corners: lights, plates, windows, wheels) plus dense-flow
                   triangle points and wide-baseline photo-consistency points, which also reach the textureless roof,
                   all with the refined poses -> colours from the video (car_models.build).
Checked against the car's LiDAR (evaluation only): points and models before and after the bundle adjustment.

Writes docs/ba_points_<tid>.png, docs/ba_models_cars_*.png, docs/ba_metrics.json and the viewer data in docs/viewer/.
"""
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import car_flow_planes as cfp
import car_fusion as cfu
import car_models as cm
import mesh_refine as mr
import mix3d as mx
import tri_tracks as tt
import v2x_seq_demo as vd
import v2x_seq_shape as vs

REF_STEP = 3
SPAN = 10
SIGMA_POS, SIGMA_YAW = 0.2, np.radians(1.0)       # prior towards the smoothed cuboid poses
MAX_RES = 2.0              # px
MIN_ANGLE = 1.5            # deg
INK, MUTED = "#1f1f1e", "#8a8984"


def ba_sparsity(n_frames, n_points, frame_idx, point_idx):
    m = len(frame_idx)
    obs_rows = np.repeat(np.arange(2 * m), 6)                              # every residual row touches 6 columns
    f = np.repeat(frame_idx, 2)
    p = np.repeat(point_idx, 2)
    cols = np.c_[3 * f, 3 * f + 1, 3 * f + 2, 3 * n_frames + 3 * p, 3 * n_frames + 3 * p + 1, 3 * n_frames + 3 * p + 2]
    prior_rows = 2 * m + np.repeat(np.arange(3 * n_frames), 3)          # a prior row may mix the 3 columns of its frame
    prior_cols = np.repeat(3 * np.arange(n_frames), 9) + np.tile(np.arange(3), 3 * n_frames)
    rows = np.r_[obs_rows, prior_rows]
    cols = np.r_[cols.ravel(), prior_cols]
    return sp.coo_matrix((np.ones(len(rows), int), (rows, cols)),
                         shape=(2 * m + 3 * n_frames, 3 * n_frames + 3 * n_points)).tocsr()


def project_obs(cams, obs, d, X, fi, pi):
    """Pixels of points X[pi] in frames fi with the pose corrections d (n_frames, 3)."""
    pos = np.array([o["pos"] for o in obs])[fi] + d[fi, :2]
    yaw = np.array([o["yaw"] for o in obs])[fi] + d[fi, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    P = X[pi]
    W = np.c_[pos[:, 0] + c * P[:, 0] - s * P[:, 1], pos[:, 1] + s * P[:, 0] + c * P[:, 1], P[:, 2]]
    KR = np.stack([cams[o["k"]][0] @ cams[o["k"]][1] for o in obs])[fi]
    Kt = np.stack([cams[o["k"]][0] @ cams[o["k"]][2] for o in obs])[fi]
    x = np.einsum("nij,nj->ni", KR, W) + Kt
    return x[:, :2] / x[:, 2:3]


def anisotropic_prior(d, ray_dirs, sigma_along, sigma_across, sigma_yaw):
    """Prior residuals of the pose corrections: along / across each frame's horizontal viewing ray, and heading."""
    perp = np.c_[-ray_dirs[:, 1], ray_dirs[:, 0]]
    return np.c_[np.einsum("ij,ij->i", d[:, :2], ray_dirs) / sigma_along,
                 np.einsum("ij,ij->i", d[:, :2], perp) / sigma_across, d[:, 2] / sigma_yaw]


def bundle_adjust(cams, obs, X0, fi, pi, uv, d0=None, anchor=None):
    """anchor = sigma (m) along the viewing ray: the trajectory and the points can otherwise slide together along the
    rays (weakly observed); None keeps the isotropic prior."""
    F, N = len(obs), len(X0)
    sigma = np.array([SIGMA_POS, SIGMA_POS, SIGMA_YAW])
    if anchor is not None:
        foot = np.array([-np.linalg.solve(cams[o["k"]][1], cams[o["k"]][2])[:2] for o in obs])
        rays = np.array([o["pos"] for o in obs]) - foot
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    def fun(p):
        d = p[:3 * F].reshape(F, 3)
        X = p[3 * F:].reshape(N, 3)
        r = (project_obs(cams, obs, d, X, fi, pi) - uv).ravel()
        prior = (d / sigma) if anchor is None else anisotropic_prior(d, rays, anchor, SIGMA_POS, SIGMA_YAW)
        return np.r_[r, prior.ravel()]

    p0 = np.r_[np.zeros(3 * F) if d0 is None else d0.ravel(), X0.ravel()]
    sol = least_squares(fun, p0, jac_sparsity=ba_sparsity(F, N, fi, pi), loss="huber", f_scale=1.0, x_scale="jac",
                        method="trf", tr_solver="lsmr", max_nfev=300)
    d = sol.x[:3 * F].reshape(F, 3)
    X = sol.x[3 * F:].reshape(N, 3)
    res = np.linalg.norm(project_obs(cams, obs, d, X, fi, pi) - uv, axis=1)
    return d, X, res


def collect_tracks(seq, obs):
    tracks = []
    for r in range(0, len(obs), REF_STEP):
        o = obs[r]
        P, _ = tt.reference_mesh(cv2.cvtColor(seq.frames[o["k"]], cv2.COLOR_BGR2GRAY), seq.mask(o) > 0)
        uv = tt.track(seq, obs, r, P, span=SPAN)
        tracks.append(uv[(~np.isnan(uv[:, :, 0])).sum(1) >= 3])
    return np.concatenate(tracks)


def triangulate(seq, obs, U):
    C = np.full(U.shape[:2] + (3,), np.nan)
    D = np.full(U.shape[:2] + (3,), np.nan)
    for j, o in enumerate(obs):
        ok = ~np.isnan(U[:, j, 0])
        if ok.any():
            C[ok, j], D[ok, j] = tt.rays(seq.cams[o["k"]], o, U[ok, j].astype(float))
    return tt.intersect_rays(C, D)


def with_corrections(obs, d):
    out = []
    for o, dd in zip(obs, d):
        q = dict(o)
        q["pos"] = np.asarray(o["pos"]) + dd[:2]
        q["yaw"] = float(o["yaw"] + dd[2])
        out.append(q)
    return out


def point_scores(S, L):
    if len(S) == 0:
        return {"points": 0, "accuracy_cm": float("nan"), "completeness_cm": float("nan")}
    return {"points": int(len(S)), "accuracy_cm": float(100 * np.median(cKDTree(L).query(S)[0])),
            "completeness_cm": float(100 * np.median(cKDTree(S).query(L)[0]))}


def motion_evidence(seq, gray, obs, train, hull, axes):
    """Dense-flow triangle points and wide-baseline photo-consistency points, with the refined poses: they reach the
    textureless top surfaces where no corner can be tracked."""
    by_k = {o["k"]: o for o in obs}
    V0, F0, box, _ = mr.coarse_mesh(hull, axes)
    V2, F2 = mr.refine(V0, F0, hull, axes, np.zeros((0, 3)), box, levels=2)[-1]
    photo = cm.photo_points(seq, gray, V2, F2, train)
    flow = []
    for o in train:
        if o["k"] + 1 in by_k:
            f = cfu.frame_model(seq, o, by_k[o["k"] + 1])
            if len(f["V"]):
                flow.append(mx.to_ground_frame(f["V"], o).mean(1))
    return photo, np.vstack(flow) if flow else np.zeros((0, 3))


def run_car(seq, tid, ground, gray):
    obs0 = tt.smooth_poses([o for o in seq.tracks[tid] if seq.usable(o)])
    l, w, h = (float(np.median([o["dims"][i] for o in obs0])) for i in range(3))
    lo, hi = np.array([-l / 2 - 0.3, -w / 2 - 0.3, -0.2]), np.array([l / 2 + 0.3, w / 2 + 0.3, max(h, 1.6) + 0.6])
    U = collect_tracks(seq, obs0)
    X0, ang0 = triangulate(seq, obs0, U)
    ok = ~np.isnan(X0[:, 0]) & np.all((X0 >= lo) & (X0 <= hi), axis=1)
    U, X0 = U[ok], X0[ok]
    pi, fi = np.nonzero(~np.isnan(U[:, :, 0]))
    uv = U[pi, fi]
    # initial points as the smoothed poses give them (for the comparison)
    res0 = np.linalg.norm(project_obs(seq.cams, obs0, np.zeros((len(obs0), 3)), X0, fi, pi) - uv, axis=1)
    worst0 = np.zeros(len(X0))
    np.maximum.at(worst0, pi, res0)
    keep0 = (worst0 < MAX_RES) & (ang0[ok] >= MIN_ANGLE)
    # bundle adjustment, two rounds
    d, X, res = bundle_adjust(seq.cams, obs0, X0, fi, pi, uv)
    good = res < 3 * MAX_RES
    d, X, res = bundle_adjust(seq.cams, obs0, X, fi[good], pi[good], uv[good], d0=d)
    fi, pi, uv = fi[good], pi[good], uv[good]
    obs = with_corrections(obs0, d)
    worst = np.zeros(len(X))
    np.maximum.at(worst, pi, res)
    n_obs = np.bincount(pi, minlength=len(X))
    Ub = np.full_like(U, np.nan)
    Ub[pi, fi] = uv
    _, ang = triangulate(seq, obs, Ub)
    keep = (worst < MAX_RES) & (n_obs >= 3) & (ang >= MIN_ANGLE) & np.all((X >= lo) & (X <= hi), axis=1)
    L0 = np.vstack([mx.to_ground_frame(cfp.lidar_on_car(seq, o, ground), o) for o in obs0])
    L = np.vstack([mx.to_ground_frame(cfp.lidar_on_car(seq, o, ground), o) for o in obs])
    pts = {"smoothed poses": point_scores(X0[keep0], L0), "bundle-adjusted": point_scores(X[keep], L)}
    pts["smoothed poses"]["median_angle_deg"] = float(np.median(ang0[ok][keep0])) if keep0.any() else 0.0
    pts["bundle-adjusted"]["median_angle_deg"] = float(np.median(ang[keep])) if keep.any() else 0.0
    pts["reprojection_px"] = {"before": float(np.median(res0)), "after": float(np.median(res))}
    pts["pose_change"] = {"pos_cm": float(100 * np.median(np.linalg.norm(d[:, :2], axis=1))),
                          "yaw_deg": float(np.degrees(np.median(np.abs(d[:, 2]))))}
    # full model on the refined poses
    train = [o for i, o in enumerate(obs) if i % 4]
    test = [o for i, o in enumerate(obs) if i % 4 == 0]
    axes = vs.grid(l, w, 1.6)
    hull = vs.height_field(vs.carve_votes(axes, [seq.view(o, cm.shaved(seq, o, cm.MASK_SHAVE)) for o in train])
                           >= vs.VOTE)
    photo, flow = motion_evidence(seq, gray, obs, train, hull, axes)
    E = np.vstack([X[keep], photo, flow])
    ev = {"obs": obs, "train": train, "test": test, "axes": axes, "hull": hull, "dims": (l, w), "E": E,
          "photo": photo, "flow": flow, "n_top": int((E[:, 2] > 0.8).sum())}
    model = cm.build(seq, ev, ground)
    return {"obs0": obs0, "obs": obs, "X0": X0[keep0], "X": X[keep], "L": L, "L0": L0, "points": pts, "model": model,
            "ev": ev}


def points_figure(path, tid, r):
    fig = plt.figure(figsize=(18, 5.6))
    Ls = r["L"][np.random.default_rng(0).permutation(len(r["L"]))[:3000]]
    lim = max(np.ptp(r["L"][:, 0]), np.ptp(r["L"][:, 1])) / 2 + 0.4
    for c, (name, X, key) in enumerate((("smoothed label poses", r["X0"], "smoothed poses"),
                                         ("bundle-adjusted poses", r["X"], "bundle-adjusted"))):
        for v, (elev, azim) in enumerate(((20, -60), (85, -90))):
            ax = fig.add_subplot(1, 4, 2 * c + v + 1, projection="3d")
            ax.scatter(*Ls.T, s=0.3, c="#d03b3b", alpha=0.25)
            ax.scatter(*X.T, s=2, c="#2a78d6")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-0.2, 2.2)
            ax.set_box_aspect((1, 1, 0.45))
            ax.view_init(elev, azim)
            ax.tick_params(labelsize=6)
            p = r["points"][key]
            ax.set_title(f"{name}: {p['points']} points\nto LiDAR {p['accuracy_cm']:.1f} cm, LiDAR to points "
                         f"{p['completeness_cm']:.1f} cm, rays {p['median_angle_deg']:.1f} deg", fontsize=9, loc="left",
                         color=INK)
    fig.suptitle(f"car {tid}: tracked points (blue) vs LiDAR (red); reprojection "
                 f"{r['points']['reprojection_px']['before']:.2f} -> {r['points']['reprojection_px']['after']:.2f} px, "
                 f"poses moved by {r['points']['pose_change']['pos_cm']:.0f} cm / {r['points']['pose_change']['yaw_deg']:.2f} deg",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def main(tids):
    seq = vs.Sequence()
    seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    ground = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))[2]
    tids = tids or mr.suitable_cars(seq)
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in seq.frames]
    prev = json.load(open(os.path.join(vd.OUT_DIR, "models_metrics.json")))["per_car"]
    results, models = {}, {}
    for tid in tids:
        r = run_car(seq, tid, ground, gray)
        p, m = r["points"], r["model"]["rows"]
        f = m[f"level {cm.LEVELS}"]
        old = prev.get(tid, {}).get(f"level {cm.LEVELS}", {})
        print(f"car {tid}: reprojection {p['reprojection_px']['before']:.2f} -> {p['reprojection_px']['after']:.2f} px, poses "
              f"moved {p['pose_change']['pos_cm']:.0f} cm / {p['pose_change']['yaw_deg']:.2f} deg | points "
              f"{p['smoothed poses']['points']} -> {p['bundle-adjusted']['points']}, to LiDAR "
              f"{p['smoothed poses']['accuracy_cm']:.1f} -> {p['bundle-adjusted']['accuracy_cm']:.1f} cm, rays "
              f"{p['smoothed poses']['median_angle_deg']:.1f} -> {p['bundle-adjusted']['median_angle_deg']:.1f} deg | model: "
              f"compl {old.get('completeness_cm', float('nan')):.1f} -> {f['completeness_cm']:.1f} cm, top "
              f"{old.get('top_err_cm', float('nan')):+.0f} -> {f['top_err_cm']:+.0f} cm, IoU {old.get('iou', float('nan')):.2f} -> "
              f"{f['iou']:.2f}", flush=True)
        results[tid] = {"points": p, "model": m}
        models[tid] = (r["model"], r["ev"])
        points_figure(os.path.join(vd.OUT_DIR, f"ba_points_{tid}.png"), tid, r)
    keys = ("completeness_cm", "top_err_cm", "iou", "precision_cm")
    summary = {"model (bundle-adjusted)": {k: float(np.mean([results[t]["model"][f"level {cm.LEVELS}"][k] for t in results]))
                                           for k in keys},
               "model (previous, label poses)": {k: float(np.mean([prev[t][f"level {cm.LEVELS}"][k] for t in results
                                                                   if t in prev])) for k in keys},
               "points to LiDAR, smoothed poses (cm)": float(np.nanmean([results[t]["points"]["smoothed poses"]["accuracy_cm"]
                                                                         for t in results])),
               "points to LiDAR, bundle-adjusted (cm)": float(np.nanmean([results[t]["points"]["bundle-adjusted"]["accuracy_cm"]
                                                                          for t in results]))}
    summary["model (bundle-adjusted)"]["abs_top_err_cm"] = float(np.mean([abs(results[t]["model"][f"level {cm.LEVELS}"]["top_err_cm"])
                                                                          for t in results]))
    summary["model (previous, label poses)"]["abs_top_err_cm"] = float(np.mean([abs(prev[t][f"level {cm.LEVELS}"]["top_err_cm"])
                                                                                for t in results if t in prev]))
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "per_car": results}, open(os.path.join(vd.OUT_DIR, "ba_metrics.json"), "w"), indent=1)
    t_list = list(models)
    for i in range(0, len(t_list), 8):
        cm.cars_figure(os.path.join(vd.OUT_DIR, f"ba_models_cars_{i // 8 + 1}.png"), seq, models, t_list[i:i + 8])
    if len(models) > 2:
        cm.export_meshes(os.path.join(vd.OUT_DIR, "viewer", "meshes.json"), models)
        names = ["silhouette hull", "silhouette-guarded"] + [f"level {k}" for k in range(cm.LEVELS + 1)]
        json.dump({n: {k: {"mean": float(np.mean([m["rows"][n][k] for m, _ in models.values()]))}
                       for k in ("precision_cm", "completeness_cm", "top_err_cm", "iou")} for n in names},
                  open(os.path.join(vd.OUT_DIR, "viewer", "summary.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
