"""Build whole_scene-format dumps from the precise annotation (masks + cuboids), so whole_scene_cars.py
fuses them unchanged: the same windows, track solve, cabin and colours as the main pipeline, seeded by
the improved masks instead of its own detection.

    V2X_SEQ_DIR=<seq> python make_dumps.py NAME [TRACK ...]      # default: every car track with enough masks

Per track: a window of 17 consecutive video frames at every STRIDE-th annotated frame (default 2; the reported
runs use STRIDE=1); the vertices are
seeded inside the tool's mask at the reference frame and tracked by the pipeline's own tracker
(ws.track_all); the joint solve (ws.solve_object, the paper's two rounds with the residual rejection)
places them; the det records follow whole_scene.dump_objects' format exactly (including ba.U, which
solved_motion reads even with BA off).  The cuboid bounds the solved points: its footprint (the image-clock
pose) filters the depth-ambiguous outliers before the fusion.
"""
import json
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.spatial import Delaunay

import v2x_seq_demo as vd
import v2x_seq_shape as vs
import whole_scene as ws
import ego_sequence as es

NAME = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "Adense"
TRACKS = sys.argv[2:] or None
D = os.path.join("docs", f"precise_ann_{NAME}")
OUT = os.path.join("docs", f"ann_objects_{NAME}")
CAR = ("Car", "Van", "Truck", "Bus")
WIN = 8                                              # frames on each side of the reference (17 frames)
STRIDE = int(os.environ.get("STRIDE", "2"))          # a window at every STRIDE-th annotated frame
GLASS_OFF = os.environ.get("GLASS_OFF", "0") == "1"  # seed the painted body only (no gain: the tyres anchor the solve)
VIS_MIN = float(os.environ.get("VIS_MIN", "0.55"))  # reference frames: the car's mask over its cuboid's projection
KNOWN_MOTION = os.environ.get("KNOWN_MOTION", "0") == "1"   # the window's motion fixed to the cuboids' (too coarse at 60 m)
CUBOID_SCALE = os.environ.get("CUBOID_SCALE", "1") == "1"   # the window's scale (about the camera) from its cuboid


def corners_of(center, yaw, lwh):
    c, s = np.cos(yaw), np.sin(yaw)
    f, n = np.array([c, s]), np.array([-s, c])
    L, W, H = lwh
    return np.array([[*(center[:2] + dx * f + dy * n), center[2] + dz] for dx in (-L / 2, L / 2)
                     for dy in (-W / 2, W / 2) for dz in (-H / 2, H / 2)])


def cuboid_motion(poses, ks, r):
    """The window's planar motion from the cuboid poses {k: (xy, yaw)} (missing frames interpolated), in
    whole_scene.solve_object's form: m_init(c) -> (F, 3) of (theta, tx, ty) about the centre c, X_j = R (X - c) + c + t;
    from X_j = R(yaw_j - yaw_r) (X - p_r) + p_j: theta = yaw_j - yaw_r, t = p_j - c + R(theta) (c - p_r).
    None if the cuboids cover less than 80 % of the window."""
    kk = np.array(sorted(poses))
    if len(kk) < 2 or np.isin(ks, kk).mean() < 0.8:
        return None
    P = np.array([poses[k][0] for k in kk], float)
    Y = np.unwrap([poses[k][1] for k in kk])
    pk = np.c_[np.interp(ks, kk, P[:, 0]), np.interp(ks, kk, P[:, 1])]
    yk = np.interp(ks, kk, Y)

    def m_init(c):
        M = np.zeros((len(ks), 3))
        for j in range(len(ks)):
            th = yk[j] - yk[r]
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            M[j] = (th, *(pk[j] - c + R @ (c - pk[r])))
        return M
    return m_init


def solve_window(cams, fids, k0, m0, grays_cache, poses=None):
    """The window's joint solve seeded by the tool's mask: (sol, uv, P, tri, lab, img0, ks) or None.  With the
    track's cuboid poses (KNOWN_MOTION) the motion is the cuboids' and only the depths are solved: a triangulation
    along the reference rays with the car's own travel as the baseline (a lone window of a side-on car 60 m away
    cannot fix its depth: its own motion estimate put it 1-4 m off the cuboid)."""
    grays = {}
    for k in range(max(0, k0 - WIN), min(len(fids), k0 + WIN + 1)):
        if k not in grays_cache:
            img = cv2.imread(os.path.join(vd.DATA, "image", f"{fids[k]}.jpg"))
            if img is None:
                continue                         # a frame without its image (DAIR-V2X-C skips some)
            grays_cache[k] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ks = sorted(grays_cache)
    ks = [k for k in ks if k0 - WIN <= k <= k0 + WIN]
    if len(ks) < 9 or k0 not in ks:
        return None
    r = ks.index(k0)
    fid0 = fids[k0]
    img0 = cv2.imread(os.path.join(vd.DATA, "image", f"{fid0}.jpg"))
    # the vertices on the painted body only, as in the main pipeline (whole_scene.body_mask): the glass reflects the
    # sky and the scene, its tracked texture is no surface; the cabin comes from the silhouettes
    body0 = ws.body_mask(img0, m0, de_merge=ws.BODY_DE) if GLASS_OFF else m0
    er = cv2.erode(body0.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    yy, xx = np.nonzero(er)
    sel = (yy % 5 == 0) & (xx % 5 == 0)
    P = np.c_[xx[sel], yy[sel]].astype(np.float32)
    if len(P) < ws.MIN_VERTS:
        return None
    if len(P) > 900:                     # the solve's cost grows with the vertex count; the fusion averages
        st = int(np.ceil(len(P) / 900))  # many windows, so 900 well-tracked points are plenty
        P = P[::st]
    uv = ws.track_all({i: grays_cache[k] for i, k in enumerate(ks)}, r, P)
    tl = (~np.isnan(uv).any(2)).sum(1)
    good = tl >= max(6, int(0.6 * len(ks)))
    if good.sum() < ws.MIN_VERTS:
        return None
    P, uv = P[good], uv[good]
    tri = Delaunay(P).simplices
    lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)[P[:, 1].astype(int), P[:, 0].astype(int)].astype(float)
    contact_y = float(np.nonzero(m0.any(1))[0].max()) - 1.0     # the wheels' line (the wet road mirrors below)
    # the anchors: the tyres' contact row; without the tyres (GLASS_OFF seeds the colour body only) the body's lower
    # edge at the ride height, as in the main pipeline
    anchor = dict(ride=ws.RIDE) if GLASS_OFF else dict(contact_y=contact_y)
    m_init = cuboid_motion(poses, ks, r) if KNOWN_MOTION and poses else None
    if m_init is not None:
        anchor.update(m_init=m_init, fix_motion=True)
    cams = [cams[k] for k in ks]

    def solve_once(uv_, tri_, lab_):
        return ws.solve_object(cams, list(range(len(ks))), r, uv_, tri_, lab_, **anchor)

    sol = solve_once(uv, tri, lab)
    if sol is None:
        return None
    keep = sol["vertex_res"] <= max(1.5, 2.5 * np.median(sol["vertex_res"]))   # the paper's second round
    verts2 = sol["verts"][keep]
    uv, P, lab = uv[verts2], P[verts2], lab[verts2]
    tri = Delaunay(P).simplices
    sol = solve_once(uv, tri, lab)
    if sol is None or sol["speed_mps"] < ws.MIN_SPEED or len(sol["verts"]) < ws.MIN_VERTS:
        return None
    return sol, uv, P, tri, lab, img0, ks, r


def place_in_cuboid(sol, e0, margin=0.15):
    """The window's scale from its cuboid.  A static camera sees a window and the same window scaled about the camera
    centre identically in every frame (motion and depths scaled alike): the image fixes the shape, not the scale.
    The ground anchors guess it, and at 60-70 m a few per cent of scale move the car by metres; the cuboid knows
    where the car is.  The scale (0.8-1.2) that puts the most body vertices inside the cuboid (+margin), the middle
    of the best range; sol is rescaled in place (depths, motion, centre, the track solve's problem).  Returns it."""
    p = e0.get("image_clock", e0)
    c, yw, (L, W, H) = np.asarray(p["center"][:2], float), float(p["yaw"]), e0["lwh"]
    Rw = np.array([[np.cos(yw), np.sin(yw)], [-np.sin(yw), np.cos(yw)]])
    O = np.asarray(sol["ba"]["Cr"], float)[0]          # the rays' origin: the camera centre
    X = sol["X"][~np.isin(sol["verts"], sol["mirror"])]
    grid = np.arange(0.8, 1.2001, 0.002)
    inside = []
    for k in grid:
        Y = O + k * (X - O)
        d = (Y[:, :2] - c) @ Rw.T
        inside.append(np.mean((np.abs(d[:, 0]) <= L / 2 + margin) & (np.abs(d[:, 1]) <= W / 2 + margin) &
                              (Y[:, 2] >= -margin) & (Y[:, 2] <= H + margin)))
    inside = np.array(inside)
    k = float(np.median(grid[inside >= inside.max() - 0.02]))
    sol["X"] = O + k * (sol["X"] - O)
    M = np.asarray(sol["M"], float).copy()
    M[:, 1:] *= k                                       # X_j = R (X - c) + c + t  ->  c' = O + k (c - O), t' = k t
    sol["M"], sol["centre"] = M, O[:2] + k * (np.asarray(sol["centre"], float) - O[:2])
    sol["speed_mps"] *= k
    b = dict(sol["ba"])
    b["s"], b["lo"], b["hi"] = k * np.asarray(b["s"]), k * np.asarray(b["lo"]), k * np.asarray(b["hi"])
    b["z_anchor"] = O[2] + k * (b["z_anchor"] - O[2])
    b["Xc"] = O + k * (np.asarray(b["Xc"], float).reshape(-1, 3) - O) if len(b["Xc"]) else b["Xc"]
    sol["ba"] = b
    sol["cuboid_scale"], sol["cuboid_inside"] = k, float(inside.max())
    return k


def build_det(k0, ks, r, sol, uv, tri, img0, m0, e0, cam_c, cams):
    """One det record, whole_scene.dump_objects' format: the rows are sol["verts"] (sorted) into the round-2
    arrays, the triangles re-indexed onto them; then the cuboid's footprint drops the depth outliers."""
    if CUBOID_SCALE:
        place_in_cuboid(sol, e0)
    n = len(sol["verts"])
    v = sol["verts"]                                    # sorted row indices into the round-2 arrays
    uv_v = uv[v]                                        # (n, F, 2)
    # sol["mirror"] and sol["bottom"] hold row indices into the round-2 arrays (as sol["verts"]): local ones here
    body = ~np.isin(v, sol["mirror"])                   # the mirror vertices are not the car's body
    bottom = np.isin(v, sol["bottom"])                  # the ground anchors (the body's lowest vertices)
    X = sol["X"]                                        # (n, 3) at the reference frame
    # the cuboid's footprint filters the depth-ambiguous outliers (its image-clock pose)
    cc = corners_of(e0["image_clock"]["center"] if "image_clock" in e0 else e0["center"],
                    e0["image_clock"]["yaw"] if "image_clock" in e0 else e0["yaw"], e0["lwh"])
    uvc, zc = vs.project(cams[k0], cc)
    keep_rows = np.ones(n, bool)
    if (zc > 0.3).all():
        c_cub = np.array(e0["image_clock"]["center"][:2] if "image_clock" in e0 else e0["center"][:2])
        yw = float(e0["image_clock"]["yaw"] if "image_clock" in e0 else e0["yaw"])
        d_xy = (X[:, :2] - c_cub) @ np.array([[np.cos(yw), np.sin(yw)], [-np.sin(yw), np.cos(yw)]])
        keep_rows = (np.abs(d_xy[:, 0]) <= e0["lwh"][0] / 2 + 1.0) & (np.abs(d_xy[:, 1]) <= e0["lwh"][1] / 2 + 0.8)
    # the vertices whose reference projection falls outside the tool's mask are the along-ray fringe:
    # their colours are the road and their depths the noise - drop them before the fusion
    uv_all_ref = np.round(uv_v[:, r]).astype(int)
    q_all = np.clip(uv_all_ref, [0, 0], [img0.shape[1] - 1, img0.shape[0] - 1])
    keep_rows = keep_rows & (m0[q_all[:, 1], q_all[:, 0]] > 0)
    keep_rows |= bottom                                 # the anchors fix the scale of the track solve: kept
    if keep_rows.sum() < ws.MIN_VERTS:
        return None
    X, body, bottom = X[keep_rows], body[keep_rows], bottom[keep_rows]
    uv_v = uv_v[keep_rows]
    tri_keep = keep_rows[tri].all(1)
    remap = -np.ones(len(keep_rows), int)
    remap[np.nonzero(keep_rows)[0]] = np.arange(int(keep_rows.sum()))
    tri_det = remap[tri[tri_keep]]
    uv_det = uv_v                                       # already filtered by the cuboid's footprint
    uv_ref = np.round(uv_det[:, r]).astype(int)
    q = np.clip(uv_ref, [0, 0], [img0.shape[1] - 1, img0.shape[0] - 1])
    bgr = img0[q[:, 1], q[:, 0]]
    M = np.asarray(sol["M"], float)
    F = M.shape[0]
    a, b = max(0, r - 2), min(F - 1, r + 2)
    vel = (M[b, 1:] - M[a, 1:]) / max(ks[b] - ks[a], 1)
    return {"k": k0, "track_tid": str(e0.get("track")), "X": X, "uv": uv_ref.astype(float), "bgr": bgr, "tri": tri_det,
            "tri_bgr": ws.triangle_colours(img0, uv_ref.astype(float), tri_det),
            "body": body, "filled": np.zeros(len(X), bool), "bottom": bottom,
            "vel": vel, "speed_mps": float(sol["speed_mps"]), "camera": cam_c,
            "reprojection_px": float(sol["reprojection_px"]),
            "ks": list(ks), "r": r, "M": M, "centre": np.asarray(sol["centre"], float),
            "cuboid_scale": sol.get("cuboid_scale", 1.0), "cuboid_inside": sol.get("cuboid_inside", np.nan),
            "ba": ba_subset(sol["ba"], keep_rows)}


def ba_subset(ba, keep):
    """The window's track-solve problem (whole_scene.solve_object's "ba") on the kept vertices only, so that
    car_ba.py can solve the track as for the main pipeline's dumps: the per-vertex arrays filtered, the surface
    pairs and ground anchors whose vertices all survive re-indexed, the contacts unchanged."""
    remap = -np.ones(len(keep), int)
    remap[np.nonzero(keep)[0]] = np.arange(int(keep.sum()))
    pk = keep[ba["t1"]].all(1) & keep[ba["t2"]].all(1) if len(ba["t1"]) else np.zeros(0, bool)
    bi = np.asarray(ba["bi"], int)
    return {**{k: np.asarray(ba[k])[keep] for k in ("U", "Cr", "Dr", "s", "lo", "hi")},
            "t1": remap[ba["t1"][pk]], "t2": remap[ba["t2"][pk]], "w": np.asarray(ba["w"])[pk],
            "bi": remap[bi[keep[bi]]], "z_anchor": ba["z_anchor"], "Xc": ba["Xc"], "Uc": ba["Uc"]}


def main():
    os.makedirs(OUT, exist_ok=True)
    import ego_sequence as es
    seq, ground = es.open_sequence()
    fids, cams = seq.fids, seq.cams
    K0, R0, t0 = cams[0]
    cam_c = -np.linalg.solve(R0, t0)                   # the camera centre in the world (the ground frame)
    tracks = {}
    for fn in sorted(os.listdir(os.path.join(D, "cuboids"))):
        fid = fn[:-5]
        for i, e in enumerate(json.load(open(os.path.join(D, "cuboids", fn)))):
            if e["type"] in CAR and (TRACKS is None or str(e.get("track")) in TRACKS):
                tracks.setdefault(str(e.get("track")), {})[fid] = (i, e)
    grays_cache = {}
    dets_by_ref = {}
    n_win = 0
    for tid, entries in sorted(tracks.items()):
        dets_by_track = {}
        refs = []
        for fid, (i, e) in sorted(entries.items()):
            if e.get("source") not in ("fitted", "interpolated") or e.get("visible", 1.0) < VIS_MIN:
                continue                                 # no measured cuboid, or the car occluded / cut off here
            p = os.path.join(D, "masks", fid + ".png")
            if not os.path.exists(p):
                continue
            m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            refs.append((fids.index(fid), fid, i, int((m == i + 1).sum()), m == i + 1))
        refs.sort()
        poses = {}                                       # the track's measured cuboid poses (image clock)
        for fid, (i, e) in entries.items():
            if e.get("source") in ("fitted", "interpolated"):
                q = e.get("image_clock", e)
                poses[fids.index(fid)] = (np.asarray(q["center"][:2], float), float(q["yaw"]))
        made = 0
        for k0, fid0, i0, area, m0 in refs[::STRIDE]:  # a window at every STRIDE-th annotated frame
            if area < 400:
                continue
            out = solve_window(cams, fids, k0, m0, grays_cache, poses)
            if out is None:
                continue
            sol, uv, P, tri, lab, img0, ks, r = out
            det = build_det(k0, ks, r, sol, uv, tri, img0, m0, entries[fid0][1], cam_c, cams)
            if det is None:
                continue
            dets_by_track.setdefault(k0, []).append(det)
            made += 1
        for k0, det in dets_by_track.items():
            others = [d for d in dets_by_ref.get(k0, []) if d.get("track_tid") != tid]
            dets_by_ref[k0] = others + det
            with open(os.path.join(OUT, f"{k0:04d}.pkl"), "wb") as f:
                pickle.dump(dets_by_ref[k0], f)
        print(f"{tid}: {made} windows", flush=True)
        n_win += made
    print(f"{n_win} windows over {len(dets_by_ref)} reference frames -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
