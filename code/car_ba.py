"""One rigid car per track: the windows of a track solved together (rigid structure from motion of a moving object,
as in Ozden et al. 2004).

Every window solved its own planar motion (yaw, x, y per frame) and the depths of its vertices along the rays of its
reference frame (whole_scene.solve_object). The windows of one car overlap in time, but each had its own motion, and
the fusion then had to align them (ICP); along the car's own axis the point cloud is self-similar, so the alignment
slid. Here one planar pose per FRAME is shared by all windows of the track:

    min over {Q_g} and the depths of all windows
        sum rho(project_g(Q_g Q_r^-1 X) - u)                  every window's tracks (soft L1, px)
      + the windows' own surface smoothness and ground anchors
      + W_MOTION * second differences of Q over time + W_NONHOLO * sideways steps (a car drives along its heading)

A window seen side-on (length across the ray) then fixes the motion that a window seen end-on shares; every window
lands in ONE car frame through the pose of its reference frame, no ICP needed. One pose is held fixed (the gauge).

    adjust(seq, track, pos, head) -> True when solved: every det gets "X" (new depths) and "ba_pose" (yaw, x, y)
"""
import os
import pickle

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares

import tri_smooth as ts

LAMBDA = 10.0                      # as in whole_scene.py
W_ANCHOR, W_MOTION = 30.0, 20.0
W_CONTACT = 3.0
W_NONHOLO = 30.0
MAX_NFEV = int(os.environ.get("MAX_NFEV", "40"))    # cap on the track solve's evaluations (it stops there: see ba_info)


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def project_many(P, X):
    """Pinhole projection with one 3x4 camera matrix per point: P (n, 3, 4), X (n, 3) -> pixels (n, 2)."""
    x = np.einsum("nij,nj->ni", P[:, :, :3], X) + P[:, :, 3]
    return x[:, :2] / x[:, 2:3]


def boundary_vertices(T, n):
    """Vertices on the outline of a triangulation: an edge used by exactly one triangle lies on the outline (the
    mask contour, or the rim of a hole). T (m, 3) int vertex indices, n the number of vertices. Returns bool (n,)."""
    out = np.zeros(n, bool)
    if not len(T):
        return out
    e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    out[u[c == 1].ravel()] = True
    return out


def outline(d):
    """The window's outline vertices: on its mask contour the tracker follows the moving EDGE of the car, not a point
    of its surface (it slides with the view), so their depths are no surface."""
    T = d["tri"][d["body"][d["tri"]].all(1)]
    return boundary_vertices(T, len(d["X"]))


def solved_motion(d):
    """The window's own motion {frame: (th, tx, ty)} over the frames it solved: world_g = R(th) (X - c) + c + t."""
    U = d["ba"]["U"]
    return {g: d["M"][j] for j, g in enumerate(d["ks"]) if j == d["r"] or not np.isnan(U[:, j, 0]).all()}


def compose(pose, m, c):
    """The car frame's pose at a frame where the window moved it by m = (th, tx, ty) about c, from its reference
    pose: world = R(h) Y + p at the reference, so R(th) (R(h) Y + p - c) + c + t later."""
    th, tx, ty = m
    return np.r_[pose[0] + th, rot(th) @ (pose[1:] - c) + c + np.array([tx, ty])]


def initial_poses(track, pos, head):
    """Pose (yaw, x, y) of the car frame in every frame the track's windows saw. The reference poses are chained
    outward from the middle window through the windows' own motions (window i places the reference frame of window
    i + 1), so that each window starts consistent with its own points; where the chain breaks (a gap), the smoothed
    path restarts it. Every frame then takes the pose of the window whose reference is nearest in time."""
    mot = [solved_motion(d) for _, d in track]
    refs = [d["ks"][d["r"]] for _, d in track]
    n = len(track)
    ref = [None] * n
    mid = n // 2
    ref[mid] = np.r_[head[mid], pos[mid]]
    for i in list(range(mid + 1, n)) + list(range(mid - 1, -1, -1)):
        j = i - 1 if i > mid else i + 1                        # the neighbour toward the middle
        if refs[i] in mot[j]:
            ref[i] = compose(ref[j], mot[j][refs[i]], track[j][1]["centre"])
        else:
            ref[i] = np.r_[head[i], pos[i]]
    best = {}
    for i, (_, d) in enumerate(track):
        for g, m in mot[i].items():
            if g not in best or abs(g - refs[i]) < abs(g - refs[best[g][0]]):
                best[g] = (i, compose(ref[i], m, d["centre"]))
    frames = sorted(best)
    Q = np.array([best[g][1] for g in frames])
    Q[:, 0] = np.unwrap(Q[:, 0])
    return frames, Q


def ba_cache_path(cache_dir, dumps, track, tag=""):
    """File of the cached track solve of one track: a readable prefix (dump folder, first window frame, number of
    windows) and a digest of the dump folder's full path and every window's frame, vertex count and first vertex pixel
    - two tracks of one run can share the prefix (`tag`: the solve's settings)."""
    import hashlib
    key = os.path.abspath(dumps) + "|" + ",".join(f"{k}:{len(d['X'])}:{d['uv'][0][0]:.2f}" for k, d in track)
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"{os.path.basename(os.path.normpath(dumps))}_{track[0][0]:04d}_{len(track)}_{digest}{tag}.pkl")


def ba_cached(cache_dir, dumps, track, solve, tag=""):
    """The track solve, from the cache when there, else solve() and store it. solve() adjusts the dets of `track` in
    place (new det["X"], det["ba_pose"], and track[0][1]["ba_info"]); after this call the dets carry the solved values
    either way. Returns True when they carry a solution."""
    p = ba_cache_path(cache_dir, dumps, track, tag)
    if os.path.exists(p):
        with open(p, "rb") as f:
            data = pickle.load(f)
        for (_, d), X, pose in zip(track, data["X"], data["pose"]):
            d["X"], d["ba_pose"] = X, pose
        track[0][1]["ba_info"] = data["info"]
        return True
    ok = solve()
    if ok and all("ba_pose" in d for _, d in track):
        os.makedirs(cache_dir, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"X": [d["X"] for _, d in track], "pose": [d["ba_pose"] for _, d in track],
                         "info": track[0][1].get("ba_info", {})}, f)
    return bool(ok)


def adjust(seq, track, pos, head, max_nfev=MAX_NFEV, drop_outline=False):
    if not all("ba" in d for _, d in track):
        return False
    frames, Q0 = initial_poses(track, pos, head)
    fidx = {g: i for i, g in enumerate(frames)}
    nf = len(frames)
    # the gauge: the reference frame of the middle window keeps its initial pose (the start of the chain)
    fixed = fidx[track[len(track) // 2][1]["ks"][track[len(track) // 2][1]["r"]]]
    qcol = -np.ones(nf, int)
    qcol[np.arange(nf) != fixed] = np.arange(nf - 1)
    nq = 3 * (nf - 1)
    Pcam = np.array([seq.cams[g][0] @ np.c_[seq.cams[g][1], seq.cams[g][2]] for g in frames])

    # --- every window's problem in global indices
    Cr, Dr, s0, lo, hi, T1, T2, W, BI, ZA = [], [], [], [], [], [], [], [], [], []
    ov, og, orf, ouv = [], [], [], []           # observations: vertex, frame, window's reference frame, pixel
    cx, cg, crf, cuv = [], [], [], []           # tyre contacts: point (world at the reference), frame, ref, pixel
    nv = 0
    spans = []
    for _, d in track:
        b = d["ba"]
        n = len(b["s"])
        Cr.append(b["Cr"]), Dr.append(b["Dr"]), s0.append(b["s"]), lo.append(b["lo"]), hi.append(b["hi"])
        T1.append(b["t1"] + nv), T2.append(b["t2"] + nv), W.append(b["w"])
        BI.append(b["bi"] + nv), ZA.append(np.full(len(b["bi"]), b["z_anchor"]))
        r = fidx[d["ks"][d["r"]]]
        vi, fj = np.nonzero(~np.isnan(b["U"][:, :, 0]))
        keep = fj != d["r"]
        if drop_outline:                        # the outline's depths follow from the surface terms alone
            keep &= ~outline(d)[:n][vi]
        vi, fj = vi[keep], fj[keep]
        ov.append(vi + nv), og.append([fidx[d["ks"][j]] for j in fj]), orf.append(np.full(len(vi), r))
        ouv.append(b["U"][vi, fj].astype(float))
        if len(b["Xc"]):
            ci, cj = np.nonzero(~np.isnan(b["Uc"][:, :, 0]))
            keep = cj != d["r"]
            ci, cj = ci[keep], cj[keep]
            cx.append(b["Xc"][ci]), cg.append([fidx[d["ks"][j]] for j in cj]), crf.append(np.full(len(ci), r))
            cuv.append(b["Uc"][ci, cj])
        spans.append((nv, n))
        nv += n
    Cr, Dr = np.vstack(Cr), np.vstack(Dr)
    Dr = Dr / np.linalg.norm(Dr, axis=1, keepdims=True)
    s0, lo, hi = np.concatenate(s0), np.concatenate(lo), np.concatenate(hi)
    T1, T2, W = np.vstack(T1).astype(int), np.vstack(T2).astype(int), np.concatenate(W)
    BI, ZA = np.concatenate(BI).astype(int), np.concatenate(ZA)
    ov, og, orf = np.concatenate(ov), np.concatenate(og).astype(int), np.concatenate(orf)
    ouv = np.vstack(ouv)
    if len(cx):
        cx, cg, crf, cuv = np.vstack(cx), np.concatenate(cg).astype(int), np.concatenate(crf), np.vstack(cuv)
    else:
        cx, cg, crf, cuv = np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2))
    if len(ov) < 50:
        return False
    Po, Pc = Pcam[og], Pcam[cg]
    kf = np.array(frames, float)
    trip = np.array([(a, a + 1, a + 2) for a in range(nf - 2)]).reshape(-1, 3)
    pairs = np.array([(a, a + 1) for a in range(nf - 1)]).reshape(-1, 2)

    def unpack(x):
        Q = Q0.copy()
        Q[qcol >= 0] = x[:nq].reshape(-1, 3)
        return Q, x[nq:]

    def moved(Xr, g, r, Q):
        """World points at the reference frame r -> world points at frame g (the car's rigid planar motion)."""
        dth = Q[g, 0] - Q[r, 0]
        c, s = np.cos(dth), np.sin(dth)
        a = Xr[:, :2] - Q[r, 1:]
        return np.c_[c * a[:, 0] - s * a[:, 1] + Q[g, 1], s * a[:, 0] + c * a[:, 1] + Q[g, 2], Xr[:, 2]]

    def fun(x):
        Q, s = unpack(x)
        X = Cr + s[:, None] * Dr
        res = project_many(Po, moved(X[ov], og, orf, Q)) - ouv
        cr = W_CONTACT * (project_many(Pc, moved(cx, cg, crf, Q)) - cuv) if len(cx) else np.zeros((0, 2))
        sm = LAMBDA * W[:, None] * (ts.normals(X, T1) - ts.normals(X, T2))
        an = W_ANCHOR * (X[BI, 2] - ZA)
        a, b, c = trip.T
        acc = ((Q[c] - Q[b]) / (kf[c] - kf[b])[:, None] - (Q[b] - Q[a]) / (kf[b] - kf[a])[:, None]) * \
            ((kf[c] - kf[a]) / 2)[:, None]                       # = M_a - 2 M_b + M_c on consecutive frames
        acc = W_MOTION * acc * np.array([8.0, 1.0, 1.0])         # yaw in rad ~ 8 m at car size
        a, b = pairs.T
        mid = (Q[a, 0] + Q[b, 0]) / 2
        step = Q[b, 1:] - Q[a, 1:]
        nh = W_NONHOLO * (-np.sin(mid) * step[:, 0] + np.cos(mid) * step[:, 1])   # sideways step in the car frame
        return np.r_[res.ravel(), cr.ravel(), sm.ravel(), an, acc.ravel(), nh]

    # --- Jacobian sparsity
    rows, cols = [], []
    base = 0

    def qcols(f):                                  # the three pose columns of the frames f (none for the fixed one)
        c = qcol[f]
        return [np.where(c >= 0, 3 * c + i, -1) for i in range(3)]

    def add(r, cs):
        for c in cs:
            ok = c >= 0
            rows.append(r[ok]), cols.append(c[ok])

    m = len(ov)
    for comp in range(2):
        r = base + 2 * np.arange(m) + comp
        add(r, [nq + ov] + qcols(og) + qcols(orf))
    base += 2 * m
    mc = len(cx)
    for comp in range(2):
        r = base + 2 * np.arange(mc) + comp
        add(r, qcols(cg) + qcols(crf))
    base += 2 * mc
    for comp in range(3):
        r = base + 3 * np.arange(len(T1)) + comp
        add(r, [nq + T1[:, 0], nq + T1[:, 1], nq + T1[:, 2], nq + T2[:, 2]])
    base += 3 * len(T1)
    add(base + np.arange(len(BI)), [nq + BI])
    base += len(BI)
    for comp in range(3):
        r = base + 3 * np.arange(len(trip)) + comp
        add(r, [qcols(trip[:, i])[comp] for i in range(3)])
    base += 3 * len(trip)
    r = base + np.arange(len(pairs))
    add(r, qcols(pairs[:, 0]) + qcols(pairs[:, 1]))
    base += len(pairs)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    S = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(base, nq + nv)).tocsr()

    x0 = np.r_[Q0[qcol >= 0].ravel(), np.clip(s0, lo + 1e-6, hi - 1e-6)]
    blo = np.r_[np.full(nq, -np.inf), lo]
    bhi = np.r_[np.full(nq, np.inf), hi]
    before = np.median(np.linalg.norm(fun(x0)[:2 * m].reshape(-1, 2), axis=1))
    sol = least_squares(fun, x0, jac_sparsity=S, bounds=(blo, bhi), loss="soft_l1", f_scale=1.0, method="trf",
                        tr_solver="lsmr", x_scale="jac", max_nfev=max_nfev)
    Q, s = unpack(sol.x)
    after = np.median(np.linalg.norm(fun(sol.x)[:2 * m].reshape(-1, 2), axis=1))
    X = Cr + s[:, None] * Dr
    for (_, d), (a, n) in zip(track, spans):
        d["X"] = d["X"].copy()
        d["X"][:n] = X[a:a + n]                     # filled vertices (after the solve) keep their depths
        d["ba_pose"] = Q[fidx[d["ks"][d["r"]]]].copy()
    track[0][1]["ba_info"] = {"frames": nf, "obs": int(m), "px_before": float(before), "px_after": float(after),
                              "nfev": int(sol.nfev), "max_nfev": int(max_nfev), "status": int(sol.status),
                              "cost": float(sol.cost), "optimality": float(sol.optimality)}
    return True
