"""
Normals and 3D of triangles on moving cars, seen by the FIXED roadside camera (V2X-Seq, intersection A).

The static scene does not move between frames of a fixed camera, so its triangles carry no depth information. A
moving car does: in the car's own frame the fixed camera is a moving camera, and a triangle on the car moves and
deforms between frames k and j by the plane-induced homography H = K (R + t m^T) K^-1, where (R, t) is the car's
motion expressed in camera-k coordinates (points fixed on the car: X_j = R X_k + t) and m = n / d the triangle's plane.
  * car mask in frame k: median background subtraction inside the car's box (v2x_seq_shape.Sequence.mask), eroded;
  * triangles: Delaunay triangulation of corners, a 10 px grid and the outline inside the mask;
  * flow: DIS optical flow k -> j and back, forward-backward check;
  * car motion (R, t), two variants:
      label poses : A_k, A_j = object -> camera from the labelled position and heading; (R, t) = A_j A_k^-1;
      from flow   : essential matrix of the car's own flow (rotation and direction of motion), only the length of
                    the translation from the labelled speed;
  * per triangle: m from all its flow pixels (flow_planes.solve_triangles), vertices X = q / (m . q), normal -m/|m|;
  * every solved triangle is moved into the car's own frame and collected over the whole track;
  * check: the intersection's LiDAR points on the same car (inside its labelled box, in the car's frame, collected over
    the track): distance of the triangle vertices to them, and the angle to the plane of the LiDAR points around each
    triangle - for the triangle's own normal (from its deformation) and for the normal of the local plane through
    the collected vertices around it (radius 0.35 m).

Writes docs/carflow_pairs.png, docs/carflow_3d.png, docs/carflow_accuracy.png and docs/carflow_metrics.json.
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay, cKDTree

import flow_planes as fp
import v2x_seq_demo as vd
import v2x_seq_shape as vs

TRACKS = ("003981", "004005", "004012")
STEP = 2                     # frames between the two views (0.2 s)
INK, MUTED = "#1f1f1e", "#8a8984"
VARIANTS = ("label poses", "from flow")


def object_to_camera(cam_R, cam_t, center, yaw, z_center):
    """4x4 object -> camera: X_world = Rz(yaw) X_obj + (cx, cy, z_center), x_cam = cam_R X_world + cam_t."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    A = np.eye(4)
    A[:3, :3] = cam_R @ Rz
    A[:3, 3] = cam_R @ np.array([center[0], center[1], z_center]) + cam_t
    return A


def car_triangles(gray, mask, grid=10):
    m = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(m)
    if len(xs) < 200:
        return None
    pts = [np.c_[xs[(xs % grid == 0) & (ys % grid == 0)], ys[(xs % grid == 0) & (ys % grid == 0)]]]
    c = cv2.goodFeaturesToTrack(gray, 300, 0.01, 5, mask=m)
    if c is not None:
        pts.append(c.reshape(-1, 2))
    cnt = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:, 0]
    pts.append(cnt[::6])
    P = np.vstack(pts).astype(float)
    _, idx = np.unique((P // 3).astype(int), axis=0, return_index=True)
    P = P[np.sort(idx)]
    tri = Delaunay(P).simplices
    cen = P[tri].mean(1).round().astype(int)
    keep = m[cen[:, 1], cen[:, 0]] > 0
    tri = tri[keep]
    lab = np.full(gray.shape, -1, np.int32)
    for i, t in enumerate(tri):
        cv2.fillConvexPoly(lab, P[t].round().astype(np.int32), int(i))
    lab[m == 0] = -1
    return P, tri, lab


def crop_flow(g1, g2, box1, box2, pad=40):
    H, W = g1.shape
    x0 = int(max(0, min(box1[0], box2[0]) - pad))
    y0 = int(max(0, min(box1[1], box2[1]) - pad))
    x1 = int(min(W, max(box1[2], box2[2]) + pad))
    y1 = int(min(H, max(box1[3], box2[3]) + pad))
    f, ok = fp.flows(np.ascontiguousarray(g1[y0:y1, x0:x1]), np.ascontiguousarray(g2[y0:y1, x0:x1]))
    flow = np.zeros((H, W, 2), np.float32)
    good = np.zeros((H, W), bool)
    flow[y0:y1, x0:x1], good[y0:y1, x0:x1] = f, ok
    return flow, good


def motion_from_flow(lab, flow, ok, K, t_len):
    ys, xs = np.nonzero((lab >= 0) & ok)
    if len(xs) < 50:
        return None
    sel = np.random.default_rng(0).permutation(len(xs))[:3000]
    x1 = np.c_[xs[sel], ys[sel]].astype(np.float64)
    x2 = x1 + flow[ys[sel], xs[sel]]
    E, inl = cv2.findEssentialMat(x1, x2, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, _ = cv2.recoverPose(E, x1, x2, K, mask=inl)
    return R, t.ravel() / np.linalg.norm(t) * t_len


def lidar_on_car(seq, o, ground_z):
    import json as _json
    if hasattr(seq, "lidar_world"):              # moving camera: LiDAR already in the road-plane world
        X = seq.lidar_world(o["k"])
    else:
        d = seq.info[o["k"]]
        X = vd.read_pcd(os.path.join(vd.DATA, d["pointcloud_path"]))
        X = X[np.isfinite(X).all(1)].copy()
        X[:, 2] -= ground_z
    c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    Xo = (X - np.array([o["pos"][0], o["pos"][1], o["z"]])) @ Rz
    l, w, h = o["dims"]
    inside = (np.abs(Xo[:, 0]) < l / 2 + 0.2) & (np.abs(Xo[:, 1]) < w / 2 + 0.2) & (np.abs(Xo[:, 2]) < h / 2 + 0.3)
    return Xo[inside]


def run_car(seq, tid, ground_z):
    obs = [o for o in seq.tracks[tid] if seq.usable(o)]
    by_k = {o["k"]: o for o in obs}
    out = {v: {"V": [], "N": [], "C": []} for v in VARIANTS}
    lidar, example = [], None
    for o in obs:
        if o["k"] + STEP not in by_k:
            continue
        o2 = by_k[o["k"] + STEP]
        k, j = o["k"], o2["k"]
        K, Rc, tc = seq.cams[k]
        A_k = object_to_camera(Rc, tc, o["pos"], o["yaw"], o["z"])
        A_j = object_to_camera(seq.cams[j][1], seq.cams[j][2], o2["pos"], o2["yaw"], o2["z"])
        rel = A_j @ np.linalg.inv(A_k)
        g1 = cv2.cvtColor(seq.frames[k], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(seq.frames[j], cv2.COLOR_BGR2GRAY)
        tris = car_triangles(g1, seq.mask(o) > 0)
        if tris is None:
            continue
        P, tri, lab = tris
        flow, ok = crop_flow(g1, g2, o["box"], o2["box"])
        rgb = cv2.cvtColor(seq.frames[k], cv2.COLOR_BGR2RGB)
        colors = np.array([rgb[int(c[1]), int(c[0])] for c in P[tri].mean(1)]) / 255.0
        lidar.append(lidar_on_car(seq, o, ground_z))
        for v in VARIANTS:
            if v == "label poses":
                R, t = rel[:3, :3], rel[:3, 3]
            else:
                mf = motion_from_flow(lab, flow, ok, K, np.linalg.norm(rel[:3, 3]))
                if mf is None:
                    continue
                R, t = mf
            M, info = fp.solve_triangles(P, tri, lab, flow, ok, K, R, t, [])
            V, N = fp.vertices_from_planes(P, tri, M, K)
            good = ~np.isnan(V).any((1, 2))
            inv = np.linalg.inv(A_k)
            Vo = V[good] @ inv[:3, :3].T + inv[:3, 3]
            No = N[good] @ inv[:3, :3].T
            out[v]["V"].append(Vo)
            out[v]["N"].append(No)
            out[v]["C"].append(colors[good])
            if v == "label poses" and (example is None or good.sum() > example["solved"]):
                example = {"k": k, "j": j, "P": P, "tri": tri, "flow": flow, "N": N, "good": good,
                           "solved": int(good.sum()), "box": o["box"], "box2": o2["box"]}
    res = {}
    L = np.vstack(lidar) if lidar else np.zeros((0, 3))
    for v in VARIANTS:
        if not out[v]["V"]:
            continue
        V, N, C = np.vstack(out[v]["V"]), np.vstack(out[v]["N"]), np.vstack(out[v]["C"])
        # outliers: vertices outside a generous box around the car
        l, w, h = obs[0]["dims"]
        inside = np.all((np.abs(V[..., 0]) < l) & (np.abs(V[..., 1]) < w * 1.5) & (np.abs(V[..., 2]) < h * 1.5), axis=1)
        res[v] = {"V": V[inside], "N": N[inside], "C": C[inside], "solved": int(len(V)), "kept": int(inside.sum())}
        if len(L) and inside.any():
            tree = cKDTree(L)
            d, _ = tree.query(V[inside].reshape(-1, 3))
            ang = []
            for cen, n in zip(V[inside].mean(1), N[inside]):
                idx = tree.query_ball_point(cen, 0.35)
                if len(idx) >= 6:
                    Q = L[idx] - L[idx].mean(0)
                    _, sv, Vt = np.linalg.svd(Q)
                    if sv[2] < 0.3 * sv[1]:
                        ang.append(np.degrees(np.arccos(min(1.0, abs(Vt[2] @ n)))))
            # normals from the neighbourhood of the collected vertices (local plane of the reconstructed points)
            Vin = V[inside]
            pts = Vin.reshape(-1, 3)
            ptree = cKDTree(pts)
            ang_pca = []
            for cen in Vin.mean(1)[::3]:
                idx = tree.query_ball_point(cen, 0.35)
                pidx = ptree.query_ball_point(cen, 0.35)
                if len(idx) >= 6 and len(pidx) >= 12:
                    Q = L[idx] - L[idx].mean(0)
                    _, sv, Vt = np.linalg.svd(Q)
                    R_ = pts[pidx] - pts[pidx].mean(0)
                    _, sv2, Vt2 = np.linalg.svd(R_)
                    if sv[2] < 0.3 * sv[1]:
                        ang_pca.append(np.degrees(np.arccos(min(1.0, abs(Vt[2] @ Vt2[2])))))
            res[v].update(normal_pca_median=float(np.median(ang_pca)) if ang_pca else None,
                          normal_pca_within_20=float(np.mean(np.array(ang_pca) < 20)) if ang_pca else None)
            res[v].update(dist_median=float(np.median(d)), dist_within_20cm=float(np.mean(d < 0.2)),
                          normal_median=float(np.median(ang)) if ang else None,
                          normal_within_20=float(np.mean(np.array(ang) < 20)) if ang else None, normals_checked=len(ang))
    return res, L, example, obs[0]["dims"]


def pairs_figure(path, seq, examples):
    fig, axes = plt.subplots(len(examples), 2, figsize=(14, 4.6 * len(examples)))
    for r, (tid, ex) in enumerate(examples.items()):
        P, tri, flow, N, good = ex["P"], ex["tri"], ex["flow"], ex["N"], ex["good"]
        H, W = flow.shape[:2]
        pts = np.clip(P.round().astype(int), 0, [W - 1, H - 1])
        P2 = P + flow[pts[:, 1], pts[:, 0]]
        for c, (k, PP, box) in enumerate(((ex["k"], P, ex["box"]), (ex["j"], P2, ex["box2"]))):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(seq.frames[k], cv2.COLOR_BGR2RGB))
            for i, t in enumerate(tri):
                if good[i]:
                    col = np.clip(N[i] * 0.5 + 0.5, 0, 1)
                    ax.fill(*PP[t].T, facecolor=col, edgecolor="white", linewidth=0.2, alpha=0.75)
                else:
                    ax.fill(*PP[t].T, facecolor="none", edgecolor="#eb6834", linewidth=0.3)
            if c == 0:
                for q in range(0, len(P), 7):
                    ax.annotate("", xy=P2[q], xytext=P[q], arrowprops=dict(arrowstyle="->", color="#f2c94c", lw=0.6))
            x0, y0, x1, y1 = (min(ex["box"][0], ex["box2"][0]) - 20, min(ex["box"][1], ex["box2"][1]) - 20,
                              max(ex["box"][2], ex["box2"][2]) + 20, max(ex["box"][3], ex["box2"][3]) + 20)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.axis("off")
            ax.set_title(f"car {tid}, frame {k}: " + ("triangles coloured by their solved normal, flow (yellow)" if c == 0
                         else "the same triangles moved by the flow"), fontsize=9.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=90, facecolor="white")
    plt.close(fig)


def box_edges(l, w, h):
    C = np.array([[x, y, z] for x in (-l / 2, l / 2) for y in (-w / 2, w / 2) for z in (-h / 2, h / 2)])
    E = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return C, E


def views_figure(path, results):
    views = (("side", 5, 90), ("3/4 front", 25, 35), ("top", 85, -90), ("3/4 rear", 25, 215))
    fig = plt.figure(figsize=(18, 4.2 * len(results)))
    for r, (tid, (res, L, dims)) in enumerate(results.items()):
        rv = res.get("label poses")
        for c, (name, elev, azim) in enumerate(views):
            ax = fig.add_subplot(len(results), len(views), r * len(views) + c + 1, projection="3d")
            if rv is not None and len(rv["V"]):
                sel = np.random.default_rng(0).permutation(len(rv["V"]))[:2500]
                cols = np.clip(rv["N"][sel] * 0.5 + 0.5, 0, 1)
                ax.add_collection3d(Poly3DCollection(rv["V"][sel], facecolors=cols, edgecolors="none", alpha=0.85))
            if len(L):
                Ls = L[np.random.default_rng(1).permutation(len(L))[:3000]]
                ax.scatter(*Ls.T, s=0.4, c="k", alpha=0.35)
            Cb, E = box_edges(*dims)
            for a, b in E:
                ax.plot(*Cb[[a, b]].T, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
            l, w, h = dims
            lim = max(l, w, h) / 2 + 0.3
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev, azim)
            ax.axis("off")
            ax.set_title(f"car {tid}: {name}" + (f" ({rv['kept']} triangles)" if rv and c == 0 else ""), fontsize=9.5,
                         loc="left", color=INK)
    fig.suptitle("Triangles on moving cars from their deformation between frames of the fixed camera, collected in the "
                 "car's frame (colour = normal); black = LiDAR on the car; dashed = labelled box", fontsize=11.5, x=0.01,
                 ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def accuracy_figure(path, metrics):
    cars = list(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for j, (v, color) in enumerate(zip(VARIANTS, ("#2a78d6", "#eb6834"))):
        d = [100 * metrics[c].get(v, {}).get("dist_median", np.nan) for c in cars]
        n = [metrics[c].get(v, {}).get("normal_median") or np.nan for c in cars]
        for ax, vals, fmt in ((axes[0], d, "{:.0f}"), (axes[1], n, "{:.0f}")):
            bars = ax.bar(np.arange(len(cars)) + (j - 0.5) * 0.38, vals, 0.36, color=color, zorder=2, label=v)
            for b, x in zip(bars, vals):
                if np.isfinite(x):
                    ax.annotate(fmt.format(x), (b.get_x() + b.get_width() / 2, x), xytext=(0, 2),
                                textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    axes[0].set_ylabel("median distance of the vertices to the LiDAR points on the car (cm)", color=INK)
    axes[1].set_ylabel("median angle to the local LiDAR plane (°)", color=INK)
    for ax in axes:
        ax.set_xticks(range(len(cars)), [f"car {c}" for c in cars])
        ax.legend(fontsize=8.5, frameon=False)
        vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    seq = vs.Sequence()
    seq.info = sorted(json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    plane = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))
    results, examples, metrics = {}, {}, {}
    for tid in TRACKS:
        res, L, ex, dims = run_car(seq, tid, plane[2])
        results[tid] = (res, L, dims)
        if ex is not None:
            examples[tid] = ex
        metrics[tid] = {v: {k: val for k, val in r.items() if k not in ("V", "N", "C")} for v, r in res.items()}
        metrics[tid]["lidar_points"] = int(len(L))
        print(tid, json.dumps(metrics[tid]), flush=True)
    json.dump(metrics, open(os.path.join(vd.OUT_DIR, "carflow_metrics.json"), "w"), indent=1)
    pairs_figure(os.path.join(vd.OUT_DIR, "carflow_pairs.png"), seq, examples)
    views_figure(os.path.join(vd.OUT_DIR, "carflow_3d.png"), results)
    accuracy_figure(os.path.join(vd.OUT_DIR, "carflow_accuracy.png"), metrics)


if __name__ == "__main__":
    main()
