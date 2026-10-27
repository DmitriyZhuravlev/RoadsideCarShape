"""
A triangulated 3D model of the static scene from a fixed roadside video, with a normal per triangle (classical only).

  1. Triangulation: Delaunay triangulation of the median background, with vertices at the ends of the LSD line
     segments, the painted road markings, a regular grid and the image border.
  2. Labels per triangle (ground / vertical / sky): position relative to the horizon of the calibrated camera, colour
     against the road and the paint (CIE Lab), density of edges pointing to the vertical vanishing point; smoothed over
     the neighbours.
  3. 3D: ground triangles are mapped onto the ground plane by IPM. Connected vertical regions are stood up on their
     ground contact: every vertex of the region is followed down (towards the vertical vanishing point) to the first
     ground pixel; the IPM of these contacts is fitted with a line, and the region becomes the vertical plane through
     that line (if the contacts are too few: the IPM direction of the region's horizontal edges, which keeps its
     direction at any height; otherwise a plane facing the camera). Vertices are the intersections of their rays with
     that plane. Sky is dropped.
  4. Normals: n = (v1 - v0) x (v2 - v0) of each 3D triangle, oriented towards the camera.
  5. Accuracy against the LiDAR of the same intersection: depth along the camera ray of every static LiDAR point
     (points inside labelled objects removed) vs the triangle it falls in; ground and vertical separately, against a
     flat world (every triangle on the ground plane).

Writes docs/mesh_triangles.png, docs/mesh_views.png, docs/mesh_accuracy.png and docs/mesh_metrics.json.
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

import init_and_track as it
import v2x_coop_eval as ce
import v2x_seq_demo as vd
import world3d as w3

GROUND, VERTICAL, SKY = 0, 1, 2
LABEL_RGB = {GROUND: (120, 170, 120), VERTICAL: (215, 140, 70), SKY: (130, 170, 230)}
INK, MUTED = "#1f1f1e", "#8a8984"
MAX_HEIGHT = 40.0


def ray_triangle_plane_depth(C, rays, tri):
    """s with C + s * ray on the plane of triangle tri (3, 3); nan where parallel or degenerate."""
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    if np.linalg.norm(n) < 1e-12:
        return np.full(len(rays), np.nan)
    den = rays @ n
    s = np.full(len(rays), np.nan)
    ok = np.abs(den) > 1e-9
    s[ok] = (n @ (tri[0] - C)) / den[ok]
    return s


# --- 1. triangulation ---------------------------------------------------------------------------------
def vertices(bg, cam, grid=48):
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    segs = cv2.createLineSegmentDetector().detect(gray)[0].reshape(-1, 4)
    segs = segs[np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) > 12]
    pts = [segs[:, :2], segs[:, 2:]]
    gx, gy = np.meshgrid(np.arange(0, W, grid), np.arange(0, H, grid))
    pts.append(np.c_[gx.ravel(), gy.ravel()])
    border = np.r_[np.c_[np.linspace(0, W - 1, 41), np.zeros(41)], np.c_[np.linspace(0, W - 1, 41), np.full(41, H - 1)],
                   np.c_[np.zeros(23), np.linspace(0, H - 1, 23)], np.c_[np.full(23, W - 1), np.linspace(0, H - 1, 23)]]
    pts.append(border)
    P = np.vstack(pts).astype(float)
    P[:, 0] = np.clip(P[:, 0], 0, W - 1)
    P[:, 1] = np.clip(P[:, 1], 0, H - 1)
    keep, taken = [], set()
    for p in P:                                               # thin out points closer than 5 px
        key = (int(p[0] // 5), int(p[1] // 5))
        if key not in taken:
            taken.add(key)
            keep.append(p)
    return np.array(keep), segs


def triangle_index_image(P, tris, shape):
    lab = np.full(shape, -1, np.int32)
    for i, t in enumerate(tris):
        cv2.fillConvexPoly(lab, P[t].round().astype(np.int32), int(i))
    return lab


# --- 2. labels ----------------------------------------------------------------------------------------
def horizon_line(cam):
    K, R, t = cam
    h = np.linalg.inv(K).T @ R[:, 2]
    h = h / np.linalg.norm(h[:2])
    return h if h @ [K[0, 2], 2 * K[1, 2], 1.0] > 0 else -h      # positive below the horizon


def label_triangles(bg, cam, P, tris, lab, segs):
    H, W = lab.shape
    n = len(tris)
    labl = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    idx = lab.ravel()
    ok = idx >= 0
    count = np.bincount(idx[ok], minlength=n).astype(float)
    mean_lab = np.stack([np.bincount(idx[ok], labl[ok, c], minlength=n) for c in range(3)], 1) / np.maximum(count, 1)[:, None]
    mean_sat = np.bincount(idx[ok], hsv[ok, 1], minlength=n) / np.maximum(count, 1)
    # edges pointing to the vertical vanishing point
    K, R, t = cam
    vz = K @ R[:, 2]
    vert = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in segs:
        mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        to_vp = vz[:2] - mid * vz[2]
        d = np.array([x1 - x0, y1 - y0])
        if abs(d @ to_vp) / (np.linalg.norm(d) * np.linalg.norm(to_vp) + 1e-12) > np.cos(np.radians(2.0)) \
                and np.linalg.norm(d) > 20:
            cv2.line(vert, (int(x0), int(y0)), (int(x1), int(y1)), 1, 3)
    vert_frac = np.bincount(idx[ok], vert.ravel()[ok].astype(float), minlength=n) / np.maximum(count, 1)
    hz = horizon_line(cam)
    cent = P[tris].mean(1)
    below = np.c_[cent, np.ones(n)] @ hz
    # colour models: road (lower image, median) and white paint
    lower = labl.reshape(H, W, 3)[int(0.6 * H):].reshape(-1, 3)
    road = np.median(lower, 0)
    d_road = np.linalg.norm(mean_lab - road, axis=1)
    paint = (mean_lab[:, 0] > road[0] + 35) & (mean_sat < 50)
    green = (mean_lab[:, 1] < 120) & (mean_sat > 45)
    labels = np.full(n, GROUND)
    labels[(d_road > 22) & ~paint] = VERTICAL
    labels[green & (below > 90)] = GROUND                          # grass verges near the camera lie on the ground
    labels[vert_frac > 0.06] = VERTICAL
    labels[below < 4] = VERTICAL
    labels[(below < 4) & (mean_lab[:, 0] > 175) & (mean_sat < 40)] = SKY
    fixed = (vert_frac > 0.12) | (below < 4)
    # smoothing: majority of the neighbours (Delaunay adjacency), strong evidence kept
    neigh = Delaunay_neighbours(tris)
    for _ in range(3):
        new = labels.copy()
        for i in range(n):
            nb = [j for j in neigh[i] if j >= 0]
            if not nb or fixed[i]:
                continue
            votes = np.bincount(labels[nb], minlength=3)
            if votes.max() >= 2 and votes.argmax() != labels[i]:
                new[i] = votes.argmax()
        labels = new
    return labels, below


_NEIGH = {}


def Delaunay_neighbours(tris):
    return _NEIGH["n"]


# --- 3. 3D --------------------------------------------------------------------------------------------
def walk_to_ground(p, cam, lab, labels, max_px=500, step=3):
    """Follow an image point down towards the vertical vanishing point to the first ground pixel."""
    K, R, t = cam
    vz = K @ R[:, 2]
    H, W = lab.shape
    d = vz[:2] - p * vz[2]
    d = d / np.linalg.norm(d)
    if d[1] < 0:
        d = -d
    for s in range(0, max_px, step):
        q = p + s * d
        x, y = int(q[0]), int(q[1])
        if not (0 <= x < W and 0 <= y < H):
            return None
        i = lab[y, x]
        if i >= 0 and labels[i] == GROUND:
            return q
    return None


def build_mesh(cam, P, tris, lab, labels, segs):
    C = w3.centre(cam)
    n = len(tris)
    V = np.full((n, 3, 3), np.nan)
    # ground triangles: IPM
    for i in np.where(labels == GROUND)[0]:
        g, ok = w3.to_ground(cam, P[tris[i]])
        if ok.all() and np.hypot(*(g.mean(0) - C[:2])) < 200:
            V[i] = np.c_[g, np.zeros(3)]
    # vertical regions: connected components of vertical triangles
    neigh = _NEIGH["n"]
    region = np.full(n, -1)
    rid = 0
    for i in np.where(labels == VERTICAL)[0]:
        if region[i] >= 0:
            continue
        stack = [i]
        region[i] = rid
        while stack:
            a = stack.pop()
            for b in neigh[a]:
                if b >= 0 and labels[b] == VERTICAL and region[b] < 0:
                    region[b] = rid
                    stack.append(b)
        rid += 1
    # a small vertical region surrounded only by ground is a stain, shadow or wet patch on the road
    e1, e2 = P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]]
    area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2
    for r in range(rid):
        members = np.where(region == r)[0]
        border = {b for a in members for b in neigh[a] if b >= 0 and region[b] != r}
        if area[members].sum() < 2500 and border and all(labels[b] == GROUND for b in border):
            labels[members] = GROUND
            region[members] = -1
            for i in members:
                g, ok = w3.to_ground(cam, P[tris[i]])
                if ok.all() and np.hypot(*(g.mean(0) - C[:2])) < 200:
                    V[i] = np.c_[g, np.zeros(3)]
    planes = {}
    for r in range(rid):
        members = np.where(region == r)[0]
        if len(members) == 0:
            continue
        vids = np.unique(tris[members].ravel())
        contacts = []
        for v in vids:
            q = walk_to_ground(P[v], cam, lab, labels)
            if q is not None:
                g, ok = w3.to_ground(cam, q[None])
                if ok[0] and np.hypot(*(g[0] - C[:2])) < 200:
                    contacts.append(g[0])
        if not contacts:
            continue
        G = np.array(contacts)
        p0 = np.median(G, 0)
        direction = None
        if len(G) >= 3:
            _, sv, Vt = np.linalg.svd(G - G.mean(0))
            if sv[0] > 3 * max(sv[1], 1e-6) and np.ptp((G - G.mean(0)) @ Vt[0]) > 0.8:
                direction, p0 = Vt[0], G.mean(0)
        if direction is None:                              # horizontal edges inside the region (IPM keeps direction)
            ang = []
            box_mask = np.isin(lab, members)
            for x0, y0, x1, y1 in segs:
                mx, my = int((x0 + x1) / 2), int((y0 + y1) / 2)
                if 0 <= my < lab.shape[0] and 0 <= mx < lab.shape[1] and box_mask[my, mx]:
                    g, ok = w3.to_ground(cam, np.array([[x0, y0], [x1, y1]]))
                    if ok.all() and np.linalg.norm(g[1] - g[0]) > 0.3:
                        dd = g[1] - g[0]
                        ang.append(np.arctan2(dd[1], dd[0]) % np.pi)
            if len(ang) >= 3:
                a = np.arctan2(np.mean(np.sin(2 * np.array(ang))), np.mean(np.cos(2 * np.array(ang)))) / 2
                direction = np.array([np.cos(a), np.sin(a)])
        if direction is None:                              # plane facing the camera
            view = p0 - C[:2]
            direction = np.array([-view[1], view[0]]) / np.linalg.norm(view)
        normal = np.r_[-direction[1], direction[0], 0.0]
        planes[r] = (np.r_[p0, 0.0], normal)
        for i in members:
            rays = w3.rays(cam, P[tris[i]])
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
            den = rays @ normal
            if np.any(np.abs(den) < 1e-6):
                continue
            s = ((np.r_[p0, 0.0] - C) @ normal) / den
            X = C + s[:, None] * rays
            if np.all(s > 0) and np.all(X[:, 2] > -0.5) and np.all(X[:, 2] < MAX_HEIGHT):
                X[:, 2] = np.maximum(X[:, 2], 0.0)
                V[i] = X
    edge = np.max(np.linalg.norm(V - np.roll(V, 1, axis=1), axis=2), axis=1)
    V[edge > 15.0] = np.nan                                  # stretched slivers at grazing angles
    normals = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    flip = np.einsum("ij,ij->i", normals, C - V.mean(1)) < 0
    normals[flip] *= -1
    return V, normals, region, planes


# --- 5. LiDAR check -----------------------------------------------------------------------------------
def lidar_points(seq_dir, info_rows, ground_z, tracks_boxes, k_list):
    pts_all = []
    for k in k_list:
        d = info_rows[k]
        X = vd.read_pcd(os.path.join(seq_dir, d["pointcloud_path"]))
        X = X[np.isfinite(X).all(1)]
        X[:, 2] -= ground_z                                     # world: ground at z = 0
        pts_all.append((k, X))
    return pts_all


def depth_errors(cam, lab, labels, V, pts, boxes, flat=False):
    C = w3.centre(cam)
    uv, d = w3.project(cam, pts)
    H, W = lab.shape
    ok = (d > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    for x0, y0, x1, y1 in boxes:                               # moving and parked labelled objects removed
        ok &= ~((uv[:, 0] >= x0 - 3) & (uv[:, 0] <= x1 + 3) & (uv[:, 1] >= y0 - 3) & (uv[:, 1] <= y1 + 3))
    uv, pts = uv[ok], pts[ok]
    tri = lab[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    rows = []
    rays = pts - C
    s_true = np.linalg.norm(rays, axis=1)
    rays = rays / s_true[:, None]
    for i in np.unique(tri[tri >= 0]):
        sel = tri == i
        if labels[i] == SKY:
            continue
        if flat:
            g, okg = w3.to_ground(cam, uv[sel])
            s = np.where(okg, np.linalg.norm(np.c_[g, np.zeros(len(g))] - C, axis=1), np.nan)
        else:
            if np.isnan(V[i]).any():
                continue
            s = ray_triangle_plane_depth(C, rays[sel], V[i])
        for a, b, z in zip(s, s_true[sel], pts[sel, 2]):
            if np.isfinite(a) and a > 0:
                rows.append((labels[i], abs(a - b), abs(a - b) / b, b, z))
    return np.array(rows)


# --- rendering ----------------------------------------------------------------------------------------
def render_mesh(view, bg_rgb, P, tris, V, normals, labels, mode="texture", size=(1280, 720), boxes=()):
    K, R, t = view
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:] = np.linspace(238, 205, size[1])[:, None, None].astype(np.uint8)
    C = w3.centre(view)
    items = []
    for i in range(len(tris)):
        if np.isnan(V[i]).any() or labels[i] == SKY:
            continue
        uv, d = w3.project(view, V[i])
        if (d <= 0.3).any() or np.abs(uv).max() > 1e5:
            continue
        items.append((d.mean(), i, uv))
    for faces, color in boxes:
        for quad, nrm in faces:
            uv, d = w3.project(view, quad)
            if (d > 0.3).all() and nrm @ (C - quad.mean(0)) > 0:
                items.append((d.mean(), -1, (uv, tuple(int(c * (0.45 + 0.55 * max(0.0, nrm @ w3.LIGHT))) for c in color))))
    for _, i, uv in sorted(items, key=lambda x: -x[0]):
        if i < 0:
            quv, color = uv
            pts = quv.round().astype(np.int32)
            cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)
            cv2.polylines(img, [pts], True, tuple(int(c * 0.55) for c in color), 1, cv2.LINE_AA)
            continue
        dst = uv.astype(np.float32)
        if mode == "texture":
            src = P[tris[i]].astype(np.float32)
            x, y, w, h = cv2.boundingRect(dst)
            if w <= 0 or h <= 0 or w > size[0] * 2 or h > size[1] * 2:
                continue
            A = cv2.getAffineTransform(src, (dst - np.float32([x, y])).astype(np.float32))
            patch = cv2.warpAffine(bg_rgb, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, (dst - [x, y]).round().astype(np.int32), 1)
            X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + w, size[0]), min(y + h, size[1])
            if X1 <= X0 or Y1 <= Y0:
                continue
            m = mask[Y0 - y:Y1 - y, X0 - x:X1 - x] > 0
            img[Y0:Y1, X0:X1][m] = patch[Y0 - y:Y1 - y, X0 - x:X1 - x][m]
        else:
            nrm = normals[i]
            color = tuple(int(v) for v in ((nrm * 0.5 + 0.5) * 255).clip(0, 255))
            cv2.fillConvexPoly(img, dst.round().astype(np.int32), color, cv2.LINE_AA)
    return img


# --- figures -------------------------------------------------------------------------------------------
def triangles_figure(path, bg, P, tris, labels, normals, V):
    rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    over = rgb.copy()
    nmap = np.full_like(rgb, 255)
    for i, t in enumerate(tris):
        pts = P[t].round().astype(np.int32)
        cv2.fillConvexPoly(over, pts, LABEL_RGB[labels[i]])
        if not np.isnan(V[i]).any():
            cv2.fillConvexPoly(nmap, pts, tuple(int(v) for v in ((normals[i] * 0.5 + 0.5) * 255).clip(0, 255)))
    over = (0.55 * rgb + 0.45 * over).astype(np.uint8)
    for t in tris:
        cv2.polylines(over, [P[t].round().astype(np.int32)], True, (255, 255, 255), 1)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.2))
    axes[0].imshow(over)
    axes[0].set_title(f"Delaunay triangulation of the background ({len(tris)} triangles): green = ground, orange = "
                      f"vertical, blue = sky", fontsize=10, loc="left", color=INK)
    axes[1].imshow(nmap)
    axes[1].set_title("normal of every 3D triangle as colour (R, G, B = x, y, z of the normal; ground = up)", fontsize=10,
                      loc="left", color=INK)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=85, facecolor="white")
    plt.close(fig)


def accuracy_figure(path, err_mesh, err_flat):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    bins = ((0, 30), (30, 60), (60, 100), (100, 200))
    for ax, (cls, name) in zip(axes, (("ground", "the ground (LiDAR height < 0.3 m)"),
                                      ("elevated", "elevated structures (LiDAR height > 0.5 m)"))):
        for j, (E, tag, color) in enumerate(((err_flat, "flat world (IPM only)", "#c3c2b7"),
                                             (err_mesh, "triangulated pop-up mesh", "#2a78d6"))):
            v = []
            for lo, hi in bins:
                kind = (E[:, 4] < 0.3) if cls == "ground" else (E[:, 4] > 0.5)
                sel = kind & (E[:, 3] >= lo) & (E[:, 3] < hi) if len(E) else np.array([], bool)
                v.append(100 * np.median(E[sel, 2]) if sel.any() else np.nan)
            bars = ax.bar(np.arange(len(bins)) + (j - 0.5) * 0.38, v, 0.36, color=color, zorder=2, label=tag)
            for b, x in zip(bars, v):
                if np.isfinite(x):
                    ax.annotate(f"{x:.1f}", (b.get_x() + b.get_width() / 2, x), xytext=(0, 2),
                                textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(bins)), [f"{lo}-{hi} m" for lo, hi in bins])
        ax.set_ylabel("median depth error vs LiDAR (% of depth)", color=INK)
        ax.set_title(f"LiDAR points on {name}", fontsize=10.5, loc="left", color=INK)
        ax.legend(fontsize=8.5, frameon=False)
        vd.style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    images, times, cams, tracks, camera_xy = ce.load_spd()
    frames = [cv2.imread(p) for p in images]
    bg = np.median(np.stack(frames[::2]), axis=0).astype(np.uint8)
    tilt, _ = it.init_tilt(cams, tracks)
    cam = ce.tilt_cameras([cams[0]], tilt)[0]
    P, segs = vertices(bg, cam)
    dl = Delaunay(P)
    tris = dl.simplices
    _NEIGH["n"] = dl.neighbors
    lab = triangle_index_image(P, tris, bg.shape[:2])
    labels, below = label_triangles(bg, cam, P, tris, lab, segs)
    V, normals, region, planes = build_mesh(cam, P, tris, lab, labels, segs)
    valid = ~np.isnan(V).any((1, 2))
    print(f"{len(P)} vertices, {len(tris)} triangles: ground {np.sum(labels == GROUND)}, vertical {np.sum(labels == VERTICAL)}"
          f", sky {np.sum(labels == SKY)}; {valid.sum()} placed in 3D, {len(planes)} vertical planes", flush=True)
    # LiDAR check
    import json as _json
    info = sorted(_json.load(open(os.path.join(vd.DATA, "data_info.json"))), key=lambda d: d["frame_id"])
    plane = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, info[0]["pointcloud_path"])))
    ks = list(range(0, len(info), 25))
    errs_m, errs_f = [], []
    for k, X in lidar_points(vd.DATA, info, plane[2], None, ks):
        boxes = [s["box"] for q in tracks.values() for s in q if s["k"] == k]
        c = ce.tilt_cameras([cams[k]], tilt)[0]
        em = depth_errors(c, lab, labels, V, X, boxes)
        ef = depth_errors(c, lab, labels, V, X, boxes, flat=True)
        if len(em):
            errs_m.append(em)
        if len(ef):
            errs_f.append(ef)
    Em, Ef = np.vstack(errs_m), np.vstack(errs_f)
    metrics = {"vertices": int(len(P)), "triangles": int(len(tris)), "placed": int(valid.sum()),
               "vertical_planes": len(planes), "lidar_frames": len(ks)}
    for name, E in (("mesh", Em), ("flat", Ef)):
        # classes by the LiDAR point's true height above the ground, not by the predicted label
        for cname, sel_fn in (("ground", lambda E: E[:, 4] < 0.3), ("elevated", lambda E: E[:, 4] > 0.5)):
            sel = sel_fn(E)
            metrics[f"{name}_{cname}"] = {"points": int(sel.sum()), "median_err_m": float(np.median(E[sel, 1])),
                                          "median_err_pct": float(100 * np.median(E[sel, 2])),
                                          "within_10pct": float(np.mean(E[sel, 2] < 0.1))}
    print(json.dumps(metrics, indent=1), flush=True)
    json.dump(metrics, open(os.path.join(vd.OUT_DIR, "mesh_metrics.json"), "w"), indent=1)
    triangles_figure(os.path.join(vd.OUT_DIR, "mesh_triangles.png"), bg, P, tris, labels, normals, V)
    accuracy_figure(os.path.join(vd.OUT_DIR, "mesh_accuracy.png"), Em, Ef)
    # views: original camera, drone, far side; textured and by normals; moving cars of one frame as cuboids
    movers = w3.visible_movers(w3.moving_objects(frames, times, bg, cam), cam, frames[0].shape[:2])
    k = 150
    boxes3d = w3.scene_objects([], [], movers[k])
    Pl = np.array([s["pos"] for q in tracks.values() for s in q])
    Pl = Pl[np.hypot(*(Pl - camera_xy).T) < w3.MAX_RANGE]
    lo, hi = Pl.min(0) - 8, Pl.max(0) + 8
    mid, span = (lo + hi) / 2, float(np.max(hi - lo))
    C0 = w3.centre(cam)
    away = (mid - C0[:2]) / np.linalg.norm(mid - C0[:2])
    side = np.array([-away[1], away[0]])
    K0, R0, t0 = cam
    s = 1280 / bg.shape[1]
    views = [("original camera", (np.diag([s, s, 1.0]) @ K0, R0, t0), (1280, int(bg.shape[0] * s))),
             ("drone view", w3.look_at(np.r_[C0[:2] - 0.25 * span * away + 0.45 * span * side, 0.5 * span],
                                       np.r_[mid, 0.0]), (1280, 720)),
             ("from the far side", w3.look_at(np.r_[mid + 0.5 * span * away - 0.2 * span * side, 0.18 * span],
                                              np.r_[mid - 0.25 * away * span, 4.0]), (1280, 720))]
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 3, figsize=(21, 8.4))
    for c, (name, view, size) in enumerate(views):
        axes[0, c].imshow(render_mesh(view, bg_rgb, P, tris, V, normals, labels, "texture", size, boxes3d))
        axes[0, c].set_title(f"textured triangle mesh + moving cuboids (frame {k}), {name}", fontsize=9.5, loc="left",
                             color=INK)
        axes[1, c].imshow(render_mesh(view, bg_rgb, P, tris, V, normals, labels, "normals", size, boxes3d))
        axes[1, c].set_title(f"same, coloured by triangle normal, {name}", fontsize=9.5, loc="left", color=INK)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(vd.OUT_DIR, "mesh_views.png"), dpi=85, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
