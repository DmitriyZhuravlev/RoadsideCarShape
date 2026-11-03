"""A pedestrian or cyclist from the video, posed by its own LiDAR (diagnostic, like the Golf): per frame the object's
LiDAR points (labelled box enlarged by 0.6 m only as the search region, largest cluster), centre from the points,
heading from the smoothed path, size from the extents; the camera-time shift by the masks; masks by GrabCut in the
labelled 2D box clipped to the LiDAR box; visual hull (vote 0.8, masks +2 px, height capped by the LiDAR), symmetric
about the heading axis; coloured from the best view (outward normals, own z-buffer), mirror fill; IoU vs the LiDAR."""
import os, sys, json, pickle, numpy as np, cv2
from scipy.signal import savgol_filter
from scipy.ndimage import label as cclabel, gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
S, TID = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.expanduser('~/Documents/MyArticles/code/3dseg')); os.chdir(os.path.expanduser('~/Documents/MyArticles/code/3dseg'))
sys.path.insert(0, os.path.expanduser('~/Documents/MyArticles/RoadsideCarShape/scripts'))
import v2x_seq_shape as vs, v2x_seq_demo as vd, whole_scene_cars as w, cars_figure as cf
seq = vs.Sequence(); Hh_img, Ww_img = seq.shape
info = sorted(json.load(open(os.path.join(vd.DATA, 'data_info.json'))), key=lambda d: d['frame_id'])
plane = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, info[0]['pointcloud_path'])))
lab = {}
for k, f in enumerate(seq.fids):
    for o in json.load(open(os.path.join(vd.DATA, 'label', 'camera', f'{f}.json'))):
        if o['track_id'] == TID:
            lab[k] = dict(pos=np.array([o['3d_location']['x'], o['3d_location']['y']]), yaw=o['rotation'], type=o['type'],
                          dims=(o['3d_dimensions']['l'], o['3d_dimensions']['w'], o['3d_dimensions']['h']),
                          box=np.array([o['2d_box'][x] for x in ('xmin', 'ymin', 'xmax', 'ymax')], float))
typ = next(iter(lab.values()))['type']
rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
def raw(k):
    X = vd.read_pcd(os.path.join(vd.DATA, info[k]['pointcloud_path'])); X = X[np.isfinite(X).all(1)]
    X[:, 2] -= plane[0] * X[:, 0] + plane[1] * X[:, 1] + plane[2]; return X
kl = sorted(lab); far = [kk for kk in range(0, len(seq.fids), 10) if all(abs(kk - j) > 30 for j in kl)][:8] or [0, len(seq.fids) - 1]
STATIC = cKDTree(np.vstack([raw(kk)[:, :3] for kk in far]))       # what stays when the object is elsewhere: planters, poles
def pts(k):
    o = lab[k]; X = raw(k)
    d = (X[:, :2] - o['pos']) @ rot(o['yaw']); l, w_, h = o['dims']
    P = X[(np.abs(d[:, 0]) < l / 2 + 0.6) & (np.abs(d[:, 1]) < w_ / 2 + 0.6) & (X[:, 2] > 0.12) & (X[:, 2] < 2.4)]
    if len(P): P = P[STATIC.query(P[:, :3])[0] > 0.12]
    if len(P) < 8: return P
    g = np.floor(P[:, :2] / 0.2).astype(int); g -= g.min(0); occ = np.zeros(g.max(0) + 1, bool); occ[g[:, 0], g[:, 1]] = True
    cc, _ = cclabel(occ, structure=np.ones((3, 3))); ids = cc[g[:, 0], g[:, 1]]
    return P[ids == np.argmax(np.bincount(ids)[1:]) + 1]
P = {k: pts(k) for k in sorted(lab)}
ks = [k for k in sorted(P) if len(P[k]) >= 12]
C = np.array([(P[k][:, :2].max(0) + P[k][:, :2].min(0)) / 2 for k in ks]); kf = np.array(ks, float)
win = min(15, (len(ks) // 2) * 2 - 1)
Cs = np.c_[savgol_filter(C[:, 0], win, 2), savgol_filter(C[:, 1], win, 2)]
v = np.gradient(Cs, kf, axis=0); TH = np.unwrap(np.arctan2(v[:, 1], v[:, 0])); TH = savgol_filter(TH, win, 2)
ext = np.array([np.ptp((P[k][:, :2] - Cs[i]) @ rot(TH[i]), axis=0) for i, k in enumerate(ks)])
L, W = np.percentile(ext[:, 0], 85), np.percentile(ext[:, 1], 85); H = np.median([np.percentile(P[k][:, 2], 99) for k in ks])
ld = np.median([lab[k]['dims'] for k in lab], 0); L, W = max(L, ld[0]), max(W, ld[1])     # an unseen side is missing from the extents
H_LIDAR = H                                              # the LiDAR's own height; H may be raised to the silhouettes' top (HEAD)
print(typ, TID, 'frames with LiDAR', len(ks), 'of', len(lab), '| LiDAR box L %.2f W %.2f H %.2f (label median %.2f %.2f %.2f)' % (L, W, H, *np.median([lab[k]['dims'] for k in lab], 0)))
pose_l = lambda t: (np.array([np.interp(t, kf, Cs[:, 0]), np.interp(t, kf, Cs[:, 1])]), np.interp(t, kf, TH))
BOX = np.array([[x, y, z] for z in (0, H) for x, y in ((-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2))])
def to_world(p, a, Y): c, s = np.cos(a), np.sin(a); return np.c_[Y[:, 0] * c - Y[:, 1] * s + p[0], Y[:, 0] * s + Y[:, 1] * c + p[1], Y[:, 2]]
# masks
def background(k, half=60, step=6):
    fr = [seq.frames[j] for j in range(max(0, k - half), min(len(seq.frames), k + half + 1), step) if seq.frames[j] is not None]
    return np.median(np.stack(fr), 0).astype(np.uint8)
rawm = {}
SAM_MASKS = os.environ.get("MASKS", "sam") == "sam"           # SAM 2.1 prompted by the labelled 2D box (GPU venv); else GrabCut
if SAM_MASKS:
    import torch
    from ultralytics import SAM
    sam = SAM(os.path.expanduser("~/.cache/sam/sam2.1_l.pt"))
    from sam_masks import sam_crop
    for k in ks:
        if seq.frames[k] is None: continue
        x0, y0, x1, y1 = lab[k]['box']
        if x0 < 3 or y0 < 3 or x1 > Ww_img - 3 or y1 > Hh_img - 3: continue
        m = sam_crop(sam, seq.frames[k], [x0 - 2, y0 - 2, x1 + 2, y1 + 2], rider=typ != 'Pedestrian')   # on a crop around it
        if not m.any(): continue
        n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        if n < 2: continue
        keep = [i for i in range(1, n) if st[i, 4] >= 0.1 * st[1:, 4].max()]           # a rider and the bike may part
        rawm[k] = np.isin(cc, keep)
for k in ([] if SAM_MASKS else ks):
    if seq.frames[k] is None: continue
    x0, y0, x1, y1 = lab[k]['box']
    if x0 < 3 or y0 < 3 or x1 > Ww_img - 3 or y1 > Hh_img - 3: continue
    pad = 16; X0, Y0, X1, Y1 = int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(Ww_img, x1 + pad)), int(min(Hh_img, y1 + pad))
    crop = seq.frames[k][Y0:Y1, X0:X1]; bg = background(k)[Y0:Y1, X0:X1]
    yy, xx = np.mgrid[Y0:Y1, X0:X1]; inbox = (xx >= x0 - 2) & (xx <= x1 + 2) & (yy >= y0 - 2) & (yy <= y1 + 2)
    ring = ~inbox                                                    # exposure changes: match the background to the frame on the road around
    gain = (np.median(crop[ring].astype(float), 0) + 1) / (np.median(bg[ring].astype(float), 0) + 1) if ring.sum() > 50 else np.ones(3)
    diff = np.linalg.norm(crop.astype(float) - bg.astype(float) * gain, axis=2) > 25
    gc = np.full(diff.shape, cv2.GC_BGD, np.uint8); gc[inbox] = cv2.GC_PR_BGD; gc[inbox & diff] = cv2.GC_PR_FGD
    gc[cv2.erode((inbox & diff).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = cv2.GC_FGD
    if not (gc == cv2.GC_FGD).any(): continue
    try: cv2.grabCut(crop, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error: continue
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2: continue
    keep = [i for i in range(1, n) if st[i, 4] >= 0.1 * st[1:, 4].max()]               # a rider and the bike may part
    seg = np.isin(cc, keep) & (cv2.dilate(diff.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)   # road seen through wheels stays out
    full = np.zeros((Hh_img, Ww_img), bool); full[Y0:Y1, X0:X1] = seg; rawm[k] = full
def box_iou(dt):
    vals = []
    for k, m in rawm.items():
        if not (kf[0] <= k + dt <= kf[-1]): continue
        q = np.round(vs.project(seq.cams[k], to_world(*pose_l(k + dt), BOX))[0]).astype(np.int32)
        mm = np.zeros(seq.shape, np.uint8); cv2.fillConvexPoly(mm, cv2.convexHull(q), 1)
        vals.append(((mm > 0) & m).sum() / max(((mm > 0) | m).sum(), 1))
    return np.mean(vals) if vals else 0
DT = max(np.arange(-3, 1.01, 0.25), key=box_iou)
print('camera-time shift %.2f frames (box-mask IoU %.2f, without shift %.2f)' % (DT, box_iou(DT), box_iou(0)))
pose = {k: pose_l(k + DT) for k in rawm if kf[0] <= k + DT <= kf[-1]}
masks = {}
for k in pose:
    hb = np.zeros(seq.shape, np.uint8); cv2.fillConvexPoly(hb, cv2.convexHull(np.round(vs.project(seq.cams[k], to_world(*pose[k], BOX))[0]).astype(np.int32)), 1)
    masks[k] = rawm[k] & (cv2.dilate(hb, np.ones((9, 9), np.uint8)) > 0)
# hull
step = 0.02
HEAD = os.environ.get('HEAD_FIX', '1') == '1'          # the top from the silhouettes: the LiDAR's p99 misses the sparse head
xs, ys, zs = np.arange(-L / 2 - 0.3, L / 2 + 0.3, step), np.arange(-W / 2 - 0.25, W / 2 + 0.25 + 1e-9, step), np.arange(0, H + (0.5 if HEAD else 0.1), step)
ys = ys - ys.mean()
G = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1).reshape(-1, 3)
votes = np.zeros(len(G)); seen = np.zeros(len(G))
for k, m in masks.items():
    mm = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    q = np.round(vs.project(seq.cams[k], to_world(*pose[k], G))[0]).astype(int)
    ok = (q[:, 0] >= 0) & (q[:, 0] < Ww_img) & (q[:, 1] >= 0) & (q[:, 1] < Hh_img); seen += ok; votes[ok] += mm[q[ok, 1], q[ok, 0]]
occ = ((votes >= np.round(0.8 * seen)) & (seen > 0)).reshape(len(xs), len(ys), len(zs)); occ &= occ[:, ::-1, :]
if HEAD:
    # the highest slice the masks still fill (>= 20 voxels at the vote), at most 0.4 m over the LiDAR's p99
    nz = occ.sum((0, 1)); top = zs[np.nonzero(nz >= 20)[0].max()] if (nz >= 20).any() else H
    print('top from the silhouettes %.2f m (LiDAR p99 %.2f m)' % (top, H)); H = max(H, float(top))
occ &= (zs <= H)[None, None, :]
if typ == 'Pedestrian' and os.environ.get('TEMPLATE', '1') == '1':
    # a side-on carve cannot narrow the body along the line of sight (shoulders -> neck -> head run across the walk,
    # i.e. along the rays): the measured carve keeps its side profile, its depth across the walk follows a human
    # half-width profile (anthropometric, scaled to the measured height; arms included at the shoulders)
    zf = [0.00, 0.25, 0.47, 0.55, 0.70, 0.80, 0.83, 0.855, 0.875, 0.90, 0.95, 1.00]
    hw = [0.13, 0.13, 0.17, 0.17, 0.19, 0.22, 0.22, 0.075, 0.07, 0.085, 0.085, 0.05]
    lim = np.interp(zs / H, zf, hw) * H / 1.75
    occ &= np.abs(ys)[None, :, None] <= lim[None, None, :]
occ &= (np.abs(xs) <= L / 2 + 0.05)[:, None, None] & (np.abs(ys) <= W / 2 + 0.05)[None, :, None]
if os.environ.get("OCC_OVERRIDE"):          # a refined occupancy on the same grid (tri_vru.py: the triangulated surface)
    occ = np.load(os.environ["OCC_OVERRIDE"]); print("occupancy from", os.environ["OCC_OVERRIDE"])
if os.environ.get("HULL_DEBUG"):
    np.savez(os.environ["HULL_DEBUG"], votes=votes.reshape(len(xs), len(ys), len(zs)), seen=seen.reshape(len(xs), len(ys), len(zs)), occ=occ, xs=xs, ys=ys, zs=zs, H=H)
# LiDAR reference: the points in the object frame of each frame (no shift: the LiDAR's own time), mirrored
cl = np.vstack([np.c_[(P[k][:, :2] - Cs[i]) @ rot(TH[i]), P[k][:, 2]] for i, k in enumerate(ks)])
cl = cl[(np.abs(cl[:, 0]) < L / 2 + 0.1) & (np.abs(cl[:, 1]) < W / 2 + 0.1)]; cl = np.vstack([cl, cl * [1, -1, 1]])
gs = 0.05; rx, ry = np.arange(-1.5, 1.5, gs) + gs / 2, np.arange(-1.0, 1.0, gs) + gs / 2
def hf(Pts):
    Hm = np.zeros((len(rx), len(ry))); ix = np.floor((Pts[:, 0] + 1.5) / gs).astype(int); iy = np.floor((Pts[:, 1] + 1.0) / gs).astype(int)
    ok = (ix >= 0) & (ix < len(rx)) & (iy >= 0) & (iy < len(ry)); np.maximum.at(Hm, (ix[ok], iy[ok]), Pts[ok, 2]); return Hm
Lr = w.close_lidar_holes(hf(cl), gs, radius=0.08)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij'); Hm = hf(np.c_[X[occ], Y[occ], Z[occ]])
iou = w.shape_scores(Hm, rx, ry, Lr, rx, ry)['iou'] if occ.any() else np.nan
print('hull voxels', occ.sum(), 'size L %.2f W %.2f H %.2f' % ((np.ptp(X[occ]) + step, np.ptp(Y[occ]) + step, Z[occ].max()) if occ.any() else (0, 0, 0)), '| IoU vs its LiDAR %.3f' % iou)
V, F, _, _ = marching_cubes(gaussian_filter(np.pad(occ, 2).astype(float), 0.8), 0.5); V = (V - 2) * step + [xs[0], ys[0], zs[0]]
n = np.zeros_like(V); fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
for i in range(3): np.add.at(n, F[:, i], fn)
n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
if np.mean(((V - V.mean(0)) * n).sum(1)) < 0: n = -n
def visible(k, Vw, q):
    cm = seq.cams[k]; z = (cm[1] @ Vw.T + cm[2][:, None])[2]; key = np.floor(q[:, 0]).astype(int) * 100000 + np.floor(q[:, 1]).astype(int); zmin = {}
    for kk, zz in zip(key, z): zmin[kk] = min(zz, zmin.get(kk, 1e9))
    return z <= np.array([zmin[k_] for k_ in key]) + 0.06
best = np.full(len(V), -1.0); col = np.full((len(V), 3), 190.0); seenv = np.zeros(len(V), bool)
for k in sorted(masks):
    cm = seq.cams[k]; Ck = -np.linalg.solve(cm[1], cm[2]); p, a = pose[k]; Vw = to_world(p, a, V); q, _ = vs.project(cm, Vw); qi = np.round(q).astype(int)
    vd_ = Ck - Vw; dist = np.linalg.norm(vd_, axis=1); vd_ /= dist[:, None]; nw = np.c_[n[:, :2] @ rot(a).T, n[:, 2]]; cosv = (nw * vd_).sum(1)
    ok = (qi[:, 0] >= 0) & (qi[:, 0] < Ww_img) & (qi[:, 1] >= 0) & (qi[:, 1] < Hh_img) & (cosv > 0.05)
    ok[ok] &= cv2.dilate(masks[k].astype(np.uint8), np.ones((3, 3), np.uint8))[qi[ok, 1], qi[ok, 0]] > 0; ok &= visible(k, Vw, q)
    sc = cosv * (30.0 / dist); u = ok & (sc > best); best[u] = sc[u]; col[u] = seq.frames[k][qi[u, 1], qi[u, 0]]; seenv |= u
if seenv.any():
    d, j = cKDTree(V[seenv]).query(V * [1, -1, 1]); fill = (~seenv) & (d < 0.04); col[fill] = col[seenv][j[fill]]
    rest = ~(seenv | fill); d2, j2 = cKDTree(V[seenv | fill]).query(V[rest]); col[rest] = col[seenv | fill][j2]
# figure: frame | mask | box | three views of the model | from above vs LiDAR
k = int(os.environ["SHOW_K"]) if os.environ.get("SHOW_K") else max(masks, key=lambda k: masks[k].sum()); m = masks[k]; yy, xx = np.nonzero(m)
pad = int(0.35 * (yy.max() - yy.min())); X0, Y0, X1, Y1 = max(0, xx.min() - pad), max(0, yy.min() - pad // 2), min(Ww_img, xx.max() + pad), min(Hh_img, yy.max() + pad // 3)
crop = lambda im: im[Y0:Y1, X0:X1].copy()
ov = seq.frames[k].copy(); ov[m] = (0.45 * ov[m] + 0.55 * np.array([60, 200, 60])).astype(np.uint8)
cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cnt, -1, (0, 150, 0), 1, cv2.LINE_AA)
bi = seq.frames[k].copy(); qb = vs.project(seq.cams[k], to_world(*pose[k], BOX))[0]
for a_, b_ in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
    cv2.line(bi, tuple(np.round(qb[a_]).astype(int)), tuple(np.round(qb[b_]).astype(int)), (40, 200, 40), 1, cv2.LINE_AA)
Cw = -np.linalg.solve(seq.cams[k][1], seq.cams[k][2]); p_, a_ = pose[k]; Cc = (Cw[:2] - p_) @ rot(a_); az = np.arctan2(Cc[1], Cc[0])
Kn = np.array([[1100.0, 0, 160], [0, 1100.0, 140], [0, 0, 1]]); views = []
for dz, t in ((0, 'as the camera saw it'), (90, 'turned 90 deg'), (180, 'from behind (mirror colours)')):
    aa = az + np.radians(dz); camn = cf.look_at(Kn, np.array([9.0 * np.cos(aa), 9.0 * np.sin(aa), 3.0]), np.array([0, 0, H / 2]))
    views.append(cf.panel(cf.render_mesh(np.full((280, 320, 3), 255, np.uint8), camn, V, F, col), t, size=(320, 280)))
row = np.hstack([cf.panel(crop(seq.frames[k]), f'{typ} {TID}: frame {k}', size=(320, 280)), cf.panel(crop(ov), 'mask (SAM 2.1, LiDAR box)', size=(320, 280)),
                 cf.panel(crop(bi), f'LiDAR box {L:.2f} x {W:.2f} x {H_LIDAR:.2f} m', size=(320, 280))] + views)
cv2.putText(row, f'IoU vs its LiDAR {iou:.2f}; {len(masks)} masks; label size {np.median([lab[k]["dims"] for k in lab], 0).round(2).tolist()}', (8, row.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
cv2.imwrite(f'{S}/vru_{TID}.jpg', row)
def _crop(m):
    yy, xx = np.nonzero(m); return (int(yy.min()), int(xx.min()), m[yy.min():yy.max() + 1, xx.min():xx.max() + 1]) if len(yy) else None
pickle.dump({'V': V, 'F': F, 'col': col, 'occ': occ, 'iou': iou, 'dims': (L, W, H), 'DT': DT, 'typ': typ, 'pose': pose,
             'masks': {k: _crop(m) for k, m in masks.items()}, 'cl': cl, 'axes': (xs, ys, zs)}, open(f'{S}/vru_{TID}.pkl', 'wb'))
