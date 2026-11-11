"""Models reconstructed from the classical input (masks and cuboids obtained from the video alone), one car per row:
(a) the video frame with the classical mask (green outline) and cuboid (blue), (b) the model's triangle mesh, coarsened,
(c, d) the model with the colours of the video from the front left and the rear right.
  RUN=ann_Acls_e2e ANN=docs/precise_ann_Acls [N=6] [OUT=...] python3 scripts/classical_models_figure.py"""
import json
import os
import sys

RSC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RSC, "scripts"))
import cars_figure as cf  # noqa: E402  (switches to the 3dseg folder)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import v2x_seq_demo as vd  # noqa: E402

D = os.path.expanduser("~/Documents/MyArticles/code/3dseg/docs")
RUN = os.environ.get("RUN", "ann_Acls_e2e")
ANN = os.environ.get("ANN", "docs/precise_ann_Acls")
N = int(os.environ.get("N", "6"))
OUT = os.environ.get("OUT", os.path.expanduser("~/Documents/MyArticles/improve_runs/figs/classical_models.png"))
GRID = float(os.environ.get("GRID", "0.14"))
W, H = 1000, 563
BLUE, GREEN = (168, 95, 31), (60, 200, 60)                    # BGR
K0 = np.array([[2.3 * W, 0, W / 2], [0, 2.3 * W, H / 2 - 30], [0, 0, 1]])
TARGET = np.array([0.0, 0.0, 0.7])


def compose_grid(rows, col_titles, row_labels, out_path, width_in=10.0, dpi=200, title_size=10):
    """Save a figure with one row of panels per car."""
    R, C = len(rows), len(col_titles)
    if len(row_labels) != R or any(len(r) != C for r in rows):
        raise ValueError("rows, col_titles and row_labels do not agree")
    h0, w0 = rows[0][0].shape[:2]
    height = R * (width_in / C) * h0 / w0 + 0.35
    fig, axes = plt.subplots(R, C, figsize=(width_in, height), squeeze=False)
    fig.subplots_adjust(left=0.004, right=0.996, top=1 - 0.28 / height, bottom=0.004, wspace=0.02, hspace=0.03)
    for i in range(R):
        for j in range(C):
            axes[i][j].imshow(rows[i][j])
            axes[i][j].axis("off")
        axes[i][0].text(0.02, 0.04, row_labels[i], transform=axes[i][0].transAxes, fontsize=title_size - 2,
                        color="white", bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))
    for j in range(C):
        axes[0][j].set_title(col_titles[j], loc="left", pad=4, fontsize=title_size)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def cluster(V, F, g):
    """Vertex clustering on a grid of g metres (as in teaser_figure.py)."""
    _, inv = np.unique(np.floor(V / g).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    Vc = np.zeros((inv.max() + 1, 3))
    np.add.at(Vc, inv, V)
    Vc /= np.bincount(inv)[:, None]
    Fc = inv[F]
    Fc = Fc[(Fc[:, 0] != Fc[:, 1]) & (Fc[:, 1] != Fc[:, 2]) & (Fc[:, 2] != Fc[:, 0])]
    _, u = np.unique(np.sort(Fc, axis=1), axis=0, return_index=True)
    Fc = Fc[np.sort(u)]
    n = np.cross(Vc[Fc[:, 1]] - Vc[Fc[:, 0]], Vc[Fc[:, 2]] - Vc[Fc[:, 0]])
    if np.einsum("ij,ij->i", n, Vc[Fc].mean(1) - Vc.mean(0)).sum() < 0:
        Fc = Fc[:, ::-1]
    return Vc, Fc


def draw_wire(img, cam, eye, V, F, color=BLUE, thick=1, scale=4):
    K, R, t = cam
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    F = F[np.einsum("ij,ij->i", n, eye - V[F].mean(1)) > 0]
    p = (V @ R.T + t) @ K.T
    ok = p[:, 2] > 1e-6
    F = F[ok[F].all(1)]
    uv = p[:, :2] / np.where(ok, p[:, 2], 1)[:, None]
    E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1), axis=0)
    q = np.round(uv * scale).astype(np.int32)
    for a, b in E:
        cv2.line(img, (int(q[a, 0]), int(q[a, 1])), (int(q[b, 0]), int(q[b, 1])), color, thick, cv2.LINE_AA,
                 shift=int(np.log2(scale)))
    return img


def hcam(a_deg, elev=30.0, dist=20.0, zoom=1.3):
    e, a = np.radians(elev), np.radians(a_deg)
    eye = TARGET + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    Kz = K0.copy()
    Kz[0, 0] = Kz[1, 1] = zoom * K0[0, 0]
    return cf.look_at(Kz, eye, TARGET), eye


def frame_panel(fids, tid, k0, k1):
    """The frame of the track (window indices k0..k1) where the classical mask covers most of the cuboid."""
    best = None
    for k in range(k0, k1 + 1):
        p = os.path.join(ANN, "cuboids", fids[k] + ".json")
        if not os.path.exists(p):
            continue
        E = json.load(open(p))
        i = next((i for i, e in enumerate(E) if str(e.get("track")) == tid), None)
        if i is not None and (best is None or E[i].get("visible", 0) > best[2].get("visible", 0)):
            best = (k, i, E[i])
    k, i, e = best
    img = cv2.imread(os.path.join(vd.DATA, "image", fids[k] + ".jpg"))
    m = cv2.imread(os.path.join(ANN, "masks", fids[k] + ".png"), cv2.IMREAD_UNCHANGED)
    uv = np.array(e["corners_uv"])
    cnt = cv2.findContours((m == i + 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(img, cnt, -1, GREEN, 2, cv2.LINE_AA)
    for a in range(8):
        for bit in (1, 2, 4):
            if a < a ^ bit:
                cv2.line(img, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[a ^ bit]), BLUE, 2, cv2.LINE_AA)
    (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
    cw = max(1.5 * (x1 - x0), 1.5 * (y1 - y0) * W / H)
    ch = cw * H / W
    X0 = int(min(max(0, (x0 + x1) / 2 - cw / 2), max(0, img.shape[1] - cw)))
    Y0 = int(min(max(0, (y0 + y1) / 2 - ch / 2), max(0, img.shape[0] - ch)))
    return cv2.resize(img[Y0:Y0 + int(ch), X0:X0 + int(cw)], (W, H), interpolation=cv2.INTER_CUBIC), fids[k]


res = [r for r in json.load(open(f"{D}/whole_scene_cars_{RUN}.json")) if r.get("ann_track")]
meshes = json.load(open(f"{D}/cars_viewer_{RUN}/meshes.json"))
res = [r for r in res if f"track {r['track']}" in meshes]
res.sort(key=lambda r: -r.get("fused closed: iou", -1))
pick = os.environ.get("TRACKS", "").split()
res = [r for r in res if r["ann_track"] in pick] if pick else res[:N]
fids = vd.load_sequence()[0]
rows, labels = [], []
white = lambda: np.full((H, W, 3), 255, np.uint8)
for r in res:
    m = meshes[f"track {r['track']}"]
    V, F = np.array(m["v"]).reshape(-1, 3), np.array(m["f"]).reshape(-1, 3)
    C = np.array(m["c"]).reshape(-1, 3)[:, ::-1] * 255
    k0, k1 = (int(x) for x in r["frames"].split("-"))
    crop, fid = frame_panel(fids, r["ann_track"], k0, k1)
    Vc, Fc = cluster(V, F, GRID)
    cam, eye = hcam(45.0)
    mesh = draw_wire(cf.render_mesh(white(), cam, Vc, Fc, np.full((len(Vc), 3), 232.0)), cam, eye, Vc, Fc)
    rows.append([cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in
                 (crop, mesh, cf.render_mesh(white(), hcam(45.0)[0], V, F, C), cf.render_mesh(white(), hcam(225.0)[0], V, F, C))])
    iou = r.get("fused closed: iou")
    labels.append(f"{r['windows']} frames, {r['length']:.1f} x {r['width']:.1f} x {r['height']:.1f} m"
                  + (f", IoU {iou:.2f}" if iou is not None else ""))
    print(r["ann_track"], "frame", fid, labels[-1], "label", r.get("label"), flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
compose_grid(rows, ["(a) frame, mask and cuboid", "(b) triangle mesh", "(c) model, front left",
                    "(d) model, rear right"], labels, OUT)
print(OUT)
