"""Opening figure: one car of DAIR-V2X-C (clip 114_0, car 000046): (a) the video frame, (b) the triangle mesh of the
fused model (shown coarsened, dark-blue edges on a grey surface), (c, d) the model with the colours of the video from the
front left (as (b)) and the rear right, all from 30 deg above, rendered at high resolution.
Run with V2X_SEQ_DIR=<dataset>/dair_c_spd/clip_114_0. Writes figures/teaser.png."""
import json
import os
import pickle
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
RUN, LAB, K_WIN = "ann_P_clip_114_0_all", "000046", int(os.environ.get("K_WIN", "142"))
RUN_COL = RUN + "_fill"     # the same model, subdivided and coloured (MESH_SUBDIV=2, FILL_UP=1)
DUMPS = f"{D}/ann_objects_P_clip_114_0_all"
GRID = float(os.environ.get("GRID", "0.14"))               # m: vertex clustering of the shown mesh
W, H = 1400, 788                                            # panel size, 16:9
BLUE = (168, 95, 31)                                        # BGR of #1f5fa8


def compose_2x2(images, titles, out_path, width_in=10.0, dpi=200, title_size=11):
    if len(images) != 4 or len(titles) != 4:
        raise ValueError("len(images) != 4 or len(titles) != 4")
    h0, w0, _ = images[0].shape
    height = 2 * (width_in / 2) * h0 / w0 + 0.7
    fig, axes = plt.subplots(2, 2, figsize=(width_in, height))
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.3 / height, bottom=0.005, wspace=0.02, hspace=0.12)
    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.imshow(img)
        ax.set_title(title, loc="left", fontsize=title_size, pad=4)
        ax.axis("off")
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def cluster(V, F, g):
    """Vertex clustering on a grid of g metres: one vertex per occupied cell (the mean), degenerate faces dropped."""
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
    if np.einsum("ij,ij->i", n, Vc[Fc].mean(1) - Vc.mean(0)).sum() < 0:   # wound inward: turn outward
        Fc = Fc[:, ::-1]
    return Vc, Fc


def draw_wire(img, cam, eye, V, F, color=BLUE, thick=1, scale=4):
    """Edges of the front-facing triangles of a mesh, drawn onto img."""
    K, R, t = cam
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    F = F[np.einsum("ij,ij->i", n, eye - V[F].mean(1)) > 0]
    p = (V @ R.T + t) @ K.T
    ok = p[:, 2] > 1e-6
    F = F[ok[F].all(1)]
    uv = p[:, :2] / np.where(ok, p[:, 2], 1)[:, None]
    E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1), axis=0)
    q = np.round(uv * scale).astype(np.int32)
    sh = int(np.log2(scale))
    for a, b in E:
        cv2.line(img, (int(q[a, 0]), int(q[a, 1])), (int(q[b, 0]), int(q[b, 1])), color, thick, cv2.LINE_AA, shift=sh)
    return img


def model(run):
    rows = [x for x in json.load(open(f"{D}/whole_scene_cars_{run}.json")) if x.get("label") == LAB and "fused closed: iou" in x]
    r = max(rows, key=lambda x: x["fused closed: iou"])
    m = json.load(open(f"{D}/cars_viewer_{run}/meshes.json"))[f"track {r['track']}"]
    return r, np.array(m["v"]).reshape(-1, 3), np.array(m["f"]).reshape(-1, 3), np.array(m["c"]).reshape(-1, 3)


# (a) the video frame around the car, framed by its body in one window
o = next(x for x in pickle.load(open(f"{DUMPS}/{K_WIN:04d}.pkl", "rb")) if x["track_tid"] == LAB)
fids = vd.load_sequence()[0]
frame = cv2.imread(os.path.join(vd.DATA, "image", f"{fids[K_WIN]}.jpg"))
b = o["uv"][o["body"]]
(x0, y0), (x1, y1) = b.min(0), b.max(0)
cw = 1.45 * (x1 - x0)
ch = cw * H / W
X0 = min(max(0, int(round((x0 + x1) / 2 - cw / 2))), frame.shape[1] - int(cw))
Y0 = min(max(0, int(round((y0 + y1) / 2 - ch / 2))), frame.shape[0] - int(ch))
crop = cv2.resize(frame[Y0:Y0 + int(round(ch)), X0:X0 + int(round(cw))], (W, H), interpolation=cv2.INTER_CUBIC)

K = np.array([[2.3 * W, 0, W / 2], [0, 2.3 * W, H / 2 + 10], [0, 0, 1]])
target = np.array([0.0, 0.0, 0.7])


def hcam(a_deg, elev=30.0, dist=20.0, zoom=1.3):
    """Camera at azimuth a_deg (0: the front, 90: the left, 180: the rear) and elevation elev above the road."""
    e, a = np.radians(elev), np.radians(a_deg)
    eye_ = target + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    Kz = K.copy()
    Kz[0, 0] = Kz[1, 1] = zoom * K[0, 0]
    Kz[1, 2] = H / 2 - 40                                        # centred: the near end reaches lower
    return cf.look_at(Kz, eye_, target), eye_


A_VIEW = 45.0                                                    # front left; (d) turned by 180 deg

# (b) the model's triangle mesh, coarsened for display
r, Vm, Fm, _ = model(RUN)
n_full = len(Fm)
Vm, Fm = cluster(Vm, Fm, GRID)
cam, e_b = hcam(A_VIEW)
mesh = cf.render_mesh(np.full((H, W, 3), 255, np.uint8), cam, Vm, Fm, np.full((len(Vm), 3), 232.0))
draw_wire(mesh, cam, e_b, Vm, Fm)

# (c, d) the coloured model
_, V, F, C = model(RUN_COL)
C = C[:, ::-1] * 255
high = lambda a: cf.render_mesh(np.full((H, W, 3), 255, np.uint8), hcam(a)[0], V, F, C)

imgs = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in (crop, mesh, high(A_VIEW), high(A_VIEW + 180))]
titles = ["(a) video frame", "(b) model, triangle mesh, front left", "(c) model, front left",
          "(d) model, rear right"]
out = os.path.join(RSC, "figures", "teaser.png")
compose_2x2(imgs, titles, out)
print(out, "frame", fids[K_WIN], "mesh shown", len(Fm), "of", n_full, "triangles",
      {k: r[k] for k in ("track", "windows", "length", "label_length", "fused closed: iou")})
