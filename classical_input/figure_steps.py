"""Fig. 2 of the article (figures/classical_steps.png): the classical input around one car of intersection A, frame
010871 - frame, foreground, flow, segments, the tangent cuboid and the window triangulation (as code/make_dumps.py).
    python classical_input/figure_steps.py"""
import os, sys
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.argv = sys.argv[:1]
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D); os.chdir(D)
import camera, ofc, foreground, run_example as rx
FIG = os.path.join(D, "..", "figures")
fids = sorted(f[:-4] for f in os.listdir(os.path.join(rx.ROOT, "image")))
K = fids.index("010871")
cam = camera.Camera(rx.ROOT, fids[1])
paths = [os.path.join(rx.ROOT, "image", f + ".jpg") for f in fids]
fgs, bg = foreground.masks("gsoc", paths, {K})
prev, img = cv2.imread(paths[K - 1]), cv2.imread(paths[K])
objs, dbg = ofc.detect(cam, prev, img, bg, fgs[K])
for o in objs: o["track"] = objs.index(o); o["track_type"] = o["type"]
print([(o["type"], np.round(o["lwh"], 2).tolist(), [int(v) for v in o["box"]]) for o in objs])
car = min(objs, key=lambda o: abs(0.5 * (o["box"][0] + o["box"][2]) - 0.5 * (o["box"][0] + o["box"][2])) + 0 if False else -o["box"][3])  # the nearest (lowest) object
pal = (np.array(plt.get_cmap("tab20").colors) * 255).astype(np.uint8)
full = lambda x: cv2.resize(x, (1920, 1080), interpolation=cv2.INTER_NEAREST)
hsv = cv2.cvtColor(ofc.flow_hsv(dbg["flow"]), cv2.COLOR_HSV2RGB)
fg = np.where(dbg["foreground"][..., None], cv2.cvtColor(ofc.half(img), cv2.COLOR_BGR2RGB), 240).astype(np.uint8)
segs = np.full(dbg["flow_labels"].shape + (3,), 240, np.uint8)
for i, m in enumerate(dbg["segments"]):
    segs[m] = pal[i % 20]
# (e): the car's mask, its silhouette, the tangents of the three directions and the cuboid built from them
o = car
ax3 = ofc.axes(o["yaw"]); rays = cam.rays(o["hull"])
planes = {k: ofc.tangent_planes(cam, rays, d, o["occluded"]) for k, d in ax3.items()}
u, v = ax3["u"][:2], ax3["v"][:2]
l, w, h = o["lwh"]; sc, rc = o["centre"] @ u, o["centre"] @ v
pp = np.array([sc - l / 2, sc + l / 2, rc - w / 2, rc + w / 2, h])
X = np.array([pp[i] * ax3["u"] + pp[2 + j] * ax3["v"] + k * pp[4] * ax3["z"] for i, j, k in ofc.CORNERS])
Xc = cam.to_cam(X)
tg = img.copy()
mk = cv2.resize(o["mask"].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST) > 0
tg[mk] = (0.55 * tg[mk] + 0.45 * np.array([230, 160, 60])).astype(np.uint8)
rx.draw_cuboid(tg, cam, o, (255, 255, 255), 2)
cv2.polylines(tg, [np.round(o["hull"]).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
cols = {"u": (40, 40, 230), "v": (40, 180, 40), "z": (230, 120, 30)}
for key, (pl, (a, b)) in planes.items():
    ang = np.arctan2(Xc @ b, Xc @ a)
    for plane, c in zip(pl, (np.argmin(ang), np.argmax(ang))):
        if plane is None:
            continue
        line = np.array([X[c] + s * ax3[key] for s in np.linspace(-8, 8, 60)])
        luv = cam.project(line); luv = luv[np.isfinite(luv).all(1)]
        cv2.polylines(tg, [np.round(luv).astype(np.int32)], False, cols[key], 2, cv2.LINE_AA)
        cv2.circle(tg, tuple(np.round(cam.project(X[c][None])[0]).astype(int)), 7, cols[key], -1, cv2.LINE_AA)

# (f), (g): the window triangulation of the article (as code/make_dumps.py): grid vertices inside the eroded mask,
# tracked by the pipeline's tracker through +-8 frames, kept if followed in most, Delaunay inside the mask
sys.path.insert(0, "/home/dzhura/Documents/MyArticles/RoadsideCarShape/code")
import whole_scene as ws
from scipy.spatial import Delaunay
er = cv2.erode(mk.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
yy, xx = np.nonzero(er)
sel = (yy % 5 == 0) & (xx % 5 == 0)
P = np.c_[xx[sel], yy[sel]].astype(np.float32)
if len(P) > 900:
    P = P[::int(np.ceil(len(P) / 900))]
ks = list(range(K - 8, K + 9))
grays = {i: cv2.cvtColor(cv2.imread(paths[k]), cv2.COLOR_BGR2GRAY) for i, k in enumerate(ks)}
uv = ws.track_all(grays, 8, P)
good = (~np.isnan(uv).any(2)).sum(1) >= max(6, int(0.6 * len(ks)))
P, uv = P[good], uv[good]
tri = Delaunay(P).simplices
cen = P[tri].mean(1).round().astype(int)
tri = tri[er[cen[:, 1], cen[:, 0]]]                      # triangles inside the mask
J = 12                                                   # frame +4 of the window
last = np.nonzero(~np.isnan(uv[:, J]).any(1))[0]
def mesh(image, Q, tris, col=(168, 95, 31)):
    im = image.copy()
    for a, b, c in tris:
        if np.isfinite(Q[[a, b, c]]).all():
            cv2.polylines(im, [np.round(Q[[a, b, c]]).astype(np.int32)], True, col, 1, cv2.LINE_AA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
keep_last = np.isin(tri, last).all(1)
H_CENTRE = np.nanmean(uv[last, J], axis=0)
print("vertices", len(P), "triangles", len(tri))

x0, y0, x1, y1 = map(int, car["box"])                    # the panels around the example car
cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
crop = lambda c, wd, ht: (int(np.clip(c[0] - wd // 2, 0, 1920 - wd)), int(np.clip(c[1] - ht // 2, 0, 1080 - ht)))
small = crop((cx, cy), 600, 380)
panels = [("(a) frame", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), small),
          ("(b) foreground", full(fg), small),
          ("(c) optical flow", full(hsv), small),
          ("(d) segments", full(segs), small),
          (f"(e) cuboid {l:.2f} x {w:.2f} x {h:.2f} m", cv2.cvtColor(tg, cv2.COLOR_BGR2RGB),
           crop((cx, cy), 600, 380)),
          (f"(f) triangulation, {len(P)} vertices", mesh(img, P, tri), small)]
fig, axs = plt.subplots(2, 3, figsize=(10, 4.45))
for (title, im, (x_, y_)), a in zip(panels, axs.ravel()):
    a.imshow(im[y_:y_ + 380, x_:x_ + 600]); a.set_title(title, fontsize=9, loc="left", pad=3); a.axis("off")
    if title.startswith("(e)"):
        for key, name in (("u", "motion"), ("v", "lateral"), ("z", "vertical")):
            a.plot([], [], color=np.array(cols[key][::-1]) / 255, lw=3, label=name)
        a.legend(loc="lower left", fontsize=5.5, framealpha=0.85, handlelength=1.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "classical_steps.png"), dpi=110); plt.close(fig)
print("car", np.round(o["lwh"], 2).tolist(), [int(v) for v in o["box"]])
