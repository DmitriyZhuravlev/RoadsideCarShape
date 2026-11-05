"""Orientation from motion, intersection A: (a) every car's path and its heading from the track solve (blue), with the
LiDAR-fitted cuboids of the labels (grey, evaluation only); (b) one turning car up close. Needs figures/models_A.pkl
(scripts/capture_models.py). Writes figures/heading.png. Usage: python scripts/heading_figure.py [label]"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/Documents/MyArticles/code/3dseg")); os.chdir(os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
import v2x_seq_shape as vs
FIG = os.path.expanduser("~/Documents/MyArticles/RoadsideCarShape/figures")
TURN = sys.argv[1] if len(sys.argv) > 1 else "003981"
models = pickle.load(open(os.path.join(FIG, "models_A.pkl"), "rb"))
seq = vs.Sequence()


def draw(ax, m, every, arrow, legend=False):
    """The car's path and headings (from its track-solve poses: car xy = A world xy + b) and its labelled cuboids;
    returns the per-frame heading errors against the labels (rad, modulo 180 deg)."""
    lab = m["row"]["label"]
    lb = {o["k"]: o for o in seq.tracks.get(lab, [])}
    ks = sorted(m["poses"])
    cen = {k: m["poses"][k][0].T @ -m["poses"][k][1] for k in ks}
    head = {k: np.arctan2(m["poses"][k][0][0, 1], m["poses"][k][0][0, 0]) for k in ks}
    P = np.array([cen[k] for k in ks])
    ax.plot(P[:, 0], P[:, 1], color="tab:blue", lw=1.0, label="path and heading from motion" if legend else None)
    for i, k in enumerate(ks[::every]):
        if k in lb:
            o = lb[k]; l, w = o["dims"][:2]; c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
            Q = np.array([[l, w], [l, -w], [-l, -w], [-l, w], [l, w]]) / 2 @ np.array([[c, s], [-s, c]]) + o["pos"]
            ax.plot(Q[:, 0], Q[:, 1], color="0.6", lw=0.7,
                    label="LiDAR-fitted cuboid (evaluation)" if legend and i == 0 else None)
        ax.annotate("", xy=cen[k] + arrow * np.array([np.cos(head[k]), np.sin(head[k])]), xytext=cen[k],
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="tab:blue"))
    return [abs((head[k] - lb[k]["yaw"] + np.pi / 2) % np.pi - np.pi / 2) for k in ks if k in lb]


fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
errs = []
for j, m in enumerate(models.values()):
    if m["row"].get("label"):
        e = draw(a, m, 12, 3.0, legend=(j == 0))
        if e:
            errs.append(np.median(e))
a.set_aspect("equal"); a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
a.set_title(f"(a) all {len(errs)} cars of A: median heading error {np.degrees(np.median(errs)):.1f}$^\\circ$", fontsize=10)
a.legend(loc="best", fontsize=8)
turn = next(m for m in models.values() if m["row"].get("label") == TURN)
e = draw(b, turn, 6, 2.0)
b.set_aspect("equal"); b.set_xlabel("x (m)"); b.set_ylabel("y (m)")
b.set_title(f"(b) car {TURN} turning: median heading error {np.degrees(np.median(e)):.1f}$^\\circ$", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "heading.png"), dpi=150)
print("cars", len(errs), "median of per-car medians %.1f deg" % np.degrees(np.median(errs)), os.path.join(FIG, "heading.png"))
