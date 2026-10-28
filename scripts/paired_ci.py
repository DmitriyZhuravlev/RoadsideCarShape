"""Paired bootstrap intervals for the article: per dataset the mean per-car difference of the closed-LiDAR solid IoU
(ours - labelled box) with a 95 % bootstrap confidence interval, split by whether the car's length was measured
side-on (n_side >= 2) or came from the length prior (n_side < 2).

    python scripts/paired_ci.py            -> tables/paired_ci.tex (and the numbers on stdout)
    python scripts/paired_ci.py --boxes A|--boxes B    (internal) per-car labelled-box IoUs of the A5/B5 runs

Datasets: intersections A and B (3dseg docs/whole_scene_cars_abl_{A,B}5.json, the article's "ours"; the per-car box
scores computed exactly as scripts/box_scores.py - in a subprocess, because the sequence and its V2X_SEQ_DIR are per
dataset), and the larger evaluations (dataset/{dair_c_runs,spd_runs}; per-clip cars.json + boxes.json from pool.py).
Rows without n_side (older runs) form one "n/a" group. Bootstrap: 5000 resamples, fixed seed.
"""
import json
import os
import subprocess
import sys

import numpy as np

CODE = os.path.expanduser("~/Documents/MyArticles/code/3dseg")
RUNS = os.path.expanduser("~/Documents/MyArticles/dataset")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
N_BOOT, SEED = 5000, 0
LARGE = [("DAIR-V2X-C", "dair_c_runs"), ("V2X-Seq-SPD", "spd_runs")]


def boxes_site(site):
    """(internal, subprocess) per-car labelled-box IoU against the gap-closed LiDAR of intersection A or B, on the
    cars of the article's runs (abl_A5 / abl_B5), scored exactly like scripts/box_scores.py."""
    if site == "B":
        os.environ["V2X_SEQ_DIR"] = os.path.join(RUNS, "coop_spd")
    os.chdir(CODE)
    sys.path.insert(0, CODE)
    import ego_sequence as es  # noqa: E402
    import lidar_masks as lm  # noqa: E402
    import whole_scene_cars as w  # noqa: E402
    seq, ground = es.open_sequence()
    shapes = lm.LidarShapes(seq, ground)
    run = {"A": "A5", "B": "B5"}[site]
    rows = [r for r in json.load(open(os.path.join("docs", f"whole_scene_cars_abl_{run}.json")))
            if "fused closed: iou" in r]
    xs = np.arange(-5, 5, 0.1) + 0.05
    ys = np.arange(-1.5, 1.5, 0.1) + 0.05
    out = {}
    for r in rows:
        Lt, lx, ly, n = w.lidar_heights(shapes, r["label"])
        Lc = w.close_lidar_holes(Lt, lx[1] - lx[0])
        box = np.where((np.abs(xs)[:, None] < r["label_length"] / 2) & (np.abs(ys)[None] < r["label_width"] / 2),
                       r["label_height"], 0.0)
        out[str(r["label"])] = {"ours": r["fused closed: iou"], "box": w.shape_scores(box, xs, ys, Lc, lx, ly)["iou"],
                                "n_side": r.get("n_side")}
    print(json.dumps(out))


def site_rows(site):
    r = subprocess.run([sys.executable, __file__, "--boxes", site], capture_output=True, text=True, check=True)
    return list(json.loads(r.stdout[r.stdout.index("{"):]).values())   # past the ground-plane line


def large_rows(runs_dir):
    rows = []
    for c in json.load(open(os.path.join(runs_dir, "chosen.json"))):
        cars_p = os.path.join(runs_dir, c["clip"], "cars.json")
        boxes_p = os.path.join(runs_dir, c["clip"], "boxes.json")
        if not (os.path.exists(cars_p) and os.path.exists(boxes_p)):
            continue
        boxes = json.load(open(boxes_p))
        for r in json.load(open(cars_p)):
            if "fused closed: iou" in r and str(r["track"]) in boxes:
                rows.append({"ours": r["fused closed: iou"], "box": boxes[str(r["track"])]["box closed: iou"],
                             "n_side": r.get("n_side")})
    return rows


def boot(rows):
    """mean paired difference ours - box and its 95 % percentile bootstrap interval."""
    d = np.array([r["ours"] - r["box"] for r in rows])
    rng = np.random.default_rng(SEED)
    means = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return len(d), float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


GROUPS = ("all", "measured", "prior", "n/a")


def grouped(rows):
    """the rows in the four groups: measured n_side >= 2, prior n_side < 2, n/a without the key, and all together"""
    has = [r for r in rows if r.get("n_side") is not None]
    return {"all": rows, "measured": [r for r in has if r["n_side"] >= 2], "prior": [r for r in has if r["n_side"] < 2],
            "n/a": [r for r in rows if r.get("n_side") is None]}


def main():
    datasets = [("intersection A (tuning)", site_rows("A")), ("intersection B", site_rows("B"))]
    datasets += [(name, large_rows(os.path.join(RUNS, d))) for name, d in LARGE]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Paired difference of the mean solid IoU (ours $-$ labelled box) against the gap-closed LiDAR,"
             r" per car, with a 95\,\% bootstrap confidence interval (5000 resamples, fixed seed). \emph{measured}:"
             r" the car's length was seen side-on ($\ge 2$ side-on windows, \texttt{n\_side} $\ge 2$); \emph{prior}:"
             r" capped by the 4.6\,m length prior; \emph{n/a}: older runs without the split. Intersection A is the"
             r" tuning set of the method.}",
             r"\label{tab:paired}", r"\small", r"\begin{tabular}{llrcl}", r"\toprule",
             r"Dataset & length & cars & mean $\Delta$ IoU & 95\,\% CI \\", r"\midrule"]
    for i, (name, rows) in enumerate(datasets):
        g = grouped(rows)
        for j, key in enumerate(GROUPS):
            if not g[key]:
                continue
            n, mean, lo, hi = boot(g[key])
            ci = f"[{lo:+.3f}, {hi:+.3f}]" if n >= 2 else "--"
            lines.append(" & ".join([name if j == 0 else "", key, str(n), f"{mean:+.3f}", ci]) + r" \\")
        if i < len(datasets) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "paired_ci.tex"), "w").write("\n".join(lines) + "\n")
    for name, rows in datasets:
        g = grouped(rows)
        for key in GROUPS:
            if g[key]:
                n, mean, lo, hi = boot(g[key])
                print(f"{name:26s} {key:9s} n={n:3d}  mean {mean:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--boxes":
        boxes_site(sys.argv[2])
    else:
        main()
