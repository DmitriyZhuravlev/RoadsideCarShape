"""Paired bootstrap intervals for the article: per dataset the mean per-car difference of the closed-LiDAR solid IoU
(ours - labelled box) with a 95 % bootstrap confidence interval, split by whether the car's length was measured
side-on (n_side >= 2) or completed from its cuboid (n_side < 2).

    python scripts/paired_ci.py            -> tables/paired_ci.tex (and the numbers on stdout)

All on the precise annotation (SAM masks and LiDAR-fitted cuboids): intersections A and B are the full method's runs
(3dseg docs/whole_scene_cars_ann_{Aall1,Aall2,Ball}_full.json), the larger evaluations the per-clip runs
(docs/whole_scene_cars_ann_P_<clip>_all.json, clips of dataset/{dair_c_runs,spd_runs}/chosen.json). whole_scene_cars.py
scores the labelled box per car in the same row, so ours and the box are paired by the row; a label in two track
fragments keeps the longer track - never the higher score, which would bias the pair. Bootstrap: 5000 resamples,
fixed seed.
"""
import json
import os

import numpy as np

DOCS = os.path.expanduser("~/Documents/MyArticles/code/3dseg/docs")
RUNS = os.path.expanduser("~/Documents/MyArticles/dataset")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
N_BOOT, SEED = 5000, 0
SITES = [("intersection A (tuning)", ("Aall1", "Aall2")), ("intersection B", ("Ball",))]
LARGE = [("DAIR-V2X-C", "dair_c_runs"), ("V2X-Seq-SPD", "spd_runs")]
# clips of the larger evaluation that informed a choice during development (the conditional length cap was chosen
# on A, B and these five; seq_0082 also checked the track solve): "held-out" leaves them out
DEV_CLIPS = {"clip_52_1", "clip_106_0", "clip_21_0", "seq_0089", "seq_0082"}


def paired(path, clip=None):
    """The pairs (ours, labelled box) of one result file, one per label (its longest track)."""
    best = {}
    for r in json.load(open(path)):
        if "fused closed: iou" not in r or "box label closed: iou" not in r or not r.get("label"):
            continue
        if r["label"] not in best or r["windows"] > best[r["label"]]["windows"]:
            best[r["label"]] = r
    return [{"ours": r["fused closed: iou"], "box": r["box label closed: iou"], "n_side": r.get("n_side"),
             "clip": clip} for r in best.values()]


def site_rows(names):
    rows = []
    for n in names:
        p = os.path.join(DOCS, f"whole_scene_cars_ann_{n}_full.json")
        if os.path.exists(p):
            rows += paired(p)
    return rows


def large_rows(runs_dir):
    rows = []
    for c in json.load(open(os.path.join(runs_dir, "chosen.json"))):
        p = os.path.join(DOCS, f"whole_scene_cars_ann_P_{c['clip']}_all.json")
        if os.path.exists(p):
            rows += paired(p, c["clip"])
    return rows


def boot(rows):
    """mean paired difference ours - box and its 95 % percentile bootstrap interval."""
    d = np.array([r["ours"] - r["box"] for r in rows])
    rng = np.random.default_rng(SEED)
    means = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return len(d), float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


GROUPS = ("all", "measured", "completed", "held-out")


def grouped(rows):
    """the rows in four groups: all together, measured n_side >= 2, completed from the cuboid n_side < 2, and
    held-out: the larger evaluations' clips that informed no choice (rows of A and B carry no clip)"""
    return {"all": rows, "measured": [r for r in rows if (r["n_side"] or 0) >= 2],
            "completed": [r for r in rows if (r["n_side"] or 0) < 2],
            "held-out": [r for r in rows if r["clip"] and r["clip"] not in DEV_CLIPS]}


def main():
    datasets = [(name, site_rows(names)) for name, names in SITES]
    datasets += [(name, large_rows(os.path.join(RUNS, d))) for name, d in LARGE]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Paired difference of the mean solid IoU (ours $-$ labelled box) against the gap-closed LiDAR,"
             r" per car, with a 95\,\% bootstrap confidence interval (5000 resamples, fixed seed). \emph{measured}:"
             r" the car's length was seen side-on ($\ge 2$ side-on windows); \emph{completed}: never seen side-on,"
             r" length and profile completed from its cuboid; \emph{held-out}: without the five clips that informed a"
             r" choice during development. Intersection A is the tuning set of the method; B informed three choices"
             r" (Sect.~\ref{sec:protocol}).}",
             r"\label{tab:paired}", r"\small", r"\begin{tabular}{llrcl}", r"\toprule",
             r"Dataset & length & cars & mean $\Delta$ IoU & 95\,\% CI \\", r"\midrule"]
    for i, (name, rows) in enumerate(datasets):
        g = grouped(rows)
        first = True
        for key in GROUPS:
            if not g[key]:
                continue
            n, mean, lo, hi = boot(g[key])
            ci = f"[{lo:+.3f}, {hi:+.3f}]" if n >= 2 else "--"
            lines.append(" & ".join([name if first else "", key, str(n), f"{mean:+.3f}", ci]) + r" \\")
            first = False
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
    main()
