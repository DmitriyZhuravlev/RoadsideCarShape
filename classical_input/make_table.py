"""The article's table of the classical input (tables/classical_input.tex) from the runs on intersections A and B.

    python make_table.py      # reads results/{A,B}/metrics.json and results/{A,B}/lidar_eval.json

The runs: FG=gsoc VIDEO=0 OUT_DIR=<dir> python run_example.py <data> 1 <frames - 1>, then
OUT_DIR=<dir> python eval_lidar.py <data> 1 <frames - 1>; their metrics.json and lidar_eval.json are kept in results/.
On B (DAIR-V2X-C, its labels lag the image by the camera-LiDAR clock offset) both are matched within 6 m: GATE=6 for
eval_lidar.py, and metrics.json from run_example.evaluate(..., gate=6.0) on the saved cuboids.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tables", "classical_input.tex")
N_BOOT, SEED = 5000, 0
CAPTION = (r"The classical input (Sect.~\ref{sec:classical}) on all frames of intersections A and B. Top: against the labelled boxes of the "
           r"moving road users (at least 1~m/s, within 80~m, not truncated), matched per frame within 3~m (on B within "
           r"6~m: its labels lag the image by the camera--LiDAR clock offset, Sect.~\ref{sec:limits}); medians. "
           r"Bottom: the classical cuboid of each matched car (the median over its frames) and its labelled box "
           r"against the car's LiDAR accumulated over the clip, scored as in Table~\ref{tab:main} by a "
           r"re-implementation of the protocol; means, and the paired difference (cuboid $-$ labelled box) with a "
           r"95\,\% bootstrap interval. These cars are every labelled car matched to a classical object in at least "
           r"three frames, with at least 100 LiDAR points: a set other than the reconstructed cars of "
           r"Table~\ref{tab:main}, on which the labelled boxes therefore score differently.")


def latex_table(cols, caption, label):
    rows = [("frames", lambda c: f"{c['frames']}"),
            ("labelled object-frames", lambda c: f"{c['labelled']}"),
            ("recall / precision", lambda c: f"{c['recall']:.2f} / {c['precision']:.2f}"),
            ("centre error (m)", lambda c: f"{c['centre']:.2f}"),
            (r"heading error (mod $180^\circ$)", lambda c: f"{c['yaw']:.1f}$^\\circ$"),
            ("2D box IoU", lambda c: f"{c['box_iou']:.2f}"),
            ("size error $L$ / $W$ / $H$ (m)", lambda c: f"{c['dl']:.2f} / {c['dw']:.2f} / {c['dh']:.2f}"),
            None,
            ("cars against the LiDAR", lambda c: f"{c['cars']}"),
            ("solid IoU raw, cuboid / box", lambda c: f"{c['ours_raw']:.3f} / {c['box_raw']:.3f}"),
            ("solid IoU closed, cuboid / box",
             lambda c: f"{c['ours_closed']:.3f} / {c['box_closed']:.3f}"),
            ("footprint IoU, cuboid / box", lambda c: f"{c['ours_fp']:.2f} / {c['box_fp']:.2f}"),
            (r"paired $\Delta$ IoU closed", lambda c: f"${c['d']:+.3f}$ $[{c['lo']:+.3f}, {c['hi']:+.3f}]$")]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcc}", r"\toprule", f" & {cols[0]['name']} & {cols[1]['name']} \\\\", r"\midrule"]
    for row in rows:
        lines.append(r"\midrule" if row is None else f"{row[0]} & {row[1](cols[0])} & {row[1](cols[1])} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n"


def column(name, site):
    m = json.load(open(os.path.join(HERE, "results", site, "metrics.json")))
    rows = json.load(open(os.path.join(HERE, "results", site, "lidar_eval.json")))
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    d = np.array([r["ours closed"] - r["box closed"] for r in rows])
    boot = d[np.random.default_rng(SEED).integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return {"name": name, "frames": m["frames"], "labelled": m["labelled moving objects (object-frames)"],
            "recall": m["recall"], "precision": m["precision"], "centre": m["median centre error m"],
            "yaw": m["median yaw error deg (mod 180)"], "box_iou": m["median 2D box IoU"], "dl": m["median |dl| m"],
            "dw": m["median |dw| m"], "dh": m["median |dh| m"], "cars": len(rows), "ours_raw": mean("ours raw"),
            "box_raw": mean("box raw"), "ours_closed": mean("ours closed"), "box_closed": mean("box closed"),
            "ours_fp": mean("ours footprint"), "box_fp": mean("box footprint"), "d": float(d.mean()),
            "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}


if __name__ == "__main__":
    cols = [column("intersection A", "A"), column("intersection B", "B")]
    open(OUT, "w").write(latex_table(cols, CAPTION, "tab:classical"))
    print(open(OUT).read())
