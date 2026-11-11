"""The article's result tables: car models reconstructed from the classical input (masks and cuboids from the video
alone, 3dseg classical_input) scored against the cars' LiDAR, on intersections A and B and on three clips of
DAIR-V2X-C and four sequences of V2X-Seq-SPD. Reads 3dseg docs/whole_scene_cars_ann_<run>_<tag>.json
(improve_runs/cars_chain.sh); writes tables_cls/main_table.tex, tables_cls/clips_table.tex and tables_cls/numbers.tex
(macros \\Cls<set>...). A label the tracker split into two tracks keeps the longer one."""
import json
import os

import numpy as np

DOCS = os.path.expanduser("~/Documents/MyArticles/code/3dseg/docs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables_cls")
T = "e2e_sub2"
SETS = [("A", "A", [("Acls", "e2e")]), ("B", "B", [("B", T)]),
        ("D", "DAIR-V2X-C", [(n, T) for n in ("c114", "c105", "c52")]),
        ("S", "V2X-Seq-SPD", [(n, T) for n in ("s0054", "s0059", "s0077", "s0002")])]
CLIPS = [("intersection A", "V2X-Seq-SPD 0058", ("Acls", "e2e")), ("intersection B", "DAIR-V2X-C example", ("B", T)),
         ("DAIR-V2X-C", "clip 114", ("c114", T)), ("", "clip 105", ("c105", T)), ("", "clip 52", ("c52", T)),
         ("V2X-Seq-SPD", "seq. 0054", ("s0054", T)), ("", "seq. 0059", ("s0059", T)), ("", "seq. 0077", ("s0077", T)),
         ("", "seq. 0002", ("s0002", T))]
KEYS = [("Ours", "fused closed: iou"), ("Lab", "box label closed: iou"), ("Own", "box own closed: iou"),
        ("Input", "box input closed: iou"), ("Single", "single closed: iou"), ("OursRaw", "fused: iou"),
        ("LabRaw", "box label: iou"), ("Scanned", "fused closed: scanned_iou"),
        ("LabScanned", "box label closed: scanned_iou")]
TABLE = ("Ours", "Lab", "Own", "Input", "Single")


def paired_ci(d, n_boot=5000, seed=0):
    """Mean of the paired differences d with its 95 % bootstrap interval: (mean, lo, hi, wins, n)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if not len(d):
        return np.nan, np.nan, np.nan, 0, 0
    rng = np.random.default_rng(seed)
    boot = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), int((d > 0).sum()), len(d)


def fmt_ci(mean, lo, hi):
    return f"${mean:+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$"


def cars(runs):
    out = []
    for name, tag in runs:
        best = {}
        for r in json.load(open(os.path.join(DOCS, f"whole_scene_cars_ann_{name}_{tag}.json"))):
            if "fused closed: iou" in r and (r["label"] not in best or r["windows"] > best[r["label"]]["windows"]):
                best[r["label"]] = r
        out += list(best.values())
    return out


def column(R, key):
    return np.array([r.get(key, np.nan) for r in R], float)


def main():
    os.makedirs(OUT, exist_ok=True)
    macros, body, total = {}, [], 0
    for s, title, runs in SETS:
        R = cars(runs)
        total += len(R)
        col = {k: column(R, key) for k, key in KEYS}
        for k in col:
            macros[f"Cls{s}{k}"] = f"{np.nanmean(col[k]):.3f}"
        macros[f"Cls{s}Cars"] = str(len(R))
        ci = {x: paired_ci(col["Ours"] - col[x]) for x in ("Lab", "Input", "Own", "Single")}
        for x in ci:
            macros[f"Cls{s}D{x}"] = fmt_ci(*ci[x][:3])
        macros[f"Cls{s}WinsLab"] = str(ci["Lab"][3])
        for name, a, b in (("Lerr", "length", "label_length"), ("Werr", "width", "label_width"),
                           ("Herr", "height", "label_height")):
            macros[f"Cls{s}{name}"] = f"{np.nanmean(np.abs(column(R, a) - column(R, b))):.2f}"
        macros[f"Cls{s}FieldErr"] = f"{np.nanmean(column(R, 'fused closed: height_err_cm')):.0f}"
        macros[f"Cls{s}Heading"] = f"{np.nanmedian(column(R, 'heading_err_deg')):.1f}"
        macros[f"Cls{s}Along"] = f"{np.nanmedian(np.abs(column(R, 'pose: along-ray err m'))):.2f}"
        macros[f"Cls{s}Across"] = f"{np.nanmedian(np.abs(column(R, 'pose: across-ray err m'))):.2f}"
        macros[f"Cls{s}Side"] = str(int((column(R, "n_side") >= 2).sum()))
        macros[f"Cls{s}Filled"] = f"{np.nanmean(column(R, 'filled_share')):.2f}"
        body.append(f"{title} & {len(R)} & " + " & ".join(macros[f"Cls{s}{k}"] for k in TABLE)
                    + f" & {fmt_ci(*ci['Lab'][:3])} & {fmt_ci(*ci['Input'][:3])} \\\\")
        print(title, len(R), {k: macros[f"Cls{s}{k}"] for k in TABLE}, macros[f"Cls{s}DLab"], macros[f"Cls{s}DInput"])
    macros["ClsTotalCars"] = str(total)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Car models from the video alone against their accumulated LiDAR: mean solid IoU against the "
             r"gap-closed reference, on intersections A and B, three clips of DAIR-V2X-C and four sequences of "
             r"V2X-Seq-SPD. The labelled box, a box of our own estimated size, the input cuboid of the classical "
             r"method and the best single window are scored in the same way; the last two columns pair our models "
             r"with the labelled box and with their input cuboid on the same cars (mean difference, 95\,\% bootstrap "
             r"interval, 5000 resamples).}",
             r"\label{tab:main}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lrccccccc}", r"\toprule",
             r" & cars & ours & labelled & box of & input & single & ours $-$ labelled & ours $-$ input \\",
             r" & & & box & our size & cuboid & window & & \\", r"\midrule", *body,
             r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(OUT, "main_table.tex"), "w").write("\n".join(lines) + "\n")

    rows = []
    for group, clip, run in CLIPS:
        R = cars([run])
        m = lambda key: f"{np.nanmean(column(R, key)):.3f}"
        rows.append(f"{group} & {clip} & {len(R)} & {m('fused: iou')} & {m('box label: iou')} & "
                    f"{m('fused closed: iou')} & {m('box label closed: iou')} & {m('box input closed: iou')} & "
                    f"{np.nanmedian(np.abs(column(R, 'pose: along-ray err m'))):.2f} \\\\")
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{The same evaluation per clip, no parameter changed: mean solid IoU of our models and of the "
             r"labelled boxes against the raw and the gap-closed LiDAR, of the input cuboids against the gap-closed "
             r"one, and the median error of the car's centre along the viewing ray.}",
             r"\label{tab:clips}", r"\footnotesize", r"\begin{tabular}{llrcccccc}", r"\toprule",
             r" & & & \multicolumn{2}{c}{raw LiDAR} & \multicolumn{3}{c}{closed LiDAR} & along \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r" & clip & cars & ours & box & ours & box & input & ray (m) \\", r"\midrule", *rows,
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "clips_table.tex"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        f.write("% generated by make_tables_cls.py - do not edit\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for old in ("e2e_table.tex",):
        if os.path.exists(os.path.join(OUT, old)):
            os.remove(os.path.join(OUT, old))


if __name__ == "__main__":
    main()
