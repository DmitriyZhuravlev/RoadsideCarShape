"""The article's result tables: car models reconstructed from the classical input (masks and cuboids from the video
alone, 3dseg classical_input) scored against the cars' LiDAR. Reads 3dseg docs/whole_scene_cars_ann_<run>_<tag>.json
(improve_runs/cls_fusion.sh; tag fb_noba: convex-hull silhouettes, model confined to its input cuboid, windows\naligned without a joint solve of the track) and writes
tables_cls/main_table.tex, clips_table.tex, ablation.tex and numbers.tex (macros \\Cls<set>...).

Every car counts once: DAIR-V2X-C clips 105_0 and 52_1 show the same video as V2X-Seq-SPD sequences 0059 and 0002, so
they are listed per clip as repeats and left out of the sets. A label the tracker split into two tracks keeps the
longer one."""
import json
import os

import numpy as np

DOCS = os.path.expanduser("~/Documents/MyArticles/code/3dseg/docs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables_cls")
T = "fb_noba"
FURTHER = ("c114", "s0054", "s0059", "s0077", "s0002")
SETS = [("A", "intersection A", ("Acls",)), ("B", "intersection B", ("B",)), ("F", "further clips", FURTHER),
        ("T", "all", ("Acls", "B") + FURTHER)]
CLIPS = [("intersection A", "seq.\\ 0058", "Acls"), ("intersection B", "example clip", "B"),
         ("further clips", "clip 114\\_0", "c114"), ("", "seq.\\ 0054", "s0054"),
         ("", "seq.\\ 0059", "s0059"), ("", "seq.\\ 0077", "s0077"), ("", "seq.\\ 0002", "s0002"),
         ("repeats", "clip 105\\_0 (in 0059)", "c105"), ("", "clip 52\\_1 (in 0002)", "c52")]
KEYS = [("Ours", "fused closed: iou"), ("Lab", "box label closed: iou"), ("Own", "box own closed: iou"),
        ("Input", "box input closed: iou"), ("Single", "single closed: iou"), ("OursRaw", "fused: iou"),
        ("LabRaw", "box label: iou"), ("Scanned", "fused closed: scanned_iou"),
        ("LabScanned", "box label closed: scanned_iou"), ("FieldErr", "fused closed: height_err_cm"),
        ("LabFieldErr", "box label closed: height_err_cm"), ("InputFieldErr", "box input closed: height_err_cm")]
TABLE = ("Ours", "Lab", "Own", "Input", "Single")
ABLATION = [("full method", "fb_noba"), ("silhouette: the mask", "nb_mask"), ("not confined to the cuboid", "nb_unbound"),
            ("no cabin carve", "nb_nocabin"), ("no symmetry", "nb_nosym"), ("no roof edge", "nb_noroof"),
            ("no cuboid completion", "nb_noprior"), ("with a joint solve of the track", "fb")]
ABL_KEY = {"fb_noba": "Full", "nb_mask": "Mask", "nb_unbound": "Unbound", "nb_nocabin": "Nocabin", "nb_nosym": "Nosym",
           "nb_noroof": "Noroof", "nb_noprior": "Noprior", "fb": "Joint"}


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


def ablation_row(name, cells):
    out = []
    for cell in cells:
        out.append(f"{cell:.3f}" if isinstance(cell, float) else "--" if cell is None else cell)
    return f"{name} & {' & '.join(out)} \\\\"


def cars(runs, tag=T):
    """{run:label: row} of the scored cars of the runs."""
    out = {}
    for name in runs:
        t = tag or ("e2e" if name == "Acls" else "e2e_sub2")
        best = {}
        for r in json.load(open(os.path.join(DOCS, f"whole_scene_cars_ann_{name}_{t}.json"))):
            if "fused closed: iou" in r and (r["label"] not in best or r["windows"] > best[r["label"]]["windows"]):
                best[r["label"]] = r
        out.update({f"{name}:{k}": v for k, v in best.items()})
    return out


def column(R, key):
    return np.array([r.get(key, np.nan) for r in R], float)


def main():
    os.makedirs(OUT, exist_ok=True)
    macros, body = {}, []
    for s, title, runs in SETS:
        R = list(cars(runs).values())
        col = {k: column(R, key) for k, key in KEYS}
        for k in col:
            macros[f"Cls{s}{k}"] = f"{np.nanmean(col[k]):.0f}" if "FieldErr" in k else f"{np.nanmean(col[k]):.3f}"
        macros[f"Cls{s}Cars"] = str(len(R))
        ci = {x: paired_ci(col["Ours"] - col[x]) for x in ("Lab", "Input", "Own", "Single")}
        for x in ci:
            macros[f"Cls{s}D{x}"] = fmt_ci(*ci[x][:3])
        macros[f"Cls{s}WinsLab"], macros[f"Cls{s}WinsInput"] = str(ci["Lab"][3]), str(ci["Input"][3])
        for name, a, b in (("Lerr", "length", "label_length"), ("Werr", "width", "label_width"),
                           ("Herr", "height", "label_height")):
            macros[f"Cls{s}{name}"] = f"{np.nanmean(np.abs(column(R, a) - column(R, b))):.2f}"
        macros[f"Cls{s}Heading"] = f"{np.nanmedian(column(R, 'heading_err_deg')):.1f}"
        macros[f"Cls{s}Along"] = f"{np.nanmedian(np.abs(column(R, 'pose: along-ray err m'))):.2f}"
        macros[f"Cls{s}Across"] = f"{np.nanmedian(np.abs(column(R, 'pose: across-ray err m'))):.2f}"
        macros[f"Cls{s}Side"] = str(int((column(R, "n_side") >= 2).sum()))
        macros[f"Cls{s}Filled"] = f"{np.nanmean(column(R, 'filled_share')):.2f}"
        macros[f"Cls{s}RangeMax"] = f"{np.nanmax(column(R, 'range_m')):.0f}"
        row = (f"{title} & {len(R)} & " + " & ".join(macros[f"Cls{s}{k}"] for k in TABLE)
               + f" & {fmt_ci(*ci['Lab'][:3])} & {fmt_ci(*ci['Input'][:3])} \\\\")
        body.append(("\\midrule\n" if s == "T" else "") + row)
        print(title, len(R), {k: macros[f"Cls{s}{k}"] for k in TABLE}, macros[f"Cls{s}DLab"], macros[f"Cls{s}DInput"])
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Car models from the video alone against their accumulated LiDAR: mean solid IoU against the "
             r"gap-closed reference. The labelled box, a box of our own estimated size, the input cuboid of the "
             r"classical method and the best single window are scored in the same way; the last two columns pair our "
             r"models with the labelled box and with their input cuboid on the same cars (mean difference, 95\,\% "
             r"bootstrap interval, 5000 resamples). Every car counts once.}",
             r"\label{tab:main}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lrccccccc}", r"\toprule",
             r" & cars & ours & labelled & box of & input & single & ours $-$ labelled & ours $-$ input \\",
             r" & & & box & our size & cuboid & window & & \\", r"\midrule", *body,
             r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(OUT, "main_table.tex"), "w").write("\n".join(lines) + "\n")

    rows = []
    for group, clip, run in CLIPS:
        R = list(cars((run,)).values())
        m = lambda key: f"{np.nanmean(column(R, key)):.3f}"
        rows.append(("\\midrule\n" if group == "repeats" else "")
                    + f"{group} & {clip} & {len(R)} & {m('fused: iou')} & {m('box label: iou')} & "
                    f"{m('fused closed: iou')} & {m('box label closed: iou')} & {m('box input closed: iou')} & "
                    f"{np.nanmedian(np.abs(column(R, 'pose: along-ray err m'))):.2f} \\\\")
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{The same evaluation per clip: mean solid IoU of our models and of the labelled boxes against "
             r"the raw and the gap-closed LiDAR, of the input cuboids against the gap-closed one, and the median error "
             r"of the car's centre along the viewing ray. The two repeats are DAIR-V2X-C clips whose video is part of "
             r"a V2X-Seq-SPD sequence above, with the labels of the other release; they are in no total. Sequences (seq.) are of V2X-Seq-SPD, clips of DAIR-V2X-C.}",
             r"\label{tab:clips}", r"\footnotesize", r"\begin{tabular}{llrcccccc}", r"\toprule",
             r" & & & \multicolumn{2}{c}{raw LiDAR} & \multicolumn{3}{c}{closed LiDAR} & along \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r" & clip & cars & ours & box & ours & box & input & ray (m) \\", r"\midrule", *rows,
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "clips_table.tex"), "w").write("\n".join(lines) + "\n")

    # ablation: mean IoU per set and the paired difference to the full method on the cars scored in both
    full = {s: cars(runs) for s, _, runs in SETS}
    rows = []
    for name, tag in ABLATION:
        cells = []
        for s, _, runs in SETS:
            C = cars(runs, tag)
            cells.append(float(np.mean([r["fused closed: iou"] for r in C.values()])))
            if s == "T":
                common = sorted(set(C) & set(full[s]))
                d = paired_ci([C[k]["fused closed: iou"] - full[s][k]["fused closed: iou"] for k in common])
                cells.append("--" if tag == T else fmt_ci(*d[:3]))
                key = ABL_KEY[tag]
                macros[f"ClsAbl{key}"] = f"{cells[-2]:.3f}"
                macros[f"ClsAblD{key}"] = cells[-1]
        rows.append(ablation_row(name, cells))
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Ablation on the classical input: one part of the method changed or left out at a time, on the "
             r"same windows; mean solid IoU against the gap-closed LiDAR, and the paired difference to the full method "
             r"over all cars scored in both (95\,\% bootstrap interval).}",
             r"\label{tab:ablation}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lccccc}",
             r"\toprule", r" & A & B & further & all & difference to full \\", r"\midrule", *rows,
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "ablation.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(rows))
    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        f.write("% generated by make_tables_cls.py - do not edit\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    main()
