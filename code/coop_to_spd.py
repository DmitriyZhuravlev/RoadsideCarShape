"""DAIR-V2X-C infrastructure side -> the V2X-Seq-SPD layout the pipeline reads (evaluation labels get track ids by
linking the labelled 3D boxes frame to frame; the reconstruction never reads them).

    python coop_to_spd.py SRC DST            one sequence of all frames (the example: one short clip)
    python coop_to_spd.py SRC DST --clips    one sequence per clip (batch_id) of the full set, split where two listed
                                             frames are more than MAX_STEP apart; clips shorter than MIN_FRAMES dropped

A frame whose image is missing stays in the sequence (an empty 0.1 s slot; the pipeline skips it).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_STEP = 0.25                 # s: a longer break between two listed frames starts a new sequence
MIN_FRAMES = 40


def source_dir(src, name):
    """src/name, or the unpacked archive folder that ends with -name (e.g. ...-infrastructure-side-velodyne)."""
    if os.path.isdir(os.path.join(src, name)):
        return os.path.join(src, name)
    cand = [p for p in glob.glob(os.path.join(src, f"*-{name}")) if os.path.isdir(p)]
    return cand[0] if cand else os.path.join(src, name)


def convert(src, dst, info, seq_id):
    """Write one sequence: symlinked data, data_info and the labels with linked track ids. Returns the track count."""
    os.makedirs(os.path.join(dst, "label", "camera"), exist_ok=True)
    for d in ("image", "velodyne", "calib"):
        if not os.path.exists(os.path.join(dst, d)):
            os.symlink(source_dir(src, d), os.path.join(dst, d))
    next_id, prev = 0, []          # prev: list of (track_id, xy, type, frames unseen)
    for d in info:
        fid = os.path.splitext(os.path.basename(d["image_path"]))[0]
        d["frame_id"] = fid
        d["sequence_id"] = seq_id
        labs = json.load(open(os.path.join(src, d["label_lidar_std_path"])))
        for o in labs:
            o["3d_location"] = {k: float(v) for k, v in o["3d_location"].items()}
            o["3d_dimensions"] = {k: float(v) for k, v in o["3d_dimensions"].items()}
            o["2d_box"] = {k: float(v) for k, v in o["2d_box"].items()}
            o["truncated_state"] = int(o["truncated_state"])
            o["rotation"] = float(o["rotation"])
        xy = np.array([[o["3d_location"]["x"], o["3d_location"]["y"]] for o in labs]).reshape(-1, 2)
        cost = np.full((len(prev), len(labs)), 1e6)
        for i, (tid, p, ty, age) in enumerate(prev):
            for j, o in enumerate(labs):
                c = np.linalg.norm(p - xy[j])
                if c < 2.0 + 1.0 * age and o["type"] == ty:
                    cost[i, j] = c
        ids = [None] * len(labs)
        if len(prev) and len(labs):
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e6:
                    ids[j] = prev[i][0]
        for j in range(len(labs)):
            if ids[j] is None:
                ids[j] = f"{next_id:06d}"
                next_id += 1
            labs[j]["track_id"] = ids[j]
        seen = set(ids)
        prev = [(ids[j], xy[j], labs[j]["type"], 0) for j in range(len(labs))] + \
               [(t, p, ty, a + 1) for t, p, ty, a in prev if t not in seen and a < 5]
        json.dump(labs, open(os.path.join(dst, "label", "camera", f"{fid}.json"), "w"))
    json.dump(info, open(os.path.join(dst, "data_info.json"), "w"), indent=1)
    return next_id


def clips(info, max_step=MAX_STEP):
    """Split frames into clips: per batch_id, in time order, a new clip starts after a break longer than max_step s."""
    by = defaultdict(list)
    for d in info:
        by[d["batch_id"]].append(d)
    out = []
    for b in sorted(by, key=int):
        v = sorted(by[b], key=lambda d: int(d["image_timestamp"]))
        n, start = 0, 0
        for i in range(1, len(v) + 1):
            if i == len(v) or (int(v[i]["image_timestamp"]) - int(v[i - 1]["image_timestamp"])) / 1e6 > max_step:
                out.append((f"{b}_{n}", v[start:i]))
                n, start = n + 1, i
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    info = sorted(json.load(open(os.path.join(src, "data_info.json"))), key=lambda d: int(d["image_timestamp"]))
    if "--clips" not in sys.argv:
        n = convert(src, dst, info, "coop0")
        print(len(info), "frames,", n, "label tracks")
        return
    image_dir = source_dir(src, "image")
    index = []
    for name, frames in clips(info):
        if len(frames) < MIN_FRAMES:
            continue
        has = sum(os.path.exists(os.path.join(image_dir, os.path.basename(d["image_path"]))) for d in frames)
        n = convert(src, os.path.join(dst, f"clip_{name}"), frames, name)
        index.append({"clip": f"clip_{name}", "intersection": frames[0]["intersection_loc"], "frames": len(frames),
                      "with_image": int(has), "label_tracks": n})
    json.dump(index, open(os.path.join(dst, "clips.json"), "w"), indent=1)
    print(len(index), "clips,", sum(c["frames"] for c in index), "frames,", sum(c["with_image"] for c in index),
          "with an image")


if __name__ == "__main__":
    main()
