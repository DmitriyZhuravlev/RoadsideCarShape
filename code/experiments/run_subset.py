"""DAIR-V2X-C infrastructure clips through the 3dseg pipeline (single pass, default ride height).

    python run_subset.py [N_CLIPS] [JOBS] [PER_SITE]     default 20 clips, 3 at a time, at most 3 per intersection
    SPD=<folder with clips.json> RUNS=<results folder> TAG=<name> ...     another dataset (default DAIR-V2X-C)

Clips: the longest (frames with an image) of every intersection, round-robin over the intersections, at most 3 per
intersection. Per clip: windows at every second frame that has an image (WIN frames in from both ends), then the car
models. Resumable: a clip whose car results exist is skipped. Results: <clip>/cars.json, logs in <clip>/.
"""
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

SPD = os.environ.get("SPD", "/home/dzhura/Documents/MyArticles/dataset/dair_c_spd")     # clips (clips.json)
CODE = os.path.expanduser("~/Documents/MyArticles/code/3dseg")
HERE = os.environ.get("RUNS", os.path.dirname(os.path.abspath(__file__)))                # results per clip
TAG = os.environ.get("TAG", "dairc")                                                      # names of dumps/outputs
WIN = 8


def pick_clips(clips, per_site=3, total=20):
    """Clips spread over intersections: the longest per intersection first, round-robin."""
    groups = defaultdict(list)
    for c in clips:
        groups[c["intersection"]].append(c)
    for g in groups.values():
        g.sort(key=lambda c: -c["with_image"])
    order = sorted(groups, key=lambda s: -groups[s][0]["with_image"])
    out, rnd = [], 0
    # rounds of one clip per intersection: per_site rounds, then more only while fewer than `total` are chosen
    while len(out) < total and any(rnd < len(groups[s]) for s in order):
        for s in order:
            if rnd < len(groups[s]) and len(out) < total:
                out.append(groups[s][rnd])
        rnd += 1
    return out


def windows(clip_dir):
    """Every second frame that has an image, at least WIN frames from both ends of the clip."""
    info = sorted(json.load(open(os.path.join(clip_dir, "data_info.json"))), key=lambda d: d["frame_id"])
    have = [os.path.exists(os.path.join(clip_dir, d["image_path"])) for d in info]
    out, last = [], -10
    for k in range(WIN, len(info) - WIN):
        if have[k] and k - last >= 2:
            out.append(k)
            last = k
    return out


def run_clip(c):
    name = c["clip"]
    data = os.path.join(SPD, name)
    work = os.path.join(HERE, name)
    os.makedirs(work, exist_ok=True)
    cars = os.path.join(work, "cars.json")
    if os.path.exists(cars):
        return name, "done before"
    dumps = os.path.join("docs", f"{TAG}_objects", name)          # relative to the 3dseg folder (vd.OUT_DIR = docs)
    ks = windows(data)
    env = dict(os.environ, V2X_SEQ_DIR=data, SEG="mask", MIRROR="0", TYRES="0", FIGURES="0", DUMP="1",
               DUMP_DIR=os.path.join(f"{TAG}_objects", name), SUFFIX=f"_{TAG}_{name}")
    have = len(os.listdir(os.path.join(CODE, dumps))) if os.path.isdir(os.path.join(CODE, dumps)) else 0
    if have < len(ks):
        todo = [k for k in ks if not os.path.exists(os.path.join(CODE, dumps, f"{k:04d}.pkl"))]
        with open(os.path.join(work, "windows.log"), "a") as log:
            subprocess.run(["python", "whole_scene.py", *map(str, todo)], cwd=CODE, env=env, stdout=log,
                           stderr=subprocess.STDOUT)
    env = dict(os.environ, V2X_SEQ_DIR=data, DUMPS=dumps, VIDEO="0", OUT_SUFFIX=f"_{TAG}_{name}")
    with open(os.path.join(work, "cars.log"), "w") as log:
        r = subprocess.run(["python", "whole_scene_cars.py"], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
    src = os.path.join(CODE, "docs", f"whole_scene_cars_{TAG}_{name}.json")
    if r.returncode == 0 and os.path.exists(src):
        os.replace(src, cars)
        return name, f"{len(ks)} windows, cars done"
    return name, f"cars failed (exit {r.returncode}), see {work}/cars.log"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    per_site = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    chosen = pick_clips(json.load(open(os.path.join(SPD, "clips.json"))), per_site=per_site, total=n)
    os.makedirs(HERE, exist_ok=True)
    json.dump(chosen, open(os.path.join(HERE, "chosen.json"), "w"), indent=1)
    print(len(chosen), "clips,", sum(c["with_image"] for c in chosen), "frames with an image:",
          ", ".join(f"{c['clip']} ({c['intersection']})" for c in chosen), flush=True)
    with ThreadPoolExecutor(jobs) as ex:
        for name, status in ex.map(run_clip, chosen):
            print(name, status, flush=True)


if __name__ == "__main__":
    main()
