"""Re-run the ablation of the article (intersections A and B) with the current 3dseg code; results replace
code/3dseg/docs/whole_scene_cars_abl_<run>.json, which scripts/make_tables.py turns into the tables.

    python scripts/rerun_ablation.py [RUN ...]      default: all runs, 3 at a time

A: V2X-Seq-SPD example (default V2X_SEQ_DIR); B: DAIR-V2X-C converted by coop_to_spd.py (dataset/coop_spd).
Rows 0-4 use the windows of v6 (docs/whole_scene_objects_v2, docs/coop_objects_p2), rows 5-6 the dense windows (v7).
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CODE = os.path.expanduser("~/Documents/MyArticles/code/3dseg")
COOP = os.path.expanduser("~/Documents/MyArticles/dataset/coop_spd")
CHANGES = {"0": dict(CARVE_FOOT="1", DENS_REL="0.4", CARVE_ZMAX="2.4"),      # previous version
           "1": dict(CARVE_FOOT="0", DENS_REL="0.4", CARVE_ZMAX="2.4"),
           "2": dict(CARVE_FOOT="0", DENS_REL="0.25", CARVE_ZMAX="2.4"),
           "3": {},                                                            # ours (v6)
           "4": dict(LEN_MAX="4.6"),
           "5": {},                                                            # ours (v7, dense windows)
           "6": dict(LEN_MAX="4.6")}
SITES = {"A": dict(dumps=("docs/whole_scene_objects_v2", "docs/whole_scene_objects_dense"), env={}),
         "B": dict(dumps=("docs/coop_objects_p2", "docs/coop_objects_dense"),
                   env=dict(V2X_SEQ_DIR=COOP, MATCH_GATE="5"))}


def run(name):
    site, row = name[0], name[1]
    s = SITES[site]
    env = dict(os.environ, VIDEO="0", OUT_SUFFIX=f"_abl_{name}", DUMPS=s["dumps"][row in "56"], **s["env"],
               **CHANGES[row])
    with open(os.path.join(CODE, "docs", f"abl_{name}.log"), "w") as log:
        r = subprocess.run(["python", "whole_scene_cars.py"], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
    return name, r.returncode


if __name__ == "__main__":
    names = sys.argv[1:] or [s + r for s in "AB" for r in "0123456"]
    with ThreadPoolExecutor(3) as ex:
        for name, code in ex.map(run, names):
            print(name, "ok" if code == 0 else f"failed ({code})", flush=True)
