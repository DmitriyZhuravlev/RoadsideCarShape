"""Re-run the ablation of the article (intersections A and B) with the current 3dseg code; results replace
code/3dseg/docs/whole_scene_cars_abl2_<run>.json, which scripts/make_tables.py turns into the tables.

    python scripts/rerun_ablation.py [RUN ...]      default: all runs, 3 at a time

A: V2X-Seq-SPD example (default V2X_SEQ_DIR); B: DAIR-V2X-C converted by coop_to_spd.py (dataset/coop_spd).
All rows use the windows of the current code (docs/ba_objects/<site>, whole_scene.py with DUMP=1); row 5 is the
current method, rows 0-4 switch its later steps off one at a time.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CODE = os.path.expanduser("~/Documents/MyArticles/code/3dseg")
COOP = os.path.expanduser("~/Documents/MyArticles/dataset/coop_spd")
OLD = dict(BA="0", EDGE_DROP="0", LEN_MAX="0", CARVE_VOTE_PRIOR="0.95", BOX_PRIOR="0", BOX_SLOPES="0")
CHANGES = {"0": dict(OLD, CABIN_TAPER="0"),                                   # the previous version
           "1": dict(OLD),                                                   # + cabin taper
           "2": dict(BA="0", EDGE_DROP="0", CARVE_VOTE_PRIOR="0.95", BOX_PRIOR="0", BOX_SLOPES="0"),   # + length prior
           "3": dict(CARVE_VOTE_PRIOR="0.95", BOX_PRIOR="0", BOX_SLOPES="0"),  # + one rigid car per track
           "4": dict(BOX_PRIOR="0", BOX_SLOPES="0"),                           # + softer hull vote for prior cars
           "5": {}}                                                            # + body-type profile: the current method
SITES = {"A": dict(dumps="docs/ba_objects/A", env={}),
         "B": dict(dumps="docs/ba_objects/B", env=dict(V2X_SEQ_DIR=COOP, MATCH_GATE="5"))}


def run(name):
    site, row = name[0], name[1]
    s = SITES[site]
    env = dict(os.environ, VIDEO="0", POISSON="0", BA_CACHE="docs/ba_cache", OUT_SUFFIX=f"_abl2_{name}",
               DUMPS=s["dumps"], **s["env"], **CHANGES[row])
    with open(os.path.join(CODE, "docs", f"abl2_{name}.log"), "w") as log:
        r = subprocess.run(["python", "whole_scene_cars.py"], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
    return name, r.returncode


if __name__ == "__main__":
    names = sys.argv[1:] or [s + r for s in "AB" for r in "012345"]
    with ThreadPoolExecutor(3) as ex:
        for name, code in ex.map(run, names):
            print(name, "ok" if code == 0 else f"failed ({code})", flush=True)
