"""Redo the car stage of every chosen clip (whole_scene_cars.py on its existing window dumps) after a change that only
affects the models' appearance, e.g. CAR_COLOUR=best. 3 clips at a time; results replace <clip>/cars.json."""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from run_subset import CODE, HERE, SPD


def redo(c):
    name = c["clip"]
    env = dict(os.environ, V2X_SEQ_DIR=os.path.join(SPD, name), DUMPS=os.path.join("docs", "dairc_objects", name),
               VIDEO="0", OUT_SUFFIX=f"_dairc_{name}")
    with open(os.path.join(HERE, name, "cars.log"), "w") as log:
        r = subprocess.run(["python", "whole_scene_cars.py"], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
    src = os.path.join(CODE, "docs", f"whole_scene_cars_dairc_{name}.json")
    if r.returncode == 0 and os.path.exists(src):
        os.replace(src, os.path.join(HERE, name, "cars.json"))
        return name, "ok"
    return name, f"failed (exit {r.returncode})"


if __name__ == "__main__":
    chosen = json.load(open(os.path.join(HERE, "chosen.json")))
    with ThreadPoolExecutor(3) as ex:
        for name, status in ex.map(redo, chosen):
            print(name, status, flush=True)
