"""Run the car fusion of intersection A (the article's configuration) and keep, per car, what the figures need: the
model mesh with its video colours, the car's pose in every window, the body pixels of every window, the LiDAR top
and the scores. Writes figures/models_A.pkl (large; not part of the article)."""
import os
import pickle
import sys

os.environ.setdefault("DUMPS", "docs/whole_scene_objects_v2")
os.environ.setdefault("VIDEO", "0")
sys.path.insert(0, os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
os.chdir(os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
import whole_scene_cars as w

OUT = os.environ.get("MODELS_PKL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures",
                                                   "models_A.pkl")
for fn in ("types_figure", "gallery", "triangulation_figure", "scene_video"):
    setattr(w, fn, lambda *a, **k: None)


def keep(dirname, models):
    out = {}
    for n, (f, row) in models.items():
        if f.get("painted") is None:
            continue
        V, F, _ = f["painted"]
        out[n] = {"V": V, "F": F, "vcol": f.get("vertex_colours"), "poses": f["poses"], "row": row,
                  "windows": [(k, d["uv"][d["body"]]) for k, d in f["track"]], "lidar": f.get("lidar"),
                  "H": f["H"], "xs": f["xs"], "ys": f["ys"]}
    pickle.dump(out, open(OUT, "wb"))
    print("kept", len(out), "models ->", OUT)


w.export = keep
w.OUT_SUFFIX = "_figcapture"
w.main()
