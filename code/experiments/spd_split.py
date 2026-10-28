"""V2X-Seq-SPD (full, infrastructure side) -> one folder per sequence in the layout the 3dseg scripts read
(data_info.json of that sequence + links to the shared image, velodyne, label, calib folders), and clips.json.

    python spd_split.py        spd_full/V2X-Seq-SPD/infrastructure-side -> spd_seq/seq_<id>/
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "spd_full", "V2X-Seq-SPD", "infrastructure-side")
DST = os.path.join(HERE, "spd_seq")
EXCLUDE = ("0058",)      # V2X-Seq-SPD-Example: intersection A of the article


def split_sequences(src, dst, exclude=()):
    """Every sequence of src/data_info.json (minus `exclude`) as dst/seq_<id>/ with its own data_info.json (the
    original frame records, sorted by frame_id) and relative links to src's image, velodyne, label, calib; writes and
    returns dst/clips.json = [{clip, intersection, frames, with_image}] sorted by clip."""
    os.makedirs(dst, exist_ok=True)
    seqs = defaultdict(list)
    for fr in json.load(open(os.path.join(src, "data_info.json"))):
        if fr["sequence_id"] not in exclude:
            seqs[fr["sequence_id"]].append(fr)
    clips = []
    for sid, frames in sorted(seqs.items()):
        frames.sort(key=lambda f: f["frame_id"])
        d = os.path.join(dst, f"seq_{sid}")
        os.makedirs(d, exist_ok=True)
        json.dump(frames, open(os.path.join(d, "data_info.json"), "w"), indent=1)
        for sub in ("image", "velodyne", "label", "calib"):
            link = os.path.join(d, sub)
            if not os.path.lexists(link):
                os.symlink(os.path.relpath(os.path.join(src, sub), d), link)
        clips.append({"clip": f"seq_{sid}", "intersection": frames[0]["intersection_loc"], "frames": len(frames),
                      "with_image": sum(os.path.exists(os.path.join(src, f["image_path"])) for f in frames)})
    json.dump(clips, open(os.path.join(dst, "clips.json"), "w"), indent=1)
    return clips


if __name__ == "__main__":
    os.makedirs(os.path.join(SRC, "velodyne"), exist_ok=True)          # filled when the split LiDAR archive is joined
    c = split_sequences(SRC, DST, EXCLUDE)
    print(len(c), "sequences,", sum(x["frames"] for x in c), "frames,", sum(x["with_image"] for x in c), "with an image")
