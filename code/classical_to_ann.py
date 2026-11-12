"""The classical input's output (RoadsideCarShape/classical_input: masks/<fid>.png, cuboids/<fid>.json) as a precise
annotation (docs/precise_ann_NAME), so make_dumps.py and whole_scene_cars.py run on it unchanged: the end-to-end run
without labels, LiDAR or a learned model.

    V2X_SEQ_DIR=<seq> python classical_to_ann.py <classical OUT_DIR> NAME

Every cuboid counts as measured (source "fitted": the size is the track's own estimate); "visible" is the mask's area
over the whole projected cuboid; the track ids are "c<id>" (no label ids: fuse with MATCH_TID=0).
"""
import json
import os
import sys

import cv2
import numpy as np

import ego_sequence as es
import v2x_seq_demo as vd
import v2x_seq_shape as vs


def corners_of(center, yaw, lwh):
    """The cuboid's 8 corners (as precise_annotation.corners_of)."""
    c, s = np.cos(yaw), np.sin(yaw)
    f, n = np.array([c, s]), np.array([-s, c])
    L, W, H = lwh
    return np.array([[*(center[:2] + dx * f + dy * n), center[2] + dz] for dx in (-L / 2, L / 2)
                     for dy in (-W / 2, W / 2) for dz in (-H / 2, H / 2)])


def convert_frame(objs, mask, cam, corners_of, project):
    """One frame of the classical input (cuboids + instance mask) as precise-annotation records."""
    result = []
    for i, e in enumerate(objs):
        center = [float(e["centre_virtuallidar"][0]), float(e["centre_virtuallidar"][1]), float(e["lwh"][2]) / 2.0]
        uv, z = project(cam, corners_of(np.asarray(center, float), float(e["yaw"]), e["lwh"]))
        if (z <= 0.3).any():
            corners_uv, visible = None, 0.0
        else:
            poly = cv2.convexHull(uv.round().astype(np.int32))
            area = float(cv2.contourArea(poly))
            visible = float((mask == i + 1).sum()) / max(area, 1.0)
            corners_uv = uv.tolist()
        result.append({"track": "c%04d" % int(e["track"]), "type": e["type"], "source": "fitted", "n_points": 0,
                       "center": center, "yaw": float(e["yaw"]), "lwh": [float(v) for v in e["lwh"]],
                       "box2d": [float(v) for v in e["box2d"]], "corners_uv": corners_uv, "mask_qc": 0.0,
                       "mask_source": "classical", "visible": round(visible, 4), "from_track": e["from_track"]})
    return result


def main():
    src, name = sys.argv[1], sys.argv[2]
    dst = os.path.join("docs", f"precise_ann_{name}")
    for d in ("masks", "cuboids"):
        os.makedirs(os.path.join(dst, d), exist_ok=True)
    seq, _ = es.open_sequence()
    n, vis = 0, []
    for fn in sorted(os.listdir(os.path.join(src, "cuboids"))):
        fid = fn[:-5]
        m = cv2.imread(os.path.join(src, "masks", fid + ".png"), cv2.IMREAD_UNCHANGED)
        if fid not in seq.fids or m is None:
            continue
        k = seq.fids.index(fid)
        img = cv2.imread(os.path.join(vd.DATA, "image", fid + ".jpg"))
        if img is not None and m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        E = convert_frame(json.load(open(os.path.join(src, "cuboids", fn))), m, seq.cams[k], corners_of, vs.project)
        cv2.imwrite(os.path.join(dst, "masks", fid + ".png"), m)
        json.dump(E, open(os.path.join(dst, "cuboids", fn), "w"), indent=1)
        n += 1
        vis += [e["visible"] for e in E if e["type"] in ("Car", "Van")]
    print(f"{n} frames -> {dst}; cars and vans: {len(vis)} object-frames, visible median {np.median(vis):.2f}, "
          f">= 0.55: {np.mean(np.array(vis) >= 0.55):.2f}")


if __name__ == "__main__":
    main()
