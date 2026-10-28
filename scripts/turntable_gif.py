"""A rotating car model as an animated GIF (for the repository README).

    python scripts/turntable_gif.py MESHES_JSON TRACK [OUT.gif]

MESHES_JSON is a car-viewer export of 3dseg/whole_scene_cars.py (docs/cars_viewer_<run>/meshes.json); TRACK its key
without "track " (e.g. 5). The model keeps the original video colours of its vertices; the camera circles it once.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cars_figure as cf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def save_gif(frames, path, fps=15):
    """Save BGR frames as a looping GIF with one shared palette (built from 8 evenly spaced frames, so the colours do
    not flicker), no dithering. Returns the file size in bytes."""
    rgb = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]
    idx = np.linspace(0, len(rgb) - 1, min(8, len(rgb))).astype(int)
    palette = Image.fromarray(np.hstack([rgb[i] for i in idx])).quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in rgb]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True,
                 disposal=2)
    return os.path.getsize(path)


def frames_of(m, n=48, size=(560, 360), dist=9.5, height=4.6):
    V = np.array(m["v"]).reshape(-1, 3)
    F = np.array(m["f"]).reshape(-1, 3)
    C = np.array(m["c"]).reshape(-1, 3)[:, ::-1] * 255                     # RGB 0-1 -> BGR 0-255
    V = V - [(V[:, 0].max() + V[:, 0].min()) / 2, (V[:, 1].max() + V[:, 1].min()) / 2, 0]
    w, h = size
    K = np.array([[1.25 * w, 0, w / 2], [0, 1.25 * w, h / 2 + 10], [0, 0, 1]])
    target = np.array([0.0, 0.0, 0.75])
    out = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        img = np.full((h, w, 3), 255, np.uint8)
        eye = np.array([dist * np.cos(a), dist * np.sin(a), height])
        cam = cf.look_at(K, eye, target)
        # a soft ground shadow under the car
        g = np.array([[x, y, 0.0] for x in np.linspace(-2.6, 2.6, 27) for y in np.linspace(-1.2, 1.2, 13)])
        q, _ = cf.vs.project(cam, g)
        hull = cv2.convexHull(np.round(q).astype(np.int32))
        shadow = img.copy()
        cv2.fillConvexPoly(shadow, hull, (225, 225, 225), cv2.LINE_AA)
        img = cv2.addWeighted(shadow, 0.7, img, 0.3, 0)
        out.append(cf.render_mesh(img, cam, V, F, C))
    return out


def main():
    meshes, track = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "figures", "turntable.gif")
    m = json.load(open(meshes))[f"track {track}"]
    size = save_gif(frames_of(m), out)
    r = m["row"]
    print(f"{out}: {size / 1e6:.2f} MB; {r.get('body_type')} IoU {r.get('fused closed: iou')}, "
          f"L {r.get('length')}/{r.get('label_length')} m")


if __name__ == "__main__":
    main()
