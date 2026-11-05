"""The window triangulations of intersection B for the article: rows 0, 1 and 5 of whole_scene_cars.py's
car_triangulations figure of the precise run on B (cars 000013, 000000, 000024), the first column's title shortened.
Writes figures/triangulation_B.png. Usage: python scripts/triangulation_B_figure.py [source png]"""
import glob, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Documents/MyArticles/code/3dseg/docs/car_triangulations_ann_Ball_full.png")
src = Image.open(SRC).convert("RGB")
H = src.height // 6
f = ImageFont.truetype(glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)[0], 9)
rows = []
for r, title in ((0, "car 000013, frame 11: triangulation on the frame"), (1, "car 000000, frame 7: triangulation on the frame"),
                 (5, "car 000024, frame 19: triangulation on the frame")):
    im = src.crop((0, r * H, src.width, (r + 1) * H)).copy()
    a = np.asarray(im.crop((0, 0, 352, 30)).convert("L"))     # the first column's title: its rows of dark pixels
    ys = np.nonzero((a < 200).sum(1) > 0)[0]
    y0 = max(0, ys.min() - 2) if len(ys) else 8
    d = ImageDraw.Draw(im)
    d.rectangle((0, y0, 352, y0 + 13), fill="white")
    d.text((12, y0 + 1), title, fill=(20, 20, 20), font=f)
    rows.append(im)
out = Image.new("RGB", (src.width, 3 * H), "white")
for i, im in enumerate(rows):
    out.paste(im, (0, i * H))
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "triangulation_B.png")
out.save(path)
print(path, out.size)
