"""python vru_photo.py NPZ OUT_JPG  (V2X_SEQ_DIR set): the video frame where the pedestrian/cyclist model projects
largest (middle half of its frames), cropped around the model's projection, the object framed. Label-free."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
os.chdir(os.path.expanduser("~/Documents/MyArticles/code/3dseg"))
import cv2, numpy as np
import v2x_seq_demo as vd, v2x_seq_shape as vs, vru_shape as vr
z = np.load(sys.argv[1])
fids, _, cams, _, _ = vd.load_sequence()
ks, poses, V = z["ks"], z["poses"], z["V"]
best = None
for i in range(len(ks) // 4, 3 * len(ks) // 4 + 1):
    k = int(ks[i]); x, y, yaw = poses[i]
    q, zz = vs.project(cams[k], vs.world(V, np.array([x, y]), yaw))
    q = q[zz > 0]
    if not len(q): continue
    x0, y0 = q.min(0); x1, y1 = q.max(0)
    if x0 < 5 or y0 < 5 or x1 > 1915 or y1 > 1075: continue
    if best is None or (y1 - y0) > best[0]:
        best = (y1 - y0, k, x0, y0, x1, y1)
_, k, x0, y0, x1, y1 = best
img = vr.frame(fids, k)
H, W = img.shape[:2]
cv2.rectangle(img, (int(x0) - 3, int(y0) - 3), (int(x1) + 3, int(y1) + 3), (40, 200, 255), 2, cv2.LINE_AA)
h = y1 - y0; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; half = max(1.4 * h, 90)
X0, Y0, X1, Y1 = int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half))
cv2.imwrite(sys.argv[2], img[Y0:Y1, X0:X1])
print("frame", k, "box", int(x1 - x0), "x", int(y1 - y0), "px")
