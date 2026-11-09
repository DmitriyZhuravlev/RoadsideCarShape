"""Moving-object instance masks by the classical pipeline of the research code (code/whole_scene.mask_labels with
SEG=mask, code/colour_flow_split.py, code/v2x_seq_classical.py, code/ipm_track_models.py), at full resolution:

  1. foreground against the video's median frame (colour distance), morphology;
  2. the wet-road reflection cut at the contact line (the sharp underside against the blurred mirror image);
  3. the blobs split by their optical flow (k-means, 1-3 motions) with votes of colour regions, queues split by colour;
  4. touching parts with one affine motion merged again;
  5. the cast shadow removed (HSV test, Cucchiara et al. 2003), moving regions kept.

classical_input lifts these instances by the tangent construction and splits the cars they merge (ofc.detect_instances).
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import colour_flow_split as cfs                                   # noqa: E402
import ipm_track_models as itm                                    # noqa: E402
import whole_scene as ws                                          # noqa: E402


def median_background(paths):
    """The video's median frame at full resolution (every second frame, as the research code)."""
    return np.median(np.stack([cv2.imread(p) for p in paths[::2]]), axis=0).astype(np.uint8)


def instance_masks(prev_bgr, bgr, bg):
    """The moving objects' masks of one frame (full resolution, bool)."""
    H, W = bgr.shape[:2]
    F = cfs.flows([prev_bgr, bgr])[1]
    Ff = cfs.full_res(F, (H, W))
    out = []
    for m in ws.merge_parts(cfs.detect(bgr, bg, F), Ff, (H, W)):
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        X0, Y0, X1, Y1 = max(0, x0 - 4), max(0, y0 - 4), min(W, x1 + 4), min(H, y1 + 4)
        sub = itm.shadow_free(bgr[Y0:Y1, X0:X1], bg[Y0:Y1, X0:X1], m[Y0:Y1, X0:X1])
        mm = np.zeros((H, W), bool)
        mm[Y0:Y1, X0:X1] = sub
        if mm.sum() < cfs.MIN_AREA or np.nanmean(np.linalg.norm(Ff[mm], axis=1)) < ws.SV_MOVE:
            continue
        out.append(mm)
    return out
