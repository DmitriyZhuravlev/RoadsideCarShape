"""Foreground masks of a fixed camera's video by classical background subtraction (no learned component).

    method    background model                                                     source
    mean      the mean of the whole video, colour difference                       here (ofc.foreground)
    median    the per-pixel median of the whole video, colour difference           here (ofc.foreground)
    gsoc      GSOC: per pixel a set of samples of colour and local binary          OpenCV contrib (bgsegm)
              similarity patterns, thresholds adapted by feedback (the ideas of
              SuBSENSE, St-Charles et al., IEEE TIP 2015)
    lsbp      LSBP (Guo et al., 2016): local SVD binary patterns, sample-based     OpenCV contrib (bgsegm)

SuBSENSE and PAWCS themselves are in BGSLibrary (pybgs), which does not build against OpenCV 5 here. The online models
(gsoc, lsbp) see the whole video twice, every frame in order: the first pass builds the model, the second gives the
masks, so that the first frames of a clip do not meet a model that is still learning. All masks at half resolution
(ofc.SCALE), cleaned by a morphological opening and closing.
"""
import cv2
import numpy as np

import ofc


def clean(m):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


def subtractor(method):
    return {"gsoc": cv2.bgsegm.createBackgroundSubtractorGSOC,
            "lsbp": cv2.bgsegm.createBackgroundSubtractorLSBP}[method]()


def masks(method, paths, keep):
    """Foreground masks (half resolution, bool) of the frames whose indices are in keep, and the background image
    (half resolution) of the model; paths: all frames of the video in order."""
    frames = [ofc.half(cv2.imread(p)) for p in paths]
    if method in ("mean", "median"):
        stack = np.stack(frames)
        bg = (stack.mean(0) if method == "mean" else np.median(stack, axis=0)).round().astype(np.uint8)
        return {k: ofc.foreground(frames[k], bg) for k in keep}, bg
    cv2.setRNGSeed(0)                                    # the sample-based models draw random samples: reproducible
    sub = subtractor(method)
    out = {}
    for rnd in range(2):
        for k, f in enumerate(frames):
            m = sub.apply(f)
            if rnd == 1 and k in keep:
                out[k] = clean(m > 127)
    return out, sub.getBackgroundImage()
