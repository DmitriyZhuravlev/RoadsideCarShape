"""Moving-object masks and 3D cuboids by optical-flow clustering, after

    D. Zhuravlev, Moving object 3D detection and segmentation using optical flow clustering. CSOC 2023, LNNS 724,
    pp. 426-443, Springer, 2023. https://doi.org/10.1007/978-3-031-35314-7_38

No step is learned. Per frame (Alg. 1 of the paper, with the segments of step 2 from three cues):
  1. dense optical flow F to the previous frame (Farneback; the paper allows any flow);
  2. segments. Three segmentations of the frame:
       - background: the mean of the whole video; a pixel is foreground where the frame differs from it;
       - colour: k-means quantisation of the frame's Lab colours, connected regions of one colour;
       - flow: HSV coding of the flow, direction as hue and magnitude as value at a fixed saturation, k-means
         quantisation of it (KMEANS of the paper), connected regions of one moving colour;
     a segment is a connected region that is foreground and lies wholly inside one colour segment and one moving
     flow segment (the intersection of the three); segments are sorted by area;
  3. starting from the largest segment, LIFT its 2D box to a cuboid: for every type t, a cuboid of the type's
     average size d_t, oriented along the direction of motion (the IPM of the box's bottom centre moved by the mean
     flow), is placed on the road where its projection's 2D box matches the box best; the type is the one with the
     smallest mismatch, and that mismatch is the fitting error e (see lift() for why the centre alone is fitted);
  4. neighbouring segments moving alike (direction and image speed), and not another object's body (a large segment of another
     colour beside the object), are merged one at a time and LIFT repeated: at every step
     the neighbour whose merge fits a type best; of all merges the one with the smallest error is kept as one object
     (of merges fitting about equally well, the larger), its mask and its cuboid;
  5. post-processing: an object's mask must be sufficiently convex, the footprints of two objects cannot intersect
     on the road, an object cut by the image border is skipped (its cuboid cannot fit inside its 2D box), two
     overlapping objects that move alike are merged if their union fits as well, and the mirror image of a car on the
     wet road (directly below it, moving alike, its footprint touching the car's) is dropped.
Over a clip, objects are linked into tracks by their footprint centres and each track keeps its most frequent type
(the tracking and type persistence the paper proposes).
"""
import cv2
import numpy as np

# average sizes (length, width, height) in m of the object types
TYPES = {"Car": (4.5, 1.8, 1.5), "Van": (5.0, 1.95, 1.95), "Truck": (8.0, 2.5, 3.0), "Bus": (11.0, 2.55, 3.2),
         "Cyclist": (1.7, 0.6, 1.7), "Pedestrian": (0.6, 0.6, 1.7)}
SCALE = 0.5            # flow, segmentations and masks at half resolution
K = 16                 # k-means colours of the flow (over-segmentation is allowed, merging undoes it)
K_COLOUR = 12          # k-means colours of the image
F_MAX = 6.0            # px/frame at full resolution that saturate the value channel
F_MIN = 0.6            # px/frame: a flow colour whose centre moves less is static
FG_DIFF = 25           # grey levels: a pixel differing from the mean background by more is foreground
MIN_AREA = 20          # px at half resolution: smaller segments are dropped
BODY_DE = 30.0         # Lab distance (8-bit OpenCV Lab) between two bodies of different colour
BODY_SHARE = 0.3       # a segment this large compared with an object's body could be another object's body
BODY_WIDEN = 0.25      # ... if it widens the object by more than this share of its width
SMALL = 0.02           # share of a car's image area at the segment's distance: smaller segments join a neighbour
E_ACCEPT = 0.30        # an object whose measured size deviates more from every type's average (relative RMS) is
                       # rejected
TANGENT_RMS = 0.30     # m: the six tangents disagree more (RMS): no box
PRIOR_WEIGHT = 0.1     # weight of the type's average size against the tangents (both in m)
MAX_MERGE = 40         # the segments are small now: a car is its body, windows, lights and tyres
MERGE_ANGLE = 35.0     # deg: a neighbour moving in a direction farther from the object's is not merged
SPEED_TOL = 0.05       # relative image speed two parts of one object may differ by, beyond the depth term ...
CAR_DEPTH = 4.5        # m: ... CAR_DEPTH / distance (the depth across a car makes its flow vary)
COLOUR_SAME = 20.0     # Lab distance within which a part with a different image speed is still the same body
SPLIT_SHARE = 0.15     # an object is split into two bodies only if each holds this share of its area
SPLIT_SPEED = 0.02     # margin of the two pixel-speed clusters of a split beyond CAR_DEPTH / distance
MIN_STEP = 0.03        # m/frame on the road: slower objects have no usable direction of motion
MIN_SOLIDITY = 0.6     # an object's mask must be sufficiently convex (the paper's post-processing)
BORDER = 2             # px: a box this close to the image border is cut, and the cuboid would not fit inside it
TRIALS = 8             # neighbours tried per merge step (the largest of those moving in a close direction)
E_TIE = 0.03           # of merges fitting about equally well as the same type, the larger is kept
DUPLICATE = 0.3        # two objects whose 2D boxes overlap by this share of the smaller are tried as one
TRACK_MIN = 3          # objects per track for the track's heading and size
REFLECTION_GAP = 1.0   # m: a mirror image below an object lifts to a footprint touching the object's
CONTAINED = 0.7        # share of an object's 2D box inside a larger object's box that makes it a piece of that one
MIRROR_NCC = 0.55      # the rows below a candidate contact line correlate this well with the rows above, mirrored:
                       # the mask below it is the object's mirror image on the wet road
TRACK_PATH = 1.5       # m: a track that moved less keeps its per-frame headings
TRACK_SPEED = 0.3      # m/frame: slower along its path, an object keeps its flow's direction
EXTRAP = 5             # frames a track is extrapolated beyond its ends to split a merged object
LINK_GAP = 3           # frames a track survives without an object
LINK_IOU = 0.3         # IoU of an object's 2D box with a track's box moved by its flow that links them
VISIBLE = 0.25         # a split part covering less of its track's predicted silhouette keeps the prediction
INSIDE = 0.5           # share of a track's predicted 2D box's columns inside an object's box that places it there


def half(bgr):
    return cv2.resize(bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def mean_background(paths):
    """The mean of all frames of the video at half resolution (uint8)."""
    acc = None
    for p in paths:
        img = half(cv2.imread(p)).astype(np.float32)
        acc = img if acc is None else acc + img
    return np.round(acc / len(paths)).astype(np.uint8)


def foreground(img, bg):
    """Half-resolution frame and mean background -> bool mask of the pixels that differ from the background."""
    # no shadow test: on the grey road a dark car or its tyres pass it as well, and the box loses its bottom
    d = cv2.absdiff(cv2.GaussianBlur(img, (5, 5), 0), cv2.GaussianBlur(bg, (5, 5), 0)).max(axis=2)
    m = (d > FG_DIFF).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


def colour_labels(img, k=K_COLOUR, seed=0):
    """k-means colour quantisation of the (slightly smoothed) half-resolution frame in Lab: the label image."""
    lab = cv2.cvtColor(cv2.bilateralFilter(img, 7, 30, 5), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    return cv2.kmeans(lab, k, None, crit, 1, cv2.KMEANS_PP_CENTERS)[1].reshape(img.shape[:2])


def optical_flow(prev_img, img):
    """Dense Farneback flow of two half-resolution frames, in full-resolution px/frame (H/2, W/2, 2)."""
    a, b = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (prev_img, img))
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 15, 3, 5, 1.2, 0) / SCALE


def flow_hsv(F):
    """HSV image of the flow: hue = direction, value = magnitude / F_MAX (clipped), saturation fixed."""
    mag, ang = cv2.cartToPolar(F[..., 0], F[..., 1])
    hsv = np.zeros(F.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / F_MAX * 255, 0, 255).astype(np.uint8)
    return hsv


def quantise(F, k=K, seed=0):
    """k-means colour quantisation of the HSV-coded flow. The colours are clustered as points of the HSV cone at the
    fixed saturation, (v cos h, v sin h), so that hue wraps around. Returns the label image and the mean flow of each
    colour in px/frame."""
    mag, ang = cv2.cartToPolar(F[..., 0], F[..., 1])
    v = np.clip(mag / F_MAX, 0, 1)
    X = np.stack([v * np.cos(ang), v * np.sin(ang)], -1).reshape(-1, 2).astype(np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    _, lab, _ = cv2.kmeans(X, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    lab = lab.reshape(F.shape[:2])
    mean = np.array([F[lab == i].mean(0) if (lab == i).any() else (0.0, 0.0) for i in range(k)])
    return lab, mean


def joint_segments(fg, colour_lab, flow_lab, moving, min_area):
    """Segments that are foreground and lie wholly inside one colour segment and one moving flow segment: bool masks,
    largest first."""
    valid = fg & moving[flow_lab]
    joint = colour_lab.astype(np.int64) * (flow_lab.max() + 1) + flow_lab
    masks = []
    for g in np.unique(joint[valid]):
        n, cc, st, _ = cv2.connectedComponentsWithStats((valid & (joint == g)).astype(np.uint8), connectivity=8)
        masks += [cc == i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]
    return sorted(masks, key=lambda m: -int(m.sum()))


def segment_records(masks, F, lab):
    """Per segment: its box (half resolution), cropped mask, area, mean flow and mean Lab colour (lab: the frame in
    Lab, float); and the neighbour sets (segments within 2 px of each other)."""
    ids = -np.ones(F.shape[:2], np.int32)
    segs = []
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        segs.append({"box": (x0, y0, x1, y1), "crop": m[y0:y1, x0:x1], "area": len(xs), "flow": F[m].mean(0),
                     "lab": lab[m].mean(0)})
        ids[m] = i
    H, W = ids.shape
    for i, s in enumerate(segs):
        x0, y0, x1, y1 = s["box"]
        X0, Y0, X1, Y1 = max(0, x0 - 2), max(0, y0 - 2), min(W, x1 + 2), min(H, y1 + 2)
        c = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        c[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = s["crop"]
        ring = cv2.dilate(c, np.ones((5, 5), np.uint8)) > 0
        s["near"] = set(np.unique(ids[Y0:Y1, X0:X1][ring])) - {-1, i}
    return segs


def absorb_small(cam, masks, F, lab):
    """A small segment joins its largest neighbour that moves alike (within MERGE_ANGLE): a near car breaks into a
    hundred pieces of one colour and one motion, and the small ones carry nothing of their own. Small is relative to
    the distance: below SMALL of the image area a car would have there (its bottom centre on the road)."""
    segs = segment_records(masks, F, lab)
    f = 0.5 * (cam.K[0, 0] + cam.K[1, 1]) * SCALE
    for s in segs:
        x0, y0, x1, y1 = s["box"]
        g = cam.ipm(np.array([[0.5 * (x0 + x1) / SCALE, y1 / SCALE]]))[0]
        d = np.hypot(np.linalg.norm(g - cam.ground_c), cam.height) if np.isfinite(g).all() else np.inf
        s["dist"] = d
        car = (f * TYPES["Car"][0] / d) * (f * TYPES["Car"][2] / d)
        s["small"] = s["area"] < max(MIN_AREA, SMALL * car)
    parent = list(range(len(segs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in sorted(range(len(segs)), key=lambda i: segs[i]["area"]):
        if not segs[i]["small"]:
            continue
        cand = [j for j in segs[i]["near"] if segs[j]["area"] >= segs[i]["area"] and
                angle(segs[i]["flow"], segs[j]["flow"]) <= MERGE_ANGLE]   # small pieces: noisy speed and colour
        if cand:
            parent[root(i)] = root(max(cand, key=lambda j: segs[j]["area"]))
    groups = {}
    for i, m in enumerate(masks):
        g = root(i)
        groups[g] = m if g not in groups else (groups[g] | m)
    return sorted(groups.values(), key=lambda m: -int(m.sum()))


def corners(c, heading, l, w, h=0.0):
    """The cuboid's 8 corners in the road frame: bottom face first (counter-clockwise), then the top face."""
    u = np.array([np.cos(heading), np.sin(heading)])
    v = np.array([-np.sin(heading), np.cos(heading)])
    f = np.array([c + sl * l / 2 * u + sw * w / 2 * v for sl, sw in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
    return np.r_[np.c_[f, np.zeros(4)], np.c_[f, np.full(4, h)]]


def axes(heading):
    """The box's directions in the road frame: along the motion, lateral and vertical."""
    return {"u": np.array([np.cos(heading), np.sin(heading), 0.0]),
            "v": np.array([-np.sin(heading), np.cos(heading), 0.0]), "z": np.array([0.0, 0.0, 1.0])}


def tangent_planes(cam, rays, d, occluded=None):
    """The two planes through the camera centre that contain the direction d (road frame) and touch the silhouette
    whose rays (camera frame) are given: the images of the tangents from d's vanishing point to the silhouette. Also
    the frame (dc, a, b) in which the angle about d is measured; the plane of the smallest angle comes first. A tangent
    that touches the silhouette at an occluded point (occluded: bool per ray) touches another object's edge, not the
    box, and is None."""
    dc = cam.M @ (cam.E @ d)
    dc /= np.linalg.norm(dc)
    m = rays.mean(0)
    a = m - (m @ dc) * dc
    a /= np.linalg.norm(a)
    b = np.cross(dc, a)
    ang = np.arctan2(rays @ b, rays @ a)
    return [None if occluded is not None and occluded[i] else cam.plane_g(np.cross(dc, rays[i]))
            for i in (np.argmin(ang), np.argmax(ang))], (a, b)


CORNERS = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]     # (s0|s1, r0|r1, ground|top)


def tangent_box(cam, planes, ax, p, prior, lam=PRIOR_WEIGHT, fixed=None, iters=6):
    """The box of the tangent construction (Dubska et al.): p = (s0, s1, r0, r1, h), the extents along u and v and
    the height. Every tangent plane touches one edge of the box parallel to its direction; which edge is decided by
    the angle of the current box's corners about that direction, and then the plane gives one linear equation in p.
    Six tangents, five unknowns: least squares, the edges re-assigned and solved again. prior (l, w, h) is added with
    a small weight, so that a size the view leaves undetermined takes the prior. fixed (l, w, h): only the centre is
    solved. Returns p and the RMS of the tangent equations (m)."""
    u, v, z = ax["u"], ax["v"], ax["z"]
    T = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0]], float)         # fixed size: p = T (sc, rc) + p0
    for _ in range(iters):
        rows, rhs = [], []
        X = np.array([p[i] * u + p[2 + j] * v + k * p[4] * z for i, j, k in CORNERS])
        Xc = cam.to_cam(X)
        for key, (pl, (a, b)) in planes.items():
            ang = np.arctan2(Xc @ b, Xc @ a)
            for plane, c in zip(pl, (np.argmin(ang), np.argmax(ang))):
                if plane is None:                        # an occluded side: the type's average size fills it
                    continue
                n, d = plane
                i, j, k = CORNERS[c]
                row = np.zeros(5)
                row[i], row[2 + j], row[4] = n @ u, n @ v, k * n[2]
                s = np.linalg.norm(row)
                if s > 1e-9:
                    rows.append(row / s)
                    rhs.append(-d / s)
        R, y = np.array(rows).reshape(-1, 5), np.array(rhs)
        if fixed is not None:
            l, w, h = fixed
            p0 = np.array([-l / 2, l / 2, -w / 2, w / 2, h])
            q = np.linalg.lstsq(R @ T, y - R @ p0, rcond=None)[0]
            p = T @ q + p0
        else:
            P = np.array([[-1, 1, 0, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, 0, 1]], float) * lam
            p = np.linalg.lstsq(np.r_[R, P], np.r_[y, lam * np.asarray(prior)], rcond=None)[0]
        if p[1] - p[0] < 0.1 or p[3] - p[2] < 0.1 or p[4] < 0.1:
            return None, np.inf
    return p, float(np.sqrt(np.mean((R @ p - y) ** 2))) if len(y) else np.inf


def size_deviation(lwh, t):
    return float(np.sqrt(np.mean(((np.asarray(lwh) - TYPES[t]) / TYPES[t]) ** 2)))


def lift(cam, hull, box, heading, fixed=None, centre=None, occluded=None):
    """LIFT by the tangent construction of Dubska et al. with the calibrated camera: the 3D box around the
    silhouette (its convex hull, full-resolution pixels), with the axes of the motion, the lateral and the vertical
    direction, built from the six tangents of the three vanishing points to the silhouette. The size is measured; the
    type is the one whose average size it deviates least from (relative RMS deviation = the error). With fixed size
    only the centre is solved. occluded (bool per hull point): where the silhouette borders another object; a tangent
    touching there is left out. The best type as a dict, or None."""
    ax = axes(heading)
    rays = cam.rays(hull)
    planes = {k: tangent_planes(cam, rays, d, occluded) for k, d in ax.items()}
    if centre is None:                                   # start: the bottom centre of the box, half a car behind it
        g = cam.ipm(np.array([[0.5 * (box[0] + box[2]), box[3]]]))[0]
        if not np.isfinite(g).all():
            return None
        centre = g + 1.2 * (g - cam.ground_c) / np.linalg.norm(g - cam.ground_c)
    sc, rc = centre @ ax["u"][:2], centre @ ax["v"][:2]
    best = None
    for t, d_t in ([("fixed", fixed)] if fixed is not None else TYPES.items()):
        l, w, h = d_t
        p, rms = tangent_box(cam, planes, ax, np.array([sc - l / 2, sc + l / 2, rc - w / 2, rc + w / 2, h]), d_t,
                             fixed=fixed)
        if p is None:
            continue
        lwh = (p[1] - p[0], p[3] - p[2], p[4])
        e = 0.0 if fixed is not None else size_deviation(lwh, t)
        if best is None or e < best["error"]:
            c = 0.5 * (p[0] + p[1]) * ax["u"][:2] + 0.5 * (p[2] + p[3]) * ax["v"][:2]
            best = {"type": t, "error": e, "centre": c, "yaw": heading, "lwh": tuple(map(float, lwh)),
                    "tangent_rms": rms}
    if best is None or best["tangent_rms"] > TANGENT_RMS:
        return None
    return best


def direction(cam, box, flow):
    """Direction of motion on the road: the IPM of the box's bottom centre moved by the object's mean flow."""
    p = np.array([0.5 * (box[0] + box[2]), box[3]])
    a, b = cam.ipm(np.array([p, p + flow]))
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.linalg.norm(b - a) < MIN_STEP:
        return None
    return float(np.arctan2(*(b - a)[::-1]))


def union(segs, ids, shape):
    m = np.zeros(shape, bool)
    for i in ids:
        x0, y0, x1, y1 = segs[i]["box"]
        m[y0:y1, x0:x1] |= segs[i]["crop"]
    return m


def contact_line(gray, m, x0, y0, x1, y1):
    """The row (half resolution) where the object meets the wet road: below it the image is the object mirrored
    (blurred). The row in the lower half of the box whose rows below correlate best with the rows above, mirrored,
    inside the mask; None if no row correlates MIRROR_NCC or better."""
    H = y1 - y0
    g = cv2.GaussianBlur(gray[y0:y1, x0:x1], (3, 3), 0).astype(np.float32)
    g -= g.mean(1, keepdims=True)                        # the structure across the columns, not the brightness
    c = m[y0:y1, x0:x1]
    best, row = MIRROR_NCC, None
    for r in range(H // 2, H - 3):
        k = min(max(4, H // 6), r, H - r)
        both = c[r - k:r][::-1] & c[r:r + k]
        if both.sum() < 30:
            continue
        a, b = g[r - k:r][::-1][both], g[r:r + k][both]
        a, b = a - a.mean(), b - b.mean()
        ncc = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
        if ncc > best:
            best, row = ncc, y0 + r
    return row


def fit_mask(cam, m, flow, shape, gray, occ=None, cut_mirror=True):
    """Box, direction and LIFT of a half-resolution mask m moving with the mean flow `flow`, its mirror image on the
    wet road cut off at the contact line. occ (bool, half resolution): where the silhouette borders it, it is
    occluded, and the tangents touching there are left out."""
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    if x0 < BORDER or y0 < BORDER or x1 > shape[1] - BORDER or y1 > shape[0] - BORDER:
        return None
    m = m.copy()
    cut = contact_line(gray, m, x0, y0, x1, y1) if cut_mirror else None
    if cut is not None:
        m[cut:, x0:x1] = False
        ys = np.nonzero(m[y0:cut, x0:x1].any(1))[0]
        if len(ys) == 0:
            return None
        y1 = y0 + int(ys.max()) + 1
    # convexity with the gaps between its segments closed (a segment smaller than MIN_AREA is dropped)
    ys, xs = np.nonzero(cv2.morphologyEx(m[y0:y1, x0:x1].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)))
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    if len(xs) < MIN_SOLIDITY * max(cv2.contourArea(hull), 1.0):
        return None
    hull = hull.reshape(-1, 2) + [x0, y0]
    occluded = np.zeros(len(hull), bool)
    if occ is not None:
        q = np.clip(hull.astype(int), 0, [shape[1] - 1, shape[0] - 1])
        occluded = occ[q[:, 1], q[:, 0]]
    hull = (hull + 0.5) / SCALE                                          # the silhouette in full-resolution pixels
    box = (x0 / SCALE, y0 / SCALE, x1 / SCALE, y1 / SCALE)
    yaw = direction(cam, box, flow)
    r = lift(cam, hull, box, yaw, occluded=occluded) if yaw is not None else None
    if r is not None:
        r.update(mask=m, box=box, flow=flow, hull=hull, occluded=occluded)
    return r


def fit(cam, segs, ids, shape, gray, occ=None):
    """fit_mask of the union of the segments ids."""
    flow = sum(segs[i]["flow"] * segs[i]["area"] for i in ids) / sum(segs[i]["area"] for i in ids)
    r = fit_mask(cam, union(segs, ids, shape), flow, shape, gray, occ)
    if r is not None:
        r["ids"] = list(ids)
    return r


def footprint_overlap(a, b):
    pa = corners(a["centre"], a["yaw"], *a["lwh"][:2])[:4, :2].astype(np.float32)
    pb = corners(b["centre"], b["yaw"], *b["lwh"][:2])[:4, :2].astype(np.float32)
    inter = cv2.intersectConvexConvex(pa, pb)[0]
    return inter / min(cv2.contourArea(pa), cv2.contourArea(pb))


def moves_alike(f0, f1, dist, lab0=None, lab1=None):
    """Two parts of one rigid object: their mean flows point within MERGE_ANGLE, and their image speeds agree within
    SPEED_TOL + CAR_DEPTH / dist (the flow of one object varies only with the depth across it, about its length over
    its distance; two cars side by side at 60 m differ by 15-20 %) unless their colours (lab0 of the object's body,
    lab1 of the part) agree within COLOUR_SAME: the flow at a segment's border is blended with its neighbour's, so
    the speed alone would tear pieces off a car; the speed and the colour together tell a second car."""
    if angle(f0, f1) > MERGE_ANGLE:
        return False
    a, b = np.linalg.norm(f0), np.linalg.norm(f1)
    if abs(a - b) <= (SPEED_TOL + CAR_DEPTH / max(dist, 1.0)) * max(a, b):
        return True
    return lab0 is not None and np.linalg.norm(np.asarray(lab0) - lab1) <= COLOUR_SAME


def body_lab(segs, ids):
    """The mean Lab colour of an object's body: its largest segment."""
    return segs[max(ids, key=lambda i: segs[i]["area"])]["lab"]


def union_distance(cam, segs, ids):
    """Distance (m) from the camera to the road point below the bottom centre of the segments ids."""
    x0 = min(segs[i]["box"][0] for i in ids)
    x1 = max(segs[i]["box"][2] for i in ids)
    y1 = max(segs[i]["box"][3] for i in ids)
    g = cam.ipm(np.array([[0.5 * (x0 + x1) / SCALE, y1 / SCALE]]))[0]
    return float(np.hypot(np.linalg.norm(g - cam.ground_c), cam.height)) if np.isfinite(g).all() else np.inf


def angle(f0, f1):
    c = f0 @ f1 / (np.linalg.norm(f0) * np.linalg.norm(f1) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def other_body(segs, ids, new):
    """The segments new hold another object's body: their largest segment is large (BODY_SHARE of the body of ids, its
    largest segment), clearly of another colour (Lab distance above BODY_DE) and lies beside ids (less than half of its
    width inside their columns), widens them by more than BODY_WIDEN (a car partly hidden behind another) or lies
    outside their box altogether. A car's windows, lights and tyres differ in colour from its body but lie within its
    columns and do not widen it."""
    body = max(ids, key=lambda i: segs[i]["area"])
    cand = max(new, key=lambda i: segs[i]["area"])
    if segs[cand]["area"] < BODY_SHARE * segs[body]["area"] or \
            np.linalg.norm(segs[cand]["lab"] - segs[body]["lab"]) <= BODY_DE:
        return False
    X0 = min(segs[i]["box"][0] for i in ids)
    Y0 = min(segs[i]["box"][1] for i in ids)
    X1 = max(segs[i]["box"][2] for i in ids)
    Y1 = max(segs[i]["box"][3] for i in ids)
    x0, y0, x1, y1 = segs[cand]["box"]
    inside = max(0, min(x1, X1) - max(x0, X0)) / (x1 - x0)
    wider = (max(x1, X1) - min(x0, X0)) - (X1 - X0) > BODY_WIDEN * (X1 - X0)
    apart = min(x1, X1) <= max(x0, X0) or min(y1, Y1) <= max(y0, Y0)
    return inside < 0.5 or wider or apart


def box_overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0) / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def merge_duplicates(cam, segs, objects, shape, gray):
    """Two objects whose 2D boxes overlap and that move alike are one object if their union fits a type about as
    well as the worse of the two (a near car whose pieces formed two groups); the parts of a split object are not
    merged back."""
    changed = True
    while changed:
        changed = False
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                if a.get("split") or b.get("split") or box_overlap(a["box"], b["box"]) < DUPLICATE or \
                        not moves_alike(a["flow"], b["flow"], float(np.linalg.norm(a["centre"])),
                                        body_lab(segs, a["ids"]), body_lab(segs, b["ids"])) or \
                        other_body(segs, a["ids"], b["ids"]) or other_body(segs, b["ids"], a["ids"]):
                    continue
                r = fit(cam, segs, a["ids"] + b["ids"], shape, gray)
                if r is not None and r["error"] <= min(E_ACCEPT, max(a["error"], b["error"]) + E_TIE):
                    r["segments"] = a["segments"] + b["segments"]
                    objects = [o for k, o in enumerate(objects) if k not in (i, j)] + [r]
                    changed = True
                    break
            if changed:
                break
    return objects


def footprint_gap(a, b):
    """Distance (m) between two footprints on the road, 0 if they overlap."""
    pa = corners(a["centre"], a["yaw"], *a["lwh"][:2])[:4, :2]
    pb = corners(b["centre"], b["yaw"], *b["lwh"][:2])[:4, :2]
    if footprint_overlap(a, b) > 0:
        return 0.0
    return float(min(min(np.linalg.norm(pb - q, axis=1).min() for q in pa),
                     min(np.linalg.norm(pa - q, axis=1).min() for q in pb)))


def drop_reflections(objects):
    """On a wet road a car's mirror image moves with it: an object directly below another one in the image (sharing
    most of its columns, its top in the lower part of the other or just under it), moving alike, whose footprint on
    the road touches the other's (gap < REFLECTION_GAP), is that object's reflection and is dropped."""
    keep = []
    for b in objects:
        x0, y0, x1, y1 = b["box"]
        refl = False
        for a in objects:
            if a is b:
                continue
            X0, Y0, X1, Y1 = a["box"]
            cols = max(0.0, min(x1, X1) - max(x0, X0)) / (x1 - x0)
            if cols >= 0.5 and Y0 + 0.4 * (Y1 - Y0) < y0 <= Y1 + 0.1 * (Y1 - Y0) and y1 > Y1 and \
                    angle(a["flow"], b["flow"]) <= MERGE_ANGLE and footprint_gap(a, b) < REFLECTION_GAP:
                refl = True
                break
        if not refl:
            keep.append(b)
    return keep


def drop_contained(objects):
    """An object whose 2D box lies mostly inside a larger object's box is a piece of it (a roof, a window); an
    object whose bottom edge lies inside a nearer object's box (sharing most of its columns) has no visible ground
    contact, which LIFT needs (the bottom of the box is on the road)."""
    area = lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])

    def hidden(b, a):
        x0, y0, x1, y1 = b["box"]
        X0, Y0, X1, Y1 = a["box"]
        cols = max(0.0, min(x1, X1) - max(x0, X0)) / (x1 - x0)
        return cols >= 0.5 and Y0 < y1 < Y1 - 0.1 * (Y1 - Y0)

    return [b for b in objects if not any(
        (area(a) > area(b) and box_overlap(a["box"], b["box"]) >= CONTAINED) or hidden(b, a)
        for a in objects if a is not b)]


def two_parts(m, values, shape):
    """2-means of the per-pixel values (N, d) of the mask m, the labels smoothed (median 5) and each part reduced to
    its largest connected region: (part masks, cluster centres), or None."""
    X = values.astype(np.float32).reshape(len(values), -1)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
    cv2.setRNGSeed(0)
    _, lb, c = cv2.kmeans(X, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lim = np.zeros(shape, np.uint8)
    lim[m] = 1 + lb.ravel()
    lim = cv2.medianBlur(lim, 5)
    lim[~m] = 0
    parts = []
    for v in (1, 2):
        n, cc, st, _ = cv2.connectedComponentsWithStats((lim == v).astype(np.uint8), connectivity=8)
        if n < 2:
            return None
        parts.append(cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    return parts, c


def side_by_side(p, q):
    """Two masks lie side by side: less than half of the narrower one's columns are the other's."""
    xp, xq = np.nonzero(p.any(0))[0], np.nonzero(q.any(0))[0]
    common = max(0, min(xp.max(), xq.max()) - max(xp.min(), xq.min()) + 1)
    return common < 0.5 * min(xp.max() - xp.min() + 1, xq.max() - xq.min() + 1)


def split_two_bodies(cam, obj, F, lab, shape, gray):
    """An accepted object holding two cars, by one of two partitions of its pixels (2-means, smoothed, each part its
    largest connected region, each at least SPLIT_SHARE of the object):
      - by image speed: the two speeds differ by more than one rigid object allows at its distance (SPLIT_SPEED +
        CAR_DEPTH / distance) and the two parts differ in colour (beyond COLOUR_SAME);
      - by colour (Lab): the parts differ beyond BODY_DE and lie side by side (a car's windows lie within its body's
        columns; another car's body beside it).
    Each part is then fitted on its own, the tangents along the border with the other left out (occluded). The parts
    that fit are returned; if neither does, nothing: two cars are not shown as one box. (A near car's front and back
    move at different speeds too, but within that tolerance and in one colour.)"""
    m = obj["mask"]
    if m.sum() < 4 * MIN_AREA:
        return [obj]
    dist = float(np.hypot(np.linalg.norm(obj["centre"] - cam.ground_c), cam.height))
    found = None
    r = two_parts(m, np.linalg.norm(F[m], axis=1), shape)
    if r is not None:
        parts, c = r
        lo, hi = sorted(c.ravel())
        if hi - lo > (SPLIT_SPEED + CAR_DEPTH / dist) * hi and \
                np.linalg.norm(lab[parts[0]].mean(0) - lab[parts[1]].mean(0)) > COLOUR_SAME:
            found = parts
    if found is None:
        r = two_parts(m, lab[m], shape)
        if r is not None:
            parts, c = r
            if np.linalg.norm(c[0] - c[1]) > BODY_DE and side_by_side(*parts):
                found = parts
    if found is None or min(p.sum() for p in found) < SPLIT_SHARE * m.sum():
        return [obj]
    out = []
    for p, q in ((found[0], found[1]), (found[1], found[0])):
        occ = cv2.dilate(q.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        r = fit_mask(cam, p, F[p].mean(0), shape, gray, occ, cut_mirror=False)
        if r is not None and r["error"] <= E_ACCEPT:
            r.update(ids=obj["ids"], segments=obj["segments"], split=True)
            out.append(r)
    return out                   # two cars that cannot be told apart are not shown as one box


def detect(cam, prev_bgr, bgr, bg):
    """Alg. 1 on one frame (full-resolution frames, half-resolution mean background bg): objects (mask, box, type,
    cuboid, error) and the intermediate images."""
    prev_img, img = half(prev_bgr), half(bgr)
    F = optical_flow(prev_img, img)
    flow_lab, mean = quantise(F)
    moving = np.linalg.norm(mean, axis=1) >= F_MIN
    colour_lab = colour_labels(img)
    fg = foreground(img, bg)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    masks = absorb_small(cam, joint_segments(fg, colour_lab, flow_lab, moving, MIN_AREA), F, lab)
    segs = segment_records(masks, F, lab)
    shape = F.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pool, objects = list(range(len(segs))), []
    while pool:
        ids = [pool[0]]                                  # the largest segment left
        states = [(fit(cam, segs, ids, shape, gray), list(ids))]
        inpool = set(pool)
        for _ in range(MAX_MERGE):
            f0 = sum(segs[i]["flow"] * segs[i]["area"] for i in ids)
            near = set().union(*(segs[i]["near"] for i in ids)) - set(ids)
            cand = sorted((-segs[j]["area"], j) for j in near & inpool
                          if angle(f0, segs[j]["flow"]) <= MERGE_ANGLE and not other_body(segs, ids, [j]))
            if not cand:
                break
            # greedy by the embedding error: the neighbour whose merge fits a type best (the largest if none fits)
            trials = [(fit(cam, segs, ids + [j], shape, gray), j) for _, j in cand[:TRIALS]]
            ok = [(r["error"], j, r) for r, j in trials if r is not None]
            j, r = (min(ok, key=lambda x: x[0])[1:] if ok else (cand[0][1], None))
            ids.append(j)
            states.append((r, list(ids)))
        fits = [(r, s) for r, s in states if r is not None]
        best = None
        if fits:
            r0 = min(fits, key=lambda x: x[0]["error"])[0]
            # of merges fitting about as well and as the same type, the larger (not a car grown into a bus)
            best = max(((r, s) for r, s in fits if r["error"] <= r0["error"] + E_TIE and r["type"] == r0["type"]),
                       key=lambda x: len(x[1]))
        if best is not None and best[0]["error"] <= E_ACCEPT and \
                all(footprint_overlap(best[0], o) < 0.2 for o in objects):
            best[0]["segments"] = len(best[1])
            objects += split_two_bodies(cam, best[0], F, lab, shape, gray)
            used = set(best[1])
            pool = [j for j in pool if j not in used]
        else:
            pool = pool[1:]
    objects = drop_contained(drop_reflections(merge_duplicates(cam, segs, objects, shape, gray)))
    return objects, {"flow": F, "flow_labels": flow_lab, "colours": mean, "colour_labels": colour_lab,
                     "foreground": fg, "segments": masks}


def box_iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def link(frames, max_gap=LINK_GAP, min_iou=LINK_IOU):
    """Linking in the image, where the motion is measured: every track's last 2D box is moved by its mean optical
    flow to the frame (for each frame it has been missing), and the objects are assigned to the moved boxes by IoU
    (Hungarian, at least min_iou); a track survives max_gap frames without an object. (On the road a far object's
    centre jumps by metres along the viewing ray from frame to frame.) Each track keeps its most frequent type.
    frames: list of object lists; sets o["track"] and o["track_type"]."""
    from scipy.optimize import linear_sum_assignment
    nxt, active = 0, []
    for k, objs in enumerate(frames):
        active = [a for a in active if k - a["k"] <= max_gap]
        moved = [tuple(np.array(a["box"]) + np.tile(a["flow"] * (k - a["k"]), 2)) for a in active]
        M = np.array([[box_iou(o["box"], m) for m in moved] for o in objs]).reshape(len(objs), len(moved))
        taken = {}
        if M.size:
            for i, j in zip(*linear_sum_assignment(-M)):
                if M[i, j] >= min_iou:
                    taken[i] = j
        for i, o in enumerate(objs):
            if i in taken:
                o["track"] = active[taken[i]]["track"]
            else:
                o["track"], nxt = nxt, nxt + 1
        active = [a for j, a in enumerate(active) if j not in taken.values()] + \
                 [{"track": o["track"], "box": o["box"], "flow": np.asarray(o["flow"], float), "k": k} for o in objs]
    votes = {}
    for objs in frames:
        for o in objs:
            votes.setdefault(o["track"], []).append(o["type"])
    for objs in frames:
        for o in objs:
            ts = votes[o["track"]]
            o["track_type"] = max(set(ts), key=ts.count)


def refine_tracks(cam, frames):
    """Per track of at least TRACK_MIN objects: the heading in every frame from the track's path (the tangent of a
    quadratic in time fitted to the centres, when the track moved at least TRACK_PATH m; a turning car turns its box)
    instead of one frame's flow, the size as the median of the measured sizes, the type as the one this size deviates
    least from, and in every frame the centre solved again by the tangents with this heading and size."""
    tracks = {}
    for k, objs in enumerate(frames):
        for o in objs:
            tracks.setdefault(o["track"], []).append((k, o))
    for ko in tracks.values():
        if len(ko) < TRACK_MIN:
            continue
        ks = np.array([k for k, _ in ko], float)
        objs = [o for _, o in ko]
        lwh = tuple(np.median([o["lwh"] for o in objs], axis=0))
        t = min(TYPES, key=lambda k: size_deviation(lwh, k))
        C = np.array([o["centre"] for o in objs])
        headings = [o["yaw"] for o in objs]
        deg = 2 if len(ks) >= 5 else 1
        px, py = (np.polyfit(ks, C[:, i], deg) for i in (0, 1))      # the smoothed path
        if np.linalg.norm(C[-1] - C[0]) >= TRACK_PATH:
            dx, dy = np.polyder(px), np.polyder(py)
            # the path's tangent where the track moves; a car that is just starting keeps its flow's direction
            headings = [float(np.arctan2(np.polyval(dy, k), np.polyval(dx, k)))
                        if np.hypot(np.polyval(dx, k), np.polyval(dy, k)) >= TRACK_SPEED else o["yaw"]
                        for k, o in zip(ks, objs)]
        for k, o, heading in zip(ks, objs, headings):
            r = lift(cam, o["hull"], o["box"], heading, fixed=lwh, centre=o["centre"], occluded=o["occluded"])
            # a silhouette that cannot hold the track's box was a piece of the object: the box on the smoothed path
            c = r["centre"] if r is not None else np.array([np.polyval(px, k), np.polyval(py, k)])
            o.update(centre=c, yaw=heading, lwh=lwh)
            o["track_type"] = t


def track_models(frames):
    """Per track of at least TRACK_MIN objects: its span (first, last frame), the smoothed path (a quadratic in time,
    a line for short tracks), the median size and the type of that size, and the median heading of a track that
    hardly moved."""
    tracks = {}
    for k, objs in enumerate(frames):
        for o in objs:
            tracks.setdefault(o["track"], []).append((k, o))
    models = {}
    for tr, ko in tracks.items():
        if len(ko) < TRACK_MIN:
            continue
        ks = np.array([k for k, _ in ko], float)
        C = np.array([o["centre"] for _, o in ko])
        lwh = tuple(np.median([o["lwh"] for _, o in ko], axis=0))
        deg = 2 if len(ks) >= 5 else 1
        models[tr] = {"k0": int(ks[0]), "k1": int(ks[-1]), "px": np.polyfit(ks, C[:, 0], deg),
                      "py": np.polyfit(ks, C[:, 1], deg), "lwh": lwh, "moved": np.linalg.norm(C[-1] - C[0]),
                      "type": ko[0][1].get("track_type") or min(TYPES, key=lambda k: size_deviation(lwh, k)),
                      "yaw": float(np.median([o["yaw"] for _, o in ko]))}
    return models


def predict(m, k, fallback_yaw=None):
    """A track's centre and heading at frame k (also outside its span: extrapolated along the path)."""
    c = np.array([np.polyval(m["px"], k), np.polyval(m["py"], k)])
    dx, dy = np.polyval(np.polyder(m["px"]), k), np.polyval(np.polyder(m["py"]), k)
    if m["moved"] >= TRACK_PATH and np.hypot(dx, dy) >= TRACK_SPEED:
        return c, float(np.arctan2(dy, dx))
    return c, m["yaw"] if fallback_yaw is None else fallback_yaw


def frac_inside(a, b):
    """Share of box a's columns inside box b, if they overlap vertically by at least a third of a's height (an object
    that holds another car's roof holds its columns, not its area), else 0."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w / max(a[2] - a[0], 1e-9) if h >= (a[3] - a[1]) / 3 else 0.0


def mask_object(cam, part, shape, centre, yaw, lwh, occ, base):
    """An object of a track in a frame from its pixels `part` (half resolution): the tangents of its silhouette with
    the track's size and heading place it (those along occ left out); if they cannot, the predicted centre stays."""
    ys, xs = np.nonzero(part)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32)).reshape(-1, 2)
    q = np.clip(hull, 0, [shape[1] - 1, shape[0] - 1])
    occluded = occ[q[:, 1], q[:, 0]] if occ is not None else np.zeros(len(hull), bool)
    hull = (hull + 0.5) / SCALE
    box = (xs.min() / SCALE, ys.min() / SCALE, (xs.max() + 1) / SCALE, (ys.max() + 1) / SCALE)
    r = lift(cam, hull, box, yaw, fixed=lwh, centre=centre, occluded=occluded)
    o = dict(base)
    o.update(centre=r["centre"] if r is not None else centre, yaw=yaw, lwh=lwh, box=box, mask=part, hull=hull,
             occluded=occluded, tangent_rms=r["tangent_rms"] if r is not None else np.inf)
    return o


def temporal_split(cam, frames, shape):
    """Cars that merge in some frames are apart in others. Every track is predicted into each frame of its span and
    up to EXTRAP frames beyond it (along its smoothed path, with its size and heading):
      - an object whose 2D box holds the predicted boxes (INSIDE of each) of two or more tracks, its own included, is
        a merge: its pixels go to the predicted silhouette they lie in (the nearer car where two overlap, the closest
        one where none); the part of its own track and of a track without an object in that frame is placed by its
        tangents with the track's size and heading, the border with the other parts left out; the part of a track
        that has its own object elsewhere is cut away;
      - a frame inside a track's span without an object of the track gets the predicted cuboid (marked predicted).
    frames: per frame the linked and refined objects; changed in place."""
    models = track_models(frames)
    for k, objs in enumerate(frames):
        preds = {}
        for tr, m in models.items():
            if m["k0"] - EXTRAP <= k <= m["k1"] + EXTRAP:
                c, yaw = predict(m, k)
                uv = cam.project(corners(c, yaw, *m["lwh"]))
                if np.isfinite(uv).all():
                    preds[tr] = (c, yaw, (*uv.min(0), *uv.max(0)), uv)
        out = []
        for o in objs:
            own = {p["track"] for p in objs if p is not o}       # tracks with an object of their own here
            inside = [tr for tr, (c, yaw, pb, uv) in preds.items() if frac_inside(pb, o["box"]) >= INSIDE]
            if o["track"] in preds and o["track"] not in inside:
                inside.append(o["track"])
            if len(inside) < 2:
                out.append(o)
                continue
            # pixels to the predicted silhouettes: inside one, the nearer car where several, else the closest
            near = sorted(inside, key=lambda tr: np.linalg.norm(preds[tr][0] - cam.ground_c))
            D, polys = [], []
            for tr in near:
                poly = np.zeros(shape, np.uint8)
                cv2.fillConvexPoly(poly, cv2.convexHull((preds[tr][3] * SCALE).astype(np.int32)), 1)
                D.append(cv2.distanceTransform(1 - poly, cv2.DIST_L2, 3))
                polys.append(poly > 0)
            lbl = np.argmin(np.stack(D) + 1e-3 * np.arange(len(near))[:, None, None], axis=0)
            for i, tr in enumerate(near):
                part = o["mask"] & (lbl == i)
                if part.sum() < MIN_AREA or (tr in own and tr != o["track"]):
                    continue                             # a neighbour with its own object: its part is cut away
                occ = cv2.dilate((o["mask"] & ~part).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                c, yaw = preds[tr][:2]
                base = {k2: v for k2, v in o.items() if k2 not in ("hull", "occluded")}
                base.update(track=tr, track_type=models[tr]["type"], type=models[tr]["type"], split_by_track=True)
                if part.sum() < VISIBLE * polys[i].sum():
                    # a thin visible strip of a hidden car places it worse than its track does
                    ys, xs = np.nonzero(part)
                    base.update(centre=c, yaw=yaw, lwh=models[tr]["lwh"], mask=part, tangent_rms=np.inf,
                                box=(xs.min() / SCALE, ys.min() / SCALE, (xs.max() + 1) / SCALE, (ys.max() + 1) / SCALE),
                                predicted=True)
                    out.append(base)
                else:
                    out.append(mask_object(cam, part, shape, c, yaw, models[tr]["lwh"], occ, base))
        present = {o["track"] for o in out}
        for tr, m in models.items():
            if m["k0"] < k < m["k1"] and tr not in present and tr in preds:
                c, yaw, pb, uv = preds[tr]
                part = np.zeros(shape, np.uint8)
                cv2.fillConvexPoly(part, cv2.convexHull((uv * SCALE).astype(np.int32)), 1)
                out.append({"track": tr, "track_type": m["type"], "type": m["type"], "centre": c, "yaw": yaw,
                            "lwh": m["lwh"], "box": pb, "mask": part > 0, "error": 0.0, "tangent_rms": np.inf,
                            "segments": 0, "flow": np.zeros(2), "predicted": True})
        frames[k] = out
