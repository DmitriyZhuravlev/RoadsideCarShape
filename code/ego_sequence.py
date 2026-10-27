"""
A MOVING camera as a sequence for the whole-scene pipeline (V2X-Seq-SPD vehicle side: the ego car's front camera).

  - frames undistorted (the calibration's plumb-bob model, same K); the ego car's own hood (rows >= HOOD of the
    height) is blanked and never used;
  - the camera's pose per frame from MONOCULAR VISUAL ODOMETRY: corners tracked frame to frame (pyramidal LK,
    forward-backward check), the essential matrix (RANSAC) gives the rotation and the direction of the step. The car
    drives on the road, so only its yaw change and its step along the road are kept - the camera's tilt and its
    height above the road are the mounting's and stay fixed (no drift in pitch, roll or height). The step's length
    comes from the road: corners below the horizon triangulated with a unit step lie at h below the camera, the road
    at the camera's height H, so the length is H / h at the most common h (the road);
  - the world: the road plane z = 0, origin under the first camera, x = its viewing direction along the road;
  - background(k): the neighbouring frames warped onto frame k by the homography of the road plane, median - the
    road agrees, a car that moves differently from the road does not;
  - static_flow(k, m): the image motion k-1 -> k the pixels of a region would have if it stood still on the road
    (every column at the depth of its lowest pixel, its contact with the road).

Calibration only (like the fixed camera): the intrinsics and the camera's mounting - its height above the road and
its tilt - from the camera-LiDAR calibration and the LiDAR's ground plane in the first frame. Labels and LiDAR are
otherwise evaluation only: lidar_world(k) puts the LiDAR points of frame k into this world through the calibration
and the camera pose found by the odometry; the labelled boxes likewise.
"""
import functools
import json
import os

import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

import v2x_seq_demo as vd

CAMERA = os.environ.get("CAMERA", "fixed")          # "moving": the ego car's camera (vehicle side)
EGO = os.environ.get("EGO", "vo")                  # "vo": visual odometry; "ins": GPS/INS poses (diagnostic only)
HOOD = float(os.environ.get("HOOD", "0.70"))       # share of the image height where the ego car's hood starts
BG_SPAN, BG_STEP = 8, 2                            # frames on each side warped into the background, step
VO_CORNERS, VO_FB = 3000, 1.0                      # corners per frame, px forward-backward tolerance
VO_RANSAC_PX = 1.0                                 # px: essential-matrix inlier threshold
ROAD_MAX = 35.0                                    # m: road corners nearer than this (assuming they are road)
STEP_MAX, ROAD_TRUNC = 3.0, 2.0                    # m per frame searched, px: truncated road transfer error
STEP_MED = int(os.environ.get("STEP_MED", "5"))    # steps in the running median of the step length
PITCH_WIN = int(os.environ.get("PITCH_WIN", "41"))  # frames: pitch and roll keep their oscillation around this mean


def load_json(path):
    return json.load(open(os.path.join(vd.DATA, path)))


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def road_homography(cam_src, cam_dst):
    """3x3 homography mapping pixels of cam_src to pixels of cam_dst for points on the road plane z = 0."""
    G = [K @ np.column_stack([R[:, 0], R[:, 1], t]) for K, R, t in (cam_src, cam_dst)]
    Hm = G[1] @ np.linalg.inv(G[0])
    return Hm / Hm[2, 2]


def mounting(info, K, n_frames=10):
    """The camera's mounting from the calibration and the LiDAR's ground plane (calibration, like the fixed camera's
    ground plane), median over n_frames frames spread over the sequence (one frame may stand on a slope): its height
    H above the road and R_m, the rotation from the car frame (x forward along the road, y left, z up) to the
    camera."""
    ex = load_json(info[0]["calib_lidar_to_camera_path"])
    R_lc, t_lc = np.array(ex["rotation"]), np.array(ex["translation"]).ravel()
    C_L = -np.linalg.solve(R_lc, t_lc)
    Hs, ups = [], []
    for d in info[:: max(1, len(info) // n_frames)][:n_frames]:
        a, b, c = vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, d["pointcloud_path"])))[:3]
        n_L = np.array([-a, -b, 1.0]) / np.linalg.norm([-a, -b, 1.0])
        Hs.append(float(n_L @ (C_L - np.array([0.0, 0.0, c]))))
        ups.append(R_lc @ n_L)
    H = float(np.median(Hs))
    up = np.median(np.array(ups), axis=0)
    up /= np.linalg.norm(up)
    fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up                          # optical axis along the road
    fwd /= np.linalg.norm(fwd)
    R_m = np.column_stack([fwd, np.cross(up, fwd), up])                   # car-frame vector -> camera coordinates
    return H, R_m, (R_lc, t_lc)


class EgoSequence:
    def __init__(self):
        self.info = sorted(load_json("data_info.json"), key=lambda d: d["frame_id"])
        self.fids = [d["frame_id"] for d in self.info]
        self.times = np.array([int(d["image_timestamp"]) for d in self.info]) / 1e6
        intr = load_json(self.info[0]["calib_camera_intrinsic_path"])
        self.K = np.array(intr["cam_K"], float).reshape(3, 3)
        self.D = np.array(intr["cam_D"], float)
        self.frames = [cv2.undistort(cv2.imread(os.path.join(vd.DATA, d["image_path"])), self.K, self.D)
                       for d in self.info]
        self.shape = self.frames[0].shape[:2]
        self.hood = int(HOOD * self.shape[0])
        for f in self.frames:
            f[self.hood:] = 0
        self.H, self.R_m, self.lidar_cam = mounting(self.info, self.K)
        print(f"camera {self.H:.2f} m above the road, looking {np.degrees(np.arcsin(-self.R_m[2, 2])):.1f} deg down",
              flush=True)
        self.pos, self.att, self.vo = self.odometry()
        self.yaw = self.att[:, 0]
        self.cams = [self.camera(p, a) for p, a in zip(self.pos, self.att)]
        if EGO == "ins":
            self.cams, self.pos, self.yaw = self.ins_cameras()
        self.tracks = self.labels()
        self.camera_xy = self.pos[0]
        self.ground = 0.0

    # --- poses ----------------------------------------------------------------------------------------------
    def camera(self, pos, att):
        """(K, R, t) world -> camera of the camera on the car at road position pos, attitude (yaw, pitch, roll)."""
        A = rot_z(att[0]) @ rot_y(att[1]) @ rot_x(att[2])                    # car -> world
        R = self.R_m @ A.T
        C = np.r_[pos, self.H]
        return self.K, R, -R @ C

    def odometry(self):
        """Visual odometry frame to frame: rotation and direction from the essential matrix, the step's length from
        the road. The car's yaw is integrated; its pitch and roll only oscillate around the mounting's (braking,
        bumps), so their integrated values are high-passed (minus their PITCH_WIN moving average). The step: the
        road homography between the two frames is linear in the step's length - the length whose homography maps
        the corners below the horizon best (truncated error), then a running median over STEP_MED steps."""
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in self.frames]
        Hh, W = self.shape
        mask = np.zeros((Hh, W), np.uint8)
        mask[: self.hood - 5] = 255
        pairs, drot = [], []
        for k in range(len(grays) - 1):
            a, b = grays[k], grays[k + 1]
            p0 = cv2.goodFeaturesToTrack(a, VO_CORNERS, 0.003, 8, mask=mask)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21, 21), maxLevel=4)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, winSize=(21, 21), maxLevel=4)
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pb - p0, axis=2)[:, 0] < VO_FB) & \
                 (p1[:, 0, 1] < self.hood - 5)
            q0, q1 = p0[ok, 0].astype(np.float64), p1[ok, 0].astype(np.float64)
            E, inl = cv2.findEssentialMat(q0, q1, self.K, cv2.RANSAC, 0.999, VO_RANSAC_PX)
            _, R, t, inl = cv2.recoverPose(E, q0, q1, self.K, mask=inl)
            inl = inl.ravel() > 0
            dA = (self.R_m.T @ R @ self.R_m).T                                # car attitude change, car axes
            rv = cv2.Rodrigues(dA)[0].ravel()                                 # (roll, pitch, yaw) increments
            drot.append(rv[::-1])
            pairs.append((q0[inl], q1[inl], -R.T @ t.ravel()))
        drot = np.array(drot)
        att = np.vstack([np.zeros(3), np.cumsum(drot, 0)])
        for a in (1, 2):                                                      # pitch, roll: oscillation only
            att[:, a] -= uniform_filter1d(att[:, a], PITCH_WIN, mode="nearest")
        cams0 = [self.camera(np.zeros(2), a_) for a_ in att]
        steps, diag = [], []
        for k, (q0, q1, c) in enumerate(pairs):
            par = np.median(np.linalg.norm(q1 - q0, axis=1)) if len(q0) else 0.0
            step = self.step_length(k, q0, q1, c, cams0) if par > 0.3 else 0.0
            steps.append(step)
            diag.append({"k": k, "inliers": len(q0), "parallax_px": float(par), "step_m": float(step),
                         "dyaw_deg": float(np.degrees(drot[k, 0])), "pitch_deg": float(np.degrees(att[k, 1]))})
        steps = np.array(steps, float)
        good = np.isfinite(steps)
        steps = np.interp(np.arange(len(steps)), np.nonzero(good)[0], steps[good])
        steps = median_filter(steps, STEP_MED, mode="nearest")
        pos = [np.zeros(2)]
        for k, (q0, q1, c) in enumerate(pairs):
            Rk = cams0[k][1]
            d = np.linalg.solve(Rk, c)                                        # step direction in the world
            dxy = d[:2] / max(np.linalg.norm(d[:2]), 1e-9)
            pos.append(pos[-1] + dxy * steps[k])
            diag[k]["step_m_smooth"] = float(steps[k])
        return np.array(pos), att, diag

    def step_length(self, k, q0, q1, c, cams0):
        """The step's length from the road plane: x1 ~ R1 (x0 + s (n . x0) / H u) for road points, R1 the rotation
        from camera k to k+1, u the step direction (unit, camera-k coordinates, along the road), n the road normal
        (up) in camera k; s searched on a grid, truncated transfer error."""
        Ki = np.linalg.inv(self.K)
        _, Rk, _ = cams0[k]
        _, Rk1, _ = cams0[k + 1]
        up = Rk[:, 2] / np.linalg.norm(Rk[:, 2])
        rays = np.c_[q0, np.ones(len(q0))] @ Ki.T
        down = -(rays @ up)
        road_d = self.H / np.maximum(down, 1e-6) * np.linalg.norm(rays, axis=1)
        sel = (down > 0.02) & (road_d < ROAD_MAX)
        if sel.sum() < 15:
            return np.nan
        u = c - (c @ up) * up
        u /= max(np.linalg.norm(u), 1e-9)
        R1 = Rk1 @ np.linalg.inv(Rk)
        x0 = rays[sel]
        A = x0 @ R1.T
        B = ((x0 @ up) / self.H)[:, None] * (R1 @ u)[None]
        grid = np.linspace(0.0, STEP_MAX, 601)
        P = A[None] + grid[:, None, None] * B[None]
        uv = P @ self.K.T
        err = np.minimum(np.linalg.norm(uv[..., :2] / uv[..., 2:3] - q1[sel][None], axis=2), ROAD_TRUNC)
        return float(grid[int(np.argmin((err ** 2).sum(1)))])

    def ins_cameras(self):
        """DIAGNOSTIC ONLY (not the method): the cameras from the GPS/INS poses and the calibration, in a world whose
        z = 0 is the road under the first camera - to tell the odometry's error from the rest."""
        R_lc, t_lc = self.lidar_cam
        cams, pos = [], []
        for d in self.info:
            ln, nw = load_json(d["calib_lidar_to_novatel_path"])["transform"], load_json(d["calib_novatel_to_world_path"])
            R_ln, t_ln = np.array(ln["rotation"]), np.array(ln["translation"]).ravel()
            R_nw, t_nw = np.array(nw["rotation"]), np.array(nw["translation"]).ravel()
            R_lw, t_lw = R_nw @ R_ln, R_nw @ t_ln + t_nw                       # LiDAR -> world (ENU)
            R = R_lc @ R_lw.T                                                 # world -> camera
            t = t_lc - R @ t_lw
            cams.append((self.K, R, t))
        C0 = -np.linalg.solve(cams[0][1], cams[0][2])
        shift = np.r_[C0[:2], C0[2] - self.H]
        out = [(K, R, t + R @ shift) for K, R, t in cams]
        pos = np.array([(-np.linalg.solve(R, t))[:2] for _, R, t in out])
        yaw = np.array([np.arctan2(*(np.linalg.solve(R, [0.0, 0.0, 1.0])[[1, 0]])) for _, R, t in out])
        return out, pos, yaw

    # --- labels and LiDAR in this world (evaluation only) -----------------------------------------------------
    def lidar_to_world(self, k, X):
        R_lc, t_lc = self.lidar_cam
        _, R, t = self.cams[k]
        return np.linalg.solve(R, (X @ R_lc.T + t_lc - t).T).T

    def lidar_world(self, k):
        X = vd.read_pcd(os.path.join(vd.DATA, self.info[k]["pointcloud_path"]))
        return self.lidar_to_world(k, X[np.isfinite(X).all(1)])

    def labels(self):
        R_lc = self.lidar_cam[0]
        tracks = {}
        for k, d in enumerate(self.info):
            _, R, _ = self.cams[k]
            for o in load_json(d["label_camera_std_path"]):
                if o["type"] not in ("Car", "Van"):
                    continue
                loc = np.array([o["3d_location"][a] for a in "xyz"], float)
                c = self.lidar_to_world(k, loc[None])[0]
                v = np.linalg.solve(R, R_lc @ np.array([np.cos(o["rotation"]), np.sin(o["rotation"]), 0.0]))
                tracks.setdefault(o["track_id"], []).append({
                    "k": k, "pos": c[:2], "yaw": float(np.arctan2(v[1], v[0])),
                    "dims": (o["3d_dimensions"]["l"], o["3d_dimensions"]["w"], o["3d_dimensions"]["h"]),
                    "z": float(c[2]), "truncated": o["truncated_state"],
                    "box": [o["2d_box"][x] for x in ("xmin", "ymin", "xmax", "ymax")]})
        return tracks

    # --- background and the image motion of a still region ------------------------------------------------------
    @functools.lru_cache(maxsize=16)
    def background(self, k):
        """Median of the neighbouring frames warped onto frame k through the road plane."""
        Hh, W = self.shape
        warped = []
        for j in range(k - BG_SPAN, k + BG_SPAN + 1, BG_STEP):
            if j == k or not 0 <= j < len(self.frames):
                continue
            Hm = road_homography(self.cams[j], self.cams[k])
            warped.append(cv2.warpPerspective(self.frames[j], Hm, (W, Hh), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REPLICATE))
        bg = np.median(np.stack(warped), axis=0).astype(np.uint8)
        bg[self.hood:] = 0
        return bg

    def static_flow(self, k, m):
        """Image motion k-1 -> k of the pixels of mask m (np.nonzero order) if the region stood still on the road:
        every column at the camera depth of its lowest pixel (where it touches the road)."""
        ys, xs = np.nonzero(m)
        if k == 0 or not len(xs):
            return np.zeros((len(xs), 2))
        K, R, t = self.cams[k]
        Ki = np.linalg.inv(K)
        low = np.zeros(self.shape[1], int)
        np.maximum.at(low, xs, ys)
        ray = Ki @ np.c_[xs, low[xs], np.ones(len(xs))].T                   # contact pixel of each column
        up_c = R[:, 2] / np.linalg.norm(R[:, 2])                            # world z axis in camera coordinates
        down = -(up_c @ ray)
        depth = np.where(down > 1e-3, self.H / np.maximum(down, 1e-3), np.nan)   # camera-z of the contact
        Xc = (Ki @ np.c_[xs, ys, np.ones(len(xs))].T) * depth
        Xw = np.linalg.solve(R, Xc - t[:, None]).T
        _, R0, t0 = self.cams[k - 1]
        x0 = K @ (R0 @ Xw.T + t0[:, None])
        uv0 = (x0[:2] / x0[2]).T
        return np.c_[xs, ys] - uv0


def open_sequence():
    """The fixed roadside camera (v2x_seq_shape.Sequence) or, with CAMERA=moving, the ego car's camera; and the
    ground offset of the LiDAR frame (0 for the moving camera: its LiDAR is put into the road-plane world)."""
    if CAMERA == "moving":
        seq = EgoSequence()
        return seq, 0.0
    import v2x_seq_shape as vs
    seq = vs.Sequence()
    seq.info = sorted(load_json("data_info.json"), key=lambda d: d["frame_id"])
    return seq, vd.fit_ground_plane(vd.read_pcd(os.path.join(vd.DATA, seq.info[0]["pointcloud_path"])))[2]
