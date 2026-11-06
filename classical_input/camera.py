"""Calibrated roadside camera and the road plane (V2X-Seq-SPD / DAIR-V2X infrastructure layout).

The camera is the dataset's calibration (intrinsics with distortion, virtual-LiDAR -> camera). The road plane is fitted
once, by RANSAC on the points of one LiDAR sweep: a calibration step equivalent to measuring the camera's height and
tilt above the road (Sect. 3 of the article). Labels are never read here.

Coordinates: V is the dataset's virtual-LiDAR frame; G is the road frame, z = 0 on the road and z up, x along V's x
projected onto the road.
"""
import json
import os

import cv2
import numpy as np


def lzf_decompress(data, out_len):
    """LibLZF decompression (PCL's DATA binary_compressed)."""
    out = bytearray()
    ip = 0
    while ip < len(data):
        ctrl = data[ip]
        ip += 1
        if ctrl < 32:                                    # literal run of ctrl + 1 bytes
            out += data[ip:ip + ctrl + 1]
            ip += ctrl + 1
        else:                                            # back reference, possibly overlapping
            n = ctrl >> 5
            if n == 7:
                n += data[ip]
                ip += 1
            ref = len(out) - ((ctrl & 0x1f) << 8) - 1 - data[ip]
            ip += 1
            for _ in range(n + 2):
                out.append(out[ref])
                ref += 1
    if len(out) != out_len:
        raise ValueError(f"LZF: {len(out)} bytes, expected {out_len}")
    return bytes(out)


def read_pcd_xyz(path):
    """x, y, z of a PCD v0.7 file (ascii, binary or binary_compressed, float32 fields)."""
    raw = open(path, "rb").read()
    head, pos = {}, 0
    while True:
        end = raw.index(b"\n", pos)
        line = raw[pos:end].decode("ascii").strip()
        pos = end + 1
        if line and not line.startswith("#"):
            key, *val = line.split()
            head[key] = val
            if key == "DATA":
                break
    fields, n = head["FIELDS"], int(head["POINTS"][0])
    idx = [fields.index(c) for c in "xyz"]
    kind = head["DATA"][0]
    if kind == "ascii":
        a = np.loadtxt(raw[pos:].decode("ascii").splitlines(), dtype=np.float32).reshape(n, len(fields))
    elif kind == "binary":
        a = np.frombuffer(raw[pos:pos + 4 * n * len(fields)], np.float32).reshape(n, len(fields))
    else:                                                # binary_compressed: field after field
        csize, usize = np.frombuffer(raw[pos:pos + 8], "<u4")
        buf = lzf_decompress(raw[pos + 8:pos + 8 + int(csize)], int(usize))
        a = np.frombuffer(buf, np.float32).reshape(len(fields), n).T
    return np.ascontiguousarray(a[:, idx])


def fit_plane(P, iters=400, tol=0.08, seed=0):
    """RANSAC plane n.x + d = 0 (n up, i.e. n_z > 0) on the lowest points, refined by least squares on its inliers."""
    rng = np.random.default_rng(seed)
    P = P[np.isfinite(P).all(1)]
    P = P[(np.linalg.norm(P[:, :2], axis=1) > 3) & (np.linalg.norm(P[:, :2], axis=1) < 80)]
    best = None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < 0.9:                              # the road is nearly horizontal in the LiDAR frame
            continue
        inl = np.abs(P @ n - n @ a) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = P[best]
    m = Q.mean(0)
    n = np.linalg.svd(Q - m)[2][2]
    n = n if n[2] > 0 else -n
    return n, -n @ m, best.mean()


class Camera:
    """Projection of road-frame points into the (distorted) image and the inverse perspective mapping."""

    def __init__(self, root, fid, ground=None):
        intr = json.load(open(os.path.join(root, "calib/camera_intrinsic", fid + ".json")))
        ext = json.load(open(os.path.join(root, "calib/virtuallidar_to_camera", fid + ".json")))
        self.K = np.array(intr["cam_K"], float).reshape(3, 3)
        self.D = np.array(intr["cam_D"], float)
        self.size = (intr["width"], intr["height"])
        # V -> camera: X_c = M X_v + t. M is the dataset's "rotation"; it is not orthonormal (row norms 1.00 and 0.84
        # on V2X-Seq-SPD), so it is kept as the linear map it is and never turned into a rotation vector.
        self.M = np.array(ext["rotation"], float)
        self.Minv = np.linalg.inv(self.M)
        self.t = np.array(ext["translation"], float).ravel()
        self.C = -self.Minv @ self.t                              # camera centre in V
        if ground is None:
            ground = fit_plane(read_pcd_xyz(os.path.join(root, "velodyne", fid + ".pcd")))[:2]
        self.n, self.d = np.asarray(ground[0], float), float(ground[1])
        e1 = np.array([1.0, 0, 0]) - self.n[0] * self.n
        e1 /= np.linalg.norm(e1)
        self.E = np.c_[e1, np.cross(self.n, e1), self.n]          # columns: G axes in V
        self.o = -self.d * self.n                                 # foot of V's origin on the road
        self.height = float(self.n @ self.C + self.d)
        self.ground_c = self.to_g(self.C)[:2]                    # the camera's foot point on the road

    def to_v(self, X):
        return self.o + np.asarray(X, float) @ self.E.T

    def to_g(self, X):
        return (np.asarray(X, float) - self.o) @ self.E

    def project(self, X):
        """Road-frame points (N, 3) -> distorted pixels (N, 2); NaN behind the camera."""
        Xv = self.to_v(np.atleast_2d(X))
        Xc = Xv @ self.M.T + self.t
        uv = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), self.K, self.D)[0].reshape(-1, 2)
        uv[Xc[:, 2] <= 0.1] = np.nan
        return uv

    def rays(self, uv):
        """Distorted pixels (N, 2) -> ray directions in the camera frame (N, 3), (x, y, 1) undistorted."""
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)   # the lens is strongly distorted
        xn = cv2.undistortPoints(uv, self.K, self.D, None, None, None, crit).reshape(-1, 2)
        return np.c_[xn, np.ones(len(xn))]

    def plane_g(self, n_c):
        """The plane through the camera centre with normal n_c (camera frame) as (n, d) of {X: n.X + d = 0} in G."""
        n_v, d_v = self.M.T @ n_c, float(n_c @ self.t)          # n_c.(M X_v + t) = 0
        return self.E.T @ n_v, float(n_v @ self.o + d_v)          # X_v = o + E X_g

    def to_cam(self, X):
        """Road-frame points (N, 3) -> camera frame (N, 3)."""
        return self.to_v(np.atleast_2d(X)) @ self.M.T + self.t

    def ipm(self, uv):
        """Distorted pixels (N, 2) -> their rays' intersection with the road, (N, 2) in G; NaN above the horizon."""
        dv = self.rays(uv) @ self.Minv.T                          # ray directions in V
        s = -(self.n @ self.C + self.d) / (dv @ self.n)
        X = self.C + s[:, None] * dv
        X[s <= 0] = np.nan
        return self.to_g(X)[:, :2]
