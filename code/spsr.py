"""Screened Poisson surface reconstruction (Kazhdan & Hoppe 2013) on a regular grid, as a separate smoothing step
for a reconstructed car mesh.

The indicator function chi of the solid (1 inside, 0 outside) is found from oriented points p_i with normals n_i:
its gradient should equal the field V of the inward normals spread over the grid, and it should be 1/2 at the points
(the screening term, which keeps the surface on the data instead of smoothing it away):

    min  sum |grad chi - V|^2 + alpha * sum_i (chi(p_i) - 1/2)^2        (zero boundary, chi = 0 far outside)
    ->   (-Laplace + alpha A^T A) chi = -div V + alpha A^T 1/2           (A: trilinear interpolation at the points)

solved by conjugate gradients; the surface is the iso-level chi = mean chi(p_i) (marching cubes). The original works on
an adaptive octree with B-spline bases; for a car-sized object a regular 4 cm grid is enough.

    V, F = screened_poisson(P, N, area)            P (n, 3) points, N (n, 3) unit normals (outward), area (n,)
    V, F, C = smooth_mesh(V0, F0, C0)               a mesh -> its screened Poisson surface, colours carried over
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def trilinear_weights(P, origin, h, shape):
    """Sparse interpolation matrix of points on a regular 3D grid: row i holds the 8 trilinear weights of point P[i]
    on the grid nodes around it (C order: idx = (ix * ny + iy) * nz + iz, node at origin + h * (ix, iy, iz))."""
    n, (nx, ny, nz) = len(P), shape
    g = (P - origin) / h
    i0 = np.clip(np.floor(g).astype(int), 0, np.array(shape) - 2)
    f = np.clip(g - i0, 0, 1)
    rows, cols, ws = [], [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                rows.append(np.arange(n))
                cols.append(((i0[:, 0] + dx) * ny + (i0[:, 1] + dy)) * nz + (i0[:, 2] + dz))
                ws.append(w)
    return sp.csr_matrix((np.concatenate(ws), (np.concatenate(rows), np.concatenate(cols))), shape=(n, nx * ny * nz))


def laplacian(shape, h):
    """7-point Laplacian on the grid's nodes, zero (Dirichlet) outside the grid."""
    nx, ny, nz = shape
    def lap1(m):
        return sp.diags([np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)], [-1, 0, 1])
    I = lambda m: sp.identity(m)
    L = (sp.kron(sp.kron(lap1(nx), I(ny)), I(nz)) + sp.kron(sp.kron(I(nx), lap1(ny)), I(nz))
         + sp.kron(sp.kron(I(nx), I(ny)), lap1(nz)))
    return (L / h ** 2).tocsr()


def screened_poisson(P, N, area, h=0.04, alpha=4.0, margin=6, sigma=1.0, maxiter=3000):
    """The screened Poisson surface of oriented points (outward unit normals N, area per point). Returns (V, F)."""
    lo = P.min(0) - margin * h
    shape = tuple((np.ceil((P.max(0) + margin * h - lo) / h)).astype(int) + 1)
    A = trilinear_weights(P, lo, h, shape)
    # the gradient of the indicator: the INWARD normals, area-weighted, spread over the nodes (a delta sheet of unit
    # jump across the surface), softened by one cell
    Vf = [gaussian_filter((A.T @ (-N[:, a] * area)).reshape(shape) / h ** 3, sigma) for a in range(3)]
    div = sum(np.gradient(Vf[a], h, axis=a) for a in range(3)).ravel()
    L = laplacian(shape, h)
    # screening weight: comparable to the Laplacian's diagonal per point, `alpha` as in the original's default
    w = alpha * 6.0 / h ** 2 / max(float(np.asarray(A.sum(0)).max()), 1.0)   # per node, as heavy as its Laplacian row
    M = (-L + w * (A.T @ A)).tocsr()
    b = -div + w * (A.T @ np.full(len(P), 0.5))
    pre = sp.diags(1.0 / M.diagonal())
    chi, info = spl.cg(M, b, M=pre, rtol=1e-7, maxiter=maxiter)
    chi = chi.reshape(shape)
    iso = float(np.mean(A @ chi.ravel()))
    Vm, Fm, _, _ = marching_cubes(chi, iso)
    Vm = lo + Vm * h
    # the largest connected component (stray bubbles of the indicator far from the data)
    from scipy.sparse.csgraph import connected_components
    E = np.r_[Fm[:, [0, 1]], Fm[:, [1, 2]], Fm[:, [2, 0]]]
    n, lab = connected_components(sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(Vm), len(Vm))))
    keep = lab == np.bincount(lab).argmax()
    remap = -np.ones(len(Vm), int)
    remap[keep] = np.arange(keep.sum())
    Fm = remap[Fm[keep[Fm].all(1)]]
    return Vm[keep], Fm


def vertex_normals_areas(V, F):
    """Area-weighted vertex normals (unit) and the area each vertex stands for (a third of its triangles')."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    area = np.zeros(len(V))
    for i in range(3):
        np.add.at(N, F[:, i], n)
        np.add.at(area, F[:, i], np.linalg.norm(n, axis=1) / 6)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12), area


def smooth_mesh(V0, F0, C0, h=0.06, alpha=0.0):
    """A coloured mesh -> its Poisson surface (alpha = 0: plain Poisson, Kazhdan et al. 2006 - the default for the shown
    car models; alpha > 0: screened); the colours an inverse-distance mean of the 8 nearest original vertices; nothing
    below the road (z < 0) is kept."""
    N, area = vertex_normals_areas(V0, F0)
    V, F = screened_poisson(V0, N, area, h=h, alpha=alpha)
    V[:, 2] = np.maximum(V[:, 2], 0.0)
    d, j = cKDTree(V0).query(V, k=8)                 # colours: inverse-distance mean of the 8 nearest (no blocks)
    wgt = 1.0 / np.maximum(d, 1e-3)
    C = (C0[j] * wgt[:, :, None]).sum(1) / wgt.sum(1, keepdims=True)
    return V, F, C
