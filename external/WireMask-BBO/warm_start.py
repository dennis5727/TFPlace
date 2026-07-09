"""Connectivity-based coordinate hint (drop-in for utils.random_guiding).

The greedy decoder uses the hint only as a TIE-BREAKER among equal-least-wire cells,
so it dominates the early/anchor macros and cascades to the rest. Instead of seeding
those anchors with noise (random_guiding), seed them from a spectral layout of the
macro-macro graph: connected macros get nearby coordinates, so the greedy tie-breaks
pull them together -> a better basin for the coordinate-EA to refine.

connectivity_guiding() returns the SAME dict format as utils.random_guiding (the
decoder only reads loc_x/loc_y per macro), so it is a true drop-in. Legality stays
automatic -- this only sets a target; the greedy's position_mask still enforces
in-canvas / no-overlap.
"""
import math
import numpy as np
from itertools import combinations


def _macro_weight_matrix(placedb, order):
    """Symmetric macro<->macro weight = number of shared nets (co-membership count)."""
    idx = {n: i for i, n in enumerate(order)}
    N = len(order)
    W = np.zeros((N, N))
    for net in placedb.net_info.values():
        members = [idx[n] for n in net["nodes"] if n in idx]
        for a, b in combinations(members, 2):
            W[a, b] += 1.0
            W[b, a] += 1.0
    return W


def _spectral_coords(W):
    """2-D layout from the graph Laplacian: eigenvectors 1 & 2 (skip the trivial 0th).
    Connected macros get nearby coordinates; eigenvectors spread nodes (no collapse)."""
    d = W.sum(axis=1)
    L = np.diag(d) - W                      # unnormalized Laplacian
    _, vecs = np.linalg.eigh(L)             # ascending eigenvalues; 543x543 is instant
    return vecs[:, 1], vecs[:, 2]           # Fiedler vector + next


def _to_grid(v, grid_num, margin=0.05):
    """Scale an eigenvector to [margin, 1-margin] * grid_num cells."""
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.full_like(v, grid_num / 2.0, dtype=float)
    s = (v - lo) / (hi - lo)
    return (margin + s * (1 - 2 * margin)) * grid_num


# cache the (expensive-once) embedding per placedb, keyed by id()
_EMB = {}


def connectivity_guiding(order, placedb, grid_num, grid_size, rng=None, jitter=0.0):
    """Spectral coordinate hint in random_guiding's dict format.

    jitter (cells): per-draw uniform offset for diversity (0 = pure spectral). Pass a
    small jitter on all but the first init draw so best-of-N init still explores."""
    key = id(placedb)
    if key not in _EMB:
        W = _macro_weight_matrix(placedb, order)
        ex, ey = _spectral_coords(W)
        _EMB[key] = (list(order), _to_grid(ex, grid_num), _to_grid(ey, grid_num))
    cached_order, gx, gy = _EMB[key]
    pos = {n: i for i, n in enumerate(cached_order)}

    placed = {}
    for node_id in order:
        x = placedb.node_info[node_id]["x"]
        y = placedb.node_info[node_id]["y"]
        sx = math.ceil(x / grid_size)
        sy = math.ceil(y / grid_size)
        i = pos[node_id]
        cx, cy = float(gx[i]), float(gy[i])
        if jitter and rng is not None:
            cx += rng.uniform(-jitter, jitter)
            cy += rng.uniform(-jitter, jitter)
        lx = int(min(max(0, round(cx)), grid_num - sx))   # clamp to legal bottom-left
        ly = int(min(max(0, round(cy)), grid_num - sy))
        placed[node_id] = {
            "scaled_x": sx, "scaled_y": sy, "loc_x": lx, "loc_y": ly, "x": x, "y": y,
            "center_loc_x": grid_size * lx + 0.5 * x,
            "center_loc_y": grid_size * ly + 0.5 * y,
            "bottom_left_x": lx * grid_size, "bottom_left_y": ly * grid_size}
    return placed
