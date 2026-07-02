"""Shared helpers for the local WireMask-BBO runners (run_ea.py, run_llm_order.py).

Local addition (not upstream). Reuses the vendored `utils` / `place_db` unchanged.
"""

import contextlib
import io
import os

import utils


def load_env(paths=(".env", "../.env", "../../.env")):
    """Minimal .env loader (no dependency). Reads KEY=VALUE lines and sets them in
    os.environ if not already set. Looks in this dir and up to the repo root, so
    you can keep ANTHROPIC_API_KEY in TFPlace/.env (gitignored). Real shell env
    vars always win. Returns True if a key was loaded."""
    loaded = False
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:  # skip blank/placeholder-only values
                    os.environ[k] = v
                    loaded = True
    return loaded


def quiet(fn, *a, **k):
    """Call fn while swallowing the vendored code's verbose per-decode prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def count_overlaps(placed_macros):
    """Overlapping macro pairs on the grid (edge-touching allowed) -> legality check."""
    rects = [(m["loc_x"], m["loc_y"], m["scaled_x"], m["scaled_y"])
             for m in placed_macros.values()]
    n = 0
    for i in range(len(rects)):
        xi, yi, sxi, syi = rects[i]
        for j in range(i + 1, len(rects)):
            xj, yj, sxj, syj = rects[j]
            if xi < xj + sxj and xj < xi + sxi and yi < yj + syj and yj < yi + syi:
                n += 1
    return n


def maskplace_hpwl(placedb, dataset):
    """MaskPlace's provided placement scored with the paper's cal_hpwl (their units)."""
    path = os.path.join("result", "MaskPlace", "placement", dataset + ".pl")
    if not os.path.exists(path):
        return None
    pm = {}
    with open(path) as f:
        for row in f:
            p = row.split("\t")
            if len(p) < 3:
                continue
            nid = p[0]
            if nid not in placedb.node_info:
                continue
            pm[nid] = {"center_loc_x": float(eval(p[1])), "center_loc_y": float(eval(p[2]))}
    return utils.cal_hpwl(pm, placedb)
