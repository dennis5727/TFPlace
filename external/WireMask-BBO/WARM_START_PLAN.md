# Plan: Spectral warm-start init + tightened LLM-order loop

Two **orthogonal levers** feed the same greedy engine
`utils.greedy_placer_with_init_coordinate(order, placedb, gn, gs, hint)`:

| lever | controls | set by (today) |
|---|---|---|
| **order** (`node_id_ls`) | sequence macros are placed in | `rank_macros` / random / LLM |
| **hint** (`place_record`) | target cell each macro is seeded at (tie-break/anchor) | `random_guiding` (noise) |

`init` and `ea` use the **same engine**; only the hint source differs (init = fresh
random hint; ea = swap two hint entries). So a better hint is a **drop-in replacement
for `random_guiding`** — the decoder and EA are untouched.

This plan (A) adds a connectivity-based **spectral warm-start** hint and tests it
**without the LLM** (`rank_macros` + spectral) as a control — *"does a better hint help
on its own?"*; then (B, next step) stacks it with `llm_order` to see if the order-gain
and coordinate-gain **add up**. It also (C) tightens the LLM loop (patience 3, max 15
calls, reorder **1–6 macros, using only as many as actually help**, stop on a 3×
repeated order) and (D) restricts the LLM to the top hubs.

---

## Why this should help (hypothesis)

- The hint only acts as the greedy **tie-breaker** ([`utils.py:301-309`]), so it dominates
  the early/unconstrained **anchor** macros and then **cascades** to the rest.
- **Init picks the basin; EA descends within it.** In the current runs EA is still
  descending at the budget limit (not converged), so a **better basin** (better anchor)
  should survive to the final HPWL rather than being washed out.
- A spectral layout places **connected macros near each other before the greedy runs**,
  so the tie-breaks pull them together → better anchoring → (hypothesis) lower HPWL at
  matched budget.

Honest limits: the hint only *biases* placement (the greedy still overrides with wire
cost), so expect an improvement, not a transformation. Two things to get right:
**diversity** (a single deterministic start loses best-of-N) and **collapse** (pure
attraction clumps everything at the centroid). Spectral embedding addresses collapse
(eigenvectors spread naturally); per-draw **jitter** restores diversity.

---

## A. Spectral warm start (no-LLM control)

New module `warm_start.py`. `connectivity_guiding` mirrors `random_guiding`'s **output
format** exactly (the greedy only reads `loc_x`/`loc_y` per macro), so it is a drop-in.

```python
# warm_start.py -- connectivity-based coordinate hint (drop-in for utils.random_guiding)
import math
import numpy as np
from itertools import combinations

def _macro_weight_matrix(placedb, order):
    """Symmetric macro-macro weight = # shared nets (co-membership count)."""
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
    """2-D layout from the graph Laplacian: eigenvectors 1 & 2 (skip trivial 0).
    Connected macros get nearby coordinates; eigenvectors spread nodes (no collapse)."""
    d = W.sum(axis=1)
    L = np.diag(d) - W                       # unnormalized Laplacian
    vals, vecs = np.linalg.eigh(L)           # ascending; 543x543 is instant
    return vecs[:, 1], vecs[:, 2]            # Fiedler + next

def _to_grid(v, grid_num, margin=0.05):
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.full_like(v, grid_num / 2.0, dtype=float)
    s = (v - lo) / (hi - lo)
    return (margin + s * (1 - 2 * margin)) * grid_num

# cache the (expensive-once) embedding per placedb; keyed by id()
_EMB = {}

def connectivity_guiding(order, placedb, grid_num, grid_size, rng=None, jitter=0.0):
    """Spectral coordinate hint in random_guiding's dict format.
    jitter (cells): per-draw uniform offset for diversity (0 = pure spectral)."""
    key = id(placedb)
    if key not in _EMB:
        W = _macro_weight_matrix(placedb, order)
        ex, ey = _spectral_coords(W)
        _EMB[key] = (list(order),
                     _to_grid(ex, grid_num), _to_grid(ey, grid_num))
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
        lx = int(min(max(0, round(cx)), grid_num - sx))   # clamp to legal BL range
        ly = int(min(max(0, round(cy)), grid_num - sy))
        placed[node_id] = {
            "scaled_x": sx, "scaled_y": sy, "loc_x": lx, "loc_y": ly, "x": x, "y": y,
            "center_loc_x": grid_size * lx + 0.5 * x,
            "center_loc_y": grid_size * ly + 0.5 * y,
            "bottom_left_x": lx * grid_size, "bottom_left_y": ly * grid_size}
    return placed
```

### Wire an `--init {random,spectral}` toggle into the coordinate search

The change is confined to *how the init hint is produced*. In **`coordinate_ea`**
(`run_llm_order.py`) and **`coordinate_ea_traced`** (`run_llm_order_deep.py`), replace the
init-hint line with a pluggable `init_fn`. First draw uses jitter 0 (pure spectral); the
rest jitter for diversity, preserving best-of-N.

```python
# in coordinate_ea / coordinate_ea_traced signature: add init_fn=None, jitter=8
for k in range(n_init):
    if init_fn is None:                       # --init random (baseline)
        rec = quiet(utils.random_guiding, order, placedb, gn, gs)
    else:                                     # --init spectral (warm start)
        rec = quiet(init_fn, order, placedb, gn, gs, rng, 0 if k == 0 else jitter)
    placed, hpwl = _decode(order, placedb, gn, gs, rec)
    ...
# ea phase is UNCHANGED (still swaps hint entries from the best init hint)
```

```python
# argparse (both runners)
ap.add_argument("--init", choices=["random", "spectral"], default="random")
ap.add_argument("--jitter", type=int, default=8, help="spectral per-draw offset (cells)")
# build init_fn once:
import warm_start
init_fn = warm_start.connectivity_guiding if args.init == "spectral" else None
# pass init_fn/args.jitter into every coordinate_ea(_traced) call
```

> Note: `outer_order_search` in `run_llm_order.py` uses a single `fixed_rec` probe for the
> cheap order search — leave that as `random_guiding` (it just needs a *shared* fixed
> coord set across candidates; the warm start belongs to the real coord-EA). The deep
> runner has no such probe, so nothing else to touch there.

### Experiment A (no-LLM control)

```bash
# baseline: rank_macros + random init
python run_llm_order.py --dataset adaptec1 --seeds 3 --budget 300 --outer 0 \
    --init_round 60 --methods plain --init random
# warm start: rank_macros + spectral init  (SAME order, SAME budget)
python run_llm_order.py --dataset adaptec1 --seeds 3 --budget 300 --outer 0 \
    --init_round 60 --methods plain --init spectral
```
**Read:** if spectral's mean HPWL < random's at matched budget → a better hint helps on
its own (basin quality), independent of the LLM. (`--outer 0` = no order search, so this
isolates the hint lever.)

---

## B. Combine with `llm_order` (next step)

Same toggle, method `llm_order`. This is the 2×2: {random, spectral} × {rank_macros, LLM}.

```bash
python run_llm_order.py --dataset adaptec1 --seeds 3 --budget 300 --outer 10 \
    --init_round 60 --top_n 60 --model claude-opus-4-8 \
    --methods plain llm_order --init spectral
```
**Read:** does `llm_order + spectral` beat both `llm_order + random` (order gain) and
`plain + spectral` (hint gain)? If the two gains **add up**, the levers are complementary.

Full 2×2 to report:

| | random init | spectral init |
|---|---|---|
| **rank_macros** | plain (paper) | Exp A |
| **LLM order** | current llm_order | Exp B (target) |

---

## C. Tighten the LLM-order loop

Applies to `run_llm_order_deep.py` (per-order full coord search) and defaults in both.

1. **patience 3** (was 5): stop after 3 non-improving calls.
2. **max_calls 15** (was 30).
3. **reorder 1–6 macros, minimal set**: `max_moves = 6` (was 8), AND the prompt (SYSTEM +
   `_user_msg`) now tells the LLM to **use the fewest moves that help** — "if one move is
   enough, return just one; do not pad up to the maximum." `_parse_pairs` still caps at
   `max_moves` as a hard ceiling.
4. **Stop on a 3× repeated order** — *before* spending the expensive init+ea. This directly
   fixes the observed failure mode (the LLM re-proposed the identical move set for calls
   5–11, each triggering a wasted 400-decode search under noisy scoring).

```python
# run_llm_order_deep.py, top of the call loop
from collections import Counter
seen_orders = Counter()
...
for call in range(1, args.max_calls + 1):
    fb = "LONG CONNECTIONS: " + oa.far_apart_pairs(best_placed, links)
    moves = advisor.suggest(best_hpwl, "\n".join(history_text), fb)
    if not moves:
        log(f"call {call:2d}  LLM returned no moves -> stopping.", fh); break

    order = oa.apply_order_edits(base_order, moves, hubs)
    sig = tuple(order)                       # signature of the *resulting* order
    seen_orders[sig] += 1
    if seen_orders[sig] >= 3:               # same order proposed 3x -> converged
        log(f"call {call:2d}  LLM repeated the same order 3x -> converged, stopping "
            f"(no init/ea spent).", fh)
        break

    placed, hpwl, traj = coordinate_ea_traced(
        order, placedb, gn, gs, args.n_init, args.n_ea, rng, init_fn, args.jitter)
    ...  # accept/reject + patience(3) unchanged
```

```python
# defaults
ap.add_argument("--patience", type=int, default=3)     # was 5
ap.add_argument("--max_calls", type=int, default=15)   # was 30
ap.add_argument("--max_moves", type=int, default=6)    # was 8   (reorder 1-6, minimal set)
```

> Design choice: keying the signature on the *resulting order* (`tuple(order)`) is robust —
> different move sets that yield the same order still count as a repeat. The check runs
> **before** the coord search, so a 3rd duplicate costs 1 cheap LLM call, not 400 decodes.
> Both stop conditions coexist: **patience-3** (no improvement) OR **repeat-3** (stuck),
> whichever fires first.

---

## D. Restrict the LLM to the top hubs (`--top_n 60`)

With reorder ≤ 6 and ≤ 15 calls, the LLM touches only a couple dozen hub slots total, and
the far-apart/strongest-link feedback only ever surfaces **high-degree hubs**. Feeding all
543 macros (668-line summary) is mostly dead context. Use `--top_n 60`:
- sharper, cheaper prompt; feedback is all actionable hubs;
- negligible loss (the low-degree tail is never actionable at this move budget).

Keep `--top_n -1` available for a completeness ablation, but the primary runs use 60.

---

## Files touched

- **new** `warm_start.py` — `connectivity_guiding` (+ helpers).
- `run_llm_order.py` — `--init`/`--jitter`; `init_fn` threaded into `coordinate_ea`.
- `run_llm_order_deep.py` — `--init`/`--jitter` into `coordinate_ea_traced`; repeat-3 stop;
  default `patience 3`, `max_calls 15`, `max_moves 6`.
- (docs) this file.

Nothing in the **decoder** (`utils.greedy_placer_with_init_coordinate`) or the **EA swap
loop** changes — legality (0 overlap) stays automatic, and the paper's engine is intact.

## Validation / smoke tests

```bash
# 1. spectral hint produces a legal placement, format matches random_guiding:
python -c "import place_db,utils,warm_start; from common import grid_setting as G; \
d=place_db.PlaceDB('adaptec1'); o=utils.rank_macros(d); \
import random; r=random.Random(0); \
rec=warm_start.connectivity_guiding(o,d,G['adaptec1']['grid_num'],G['adaptec1']['grid_size'],r,0); \
p,h=utils.greedy_placer_with_init_coordinate(o,d,G['adaptec1']['grid_num'],G['adaptec1']['grid_size'],rec); \
from wm_common import count_overlaps; print('HPWL',h,'overlaps',count_overlaps(p))"

# 2. deep-runner plumbing (repeat-stop + spectral) with mock LLM, no API key:
python run_llm_order_deep.py --mock --max_calls 5 --n_init 5 --n_ea 5 --init spectral

# 3. Exp A (no-LLM control), then Exp B (see sections above).
```

## Risks / caveats

- **Collapse**: if `_to_grid` range is tiny (weakly connected graph), macros bunch centrally
  → mitigated by `margin` + jitter; watch the HPWL of the first (jitter-0) init.
- **Diversity loss**: if jitter too small, best-of-N init degenerates to one point → tune
  `--jitter` (start 8 cells ≈ a few % of the 160 grid).
- **Noisy scoring still present** (same order → different HPWL): the repeat-3 stop *bounds*
  wasted re-evaluations but doesn't average the noise. A cleaner future fix is to score each
  order over k coord-seeds; out of scope here.
- **Eigendecomposition cost**: 543×543 `eigh` is instant and cached; for much larger designs
  switch to sparse `scipy.sparse.linalg.eigsh`.
```
