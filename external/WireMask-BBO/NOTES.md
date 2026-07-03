# Vendored WireMask-BBO (the paper's released pipeline)

Source: https://github.com/lamda-bbo/WireMask-BBO (NeurIPS'23, "Macro Placement by
Wire-Mask-Guided Black-Box Optimization", arXiv 2306.16844). Core files copied
**unchanged**: `place_db.py`, `utils.py`, `common.py`, `EA_swap_only.py`, `RS.py`
(+ `result/MaskPlace/` reference placements, `LICENSE`, `README.md`).

We use this to reproduce the paper's own training-free WireMask-EA number and its
head-to-head vs MaskPlace, **in the paper's own metric and setup** — the faithful
"adopt the paper's released pipeline" path.

## Local additions (not from upstream)
- `cv2.py` — a stub; upstream `utils.py` imports cv2 only for a placement-image
  dump (`write_placement_and_overlap`), which is off the EA/HPWL path. Avoids
  installing opencv. Delete it and `pip install opencv-python` for the viz.
- `run_ea.py` — a clean runner that reuses the upstream functions unchanged,
  bounds the budget (`--init_round`, `--stop_round`), tracks best HPWL on their
  `cal_hpwl`, checks legality (0 macro overlaps), and prints the vs-MaskPlace
  head-to-head. (Upstream `EA_swap_only.py` defaults `stop_round` to infinity and
  reports via CSV files; `run_ea.py` is just a bounded, self-reporting wrapper.)
- `benchmark/adaptec1` — a **real in-repo copy** of the ISPD2005 adaptec1 bookshelf
  files (`.nodes/.nets/.pl/.scl/.aux/.wts`, ~41 MB), so the pipeline is
  self-contained and portable (e.g. uploadable to Kaggle) with no external symlink.
  For other datasets, run upstream `python ispd2005.py` (downloads ISPD2005) or drop
  their bookshelf files into `benchmark/<dataset>/` the same way.

## Dependencies
`numpy`, `scipy` (already present). Runs on modern numpy/scipy despite the pinned
1.15 in `requirements.txt`. `torch`/`gpytorch` are only needed for WireMask-BO
(`BO.py`, `TuRBO/`), which we did not vendor. cv2 is stubbed.

## Run
```bash
cd external/WireMask-BBO
python run_ea.py --dataset adaptec1 --seed 2023 --init_round 100 --stop_round 400
```

## Why this matters — reconciling with our engine (see ../../README.md)

Our `tfplace/` EA runs over the project's vendored greedy engine. That engine's
greedy floor (topology order, grid 224) is ~1.47e6; our EA pushes it to ~8.2e5.
This pipeline instead uses the **paper's decoder** and reaches MaskPlace-grade
HPWL almost immediately. The gap was never the EA algorithm — it is the decoder:

| difference | our engine (`tfplace/`) | paper (`external/`) |
|---|---|---|
| grid | 224 × 224 | **160 × 160, grid_size 72** |
| HPWL units | comp_res, physical | their `cal_hpwl`, physical (different scale) |
| decode order | topology (frontier-growth) | **`rank_macros`: by descending net-area-sum** |
| greedy floor (1 decode) | ~1.47e6 | **~7.6e5 (already near MaskPlace)** |
| metric reported | HPWL + MST | HPWL |

MaskPlace adaptec1 via their `cal_hpwl` = **6.556e5** (paper ~6.38e5). A single
random-init decode here is ~7.6e5; a tiny 10-eval run reached 6.64e5 (−1.2% off
MaskPlace). Absolute HPWL here is NOT comparable to our engine's numbers
(different grid/units) — the valid comparison is WireMask-EA vs MaskPlace
**within this pipeline**.

## LLM-guided decode order (local experiment)

Since plain WireMask-EA already beats MaskPlace, the bar for the LLM is **plain
WireMask-EA itself**: does choosing the greedy **decode order** with an LLM (vs the
fixed `rank_macros` heuristic) improve it at an **equal total greedy-decode budget**?

- `order_advisor.py` — `OrderAdvisor` (Anthropic, mirrors `PromoteAdvisor`) proposes
  `[a,b]` order-edits over the top-N hub macros; `apply_order_edits` builds the decode
  order; `far_apart_pairs` is the feedback signal. `hub_names`/`macro_connections`
  computed from this pipeline's `net_info`.
- `run_llm_order.py` — three methods at matched total decodes B:
  `plain` (rank_macros), `random_order` (R random order-edits → best → coord-EA),
  `llm_order` (R LLM edits, iterative w/ feedback → best → coord-EA). Reuses the
  paper's decoder/coord-EA unchanged; asserts 0 overlaps; reports HPWL mean±std and
  vs-plain / vs-MaskPlace. `random_order` is the "is the LLM beating dumb order
  search?" control.

**API key:** put `ANTHROPIC_API_KEY=sk-...` in `TFPlace/.env` (gitignored). The
runner calls `wm_common.load_env()` which loads it (no `python-dotenv` dependency).
A real shell env var still wins. Also `pip install anthropic`.

**Whole netlist:** `--top_n -1` (the default) gives the LLM **every** macro to
reorder (all 543 for adaptec1); the connectivity summary is ~4k tokens and is
prompt-cached across calls. Use `--top_n 60` to restrict to the top hubs instead.
`--links` sets how many of the strongest macro-macro connections are shown.

Run (from this dir):
```bash
# no-API controls only:
python run_llm_order.py --dataset adaptec1 --seeds 1 --budget 40 --outer 6 \
    --init_round 10 --methods plain random_order
# full head-to-head (whole netlist to the LLM):
python run_llm_order.py --dataset adaptec1 --seeds 3 --budget 300 --outer 10 \
    --init_round 60 --top_n -1 --methods plain random_order llm_order
```

Honest expectation: `rank_macros` is already good and the coord-EA self-corrects, so
LLM-order may only tie plain WireMask-EA. Any outcome (win / tie / loss vs plain) is a
valid finding; the matched-budget design measures it cleanly.

No-API smoke (adaptec1, 1 seed, budget=20, outer=5, init=6): plain 6.944e5 vs
**random_order 6.491e5 (−6.5% vs plain, legal)** — even *random* order-search improves
plain at matched budget, i.e. the order lever has real signal. Tiny-budget/1-seed, not
conclusive; run the full multi-seed head-to-head (incl. `llm_order`) to compare the LLM
against this random-order control.

## Deep LLM-order variant (`run_llm_order_deep.py`)

A second, heavier driver that answers a sharper question: instead of scoring each
LLM-proposed order by a *single* decode under one frozen coordinate set (as
`run_llm_order.py` does during its cheap order search), **every** proposed order gets its
own **full coordinate search** (`--n_init` random-init decodes + `--n_ea` swap-only
(1+1)-EA rounds = 400 decodes/order by default). It loops up to `--max_calls` LLM
proposals on **one seed**, keeps the best HPWL across proposals, and stops after
`--patience` non-improving calls. This removes the "an order that looks best under one
coord set may not survive full refinement" confound behind `llm_order`'s seed variance.

`llm_order`-only; reuses the paper's decoder and the `order_advisor` helpers unchanged.
Every call is logged (text + PNG plots: per-call coord-EA convergence, best-HPWL-vs-call,
final placement) under `results/<run>/` (gitignored). LLM cost is tracked and printed
(warn-only at $1.55; opus-4.8 ≈ $5/$25 per MTok, so ~30 calls is well within budget).

```bash
# free plumbing test (random edits stand in for the LLM; no API key):
python run_llm_order_deep.py --mock --max_calls 3 --n_init 5 --n_ea 5
# the experiment (needs ANTHROPIC_API_KEY in TFPlace/.env):
python run_llm_order_deep.py --max_calls 30 --patience 5 --n_init 100 --n_ea 300 \
    --model claude-opus-4-8
```

At ~4s/decode a full 30-call run is ~13h — intended for a lab PC, not a bounded notebook.

### Reproduced result (adaptec1, seed 2023, 100 init + 400 EA = 500 evals)
| method | HPWL (their units) | legal |
|---|---|---|
| MaskPlace (provided placement) | 6.556e5 | — |
| **WireMask-EA (this pipeline)** | **6.299e5** | 0 overlaps |

→ **WireMask-EA beats MaskPlace by 3.9%**, training-free, legal. This
reproduces the paper's core claim. (Wall ~27 min; the curve improves steadily —
more EA rounds would widen the margin, matching the paper's "run for minutes".)
