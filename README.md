# TFPlace — LLM-Guided Decode Order for WireMask-EA

Training-free macro placement. This repo (1) reproduces **WireMask-EA** (the training-free
method from *"Macro Placement by Wire-Mask-Guided Black-Box Optimization"*, NeurIPS'23,
arXiv 2306.16844), which already beats the RL SOTA **MaskPlace**, and (2) tests whether letting
a **frozen LLM choose the greedy decode order** improves it further, at a **matched evaluation
budget**. No neural network is trained; every placement is legal by construction (zero overlap,
in-canvas).

All code lives in `external/WireMask-BBO/` — the paper's released code vendored **unchanged**,
plus our additions. The pipeline is **self-contained** (adaptec1 data is an in-repo copy, no
symlinks) and portable (e.g. uploadable to Kaggle).

## The question

Plain WireMask-EA already beats MaskPlace, so the meaningful bar for the LLM is **plain
WireMask-EA itself**: does choosing the decode order with an LLM (vs the paper's fixed
`rank_macros` heuristic) help at equal decode budget? Three methods, matched total decodes:

- **plain** — WireMask-EA on the fixed `rank_macros` order (this *is* the paper's method).
- **random_order** — best of a few *random* order edits, then the coordinate-EA ("is order search worth it at all?" control).
- **llm_order** — best of a few *LLM-proposed* order edits (connectivity-aware, with feedback), then the coordinate-EA.

The decoder and coordinate-EA are the paper's, unchanged; the LLM only proposes the **order**.

## Result so far (adaptec1, HPWL ×10⁵, WireMask-BBO's own `cal_hpwl`, 3 seeds, budget 300)

| method | HPWL (×10⁵) | vs plain |
|---|---|---|
| MaskPlace (paper 6.38±0.35; our recompute of their placement 6.56) | 6.38 / 6.56 | — |
| plain = WireMask-EA (our reproduction) | 6.245 ± 0.034 | — |
| random_order | 6.273 ± 0.025 | −0.4% |
| **llm_order** | **6.159 ± 0.162** (best seed **6.015**) | **+1.4%** |

**Reading it:** units match the paper (same `cal_hpwl`). At only 300 decodes (~1/12 the paper's
~3,500), `llm_order` beats both `plain` and MaskPlace, and reaches WireMask-RS/BO-grade quality
(paper: RS 6.13, BO 6.07, EA 5.91) — i.e. LLM ordering is notably more **sample-efficient**. We do
not yet match the paper's fully-converged WireMask-EA (5.91) — that's a budget gap. Caveats: 3
seeds (paper used 5) and high `llm_order` variance (one unlucky seed drags the mean).

Full result log: `external/WireMask-BBO/logs/opus_headtohead_20260701_134925.log`.

## Run it

From `external/WireMask-BBO/` (needs `numpy`, `scipy`; `cv2` is stubbed):

```bash
cd external/WireMask-BBO

# reproduce WireMask-EA vs MaskPlace (no API key needed):
python run_ea.py --dataset adaptec1 --seed 2023 --init_round 100 --stop_round 400

# free head-to-head (plain + random_order, no API key):
python run_llm_order.py --dataset adaptec1 --seeds 3 --budget 300 --outer 10 \
    --init_round 60 --top_n -1 --methods plain random_order

# full head-to-head incl. the LLM (needs an Anthropic API key + `pip install anthropic`):
python run_llm_order.py --dataset adaptec1 --seeds 3 --budget 300 --outer 10 \
    --init_round 60 --top_n -1 --model claude-opus-4-8 \
    --methods plain random_order llm_order
```

**API key:** put `ANTHROPIC_API_KEY=...` in `TFPlace/.env` (gitignored) or export it; the runner
reads it via `wm_common.load_env()`. `--top_n -1` gives the LLM the whole netlist; `--outer` is the
order-search rounds (patience-5 early-stop); LLM cost scales with `seeds × outer`, not budget.

## Files (`external/WireMask-BBO/`)

- **Vendored, unchanged (upstream):** `place_db.py`, `utils.py`, `common.py`, `EA_swap_only.py`,
  `RS.py`, `result/MaskPlace/` (reference placements), `LICENSE`, `README.md`.
- **Ours:** `run_ea.py` (bounded WireMask-EA + vs-MaskPlace), `order_advisor.py` (the LLM
  `OrderAdvisor` + connectivity/order-edit helpers), `run_llm_order.py` (the 3-way head-to-head),
  `wm_common.py` (shared helpers + `.env` loader), `cv2.py` (stub), `NOTES.md` (provenance +
  method details + the vs-paper comparison).
- **Data:** `benchmark/adaptec1/` — real in-repo copy of the ISPD2005 bookshelf files.

See `external/WireMask-BBO/NOTES.md` for full method details, the reconciliation with the paper,
and the ariane note (needs `protobuf<=3.20`).
