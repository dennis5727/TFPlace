"""LLM-guided decode order for WireMask-EA -- head-to-head vs plain WireMask-EA.

Bar to beat: **plain WireMask-EA** (which already beats MaskPlace). Question: does
choosing the greedy DECODE ORDER with an LLM (vs the fixed ``rank_macros`` heuristic)
improve WireMask-EA at an EQUAL total greedy-decode budget?

Three methods, matched total decodes B:
  * plain        : coordinate-EA on rank_macros order                    (B inner)
  * random_order : R random order-edits (1 decode each) -> best order,
                   then coordinate-EA on it                              (R + (B-R))
  * llm_order    : R LLM-proposed order-edits (iterative, with far-apart
                   feedback; 1 decode each) -> best order, then
                   coordinate-EA on it                                   (R + (B-R))

The decoder / coordinate-EA (the paper's, in ``utils``) are reused UNCHANGED; the
LLM only proposes the order. Legality (0 overlap) is asserted on every reported run.

Usage:
    # no-API smoke (plain + random_order):
    python run_llm_order.py --dataset adaptec1 --seeds 1 --budget 40 --outer 6 \
        --init_round 10 --methods plain random_order

    # full head-to-head (needs ANTHROPIC_API_KEY):
    python run_llm_order.py --dataset adaptec1 --seeds 3 --budget 300 --outer 10 \
        --init_round 60 --methods plain random_order llm_order
"""

import argparse
import random
import statistics
import time

import place_db
import utils
import warm_start
from common import grid_setting, my_inf
from wm_common import quiet, count_overlaps, maskplace_hpwl, load_env
import order_advisor as oa


def _decode(order, placedb, gn, gs, rec):
    """One greedy decode of ``order`` with target coords ``rec``. (placed, hpwl)."""
    return quiet(utils.greedy_placer_with_init_coordinate, order, placedb, gn, gs, rec)


def coordinate_ea(order, placedb, gn, gs, n_init, n_ea, rng, init_fn=None, jitter=8):
    """Paper's coordinate search on a FIXED decode ``order``: n_init init decodes
    (keep best) + n_ea swap-only (1+1)-EA rounds. ``init_fn`` selects the hint source:
    None = utils.random_guiding (noise); else warm_start.connectivity_guiding (spectral,
    first draw jitter 0, rest jittered for diversity). Returns (placed, hpwl, decodes)."""
    best_hpwl, best_rec, best_placed = my_inf, None, None
    for k in range(n_init):
        if init_fn is None:
            rec = quiet(utils.random_guiding, order, placedb, gn, gs)
        else:
            rec = quiet(init_fn, order, placedb, gn, gs, rng, 0 if k == 0 else jitter)
        placed, hpwl = _decode(order, placedb, gn, gs, rec)
        if hpwl < best_hpwl:
            best_hpwl, best_rec, best_placed = hpwl, rec, placed
    rec = best_rec
    for _ in range(n_ea):
        ids = list(rec.keys())
        a, b = random.sample(ids, 2)
        rec[a]["loc_x"], rec[a]["loc_y"], rec[b]["loc_x"], rec[b]["loc_y"] = (
            rec[b]["loc_x"], rec[b]["loc_y"], rec[a]["loc_x"], rec[a]["loc_y"])
        placed, hpwl = _decode(order, placedb, gn, gs, rec)
        if hpwl < best_hpwl:
            best_hpwl, best_placed = hpwl, placed
        else:  # reject -> revert
            rec[a]["loc_x"], rec[a]["loc_y"], rec[b]["loc_x"], rec[b]["loc_y"] = (
                rec[b]["loc_x"], rec[b]["loc_y"], rec[a]["loc_x"], rec[a]["loc_y"])
    return best_placed, best_hpwl, n_init + n_ea


def _links_for_feedback(placedb, hubs, n_links=40):
    """[(i, j, c, name_i, name_j)] strongest hub-hub links, for far_apart feedback."""
    out = []
    for i, j, c in oa.macro_connections(placedb, hubs, top_k=n_links):
        out.append((i, j, c, hubs[i], hubs[j]))
    return out


def outer_order_search(method, placedb, gn, gs, base_order, hubs, rounds, rng,
                       model, max_moves, n_links):
    """Search the decode order (R decodes, 1 per candidate, shared fixed init coords).
    Returns (best_order, decodes_used, advisor_or_None)."""
    fixed_rec = quiet(utils.random_guiding, base_order, placedb, gn, gs)  # same coords for all
    best_order, (best_placed, best_hpwl) = base_order, _decode(base_order, placedb, gn, gs, fixed_rec)

    advisor = None
    links = _links_for_feedback(placedb, hubs, n_links)
    history = [f"baseline rank_macros order: HPWL={best_hpwl:.4e}"]
    if method == "llm_order":
        summary = oa.build_conn_summary(placedb, hubs, gs, top_k_links=n_links)
        advisor = oa.OrderAdvisor(summary, len(hubs), model=model, max_moves=max_moves)

    PATIENCE = 5          # stop the order search after 5 rounds with no improvement
    used = 0              # decodes actually spent (may be < rounds if we stop early)
    no_improve = 0
    for r in range(rounds):
        if method == "random_order":
            moves = oa.random_order_control_edits(len(hubs), rng, n_moves=3)
        else:  # llm_order
            fb = "LONG CONNECTIONS: " + oa.far_apart_pairs(best_placed, links)
            moves = advisor.suggest(best_hpwl, "\n".join(history), fb)
            if not moves:
                break
        order = oa.apply_order_edits(base_order, moves, hubs)
        placed, hpwl = _decode(order, placedb, gn, gs, fixed_rec)
        used += 1
        accepted = hpwl < best_hpwl
        if accepted:
            best_order, best_hpwl, best_placed = order, hpwl, placed
            no_improve = 0
        else:
            no_improve += 1
        history.append(f"moves={moves} -> HPWL={hpwl:.4e} "
                       f"({'accepted' if accepted else 'rejected'})")
        if no_improve >= PATIENCE:   # stalled -> stop spending LLM calls / decodes
            break
    return best_order, used, advisor


def run_once(method, placedb, dataset, gn, gs, budget, outer, init_round, seed,
             model, max_moves, top_n, n_links, init_fn=None, jitter=8):
    random.seed(seed)
    base_order = utils.rank_macros(placedb)
    # top_n <= 0 -> give the LLM the WHOLE netlist (every macro is reorderable).
    n_hubs = len(base_order) if top_n <= 0 else min(top_n, len(base_order))
    hubs = oa.hub_names(placedb, n_hubs)
    rng = random.Random(seed)

    if method == "plain":
        order, outer_used = base_order, 0
        advisor = None
    else:
        order, outer_used, advisor = outer_order_search(
            method, placedb, gn, gs, base_order, hubs, outer, rng, model, max_moves, n_links)

    inner = budget - outer_used
    n_init = min(init_round, max(1, inner))
    n_ea = max(0, inner - n_init)
    placed, hpwl, inner_used = coordinate_ea(order, placedb, gn, gs, n_init, n_ea, rng,
                                             init_fn, jitter)
    ov = count_overlaps(placed)
    return {
        "method": method, "hpwl": hpwl, "legal": ov == 0, "overlaps": ov,
        "decodes": outer_used + inner_used,
        "llm_calls": getattr(advisor, "calls", 0) if advisor else 0,
        "in_tokens": getattr(advisor, "in_tokens", 0) if advisor else 0,
        "out_tokens": getattr(advisor, "out_tokens", 0) if advisor else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adaptec1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=300, help="total greedy decodes per run")
    ap.add_argument("--outer", type=int, default=10, help="order-search rounds (decodes)")
    ap.add_argument("--init_round", type=int, default=60)
    ap.add_argument("--top_n", type=int, default=-1,
                    help="macros the LLM may reorder; -1 = the WHOLE netlist (all macros)")
    ap.add_argument("--links", type=int, default=120,
                    help="how many of the strongest macro-macro links to show the LLM")
    ap.add_argument("--methods", nargs="+", default=["plain", "random_order", "llm_order"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--max_moves", type=int, default=6,
                    help="max order-edits per call; LLM is told to use only as many as help")
    ap.add_argument("--init", choices=["random", "spectral"], default="random",
                    help="coordinate-hint source: random noise or spectral warm start")
    ap.add_argument("--jitter", type=int, default=8,
                    help="spectral per-draw offset (cells) for best-of-N diversity")
    args = ap.parse_args()

    load_env()  # pick up ANTHROPIC_API_KEY from TFPlace/.env if present
    placedb = place_db.PlaceDB(args.dataset)
    gn = grid_setting[args.dataset]["grid_num"]
    gs = grid_setting[args.dataset]["grid_size"]
    mp = maskplace_hpwl(placedb, args.dataset)
    init_fn = warm_start.connectivity_guiding if args.init == "spectral" else None

    print(f"\nLLM-order head-to-head: {args.dataset}  budget={args.budget} "
          f"outer={args.outer} init={args.init_round} top_n={args.top_n} "
          f"seeds={args.seeds}\nMaskPlace HPWL={mp:.4e}\n" if mp else "")

    results = {m: [] for m in args.methods}
    t0 = time.time()
    for method in args.methods:
        for seed in range(args.seeds):
            r = run_once(method, placedb, args.dataset, gn, gs, args.budget, args.outer,
                         args.init_round, 1000 + seed, args.model, args.max_moves,
                         args.top_n, args.links, init_fn, args.jitter)
            results[method].append(r)
            print(f"  {method:13s} seed={seed} HPWL={r['hpwl']:.4e} legal={r['legal']} "
                  f"decodes={r['decodes']} llm_calls={r['llm_calls']}")

    print(f"\n=== summary ({args.dataset}, matched budget {args.budget} decodes) ===")
    print(f"{'method':14s} {'HPWL mean+-std':>24s} {'legal':>6s} {'vs plain':>9s} {'vs MaskPlace':>13s}")
    plain_mean = None
    for method in args.methods:
        hs = [r["hpwl"] for r in results[method]]
        m = statistics.mean(hs)
        s = statistics.pstdev(hs) if len(hs) > 1 else 0.0
        if method == "plain":
            plain_mean = m
        vp = f"{100*(plain_mean-m)/plain_mean:+.1f}%" if plain_mean else "-"
        vmp = f"{100*(mp-m)/mp:+.1f}%" if mp else "-"
        legal = all(r["legal"] for r in results[method])
        print(f"{method:14s} {m:11.4e}+-{s:8.2e} {str(legal):>6s} {vp:>9s} {vmp:>13s}")
    print(f"\nwall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
