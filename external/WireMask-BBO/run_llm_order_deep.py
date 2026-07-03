"""Deep LLM-order pipeline for WireMask-EA -- one full coord-EA per proposed order.

Difference from run_llm_order.py: instead of scoring each LLM-proposed decode order
by a single greedy decode under one frozen coordinate set, EVERY proposed order gets
its own FULL coordinate search (n_init random-init decodes + n_ea swap-only (1+1)-EA
rounds). Each order is therefore scored the way it would actually be deployed. We loop
up to --max_calls LLM proposals (single seed), keep the best HPWL across proposals, and
stop early after --patience non-improving calls.

The decoder / coordinate-EA are the paper's (in ``utils``), reused UNCHANGED; the LLM
only proposes the ORDER. Legality (0 macro overlap) is asserted on the final placement.
Every call is logged (text + PNG plots). LLM cost is tracked and printed (warn-only).

Deploy: clone the repo, run from this directory on a machine with time to spare
(a full 30-call run is ~13h at ~4s/decode; no wall-clock limit assumed).

Usage:
    # plumbing test, no API key / no cost (random edits stand in for the LLM):
    python run_llm_order_deep.py --mock --max_calls 3 --n_init 5 --n_ea 5

    # tiny real run (confirm API + cost meter):
    python run_llm_order_deep.py --max_calls 2 --n_init 5 --n_ea 5 --model claude-opus-4-8

    # the experiment (needs ANTHROPIC_API_KEY in TFPlace/.env):
    python run_llm_order_deep.py --max_calls 30 --patience 5 --n_init 100 --n_ea 300 \
        --model claude-opus-4-8
"""

import argparse
import os
import random
import time

import place_db
import utils
from common import grid_setting, my_inf
from wm_common import quiet, count_overlaps, maskplace_hpwl, load_env
import order_advisor as oa

# --- opus-4.8 pricing ($ / 1M tokens); verified via the claude-api skill 2026-07 --- #
# input $5, output $25, cache write (5m TTL) 1.25x, cache read 0.1x.
PRICE = {
    "claude-opus-4-8": (5.00, 25.00, 0.50),   # (input, output, cache_read) per MTok
    "claude-opus-4-7": (5.00, 25.00, 0.50),
    "claude-sonnet-4-6": (3.00, 15.00, 0.30),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}
BUDGET_WARN = 1.55  # print a warning once running cost crosses this (no auto-stop)


def advisor_cost(advisor, model):
    """Approximate running LLM spend ($) from the advisor's token counters.

    Uses input/output/cache-read rates; the one-time cache WRITE (~4k tokens on the
    first call, ~$0.025 for opus) is not tracked separately, so this slightly
    under-counts -- fine for a warn-only meter."""
    if advisor is None:
        return 0.0
    pin, pout, pcache = PRICE.get(model, PRICE["claude-opus-4-8"])
    return (advisor.in_tokens * pin
            + advisor.out_tokens * pout
            + advisor.cache_read_tokens * pcache) / 1e6


def _decode(order, placedb, gn, gs, rec):
    """One greedy decode of ``order`` with target coords ``rec``. (placed, hpwl)."""
    return quiet(utils.greedy_placer_with_init_coordinate, order, placedb, gn, gs, rec)


def coordinate_ea_traced(order, placedb, gn, gs, n_init, n_ea, rng):
    """Paper's coordinate search on a FIXED decode ``order``, WITH a trajectory.

    n_init random-init decodes (keep best) + n_ea swap-only (1+1)-EA rounds (revert on
    reject). Returns (best_placed, best_hpwl, trajectory) where trajectory is a list of
    (decode_idx, phase, best_hpwl_so_far), phase in {"init", "ea"} -- for plotting."""
    best_hpwl, best_rec, best_placed = my_inf, None, None
    traj = []
    idx = 0
    for _ in range(n_init):
        rec = quiet(utils.random_guiding, order, placedb, gn, gs)
        placed, hpwl = _decode(order, placedb, gn, gs, rec)
        if hpwl < best_hpwl:
            best_hpwl, best_rec, best_placed = hpwl, rec, placed
        idx += 1
        traj.append((idx, "init", best_hpwl))
    rec = best_rec
    for _ in range(n_ea):
        ids = list(rec.keys())
        a, b = random.sample(ids, 2)
        rec[a]["loc_x"], rec[a]["loc_y"], rec[b]["loc_x"], rec[b]["loc_y"] = (
            rec[b]["loc_x"], rec[b]["loc_y"], rec[a]["loc_x"], rec[a]["loc_y"])
        placed, hpwl = _decode(order, placedb, gn, gs, rec)
        if hpwl < best_hpwl:
            best_hpwl, best_placed = hpwl, placed
        else:  # reject -> revert the swap
            rec[a]["loc_x"], rec[a]["loc_y"], rec[b]["loc_x"], rec[b]["loc_y"] = (
                rec[b]["loc_x"], rec[b]["loc_y"], rec[a]["loc_x"], rec[a]["loc_y"])
        idx += 1
        traj.append((idx, "ea", best_hpwl))
    return best_placed, best_hpwl, traj


def _links_for_feedback(placedb, hubs, n_links):
    """[(i, j, c, name_i, name_j)] strongest hub-hub links, for far_apart feedback."""
    out = []
    for i, j, c in oa.macro_connections(placedb, hubs, top_k=n_links):
        out.append((i, j, c, hubs[i], hubs[j]))
    return out


# --------------------------------------------------------------------------- #
# plotting (headless; degrades gracefully if matplotlib is missing)
# --------------------------------------------------------------------------- #
def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # pragma: no cover - plotting is optional
        print(f"  [plot] matplotlib unavailable ({e}); skipping plots "
              f"(pip install matplotlib)")
        return None


def plot_convergence(traj, call_no, n_init, best_hpwl, accepted, outdir):
    """Per-call coord-EA convergence: best-HPWL-so-far vs decode index, init/ea shaded."""
    plt = _get_plt()
    if plt is None or not traj:
        return
    xs = [t[0] for t in traj]
    ys = [t[2] for t in traj]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "-", color="#1f77b4", lw=1.5)
    ax.axvspan(0, n_init, color="#cfe8ff", alpha=0.5, label=f"init ({n_init} decodes)")
    ax.axvspan(n_init, xs[-1] if xs else n_init, color="#ffe0cc", alpha=0.5,
               label=f"ea ({len(traj) - n_init} decodes)")
    tag = "accepted" if accepted else "rejected"
    ax.set_title(f"call {call_no}: coord-EA convergence  "
                 f"final={best_hpwl:.4e} ({tag})")
    ax.set_xlabel("decode # within this order's coord-EA")
    ax.set_ylabel("best HPWL so far")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"call_{call_no:02d}_convergence.png"), dpi=110)
    plt.close(fig)


def plot_best_vs_call(history_rows, mp, outdir):
    """Incumbent best HPWL vs LLM call number (call 0 = baseline). Refreshed each call."""
    plt = _get_plt()
    if plt is None or not history_rows:
        return
    calls = [r["call"] for r in history_rows]
    running = [r["running_best"] for r in history_rows]
    this = [r["hpwl"] for r in history_rows]
    acc = [r["call"] for r in history_rows if r["accepted"]]
    accy = [r["running_best"] for r in history_rows if r["accepted"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(calls, this, "o--", color="#aaaaaa", ms=4, label="this order's HPWL")
    ax.plot(calls, running, "-", color="#d62728", lw=2, label="running best")
    if acc:
        ax.plot(acc, accy, "*", color="#2ca02c", ms=12, label="accepted")
    if mp:
        ax.axhline(mp, color="#9467bd", ls=":", label=f"MaskPlace {mp:.3e}")
    ax.set_title("best HPWL vs LLM call")
    ax.set_xlabel("LLM call (0 = baseline rank_macros order)")
    ax.set_ylabel("HPWL")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "best_vs_call.png"), dpi=110)
    plt.close(fig)


def plot_placement(placed, gn, best_hpwl, outdir):
    """Draw the best layout as macro rectangles on the grid."""
    plt = _get_plt()
    if plt is None or not placed:
        return
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(6, 6))
    for m in placed.values():
        ax.add_patch(Rectangle((m["loc_x"], m["loc_y"]), m["scaled_x"], m["scaled_y"],
                               facecolor="#4c72b0", edgecolor="white", lw=0.2, alpha=0.8))
    ax.set_xlim(0, gn)
    ax.set_ylim(0, gn)
    ax.set_aspect("equal")
    ax.set_title(f"best placement  HPWL={best_hpwl:.4e}  (0 overlaps)")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "best_placement.png"), dpi=120)
    plt.close(fig)


def save_placement_pl(placed, gs, path):
    """Write the best placement as a simple .pl-like file (node, bottom_left_x/y)."""
    with open(path, "w") as f:
        f.write("# node_id\tbottom_left_x\tbottom_left_y\t: /FIXED (deep LLM-order)\n")
        for nid, m in placed.items():
            f.write(f"{nid}\t{m['bottom_left_x']}\t{m['bottom_left_y']}\t: N /FIXED\n")


def log(msg, fh):
    """Print to stdout and append to the run log file."""
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def run_deep(args):
    load_env()  # pick up ANTHROPIC_API_KEY from TFPlace/.env if present
    random.seed(args.seed)
    rng = random.Random(args.seed)

    outdir = args.outdir or os.path.join(
        "results", f"llm_deep_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    fh = open(os.path.join(outdir, "run.log"), "w")

    placedb = place_db.PlaceDB(args.dataset)
    gn = grid_setting[args.dataset]["grid_num"]
    gs = grid_setting[args.dataset]["grid_size"]
    mp = maskplace_hpwl(placedb, args.dataset)

    base_order = utils.rank_macros(placedb)
    n_hubs = len(base_order) if args.top_n <= 0 else min(args.top_n, len(base_order))
    hubs = oa.hub_names(placedb, n_hubs)
    links = _links_for_feedback(placedb, hubs, args.links)

    mode = "MOCK (random edits, no API)" if args.mock else f"LLM ({args.model})"
    log(f"\n=== deep LLM-order : {args.dataset}  seed={args.seed} ===", fh)
    log(f"mode={mode}  max_calls={args.max_calls} patience={args.patience} "
        f"n_init={args.n_init} n_ea={args.n_ea} (400/call) top_n={args.top_n} "
        f"hubs={n_hubs}", fh)
    if mp:
        log(f"MaskPlace HPWL = {mp:.4e}", fh)
    log(f"outdir = {outdir}\n", fh)

    advisor = None
    if not args.mock:
        summary = oa.build_conn_summary(placedb, hubs, gs, top_k_links=args.links)
        advisor = oa.OrderAdvisor(summary, len(hubs), model=args.model,
                                  max_moves=args.max_moves)

    t0 = time.time()

    # --- call 0: baseline rank_macros order, full coord-EA (not an LLM call) ------
    placed, hpwl, traj = coordinate_ea_traced(
        base_order, placedb, gn, gs, args.n_init, args.n_ea, rng)
    best_order, best_hpwl, best_placed = base_order, hpwl, placed
    total_decodes = args.n_init + args.n_ea
    history_text = [f"baseline rank_macros order: HPWL={best_hpwl:.4e}"]
    rows = [{"call": 0, "moves": "(baseline)", "hpwl": hpwl, "accepted": True,
             "running_best": best_hpwl}]
    log(f"call  0  baseline           HPWL={hpwl:.4e}  running_best={best_hpwl:.4e}  "
        f"decodes={total_decodes}", fh)
    plot_convergence(traj, 0, args.n_init, hpwl, True, outdir)
    plot_best_vs_call(rows, mp, outdir)

    # --- calls 1..max_calls: each proposes an order, full coord-EA scores it -------
    no_improve = 0
    for call in range(1, args.max_calls + 1):
        fb = "LONG CONNECTIONS: " + oa.far_apart_pairs(best_placed, links)
        if args.mock:
            moves = oa.random_order_control_edits(len(hubs), rng, n_moves=3)
        else:
            moves = advisor.suggest(best_hpwl, "\n".join(history_text), fb)
            if not moves:
                log(f"call {call:2d}  LLM returned no moves -> stopping.", fh)
                break

        order = oa.apply_order_edits(base_order, moves, hubs)
        placed, hpwl, traj = coordinate_ea_traced(
            order, placedb, gn, gs, args.n_init, args.n_ea, rng)
        total_decodes += args.n_init + args.n_ea
        accepted = hpwl < best_hpwl
        if accepted:
            best_order, best_hpwl, best_placed = order, hpwl, placed
            no_improve = 0
        else:
            no_improve += 1
        history_text.append(
            f"moves={moves} -> HPWL={hpwl:.4e} "
            f"({'accepted' if accepted else 'rejected'})")
        rows.append({"call": call, "moves": str(moves), "hpwl": hpwl,
                     "accepted": accepted, "running_best": best_hpwl})

        cost = advisor_cost(advisor, args.model)
        warn = "  !! OVER BUDGET" if cost > BUDGET_WARN else ""
        calls_n = getattr(advisor, "calls", 0) if advisor else 0
        log(f"call {call:2d}  n_moves={len(moves):<2d} moves={str(moves):<28.28s} "
            f"HPWL={hpwl:.4e}  {'ACCEPT' if accepted else 'reject'}  "
            f"running_best={best_hpwl:.4e}  decodes={total_decodes}  "
            f"llm_calls={calls_n}  ${cost:.3f}{warn}", fh)
        fh.write(f"    full moves (call {call}, {len(moves)} edits): {moves}\n")
        fh.flush()
        plot_convergence(traj, call, args.n_init, hpwl, accepted, outdir)
        plot_best_vs_call(rows, mp, outdir)

        if no_improve >= args.patience:
            log(f"\npatience {args.patience} reached (no improvement) -> stopping.", fh)
            break

    # --- finalize --------------------------------------------------------------- #
    ov = count_overlaps(best_placed)
    assert ov == 0, f"ILLEGAL placement: {ov} overlapping macro pairs"
    save_placement_pl(best_placed, gs, os.path.join(outdir, "best_placement.pl"))
    plot_placement(best_placed, gn, best_hpwl, outdir)

    cost = advisor_cost(advisor, args.model)
    log(f"\n=== summary ({args.dataset}, seed {args.seed}) ===", fh)
    log(f"best HPWL          : {best_hpwl:.4e}", fh)
    if mp:
        log(f"MaskPlace HPWL     : {mp:.4e}   vs MaskPlace: "
            f"{100*(mp-best_hpwl)/mp:+.1f}%", fh)
    log(f"legal (0 overlaps) : {ov == 0}", fh)
    log(f"total greedy decodes: {total_decodes}", fh)
    if advisor:
        log(f"LLM calls/tokens   : calls={advisor.calls} in={advisor.in_tokens} "
            f"out={advisor.out_tokens} cache_read={advisor.cache_read_tokens}", fh)
        log(f"approx LLM cost    : ${cost:.3f}  (budget ${BUDGET_WARN})", fh)
    log(f"wall               : {time.time()-t0:.0f}s", fh)
    log(f"artifacts          : {outdir}/ (run.log, *.png, best_placement.pl)", fh)
    fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adaptec1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_calls", type=int, default=30, help="max LLM order proposals")
    ap.add_argument("--patience", type=int, default=5,
                    help="stop after this many non-improving calls")
    ap.add_argument("--n_init", type=int, default=100, help="random-init decodes / order")
    ap.add_argument("--n_ea", type=int, default=300, help="swap-only EA rounds / order")
    ap.add_argument("--top_n", type=int, default=-1,
                    help="macros the LLM may reorder; -1 = the WHOLE netlist")
    ap.add_argument("--links", type=int, default=120,
                    help="strongest macro-macro links shown to the LLM / used for feedback")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--max_moves", type=int, default=8)
    ap.add_argument("--mock", action="store_true",
                    help="use random order-edits instead of the LLM (no API key/cost)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_deep(args)


if __name__ == "__main__":
    main()
