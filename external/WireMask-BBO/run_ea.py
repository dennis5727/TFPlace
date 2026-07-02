"""Clean runner for the vendored WireMask-EA (the paper's own pipeline).

Reuses the released functions UNCHANGED (place_db, utils, common) and reproduces
the paper's WireMask-EA loop with a bounded budget: ``init_round`` random-init
greedy decodes (keep best) then a (1+1)-EA swap-only loop for ``stop_round``.
Tracks best HPWL on the paper's own ``cal_hpwl`` (their units), checks legality
(zero macro overlap), and prints the head-to-head vs MaskPlace's provided
placement evaluated identically.

This is the faithful "adopt the paper's released pipeline" path: their decoder,
their benchmark (adaptec1 bookshelf, referenced from ../maskplace), their metric.

Usage:
    python run_ea.py --dataset adaptec1 --seed 2023 --init_round 100 --stop_round 400
"""

import argparse
import random
import time

import place_db
import utils
from common import grid_setting, my_inf
from wm_common import quiet as _quiet, count_overlaps, maskplace_hpwl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adaptec1")
    ap.add_argument("--seed", default="2023")
    ap.add_argument("--init_round", type=int, default=100)
    ap.add_argument("--stop_round", type=int, default=400)
    args = ap.parse_args()

    random.seed(args.seed)
    placedb = place_db.PlaceDB(args.dataset)
    gn = grid_setting[args.dataset]["grid_num"]
    gs = grid_setting[args.dataset]["grid_size"]
    node_id_ls = utils.rank_macros(placedb)

    mp = maskplace_hpwl(placedb, args.dataset)
    t0 = time.time()

    # --- init: random-guiding restarts, keep best -------------------------
    best_hpwl = my_inf
    best_rec = best_placed = None
    for i in range(args.init_round):
        rec = _quiet(utils.random_guiding, node_id_ls, placedb, gn, gs)
        placed, hpwl = _quiet(utils.greedy_placer_with_init_coordinate,
                              node_id_ls, placedb, gn, gs, rec)
        if hpwl < best_hpwl:
            best_hpwl, best_rec, best_placed = hpwl, rec, placed
    n_evals = args.init_round
    print(f"[init]  best HPWL after {args.init_round} random inits: {best_hpwl:.4e}")

    # --- (1+1)-EA: swap-only ---------------------------------------------
    rec = best_rec
    for it in range(args.stop_round):
        ids = list(rec.keys())
        a, b = random.sample(ids, 2)
        rec[a]["loc_x"], rec[a]["loc_y"], rec[b]["loc_x"], rec[b]["loc_y"] = (
            rec[b]["loc_x"], rec[b]["loc_y"], rec[a]["loc_x"], rec[a]["loc_y"])
        placed, hpwl = _quiet(utils.greedy_placer_with_init_coordinate,
                              node_id_ls, placedb, gn, gs, rec)
        n_evals += 1
        if hpwl < best_hpwl:
            best_hpwl, best_placed = hpwl, placed
        else:  # reject -> revert the swap
            rec[a]["loc_x"], rec[a]["loc_y"], rec[b]["loc_x"], rec[b]["loc_y"] = (
                rec[b]["loc_x"], rec[b]["loc_y"], rec[a]["loc_x"], rec[a]["loc_y"])
        if (it + 1) % 25 == 0:
            print(f"[ea {it+1:4d}] best HPWL={best_hpwl:.4e}  evals={n_evals}")

    ov = count_overlaps(best_placed)
    wall = time.time() - t0
    print("\n=== WireMask-EA (paper pipeline) : "
          f"{args.dataset} seed={args.seed} ===")
    print(f"best HPWL          : {best_hpwl:.4e}  (their units)")
    print(f"MaskPlace HPWL     : {mp:.4e}" if mp else "MaskPlace HPWL     : n/a")
    if mp:
        print(f"vs MaskPlace       : {100*(mp-best_hpwl)/mp:+.1f}%  "
              f"({'BEATS' if best_hpwl < mp else 'above'} MaskPlace)")
    print(f"legal (0 overlaps) : {ov == 0}  (overlaps={ov})")
    print(f"greedy evals       : {n_evals}   wall={wall:.0f}s")


if __name__ == "__main__":
    main()
