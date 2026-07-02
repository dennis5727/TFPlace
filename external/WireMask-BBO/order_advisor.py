"""LLM decode-order guidance for the vendored WireMask-EA (local addition).

The paper fixes the decode order to ``utils.rank_macros`` (descending net-area-sum).
Here a frozen LLM instead reorders the top-N hub macros of that baseline order via
``[a, b]`` move-edits ("place macro a immediately before macro b"); the rest keep
the ``rank_macros`` order. The paper's coordinate (1+1)-EA then refines positions on
top, unchanged. This ports the proven ``PromoteAdvisor`` / ``apply_moves`` idiom from
``tfplace/engine/integrated_search.py`` onto the WireMask (grid-160) pipeline.

Frozen model, no training. The LLM only proposes the ORDER; it never sets
coordinates or overrides the greedy, so legality stays automatic.
"""

import os
import re
from itertools import combinations


# --------------------------------------------------------------------------- #
# connectivity from the WireMask placedb (same computation as
# tfplace/engine/parse_netlist.py::macro_connections, on this place_db's net_info)
# --------------------------------------------------------------------------- #
def macro_degree(placedb):
    """{macro_name: number of nets touching it}."""
    deg = {n: 0 for n in placedb.node_info}
    for net in placedb.net_info.values():
        for node_id in net["nodes"]:
            if node_id in deg:
                deg[node_id] += 1
    return deg


def hub_names(placedb, top_n):
    """Top-N macros by degree (ties broken by name) -- the ones the LLM reorders."""
    deg = macro_degree(placedb)
    names = sorted(placedb.node_info.keys(), key=lambda n: (-deg[n], n))
    return names[:top_n]


def macro_connections(placedb, names, top_k=40):
    """Top-k strongest macro<->macro links among ``names`` as (i, j, n_shared_nets),
    i < j indices into ``names`` (same idea as parse_netlist.macro_connections)."""
    index_of = {name: i for i, name in enumerate(names)}
    pair_count = {}
    for net in placedb.net_info.values():
        members = sorted(index_of[n] for n in net["nodes"] if n in index_of)
        for i, j in combinations(members, 2):
            pair_count[(i, j)] = pair_count.get((i, j), 0) + 1
    ranked = sorted(pair_count.items(), key=lambda kv: kv[1], reverse=True)
    return [(i, j, c) for (i, j), c in ranked[:top_k]]


def build_conn_summary(placedb, hubs, grid_size, top_k_links=40):
    """Compact text summary of the hub macros + strongest links, for the LLM."""
    deg = macro_degree(placedb)
    lines = [f"There are {len(hubs)} hub macros (the most-connected), ids M0..M{len(hubs)-1}.",
             "Each: id, size in grid cells, degree (nets touching it).", ""]
    import math
    for i, name in enumerate(hubs):
        sx = math.ceil(placedb.node_info[name]["x"] / grid_size)
        sy = math.ceil(placedb.node_info[name]["y"] / grid_size)
        lines.append(f"M{i}: {sx}x{sy} cells, degree {deg[name]}")
    lines.append("")
    lines.append("STRONGEST macro-to-macro connections (shared nets):")
    for i, j, c in macro_connections(placedb, hubs, top_k=top_k_links):
        lines.append(f"  M{i} <-> M{j} : {c} nets")
    return "\n".join(lines)


def far_apart_pairs(placed_macros, links, top=12):
    """Connected hub pairs currently far apart (center L1 distance) -> LLM feedback.

    ``links`` = [(i, j, c)] hub-index pairs; ``placed_macros`` from the decoder.
    Returns a text line (a strong_search.long_links analog)."""
    scored = []
    for i, j, c, name_i, name_j in links:
        if name_i not in placed_macros or name_j not in placed_macros:
            continue
        a, b = placed_macros[name_i], placed_macros[name_j]
        d = abs(a["center_loc_x"] - b["center_loc_x"]) + abs(a["center_loc_y"] - b["center_loc_y"])
        scored.append((d, i, j, c))
    scored.sort(reverse=True)
    if not scored:
        return "(no placed hub-to-hub links)"
    return "; ".join(f"M{i}<->M{j} ~{d:.0f} apart ({c} nets)" for d, i, j, c in scored[:top])


# --------------------------------------------------------------------------- #
# order edits (port of integrated_search.apply_moves)
# --------------------------------------------------------------------------- #
def apply_order_edits(order, moves, hubs):
    """Apply 'place hubs[a] immediately before hubs[b]' edits to a decode order.

    Preserves the rest of the order (only the named hubs shift). Invalid/no-op
    moves are skipped. Returns a new full permutation of the same macro set."""
    order = list(order)
    N = len(hubs)
    for a, b in moves:
        if not (0 <= a < N and 0 <= b < N) or a == b:
            continue
        na, nb = hubs[a], hubs[b]
        if na == nb or na not in order or nb not in order:
            continue
        order.remove(na)
        order.insert(order.index(nb), na)
    return order


def random_order_control_edits(n_hubs, rng, n_moves):
    """Random [a,b] edits -- the no-LLM 'is the LLM beating dumb order search?' control."""
    return [(rng.randrange(n_hubs), rng.randrange(n_hubs)) for _ in range(n_moves)]


# --------------------------------------------------------------------------- #
# the LLM advisor (mirrors tfplace/engine/integrated_search.py::PromoteAdvisor)
# --------------------------------------------------------------------------- #
class OrderAdvisor:
    """Anthropic-backed advisor: proposes a FEW [a,b] order-edits that pull
    strongly-connected hub macros earlier/together in the greedy DECODE ORDER."""

    SYSTEM = (
        "You are an expert chip floorplanning assistant. A training-free greedy placer "
        "places macros ONE AT A TIME in a given ORDER; a macro placed earlier claims the "
        "best low-wirelength location, and placing two strongly-connected macros close "
        "together in the order tends to place them physically close. You are given a good "
        "baseline order (macros already sorted by size/connectivity) and the strongly-"
        "connected hub pairs that ended up FAR APART in the current layout. You improve the "
        "ORDER with a FEW targeted moves -- each move places one macro immediately before "
        "another in the order -- to pull connected hubs together, without reshuffling the "
        "rest. You always answer with a single JSON list of [a, b] integer pairs."
    )

    def __init__(self, summary, n_hubs, model="claude-sonnet-4-6", max_tokens=1500,
                 max_retries=2, api_key=None, max_moves=8):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.summary = summary
        self.n = n_hubs
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_moves = max_moves
        self.calls = self.in_tokens = self.out_tokens = self.cache_read_tokens = 0

    def _user_msg(self, best_hpwl, history_text, feedback):
        return (
            f"There are {self.n} hub macros with ids M0..M{self.n-1} (see summary).\n\n"
            "=== ORDER MOVES TRIED SO FAR ===\n"
            f"{history_text}\n\n"
            "=== STRONGLY-CONNECTED HUBS THAT ARE FAR APART RIGHT NOW ===\n"
            f"{feedback}\n\n"
            "=== YOUR TASK ===\n"
            f"Current best LEGAL HPWL: {best_hpwl:.4e} (lower is better).\n"
            f"Propose 1 to {self.max_moves} targeted moves to pull far-apart connected hubs "
            "together in the decode order. Each move is a pair [a, b] meaning 'place macro a "
            "immediately before macro b in the order'. Prefer moving the less-connected macro "
            "of a far-apart pair next to the more-connected one. Do NOT repeat a move set that "
            "was rejected; build on accepted ones. Think briefly, then output ONLY a JSON list "
            "of pairs, e.g. [[12,5],[30,5]]."
        )

    @staticmethod
    def _parse_pairs(text, n):
        out = []
        for a, b in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text):
            a, b = int(a), int(b)
            if 0 <= a < n and 0 <= b < n and a != b:
                out.append((a, b))
        return out

    def suggest(self, best_hpwl, history_text, feedback):
        msg = self._user_msg(best_hpwl, history_text, feedback)
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    system=[{"type": "text", "text": self.SYSTEM},
                            {"type": "text", "text": self.summary,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": [{"type": "text", "text": msg}]}],
                )
                self.calls += 1
                u = resp.usage
                self.in_tokens += getattr(u, "input_tokens", 0)
                self.out_tokens += getattr(u, "output_tokens", 0)
                self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                moves = self._parse_pairs(text, self.n)[:self.max_moves]
                if moves:
                    return moves
                print(f"  [llm] attempt {attempt+1}: no usable [a,b] pairs; retrying")
            except Exception as e:
                print(f"  [llm] attempt {attempt+1} error: {e}")
        return None
