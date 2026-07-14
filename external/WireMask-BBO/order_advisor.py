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
# The model answers by CALLING this tool, not by writing JSON in prose. That keeps
# its free-text reasoning (in a `text` block) structurally separate from its answer
# (in the `tool_use` block), so pairs written while thinking can never leak into the
# parsed move list -- the fix for the "scratchpad read as the answer" bug.
PROPOSE_MOVES_TOOL = {
    "name": "propose_moves",
    "description": (
        "Submit your chosen decode-order edits for THIS attempt. Each edit is a pair "
        "[a, b] meaning 'place hub macro a immediately before hub macro b' in the fixed "
        "baseline order. Provide between 1 and the allowed maximum edits; use the fewest "
        "that actually help. Call this exactly once, with your final answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "moves": {
                "type": "array",
                "description": "List of [a, b] integer hub-id pairs (a != b).",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
        "required": ["moves"],
    },
}


class OrderAdvisor:
    """Anthropic-backed advisor: reasons over the FULL trial history and proposes a
    FEW [a, b] order-edits (via the ``propose_moves`` tool) that pull strongly-connected
    hub macros together in the greedy DECODE ORDER.

    Every attempt is the baseline ``rank_macros`` order + the proposed edits (edits never
    stack on the previous attempt); the advisor is shown all past attempts with their HPWL
    and per-layout far-apart hubs, and answers by calling the tool -- so its reasoning text
    can never contaminate the parsed answer."""

    SYSTEM = (
        "You are an expert chip floorplanning assistant guiding a training-free greedy "
        "placer. The placer places macros ONE AT A TIME in a decode ORDER; a macro placed "
        "earlier claims the best low-wirelength location first, so placing two strongly-"
        "connected macros near each other in the ORDER tends to place them physically close. "
        "You set the order by editing a FIXED baseline order (macros pre-sorted by "
        "size/connectivity, called rank_macros). Each edit is a pair [a, b] meaning 'place "
        "hub macro a immediately before hub macro b'.\n\n"
        "CRUCIAL: every attempt is INDEPENDENT. Your edits are ALWAYS applied to the same "
        "fixed baseline order -- never to your previous attempt. Nothing carries over "
        "automatically: if an edit from an earlier attempt helped and you want to keep it, "
        "you MUST include it again in this attempt's list.\n\n"
        "You are shown every attempt you have made: its exact edit list, the HPWL it produced "
        "(lower is better), and which strongly-connected hub pairs stayed FAR APART in THAT "
        "attempt's own layout. Reason over the whole history -- which edits lowered HPWL, "
        "which raised it, which hub pairs stayed stubbornly far apart -- to choose the next "
        "edit set. Use the FEWEST edits that actually help; never pad up to the maximum, and "
        "never re-propose an edit set identical to one already tried. You answer by calling "
        "the propose_moves tool."
    )

    def __init__(self, summary, n_hubs, model="claude-sonnet-4-6", max_tokens=4096,
                 max_retries=2, api_key=None, max_moves=6):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.summary = summary
        self.n = n_hubs
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_moves = max_moves
        self.calls = self.in_tokens = self.out_tokens = self.cache_read_tokens = 0

    def _user_msg(self, best_hpwl, history_block):
        return (
            f"There are {self.n} hub macros with ids M0..M{self.n-1} (their sizes, degrees, "
            "and strongest connections are in the summary above).\n\n"
            "=== FULL TRIAL HISTORY ===\n"
            "(Every attempt below is the baseline rank_macros order with ONLY the listed "
            "edits applied -- attempts do NOT build on each other.)\n\n"
            f"{history_block}\n\n"
            "=== YOUR TASK ===\n"
            f"Current best LEGAL HPWL so far: {best_hpwl:.4e} (lower is better).\n"
            f"Propose between 1 and {self.max_moves} edits, applied to the baseline order, to "
            "pull far-apart connected hubs together. Each edit is a pair [a, b] = 'place hub a "
            "immediately before hub b'. Because every attempt restarts from the baseline, "
            "include any earlier edit you want to keep. Use only as many edits as actually "
            "help -- do not pad to the maximum, and do not repeat an edit set already tried. "
            "Call the propose_moves tool with your chosen list of [a, b] pairs."
        )

    @staticmethod
    def _moves_from_response(resp, n, max_moves):
        """Extract validated [a, b] moves from the ``propose_moves`` tool_use block ONLY.

        Reads the structured tool input, never the free-text reasoning, so pairs the model
        wrote (and maybe rejected) while thinking cannot leak in. Applies the validity
        filter (0 <= a,b < n, a != b) and caps at ``max_moves``. Returns [(a, b), ...]."""
        out = []
        for block in getattr(resp, "content", None) or []:
            if getattr(block, "type", "") != "tool_use" or getattr(block, "name", "") != "propose_moves":
                continue
            data = getattr(block, "input", None) or {}
            for pair in data.get("moves", []) or []:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                try:
                    a, b = int(pair[0]), int(pair[1])
                except (TypeError, ValueError):
                    continue
                if 0 <= a < n and 0 <= b < n and a != b:
                    out.append((a, b))
        return out[:max_moves]

    def suggest(self, best_hpwl, history_block):
        """Ask the LLM for the next edit set, given the FULL trial history.

        Returns (moves, user_msg, attempts):
          - ``moves``    : list of (a, b) tuples, or None if no usable answer after retries.
          - ``user_msg`` : the exact user message sent (for auditing / llm_calls.jsonl).
          - ``attempts`` : one dict per API call (including failed/retried ones) with the
            raw text, stop_reason, parsed moves, and token usage -- for llm_calls.jsonl."""
        msg = self._user_msg(best_hpwl, history_block)
        attempts = []
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    system=[{"type": "text", "text": self.SYSTEM},
                            {"type": "text", "text": self.summary,
                             "cache_control": {"type": "ephemeral"}}],
                    tools=[PROPOSE_MOVES_TOOL],
                    tool_choice={"type": "auto"},
                    messages=[{"role": "user", "content": [{"type": "text", "text": msg}]}],
                )
                self.calls += 1
                u = resp.usage
                in_tok = getattr(u, "input_tokens", 0)
                out_tok = getattr(u, "output_tokens", 0)
                cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
                self.in_tokens += in_tok
                self.out_tokens += out_tok
                self.cache_read_tokens += cache_read
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                stop = getattr(resp, "stop_reason", None)
                moves = self._moves_from_response(resp, self.n, self.max_moves)
                attempts.append({
                    "model": self.model, "stop_reason": stop, "raw_text": text,
                    "parsed_moves": [list(m) for m in moves],
                    "usage": {"input_tokens": in_tok, "output_tokens": out_tok,
                              "cache_read_input_tokens": cache_read},
                })
                if stop == "max_tokens":
                    print(f"  [llm] !! WARNING attempt {attempt+1}: response TRUNCATED at "
                          f"max_tokens={self.max_tokens} -- the answer may be incomplete; "
                          "raise --max_tokens.")
                if moves:
                    return moves, msg, attempts
                print(f"  [llm] attempt {attempt+1}: no propose_moves tool call / no valid "
                      "pairs; retrying")
            except Exception as e:
                attempts.append({"model": self.model, "error": str(e)})
                print(f"  [llm] attempt {attempt+1} error: {e}")
        return None, msg, attempts
