"""Unit tests for OrderAdvisor's structured move parser (local addition).

Run directly (``python test_order_advisor.py``) or with pytest. These do NOT touch the
Anthropic API: they exercise ``OrderAdvisor._moves_from_response`` on hand-built response
objects, so no API key / ``anthropic`` install is needed (that import lives in __init__,
which we never call here).

The parser reads moves ONLY from the ``propose_moves`` tool_use block, never from the
model's free-text reasoning -- so pairs written while thinking cannot leak into the answer.
"""

from types import SimpleNamespace

from order_advisor import OrderAdvisor


def _block(**kw):
    """A stand-in for an Anthropic content block (has .type and, per-type, .text /
    .name / .input attributes accessed via getattr in the parser)."""
    return SimpleNamespace(**kw)


def _resp(content, stop_reason="tool_use"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def test_scratchpad_pairs_do_not_contaminate_answer():
    """Reasoning text full of rejected [a,b] pairs must be ignored; only the tool_use
    block's moves count. (This is the bug the old whole-body regex had.)"""
    resp = _resp([
        _block(type="text",
               text="I considered [5,12] and [30,7] but rejected them; "
                    "the pair [99,1] also looked wrong. Going with the connectivity."),
        _block(type="tool_use", name="propose_moves",
               input={"moves": [[12, 5], [41, 9]]}),
    ])
    moves = OrderAdvisor._moves_from_response(resp, n=60, max_moves=6)
    assert moves == [(12, 5), (41, 9)], moves


def test_truncated_answerless_response_yields_no_moves():
    """A truncated reply with reasoning but NO tool_use block must parse to no moves,
    so suggest()'s retry path fires instead of returning scratchpad pairs."""
    resp = _resp(
        [_block(type="text", text="Let me think... maybe [5,12] then [30,7] then")],
        stop_reason="max_tokens",
    )
    assert OrderAdvisor._moves_from_response(resp, n=60, max_moves=6) == []


def test_validity_filter_and_cap():
    """Out-of-range ids, a==b, and malformed pairs are dropped; result capped at max_moves."""
    resp = _resp([
        _block(type="tool_use", name="propose_moves",
               input={"moves": [[7, 7],        # a == b -> drop
                                [60, 1],       # a out of range (n=60) -> drop
                                [1, -1],       # b out of range -> drop
                                [3],           # malformed -> drop
                                [2, 4], [5, 6], [8, 9], [10, 11]]}),  # 4 valid
    ])
    moves = OrderAdvisor._moves_from_response(resp, n=60, max_moves=3)
    assert moves == [(2, 4), (5, 6), (8, 9)], moves  # capped at 3


def test_wrong_tool_name_ignored():
    """A tool_use block that is not propose_moves must be ignored."""
    resp = _resp([
        _block(type="tool_use", name="something_else", input={"moves": [[1, 2]]}),
    ])
    assert OrderAdvisor._moves_from_response(resp, n=60, max_moves=6) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
