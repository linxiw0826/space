"""CPU unit test for the E-16e eval-side env switch ``MOPE_FEED_CAUSAL_MASK``.

The switch lives in src/eval/models/qwen3_vl_mope.py as ``_read_feed_causal_mask``.
That module has heavy top-level imports (lmms_eval, model packages) that are not
importable outside the eval environment, so we do NOT import the module. Instead
we extract the REAL source text of ``_read_feed_causal_mask`` via AST and exec it
in an isolated namespace (only ``os`` available). This tests the ACTUAL parse
convention shipped in the plugin — not a re-implementation — so it fails if the
convention drifts.

Convention under test:
  - UNSET            -> False   (byte-equivalent to pre-E-16e eval)
  - "0" / "false"    -> False
  - "1" / "true"     -> True    (case-insensitive, surrounding whitespace ok)
  - garbage          -> False   (safe default = OFF)

CPU only, no GPU, no model download.

Run:  python tests/test_feed_causal_mask_env.py   (or: pytest tests/test_feed_causal_mask_env.py)
"""

import ast
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.join(_HERE, "..", "src", "eval", "models", "qwen3_vl_mope.py")


def _load_env_helper():
    """Extract + compile ONLY the _read_feed_causal_mask function from the plugin."""
    with open(_PLUGIN, "r") as f:
        src = f.read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_read_feed_causal_mask":
            fn_src = ast.get_source_segment(src, node)
            ns = {"os": os}
            exec(compile(fn_src, _PLUGIN, "exec"), ns)
            return ns["_read_feed_causal_mask"]
    raise AssertionError("_read_feed_causal_mask not found in " + _PLUGIN)


_read_feed_causal_mask = _load_env_helper()


def _with_env(value):
    """Set MOPE_FEED_CAUSAL_MASK to value (None = unset) and return the parsed bool."""
    prev = os.environ.get("MOPE_FEED_CAUSAL_MASK")
    try:
        if value is None:
            os.environ.pop("MOPE_FEED_CAUSAL_MASK", None)
        else:
            os.environ["MOPE_FEED_CAUSAL_MASK"] = value
        return _read_feed_causal_mask()
    finally:
        if prev is None:
            os.environ.pop("MOPE_FEED_CAUSAL_MASK", None)
        else:
            os.environ["MOPE_FEED_CAUSAL_MASK"] = prev


def test_unset_is_false():
    """UNSET -> False: eval stays byte-equivalent to the pre-E-16e path."""
    assert _with_env(None) is False
    print("ok  test_unset_is_false")


def test_truthy_values():
    for v in ("1", "true", "True", "TRUE", " true ", "  1 "):
        assert _with_env(v) is True, v
    print("ok  test_truthy_values")


def test_falsy_values():
    for v in ("0", "false", "False", "", "no", "yes", "2", "on", "garbage"):
        assert _with_env(v) is False, v
    print("ok  test_falsy_values")


if __name__ == "__main__":
    test_unset_is_false()
    test_truthy_values()
    test_falsy_values()
    print("\nALL E-16e ENV-SWITCH TESTS PASSED")
