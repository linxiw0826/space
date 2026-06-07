# src/analysis/ — Space Sensing offline analysis scripts (CPU-only, no GPU).
#
# Scripts here parse artifacts produced by GPU runs (e.g. lmms-eval per-sample
# jsonl dumps) and compute derived analyses. They do NOT import lmms-eval or
# torch, and are runnable on a login node.
