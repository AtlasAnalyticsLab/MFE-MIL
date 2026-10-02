"""Exact-coordinate patch adjacency, shared by coherence_probe.py and packed_grid_fidelity.py.

Slides are not on one regular lattice (several offset lattices coexist), so
neighbours are found by looking up the exact coordinate one patch step away,
with no rounding to grid cells.
"""

import numpy as np


def patch_step(coords):
    """Patch step: mode of positive gaps between consecutive patches that share a
    row (or column). Gaps across unrelated rows/columns are excluded, so offset
    lattices cannot produce spurious small steps."""
    steps = []
    for a, b in ((0, 1), (1, 0)):
        order = np.lexsort((coords[:, a], coords[:, b]))   # sort by (other axis, this axis)
        cc = coords[order]
        same = cc[1:, b] == cc[:-1, b]
        d = (cc[1:, a] - cc[:-1, a])[same]
        steps.extend(d[d > 0].tolist())
    if not steps:
        return None
    vals, counts = np.unique(steps, return_counts=True)
    return int(vals[counts.argmax()])


def adjacent_pairs(coords, step, eight=False):
    """Unique (i, j) index pairs whose coordinates differ by exactly one step.
    4-connected: right and down. 8-connected: also the two diagonals."""
    key = {(int(x), int(y)): i for i, (x, y) in enumerate(coords)}
    offsets = [(step, 0), (0, step)] + ([(step, step), (step, -step)] if eight else [])
    pairs = []
    for i, (x, y) in enumerate(coords):
        for dx, dy in offsets:
            j = key.get((int(x) + dx, int(y) + dy))
            if j is not None:
                pairs.append((i, j))
    return pairs
