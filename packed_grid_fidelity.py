"""
Packed-grid adjacency fidelity evaluation for MFE-MIL.
Measures how well the MFE packed grid preserves true spatial adjacency.
Runs on CAM16 and PANDA test slides (no GPU, no retraining).
"""

import numpy as np
import h5py
import pandas as pd
import glob
import os
from math import ceil

from env_config import get_env
from spatial import patch_step, adjacent_pairs


# ── core metrics ───────────────────────────────────────────────────────

def evaluate_slide(coords):
    """Distances are measured on raw coordinates in units of the patch step, and true
    neighbours come from exact coordinate lookup (see spatial.py)."""
    N = len(coords)
    if N < 4:
        return None, None, None, np.nan
    S = patch_step(coords)
    if S is None:
        return None, None, None, np.nan
    C = ceil(np.sqrt(N))
    idx = np.arange(N)
    hi = idx[(idx % C != C - 1) & (idx + 1 < N)]; hj = hi + 1
    vi = idx[idx + C < N];                         vj = vi + C
    cheb = lambda i, j: np.maximum(np.abs(coords[i, 0] - coords[j, 0]),
                                   np.abs(coords[i, 1] - coords[j, 1])) / S
    eps = 1e-6

    def stats(i, j):
        if len(i) == 0:
            return None
        d = cheb(i, j)
        return dict(prec1=(np.abs(d - 1) < eps).mean(), prec2=(d <= 2 + eps).mean(),
                    med=float(np.median(d)))

    true_pairs = {(min(i, j), max(i, j)) for i, j in adjacent_pairs(coords, S)}
    if not true_pairs:
        return None, None, None, np.nan
    packed_set = {(int(i), int(j)) for i, j in zip(np.concatenate([hi, vi]), np.concatenate([hj, vj]))}
    recall = len(packed_set & true_pairs) / len(true_pairs)

    return (stats(hi, hj), stats(vi, vj),
            stats(np.concatenate([hi, vi]), np.concatenate([hj, vj])), recall)


def run(coord_arrays):
    agg = {k: [] for k in ("H_prec1", "V_prec1", "all_prec1", "all_prec2", "all_med", "recall")}
    for coords in coord_arrays:
        h, v, a, rec = evaluate_slide(coords)
        if a is None:
            continue
        if h: agg["H_prec1"].append(h["prec1"])
        if v: agg["V_prec1"].append(v["prec1"])
        agg["all_prec1"].append(a["prec1"])
        agg["all_prec2"].append(a["prec2"])
        agg["all_med"].append(a["med"])
        agg["recall"].append(rec)
    out = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in agg.items()}
    out["n_scored"] = len(agg["all_prec1"])
    return out


# ── dataset-specific loaders ───────────────────────────────────────────────

def load_coords(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return f['coords'][:]   # extraction order preserved in CLAM h5


def _warn_missing(missing):
    if missing:
        print(f"  [warn] {len(missing)} .h5 file(s) not found, e.g. {missing[0]}")


def get_cam16_test_h5s(split_csv, h5_root):
    """
    split_csv: e.g. splits/16/splits_0.csv  (columns: id, train, val, test)
    h5_root:   $CAM16_EMBEDDING_ROOT (see configs/paths.example.env); the .h5 files
               are read from <h5_root>/<normal|tumor|test>/h5_files/<slide_id>.h5
    Slide prefix (normal_, tumor_, test_) maps to subdirectory.
    """
    df = pd.read_csv(split_csv)
    slide_ids = df['test'].dropna().tolist()
    paths, missing = [], []
    for sid in slide_ids:
        prefix = sid.split('_')[0]   # normal | tumor | test
        p = os.path.join(h5_root, prefix, 'h5_files', f'{sid}.h5')
        if os.path.exists(p):
            paths.append(p)
        else:
            missing.append(p)
    _warn_missing(missing)
    return paths


def get_panda_test_h5s(split_csv, h5_root):
    """
    split_csv: e.g. splits/panda/split_0.csv
    h5_root:   $PANDA_EMBEDDING_ROOT (see configs/paths.example.env); the .h5 files
               are read from <h5_root>/h5_files/<slide_id>.h5
    """
    df = pd.read_csv(split_csv)
    slide_ids = df['test'].dropna().tolist()
    paths, missing = [], []
    for sid in slide_ids:
        p = os.path.join(h5_root, 'h5_files', f'{sid}.h5')
        if os.path.exists(p):
            paths.append(p)
        else:
            missing.append(p)
    _warn_missing(missing)
    return paths


def get_17_test_h5s(split_csv, label_csv):
    """
    split_csv: e.g. splits/17/split_1.csv  (columns: id, train, val, test)
    label_csv: splits/17/uni.csv  (columns: dir, case_id, slide_id, label)
    dir column gives the per-slide root (h5_files subfolder holds the .h5).
    """
    labels_df = pd.read_csv(label_csv).set_index('slide_id')
    df = pd.read_csv(split_csv)
    slide_ids = df['test'].dropna().tolist()
    paths, missing = [], []
    for sid in slide_ids:
        try:
            data_dir = labels_df.loc[sid, 'dir']
        except KeyError:
            print(f"  [warn] no label row for: {sid}")
            continue
        p = os.path.join(data_dir, 'h5_files', f'{sid}.h5')
        if os.path.exists(p):
            paths.append(p)
        else:
            missing.append(p)
    _warn_missing(missing)
    return paths


def get_brca_test_h5s(split_csv, label_csv):
    """
    split_csv: e.g. splits/brca/splits_0.csv  (columns: (unnamed id), train, val, test)
    label_csv: splits/brca/brcauni.csv  (columns: dir, case_id, slide_id, label)
    """
    labels_df = pd.read_csv(label_csv).set_index('slide_id')
    df = pd.read_csv(split_csv)
    slide_ids = df['test'].dropna().tolist()
    paths, missing = [], []
    for sid in slide_ids:
        try:
            data_dir = labels_df.loc[sid, 'dir']
        except KeyError:
            print(f"  [warn] no label row for: {sid}")
            continue
        p = os.path.join(data_dir, 'h5_files', f'{sid}.h5')
        if os.path.exists(p):
            paths.append(p)
        else:
            missing.append(p)
    _warn_missing(missing)
    return paths


# ── main ───────────────────────────────────────────────────────────────────

def report(h5s):
    print(f"  {len(h5s)} test slides")
    if not h5s:
        print("  skipped: no .h5 files found (see 'Data prep' in the README)")
        return
    res = run([load_coords(p) for p in h5s])
    print(f"  {res['n_scored']} slides scored")
    print(f"  prec@1 : {res['all_prec1'][0]:.4f} ± {res['all_prec1'][1]:.4f}")
    print(f"  prec@2 : {res['all_prec2'][0]:.4f} ± {res['all_prec2'][1]:.4f}")
    print(f"  recall : {res['recall'][0]:.4f} ± {res['recall'][1]:.4f}")
    print(f"  med dist: {res['all_med'][0]:.2f} ± {res['all_med'][1]:.2f}")
    print(f"  H_prec1: {res['H_prec1'][0]:.4f}  V_prec1: {res['V_prec1'][0]:.4f}")


if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))
    cam16_root = get_env('CAM16_EMBEDDING_ROOT')
    panda_root = get_env('PANDA_EMBEDDING_ROOT')

    # Datasets without data are skipped, so you can run this on just the ones you have.
    datasets = [
        ("CAM16 (split 0 test set)", cam16_root and "CAM16_EMBEDDING_ROOT",
         lambda: get_cam16_test_h5s(f'{BASE}/splits/16/splits_0.csv', cam16_root)),
        ("PANDA (split 0 test set)", panda_root and "PANDA_EMBEDDING_ROOT",
         lambda: get_panda_test_h5s(f'{BASE}/splits/panda/split_0.csv', panda_root)),
        ("CAM17 (split 1 test set)", "csv",
         lambda: get_17_test_h5s(f'{BASE}/splits/17/split_1.csv', f'{BASE}/splits/17/uni.csv')),
        ("BRCA (split 0 test set)", "csv",
         lambda: get_brca_test_h5s(f'{BASE}/splits/brca/splits_0.csv', f'{BASE}/splits/brca/brcauni.csv')),
    ]
    for title, configured, load in datasets:
        print(f"=== {title} ===")
        if not configured:
            print("  skipped: root not set in .env.local (see configs/paths.example.env)\n")
            continue
        report(load())
        print()
