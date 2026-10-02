"""
Post-hoc spatial-coherence probe (adj / rand / gap) for a trained checkpoint.

Measures whether the adapted features place TRUE spatially-adjacent patches
closer together than random pairs, beyond what generic feature dispersion alone
would predict. "True adjacent" pairs are defined from the real patch
coordinates (exact lookup one patch step away, see spatial.py), independent of the
pseudo-grid used at training time.

Feature space matches forward_mae's convention: F.normalize(raw_features) ->
shared_layers (adapter). This is the exact representation the paper's decoder
reconstructs from, and where the original Table 5/6 coherence claims are made.

Usage:
    python coherence_probe.py --dataset 16 --split 0 \
        --ckpt results/mfe/16/split0/seed1/mean/2/best_ckpt_layer2_alpha0.3_mask0.75.pt \
        --model_type mean --label "mfe_mil"
"""
import os
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F

from mfe_mil import MFE_MIL
from spatial import patch_step, adjacent_pairs


def get_test_slides(dataset, split):
    if dataset == "16":
        label_csv = "splits/16/16.csv"
        split_csv = f"splits/16/splits_{split}.csv"
    elif dataset == "panda":
        label_csv = "splits/panda/panda.csv"
        split_csv = f"splits/panda/split_{split}.csv"
    else:
        raise ValueError(dataset)

    labels_df = pd.read_csv(label_csv)
    labels_df = labels_df.set_index("slide_id")
    split_df = pd.read_csv(split_csv)
    test_ids = split_df["test"].dropna().tolist()
    return test_ids, labels_df


def load_features_and_coords(slide_id, labels_df):
    row = labels_df.loc[slide_id]
    data_dir = row["dir"]
    label = row["label"]
    pt_path = os.path.join(data_dir, "pt_files", f"{slide_id}.pt")
    h5_path = os.path.join(data_dir, "h5_files", f"{slide_id}.h5")
    features = torch.load(pt_path)
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
    return features, coords, label


def build_model(model_type, n_classes, in_dim=1024):
    args = argparse.Namespace(
        task="probe", no_existing=True, stop_grad=False, in_dim=in_dim,
        n_classes=n_classes, dropout=0.25,
    )
    model = MFE_MIL(
        args, feat_dim=in_dim, mask_ratio=0.75, shared_layers=2,
        n_classes=n_classes, mil_head_type=model_type, existing=True, window=True,
    )
    return model


def true_adjacent_pairs(coords, eight=False):
    """True-adjacent patch pairs from exact coordinates (see spatial.py)."""
    step = patch_step(coords)
    return [] if step is None else adjacent_pairs(coords, step, eight)


def slide_coherence(adapted_feat, coords, rng, max_pairs=20000, eight=False):
    """adapted_feat: [N, D] L2-unnormalized adapter output (we normalize here
    for cosine similarity). Returns (adj_sim, rand_sim) for one slide."""
    feat = F.normalize(adapted_feat, dim=1)
    N = feat.shape[0]

    adj_pairs = true_adjacent_pairs(coords, eight)
    if not adj_pairs:
        return None, None
    if len(adj_pairs) > max_pairs:
        idx = rng.choice(len(adj_pairs), size=max_pairs, replace=False)
        adj_pairs = [adj_pairs[i] for i in idx]

    n_pairs = len(adj_pairs)
    adj_i = torch.tensor([p[0] for p in adj_pairs], dtype=torch.long)
    adj_j = torch.tensor([p[1] for p in adj_pairs], dtype=torch.long)
    adj_sim = (feat[adj_i] * feat[adj_j]).sum(dim=1).mean().item()

    # random pairs, same count as adjacent pairs for a matched comparison
    rand_i = torch.from_numpy(rng.integers(0, N, size=n_pairs))
    rand_j = torch.from_numpy(rng.integers(0, N, size=n_pairs))
    valid = rand_i != rand_j
    rand_sim = (feat[rand_i[valid]] * feat[rand_j[valid]]).sum(dim=1).mean().item()

    return adj_sim, rand_sim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["16", "panda"])
    p.add_argument("--split", required=True, type=int)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_type", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--max_slides", type=int, default=None,
                    help="cap number of test slides probed, for a quick run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--connectivity", type=int, default=4, choices=[4, 8],
                    help="4: right/down neighbours; 8: also the two diagonals")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    test_ids, labels_df = get_test_slides(args.dataset, args.split)
    if args.max_slides:
        test_ids = test_ids[: args.max_slides]

    n_classes = 2 if args.dataset == "16" else 6  # PANDA: Gleason grades G0-G5
    model = build_model(args.model_type, n_classes)
    state = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only shared_layers (the adapter) matters for this probe; tolerate
    # mismatches elsewhere (e.g. proj_head) but fail loudly if the adapter
    # itself didn't load cleanly.
    bad = [k for k in missing if k.startswith("shared_layers")] + \
          [k for k in unexpected if k.startswith("shared_layers")]
    assert not bad, f"shared_layers state_dict mismatch: {bad}"
    if missing or unexpected:
        print(f"  [note] non-adapter state_dict mismatch (ignored): missing={missing} unexpected={unexpected}")
    model.to(device).eval()

    per_class = {}
    all_adj, all_rand = [], []
    n_load_fail = n_no_pairs = 0

    with torch.no_grad():
        for sid in test_ids:
            try:
                features, coords, label = load_features_and_coords(sid, labels_df)
            except Exception as e:
                print(f"  [skip] {sid}: {e}")
                n_load_fail += 1
                continue
            features = features.to(device)
            normed = F.normalize(features, dim=1)
            adapted = model.extract_shared_features(normed).cpu()

            adj_sim, rand_sim = slide_coherence(adapted, coords, rng, eight=args.connectivity == 8)
            if adj_sim is None:
                print(f"  [skip] {sid}: no adjacent patch pairs")
                n_no_pairs += 1
                continue
            all_adj.append(adj_sim)
            all_rand.append(rand_sim)
            per_class.setdefault(label, {"adj": [], "rand": []})
            per_class[label]["adj"].append(adj_sim)
            per_class[label]["rand"].append(rand_sim)

    def summarize(adj_list, rand_list):
        adj_m, rand_m = float(np.mean(adj_list)), float(np.mean(rand_list))
        ratio = adj_m / rand_m if rand_m != 0 else float("nan")
        gap = adj_m - rand_m
        return adj_m, rand_m, ratio, gap

    print(f"\n=== Coherence probe: {args.label or args.ckpt} ===")
    print(f"dataset={args.dataset} split={args.split} model={args.model_type} n_slides={len(all_adj)} "
          f"(skipped: {n_load_fail} failed to load, {n_no_pairs} without adjacent pairs) connectivity={args.connectivity}")
    adj_m, rand_m, ratio, gap = summarize(all_adj, all_rand)
    print(f"  overall: adj={adj_m:.3f} rand={rand_m:.3f} ratio={ratio:.3f}x gap={gap:.3f}")
    for label, d in sorted(per_class.items()):
        a, r, rt, g = summarize(d["adj"], d["rand"])
        print(f"  class={str(label):10s} n={len(d['adj']):3d}: adj={a:.3f} rand={r:.3f} ratio={rt:.3f}x gap={g:.3f}")


if __name__ == "__main__":
    main()
