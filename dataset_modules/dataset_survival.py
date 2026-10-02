import os
import csv
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_config import require_env

COHORT_H5_ROOT = require_env('TCGA_SURVIVAL_EMBEDDING_ROOT')
FEATURE_KEY = 'uni_v2'


def load_cohort_csv(csv_path):
    """Return list of dicts from per-cohort survival CSV."""
    return list(csv.DictReader(open(csv_path)))


def load_split(split_path):
    """Return (train_case_ids, val_case_ids) from kfold split CSV."""
    rows = list(csv.DictReader(open(split_path)))
    train_ids = {r['train'] for r in rows if r['train']}
    val_ids = {r['val'] for r in rows if r['val']}
    return train_ids, val_ids


def compute_bin_boundaries(rows, n_classes=4):
    """
    Compute quantile boundaries from event-only survival times.
    rows: list of dicts with 'survival_months' and 'censorship'.
    Returns array of (n_classes-1) boundary values.
    """
    event_times = [float(r['survival_months']) for r in rows if float(r['censorship']) == 0.0]
    if len(event_times) < n_classes:
        event_times = [float(r['survival_months']) for r in rows]
    return np.quantile(event_times, [i / n_classes for i in range(1, n_classes)])


def discretize(t, boundaries):
    for i, b in enumerate(boundaries):
        if t <= b:
            return i
    return len(boundaries)


def h5_path_for(cohort, slide_id):
    root = COHORT_H5_ROOT.format(cohort=cohort)
    return os.path.join(root, f'{slide_id}.h5')


class TCGASurvivalDataset(Dataset):
    """
    Survival dataset for one TCGA cohort using pre-extracted UNI-v2 embeddings.

    Per-slide returns:
        features    [N, 1536]  float32
        coords      [N, 5]     long
        omic        [1]        float32 (zeros — placeholder)
        label       int        discrete bin [0, n_classes)
        event_time  float      survival_months
        c           float      censorship (1=censored, 0=event)
    """

    def __init__(self, rows, cohort, boundaries, n_classes=4):
        """
        rows        : list of CSV row dicts filtered to this split
        cohort      : e.g. 'KIRC'
        boundaries  : (n_classes-1,) array computed from training rows
        n_classes   : number of discrete time bins
        """
        self.rows = rows
        self.cohort = cohort
        self.boundaries = boundaries
        self.n_classes = n_classes

        # For WeightedRandomSampler compatibility
        self.slide_cls_ids = [[] for _ in range(n_classes)]
        for i, row in enumerate(rows):
            label = discretize(float(row['survival_months']), boundaries)
            self.slide_cls_ids[label].append(i)

    def __len__(self):
        return len(self.rows)

    def getlabel(self, idx):
        return discretize(float(self.rows[idx]['survival_months']), self.boundaries)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = h5_path_for(self.cohort, row['slide_id'])

        with h5py.File(path, 'r') as f:
            features = torch.tensor(f['features'][FEATURE_KEY][:], dtype=torch.float32)
            coords = torch.tensor(f['coords'][:], dtype=torch.long)

        omic = torch.zeros(1, dtype=torch.float32)
        label = discretize(float(row['survival_months']), self.boundaries)
        event_time = float(row['survival_months'])
        c = float(row['censorship'])  # 1=censored, 0=event

        return features, coords, omic, label, event_time, c


def make_survival_datasets(cohort, csv_dir, split_dir, fold, n_classes=4):
    """
    Build train and val TCGASurvivalDataset for one fold.

    Returns (train_dataset, val_dataset, bin_boundaries)
    """
    csv_path = os.path.join(csv_dir, f'{cohort}.csv')
    split_path = os.path.join(split_dir, f'TCGA_{cohort}_survival_kfold', f'splits_{fold}.csv')

    all_rows_by_case = {}
    missing = 0
    for row in load_cohort_csv(csv_path):
        if not os.path.exists(h5_path_for(cohort, row['slide_id'])):
            missing += 1
            continue
        all_rows_by_case.setdefault(row['case_id'], []).append(row)
    if missing:
        print(f'[{cohort}] WARNING: skipped {missing} slides with no H5 embedding')

    train_ids, val_ids = load_split(split_path)

    train_rows = [r for cid in train_ids for r in all_rows_by_case.get(cid, [])]
    val_rows = [r for cid in val_ids for r in all_rows_by_case.get(cid, [])]

    boundaries = compute_bin_boundaries(train_rows, n_classes=n_classes)
    print(f'[{cohort} fold={fold}] train={len(train_rows)}, val={len(val_rows)}, '
          f'boundaries={np.round(boundaries, 1).tolist()}')

    train_dataset = TCGASurvivalDataset(train_rows, cohort, boundaries, n_classes)
    val_dataset = TCGASurvivalDataset(val_rows, cohort, boundaries, n_classes)

    return train_dataset, val_dataset, boundaries
