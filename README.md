# MFE-MIL: Window-Masking Feature-Reconstruction MIL for Whole-Slide Images

MFE-MIL is a multiple-instance-learning (MIL) framework for whole-slide
image (WSI) classification and survival prediction. An adapter module is
trained with an auxiliary window-masking feature-reconstruction (MFE) task on
top of a frozen patch-feature backbone (UNI), alongside the usual slide-level
MIL objective. This repo contains the training and evaluation code, plus
manifests for re-running the main MFE-MIL classification sweep (five MIL heads
x CAM16/CAM17/PANDA/BRCA) and the survival sweep. The paper's ablation and
baseline tables are not part of this release.

## Evidence at a glance

**Packed-grid fidelity** (coordinate-only, no GPU — see
`packed_grid_fidelity.py`):
| Dataset | Test slides | prec@1 | prec@2 | recall | H-prec@1 | V-prec@1 |
|---|---|---|---|---|---|---|
| CAM16 | 128  | 0.4873  | 0.5006  | 0.4984  | 0.967 | 0.007 |
| CAM17 | 50   | 0.4817 | 0.4947  | 0.4990  | 0.957 | 0.007 |
| PANDA | 1031 | 0.4687  | 0.5151  | 0.4995  | 0.894 | 0.043 |
| BRCA  | 96   | 0.4969  | 0.5103  | 0.5006  | 0.974 | 0.020 |

Test sets are split 0 (CAM16, PANDA, BRCA) and split 1 (CAM17). Neighbours come
from exact patch coordinates one patch step apart (no grid snapping), so the
numbers do not depend on an inferred stride. The packed grid keeps horizontal
neighbours (H-prec@1) but almost never vertical ones (V-prec@1), hence
prec@1 and recall near 0.5.

## Install

Pick one:

```bash
# Conda
conda env create -f environment.yml
conda activate mfe

# venv + pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Training and evaluation never read raw slides, so the system OpenSlide library
is optional. Install it (`apt install openslide-tools` / `brew install
openslide`) only if you use the slide-reading helpers in `utils/utils.py`.

## Data prep

This repo trains on **pre-extracted patch features**, not raw slides. Download
the datasets yourself (this repo ships no slides or features):

- CAMELYON16 / CAMELYON17: https://camelyon17.grand-challenge.org/
- PANDA: https://www.kaggle.com/c/prostate-cancer-grade-assessment
- TCGA-BRCA (and the TCGA survival cohorts KIRC/KIRP/LUAD/STAD/UCEC):
  https://portal.gdc.cancer.gov/

Then extract patches and features (this repo does not include an extractor).
You can use [CLAM](https://github.com/mahmoodlab/clam) or
[AtlasPatch](https://github.com/AtlasAnalyticsLab/AtlasPatch) for patch
extraction, with a feature extractor of your choice (we use UNI). Classification training expects a CLAM-style folder per
dataset, `<dir>` below, where `<dir>` is the value in the `dir` column of that
dataset's CSV (see the next paragraph):

```
<dir>/pt_files/<slide_id>.pt     torch tensor [num_patches, feature_dim]  (read by train_mfe.py)
<dir>/h5_files/<slide_id>.h5     HDF5 with a `coords` dataset [num_patches, 2]  (only the two analysis tools read this)
```

`feature_dim` must match `--in_dim` (default 1024). Survival training reads a
different layout: one H5 per slide at
`$TCGA_SURVIVAL_EMBEDDING_ROOT/<slide_id>.h5` with `features/uni_v2` and
`coords` datasets. Then point the config at your data:

```bash
cp configs/paths.example.env .env.local
# edit .env.local: see the comments in it for which variable each tool needs
```

`.env.local` is loaded automatically by every entrypoint (`env_config.py`)
and is gitignored — never commit it. `splits/` and `survival/` (small CSVs:
train/val/test slide-ID splits and cohort survival labels) already ship in
this repo.

**One more path to set before training (classification only):** `train_mfe.py`
reads each slide's feature directory straight from the `dir` column of the
per-dataset CSV (`splits/16/16.csv`, `splits/17/uni.csv`,
`splits/brca/brcauni.csv`, `splits/panda/panda.csv`) — it does not go through
`.env.local`. Those columns ship as placeholders (`/path/to/CAM16/features`,
etc.); either edit them to your real feature root, or create a symlink at
that exact path pointing to it, before running anything from
`configs/classification_tasks.tsv`.

## Reproduce every evaluation

Everything runs through one driver, `run_manifest.py`, reading a TSV under
`configs/`. Each row is one (dataset, model, mode, split, seed) run; already-
completed runs are skipped automatically. Use the dataset names exactly as
they appear in the TSV: `16`, `17`, `panda`, `brca` for classification and
`KIRC`, `KIRP`, `LUAD`, `STAD`, `UCEC` for survival.

Re-running a manifest trains from scratch with new random streams, so results
will not match the paper bit for bit. Expect the same trends, but small
differences in absolute numbers (we have seen up to about 1 AUC point on
CAMELYON16 between independent re-runs).

```bash
# Preview what a manifest would run, without launching anything
python run_manifest.py --task-file configs/classification_tasks.tsv --dry-run

# Launch everything in a manifest on GPU 0
python run_manifest.py --task-file configs/classification_tasks.tsv --gpu 0

# Launch just one slice (filters compose: --dataset --model --mode --split --seed)
python run_manifest.py --task-file configs/classification_tasks.tsv \
    --gpu 0 --dataset 16 --model att

# Run several jobs concurrently on one GPU
python run_manifest.py --task-file configs/classification_tasks.tsv --gpu 0 --parallel 4
```

| Table | Manifest |
|---|---|
| Main MFE-MIL table (mean/max/att/clam_sb/clam_mb x CAM16/17/PANDA/BRCA; MFE `alpha=0.3`) | `configs/classification_tasks.tsv` |
| Survival (6 models x 5 TCGA cohorts; mode `pure_mil` = MIL only, `mfe` = MIL plus the MFE reconstruction loss) | `configs/survival_tasks.tsv` |

Two standalone analysis tools (not training, so not in the manifest). Both read
patch coordinates from the `h5_files/` layout above. `packed_grid_fidelity.py`
needs no model or GPU and skips any dataset you have not set up;
`coherence_probe.py` needs a trained checkpoint and supports CAM16 and PANDA
only:

```bash
python packed_grid_fidelity.py      # coordinate-only spatial-fidelity table above
python coherence_probe.py --dataset 16 --split 0 \
    --ckpt results/mfe/16/split0/seed1/att/2/best_ckpt_*.pt \
    --model_type att --label mfe_mil   # adj/rand/gap coherence probe on a trained checkpoint
```

`coherence_probe.py` uses 4-connected neighbours by default; pass
`--connectivity 8` to include diagonals. It prints how many slides were scored
and skipped.

In `train_mfe.py`, `--alpha 0 --no_existing` trains the adapter plus MIL head
with the classification loss only; add `--shared_layer 0` for plain ABMIL
without the adapter.

## Repository map

| Path | What's in it |
|---|---|
| `train_mfe.py` | Classification entry point: arg parsing, train/val/test loop |
| `train_survival.py` | Survival entry point |
| `mfe_mil.py` | Core model (`MFE_MIL`): adapter + window-masking MFE decoder + MIL head dispatch |
| `models/` | MIL heads: ABMIL (`att`), CLAM (`clam_sb`/`clam_mb`), Mean/Max pooling, TransMIL (`trans`) |
| `dataset_modules/` | `dataset_generic.py` (classification), `dataset_survival.py` (survival) |
| `env_config.py` | `.env.local` loader (see Data prep) |
| `run_manifest.py` | Single driver for every experiment sweep |
| `configs/*.tsv` | Experiment manifests (one row per run) |
| `configs/paths.example.env` | Template for `.env.local` |
| `coherence_probe.py` | adj/rand/gap spatial-coherence probe on a trained checkpoint |
| `packed_grid_fidelity.py` | Coordinate-only packed-grid fidelity table (no GPU) |
| `splits/`, `survival/` | Train/val/test slide-ID splits; TCGA survival cohort labels |
| `nystromformer/`, `utils/` | Supporting modules (Nystrom attention, training/survival utils) |

## Checkpoints

No pretrained checkpoints are distributed with this release — every table
above is reproducible from scratch via `run_manifest.py`.

## Citation

Paper: [arXiv:2610.10225](https://arxiv.org/abs/2610.10225). See also `CITATION.cff`.

```bibtex
@article{he2026masked,
  title   = {Masked Feature Encoding for Large-Scale Whole Slide Image Representation},
  author  = {He, Haoyu and Tessier-Cloutier, Basile and Wang, Yang and Hosseini, Mahdi S.},
  journal = {arXiv preprint arXiv:2610.10225},
  year    = {2026}
}
```

