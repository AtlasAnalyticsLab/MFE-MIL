#!/usr/bin/env python3
"""
Single driver for every experiment sweep in the paper. Replaces the old
one-off per-experiment shell scripts. Reads a TSV manifest (one row per run)
and either prints what it would do (--dry-run) or launches it, skipping
anything already completed (matching the on-disk convention used throughout
the project: a non-empty training-log file means that run is done).

Usage:
    python run_manifest.py --task-file configs/classification_tasks.tsv --gpu 0
    python run_manifest.py --task-file configs/classification_tasks.tsv --gpu 0 \
        --dataset 16 --model att --dry-run
    python run_manifest.py --task-file configs/survival_tasks.tsv --gpu 0 --parallel 4

Each TSV has columns: task_group, dataset, model, mode, split, seed
  task_group  which recipe to use (see TASK_GROUPS below)
  dataset     16 / 17 / panda / brca for classification; KIRC/KIRP/LUAD/STAD/UCEC for survival
  model       mean/max/att/clam_sb/clam_mb/trans
  mode        survival mode (pure_mil / mfe); blank for classification
  split       fold/split index (note: CAM17 uses 1,2,3 not 0,1,2 — the TSV
              just lists whatever split values were actually used)
  seed        random seed
"""
import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIG = {
    "16":    dict(csv_file="splits/16/16.csv",       split_dir="splits/16/splits_{split}.csv", task="16"),
    "17":    dict(csv_file="splits/17/uni.csv",       split_dir="splits/17/split_{split}.csv",  task="172"),
    "panda": dict(csv_file="splits/panda/panda.csv",  split_dir="splits/panda/split_{split}.csv", task="panda"),
    "brca":  dict(csv_file="splits/brca/brcauni.csv", split_dir="splits/brca/splits_{split}.csv", task="brca"),
}

TASK_GROUPS = {
    # Main MFE-ABMIL table: mean/max/att/clam_sb/clam_mb x {16,17,panda,brca}
    "mfe": dict(
        script="train_mfe.py",
        out_tmpl="results/mfe/{dataset}/split{split}/seed{seed}/{model}",
        done_rel="2/training_log_mfe.txt",
        base_args=["--alpha", "0.3", "--shared_layer", "2",
                    "--in_dim", "1024", "--weighted_sample"],
    ),
    # Survival: pure_mil (alpha=0) vs mfe (alpha=0.3)
    "survival": dict(
        script="train_survival.py",
        out_tmpl="results/survival/{dataset}/{mode}/{model}_fold{split}_seed{seed}",
        done_rel="log.txt",
        base_args=["--n_classes", "4", "--max_epochs", "20"],
        mode_alpha={"pure_mil": "0", "mfe": "0.3"},
    ),
}


def build_command(row, gpu):
    group = TASK_GROUPS[row["task_group"]]
    out_dir = group["out_tmpl"].format(**row) + "/"
    cmd = [sys.executable, group["script"]]

    if group["script"] == "train_mfe.py":
        ds = DATASET_CONFIG[row["dataset"]]
        cmd += [
            "--csv_file", ds["csv_file"],
            "--split_dir", ds["split_dir"].format(**row),
            "--task", ds["task"],
            "--output_dir", out_dir,
            "--model_type", row["model"],
            "--seed", row["seed"],
        ] + group["base_args"]
    else:  # train_survival.py
        alpha = group["mode_alpha"][row["mode"]]
        cmd += [
            "--task", row["dataset"],
            "--model_type", row["model"],
            "--fold", row["split"],
            "--seed", row["seed"],
            "--alpha", alpha,
            "--output_dir", "./results/survival",
        ] + group["base_args"]

    done_marker = os.path.join(REPO_ROOT, out_dir, group["done_rel"])
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"  # so logs/manifest/*.log shows loss lines as they happen
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return cmd, done_marker, env, out_dir


def load_rows(task_file, filters):
    rows = []
    with open(task_file, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row = {k: (v or "").strip() for k, v in row.items()}
            if all(row.get(k) == v for k, v in filters.items()):
                rows.append(row)
    return rows


def run_one(row, gpu, dry_run, log_dir):
    cmd, done_marker, env, out_dir = build_command(row, gpu)
    label = f"{row['task_group']}/{row.get('mode','-')}/{row['dataset']}/{row['model']}/split{row['split']}/seed{row['seed']}"

    if os.path.exists(done_marker) and os.path.getsize(done_marker) > 0:
        print(f"[skip] {label} (already done)")
        return

    if dry_run:
        print(f"[dry-run] {label}\n    {' '.join(cmd)}")
        return

    os.makedirs(os.path.join(REPO_ROOT, out_dir), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, label.replace("/", "_") + ".log")
    print(f"[start] {label} -> {log_path}")
    with open(log_path, "a") as logf:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    print(f"[done]  {label}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-file", required=True, help="TSV manifest, e.g. configs/classification_tasks.tsv")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value for every launched job")
    p.add_argument("--parallel", type=int, default=1, help="Max concurrent jobs")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    p.add_argument("--log-dir", default="logs/manifest")
    # Optional column filters, e.g. --dataset 16 --model att --split 0 --seed 1
    for col in ("task_group", "dataset", "model", "mode", "split", "seed"):
        p.add_argument(f"--{col}")
    args = p.parse_args()

    filters = {c: getattr(args, c) for c in ("task_group", "dataset", "model", "mode", "split", "seed") if getattr(args, c)}
    rows = load_rows(args.task_file, filters)
    if not rows:
        print("No matching rows.", file=sys.stderr)
        sys.exit(1)
    print(f"{len(rows)} matching row(s) from {args.task_file}")

    if args.parallel <= 1:
        for row in rows:
            run_one(row, args.gpu, args.dry_run, args.log_dir)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            list(ex.map(lambda r: run_one(r, args.gpu, args.dry_run, args.log_dir), rows))


if __name__ == "__main__":
    main()
