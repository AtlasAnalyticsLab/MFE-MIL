"""
Survival training for MFE-MIL.

Supports two modes (controlled by --alpha):
  pure MIL  : alpha=0, direct MIL model, standard survival loss
  mfe       : alpha>0, MFE-MIL wrapper with window-masking MFE + MIL

Models: mean, max, att, clam_sb, clam_mb, trans
Tasks : KIRC, KIRP, LUAD, STAD, UCEC
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from dataset_modules.dataset_survival import make_survival_datasets
from utils.survival_utils import (
    NLLSurvLoss, CrossEntropySurvLoss, CoxSurvLoss,
    get_split_loader_survival, get_optim, concordance_index_censored,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 'survival/' ships inside this repo (cohort label CSVs + kfold splits), so this
# is a repo-relative path, not a user-configured one.
SURVIVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'survival')


# ── Args ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser('Survival training')
    p.add_argument('--task', default='KIRC',
                   choices=['KIRC', 'KIRP', 'LUAD', 'STAD', 'UCEC'])
    p.add_argument('--model_type', default='att',
                   choices=['mean', 'max', 'att', 'clam_sb', 'clam_mb', 'trans'])
    p.add_argument('--in_dim', type=int, default=1536)
    p.add_argument('--n_classes', type=int, default=4,
                   help='Discrete survival time bins')
    p.add_argument('--shared_layer', type=int, default=2,
                   help='Shared encoder layers (mfe mode only)')
    p.add_argument('--drop_out', type=float, default=0.25)
    p.add_argument('--dropout', type=float, default=0.25)   # alias used by the MFE-MIL wrapper
    p.add_argument('--bag_loss', default='nll_surv',
                   choices=['nll_surv', 'ce_surv', 'cox_surv'])
    p.add_argument('--alpha_surv', type=float, default=0.0)
    # MFE args
    p.add_argument('--alpha', type=float, default=0.0,
                   help='MFE loss weight. 0 = pure MIL, 0.3 = MFE')
    p.add_argument('--mask_ratio', type=float, default=0.75)
    # Optimiser
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--reg', type=float, default=1e-5)
    p.add_argument('--opt', default='adam', choices=['adam', 'sgd'])
    p.add_argument('--max_epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5,
                   help='Early stopping patience (epochs without val_c improvement). 0 = disabled')
    p.add_argument('--gc', type=int, default=32,
                   help='Gradient accumulation steps')
    # Run config
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--weighted_sample', action='store_true')
    p.add_argument('--testing', action='store_true')
    p.add_argument('--output_dir', default='./results/survival')
    return p.parse_args()


# ── Seed ───────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model builders ─────────────────────────────────────────────────────────────

def build_pure_mil(args):
    """Direct MIL model with survival output."""
    kw = dict(survival=True)
    if args.model_type == 'mean':
        from models.Mean_Max_MIL import MeanMIL
        return MeanMIL(in_dim=args.in_dim, n_classes=args.n_classes,
                       dropout=True, act='relu', **kw)
    elif args.model_type == 'max':
        from models.Mean_Max_MIL import MaxMIL
        return MaxMIL(in_dim=args.in_dim, n_classes=args.n_classes,
                      dropout=True, act='relu', **kw)
    elif args.model_type == 'att':
        from models.ABMIL import DAttention
        return DAttention(args.in_dim, args.n_classes,
                          dropout=args.drop_out, act='relu', **kw)
    elif args.model_type == 'trans':
        from models.TransMIL import TransMIL
        return TransMIL(args.in_dim, args.n_classes,
                        dropout=args.drop_out, act='relu', **kw)
    elif args.model_type == 'clam_sb':
        from models.CLAM import CLAM_SB
        return CLAM_SB(args, gate=True, size_arg='small', k_sample=8,
                       instance_loss_fn=nn.CrossEntropyLoss(),
                       subtyping=False, survival=True)
    elif args.model_type == 'clam_mb':
        from models.CLAM import CLAM_MB
        return CLAM_MB(args, gate=True, size_arg='small', k_sample=8,
                       instance_loss_fn=nn.CrossEntropyLoss(),
                       subtyping=False, survival=True)
    raise NotImplementedError(f'{args.model_type}')


def build_mfe(args):
    """MFE-MIL wrapper: window-masking MFE + MIL head, survival mode."""
    from mfe_mil import MFE_MIL
    # the MFE-MIL wrapper reads these flags from args
    args.no_existing = True   # use decoder (enables MAE)
    model = MFE_MIL(
        args,
        feat_dim=args.in_dim,
        mask_ratio=args.mask_ratio,
        shared_layers=args.shared_layer,
        n_classes=args.n_classes,
        mil_head_type=args.model_type,
        survival=True,
    )
    return model


# ── Train loop ─────────────────────────────────────────────────────────────────

def train_loop(epoch, model, loader, optimizer, loss_fn, alpha, gc, scaler=None):
    model.train()
    total_loss = 0.0
    all_risk = np.zeros(len(loader))
    all_c = np.zeros(len(loader))
    all_t = np.zeros(len(loader))

    optimizer.zero_grad()
    print()
    for i, (data_WSI, coords, data_omic, label, event_time, c) in enumerate(loader):
        data_WSI = data_WSI.to(device)
        label = label.to(device)
        c = c.to(device)

        # Sort patches by (y, x) so window masking captures real spatial neighbours
        if alpha > 0:
            sort_idx = torch.argsort(coords[:, 1] * 1_000_000 + coords[:, 0])
            data_WSI = data_WSI[sort_idx]

        if alpha > 0:
            mae_loss, (hazards, S, _) = model(data_WSI, mode='joint')
            surv_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            loss = (1 - alpha) * surv_loss + alpha * mae_loss
        else:
            hazards, S, _, _, _ = model(data_WSI)
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)

        if scaler is not None:
            scaler.scale(loss / gc).backward()
            if (i + 1) % gc == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            (loss / gc).backward()
            if (i + 1) % gc == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        risk = -torch.sum(S, dim=1).detach().cpu().item()
        all_risk[i] = risk if not np.isnan(risk) else 0.0
        all_c[i] = c.item()
        all_t[i] = float(event_time)

    # flush remaining gradients
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad()

    total_loss /= len(loader)
    c_index = concordance_index_censored(
        (1 - all_c).astype(bool), all_t, all_risk)[0]
    print(f'  Epoch {epoch}: train_loss={total_loss:.4f}  train_c={c_index:.4f}')


# ── Val loop ───────────────────────────────────────────────────────────────────

def validate(epoch, model, loader, loss_fn, alpha):
    model.eval()
    total_loss = 0.0
    all_risk = np.zeros(len(loader))
    all_c = np.zeros(len(loader))
    all_t = np.zeros(len(loader))

    with torch.no_grad():
        for i, (data_WSI, coords, data_omic, label, event_time, c) in enumerate(loader):
            data_WSI = data_WSI.to(device)
            label = label.to(device)
            c = c.to(device)

            if alpha > 0:
                sort_idx = torch.argsort(coords[:, 1] * 1_000_000 + coords[:, 0])
                data_WSI = data_WSI[sort_idx]

            if alpha > 0:
                hazards, S, _ = model(data_WSI, mode='mil')
            else:
                hazards, S, _, _, _ = model(data_WSI)

            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
            total_loss += loss.item()
            all_risk[i] = -torch.sum(S, dim=1).cpu().item()
            all_c[i] = c.item()
            all_t[i] = float(event_time)

    total_loss /= len(loader)
    c_index = concordance_index_censored(
        (1 - all_c).astype(bool), all_t, all_risk)[0]
    print(f'  Epoch {epoch}: val_loss={total_loss:.4f}    val_c={c_index:.4f}')
    return c_index


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    set_seed(args.seed)

    mode = 'mfe' if args.alpha > 0 else 'pure_mil'
    print(f'\ntask={args.task}  model={args.model_type}  mode={mode}  '
          f'fold={args.fold}  seed={args.seed}')
    print(f'alpha={args.alpha}  n_classes={args.n_classes}')

    # ── Datasets ──
    train_ds, val_ds, boundaries = make_survival_datasets(
        cohort=args.task,
        csv_dir=SURVIVAL_DIR,
        split_dir=SURVIVAL_DIR,
        fold=args.fold,
        n_classes=args.n_classes,
    )

    train_loader = get_split_loader_survival(
        train_ds, training=True, testing=args.testing,
        weighted=args.weighted_sample, batch_size=1)
    val_loader = get_split_loader_survival(val_ds, testing=args.testing, batch_size=1)

    # ── Loss ──
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    else:
        loss_fn = CoxSurvLoss()

    # ── Model ──
    if mode == 'mfe':
        model = build_mfe(args)
    else:
        model = build_pure_mil(args)

    if hasattr(model, 'relocate'):
        model.relocate()
    else:
        model = model.to(device)

    optimizer = get_optim(model, args)
    scaler = GradScaler() if (mode == 'mfe' and torch.cuda.is_available()) else None

    # ── Output dir ──
    out_dir = os.path.join(
        args.output_dir, args.task, mode,
        f'{args.model_type}_fold{args.fold}_seed{args.seed}')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'log.txt')

    # ── Training ──
    best_c = 0.0
    best_epoch = -1
    no_improve = 0
    with open(log_path, 'w') as log_f:
        for epoch in range(args.max_epochs):
            train_loop(epoch, model, train_loader, optimizer,
                       loss_fn, args.alpha, args.gc, scaler)
            c_index = validate(epoch, model, val_loader, loss_fn, args.alpha)

            if c_index > best_c:
                best_c = c_index
                best_epoch = epoch
                no_improve = 0
                ckpt = os.path.join(out_dir, 'best.pt')
                torch.save(model.state_dict(), ckpt)
                print(f'    >> Best c-index: {best_c:.4f} (epoch {best_epoch})')
            else:
                no_improve += 1

            log_f.write(f'epoch={epoch} val_c={c_index:.4f} best_c={best_c:.4f}\n')
            log_f.flush()

            if args.patience > 0 and no_improve >= args.patience:
                print(f'    >> Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)')
                break

    result_line = (f'task={args.task} model={args.model_type} mode={mode} '
                   f'fold={args.fold} seed={args.seed} '
                   f'best_c={best_c:.4f} best_epoch={best_epoch}\n')
    print('\n' + result_line)

    summary_path = os.path.join(args.output_dir, args.task, 'results_summary.txt')
    with open(summary_path, 'a') as f:
        f.write(result_line)


if __name__ == '__main__':
    main()
