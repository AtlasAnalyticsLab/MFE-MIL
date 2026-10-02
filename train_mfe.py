import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from sklearn.metrics import f1_score, roc_auc_score

from dataset_modules.dataset_generic import Generic_MIL_Dataset
from utils.utils import *
from mfe_mil import MFE_MIL


####
# Uses Validation AUC (binary) / F1 (multiclass) for early stopping
# to handle class imbalance.
####

def get_args_parser():
    parser = argparse.ArgumentParser('MFE-MIL training', add_help=False)
    parser.add_argument('--csv_file', default='./csv_file')
    parser.add_argument('--split_dir', default='./split_dir')
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--task', default='panda')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone', default='uni')
    parser.add_argument('--model_type', default='mean')
    parser.add_argument('--shared_layer', type=int, default=2)
    parser.add_argument('--in_dim', type=int, default=1024)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--testing', action='store_true', default=False)
    parser.add_argument('--weighted_sample', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--no_existing', action='store_false', help='Use existing decoder')
    # MFE-specific arguments
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Weight for the MFE reconstruction loss. Set 0 for pure MIL (no MFE).')
    parser.add_argument('--stop_grad', action='store_true', default=False,
                        help='Stop gradient through MAE reconstruction target (ablation).')
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_network(net):
    num_params = sum(p.numel() for p in net.parameters())
    num_params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(net)
    print('Total parameters: %d' % num_params)
    print('Trainable parameters: %d' % num_params_train)


def train_loop(args, epoch, model, loader, optimizer, scaler,
               alpha, n_classes, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    criterion = nn.CrossEntropyLoss()

    train_loss = 0.
    train_error = 0.
    correct = [0] * n_classes
    count = [0] * n_classes
    all_preds = []
    all_labels = []

    print('\n')
    for batch_idx, batch in enumerate(loader):
        data, label = batch[0], batch[1]
        data, label = data.to(device), label.to(device)

        with torch.cuda.amp.autocast():
            if alpha != 0:
                mae_loss, (logits, probs, preds) = model(data, mode="joint")
                mil_loss = criterion(logits, label)

                if torch.isnan(mil_loss):
                    print("NaN in mil_loss")
                    mil_loss = torch.tensor(0.0, device=data.device)

                total_loss = (1 - alpha) * mil_loss + alpha * mae_loss

                if batch_idx % 20 == 0:
                    print(f'  batch {batch_idx}: mil={mil_loss.item():.4f} '
                          f'mae={mae_loss.item():.4f}', end='')
            else:
                logits, probs, preds = model(data, mode="mil")
                mil_loss = criterion(logits, label)
                total_loss = mil_loss

                if batch_idx % 20 == 0:
                    print(f'  batch {batch_idx}: mil={mil_loss.item():.4f}', end='')

            if batch_idx % 20 == 0:
                print()

        loss_value = total_loss.item()
        train_loss += loss_value
        error = calculate_error(preds, label)
        train_error += error

        all_preds.append(preds.cpu())
        all_labels.append(label.cpu())

        for i in range(n_classes):
            correct[i] += ((preds == i) & (label == i)).sum().item()
            count[i] += (label == i).sum().item()

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    train_loss /= len(loader)
    train_error /= len(loader)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Epoch: {epoch}, f1: {f1:.4f}, train_loss: {train_loss:.4f}, train_error: {train_error:.4f}')
    for i in range(n_classes):
        if count[i] > 0:
            print(f'  class {i}: acc {correct[i]/count[i]:.4f}, {correct[i]}/{count[i]}')
        else:
            print(f'  class {i}: no samples')


def validate_func(args, epoch, model, loader, n_classes, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = nn.CrossEntropyLoss()

    val_loss = 0.
    val_error = 0.
    correct = [0] * n_classes
    count = [0] * n_classes
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            data, label = batch[0], batch[1]
            data, label = data.to(device), label.to(device)
            with torch.cuda.amp.autocast():
                logits, probs, preds = model(data, mode="mil")
                loss_value = criterion(logits, label).item()

            val_loss += loss_value
            val_error += calculate_error(preds, label)

            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())
            all_probs.append(probs.cpu())

            for i in range(n_classes):
                correct[i] += ((preds == i) & (label == i)).sum().item()
                count[i] += (label == i).sum().item()

        torch.cuda.empty_cache()

    val_loss /= len(loader)
    val_error /= len(loader)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    f1 = f1_score(all_labels, all_preds, average='macro')

    if args.task in ('panda', '17'):
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except ValueError:
            auc = float('nan')
    else:
        pos_probs = all_probs[:, 1] if all_probs.ndim == 2 else all_probs
        try:
            auc = roc_auc_score(all_labels, pos_probs)
        except ValueError:
            auc = float('nan')

    print(f'Epoch: {epoch}, f1: {f1:.4f}, AUC: {auc:.4f}, val_loss: {val_loss:.4f}, val_error: {val_error:.4f}')
    for i in range(n_classes):
        if count[i] > 0:
            print(f'  class {i}: acc {correct[i]/count[i]:.4f}, {correct[i]}/{count[i]}')
        else:
            print(f'  class {i}: no samples')

    return val_loss, f1, auc


def test(args, epoch, model, loader, n_classes, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = nn.CrossEntropyLoss()

    val_loss = 0.
    val_error = 0.
    correct = [0] * n_classes
    count = [0] * n_classes
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            data, label = batch[0], batch[1]
            data, label = data.to(device), label.to(device)
            with torch.cuda.amp.autocast():
                logits, probs, preds = model(data, mode="mil")
                loss_value = criterion(logits, label).item()

            val_loss += loss_value
            val_error += calculate_error(preds, label)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

            for i in range(n_classes):
                correct[i] += ((preds == i) & (label == i)).sum().item()
                count[i] += (label == i).sum().item()

        torch.cuda.empty_cache()

    val_loss /= len(loader)
    val_error /= len(loader)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    f1 = f1_score(all_labels, all_preds, average='macro')

    if args.task in ('panda', '17'):
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except ValueError:
            auc = float('nan')
    else:
        pos_probs = all_probs[:, 1] if all_probs.ndim == 2 else all_probs
        try:
            auc = roc_auc_score(all_labels, pos_probs)
        except ValueError:
            auc = float('nan')

    print(f'Epoch: {epoch}, f1: {f1}, auc: {auc}, test_loss: {val_loss:.4f}, test_error: {val_error:.4f}')
    for i in range(n_classes):
        if count[i] > 0:
            print(f'  class {i}: acc {correct[i]/count[i]:.4f}, {correct[i]}/{count[i]}')
        else:
            print(f'  class {i}: no samples')

    total_acc = sum(correct) / sum(count)
    return total_acc, f1, auc


def main(args):
    set_seed(args.seed)
    print("decoder existing:", args.no_existing)
    print("mil_head_type:", args.model_type)
    print(f"alpha={args.alpha}  seed={args.seed}")

    output_dir = os.path.join(args.output_dir, args.model_type, str(args.shared_layer))
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.task == 'panda':
        args.n_classes = 6
        dataset = Generic_MIL_Dataset(
            csv_path=args.csv_file, data_dir=None, shuffle=False,
            seed=args.seed, print_info=True,
            label_dict={0:0, 1:1, 2:2, 3:3, 4:4, 5:5},
            patient_strat=False, ignore=[])

    elif args.task == 'brca':
        args.n_classes = 2
        dataset = Generic_MIL_Dataset(
            csv_path=args.csv_file, data_dir=None, shuffle=False,
            seed=args.seed, print_info=True,
            label_dict={'IDC':0, 'ILC':1},
            patient_strat=False, ignore=[])

    elif args.task == '16':
        args.n_classes = 2
        dataset = Generic_MIL_Dataset(
            csv_path=args.csv_file, data_dir=None, shuffle=False,
            seed=args.seed, print_info=True,
            label_dict={'normal':0, 'tumor':1},
            patient_strat=False, ignore=[])

    elif args.task == '17':
        args.n_classes = 4
        dataset = Generic_MIL_Dataset(
            csv_path=args.csv_file, data_dir=None, shuffle=False,
            seed=args.seed, print_info=True,
            label_dict={'negative':0, 'itc':1, 'micro':2, 'macro':3},
            patient_strat=False, ignore=[])

    elif args.task == '172':
        args.n_classes = 2
        dataset = Generic_MIL_Dataset(
            csv_path=args.csv_file, data_dir=None, shuffle=False,
            seed=args.seed, print_info=True,
            label_dict={'negative':0, 'itc':1, 'micro':1, 'macro':1},
            patient_strat=False, ignore=[])
    else:
        raise NotImplementedError(f"Unknown task: {args.task}")

    print('output_dir:', args.output_dir)
    print('split_dir:', args.split_dir)

    train_split, val_split, test_split = dataset.return_splits(
        from_id=False, csv_path=args.split_dir)
    print(f"Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")

    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)

    scaler = torch.cuda.amp.GradScaler()
    max_epochs = 200
    patience = 10
    start_time = time.time()
    log_file = os.path.join(args.output_dir, "training_log_mfe.txt")

    masks = [0.75]
    alphas = [args.alpha]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(log_file, "w") as f:
        for alpha in alphas:
            for mask_ratio in masks:
                print(f'\nInit Model (alpha={alpha}, mask={mask_ratio})...')
                model = MFE_MIL(
                    args,
                    feat_dim=args.in_dim,
                    mask_ratio=mask_ratio,
                    shared_layers=args.shared_layer,
                    n_classes=args.n_classes,
                    mil_head_type=args.model_type
                )
                model.to(device)
                print_network(model)

                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                counter = 0
                best_auc = 0.0
                record_acc = 0
                record_auc = 0
                record_f1 = 0
                best_epoch = 0

                for epoch in range(max_epochs):
                    train_loop(args, epoch, model, train_loader, optimizer,
                               scaler, alpha, args.n_classes)
                    val_loss, f1, auc = validate_func(args, epoch, model, val_loader, args.n_classes)
                    acc, test_f1, test_auc = test(args, epoch, model, test_loader, args.n_classes)

                    if best_auc < auc:
                        best_auc = auc
                        record_acc = acc
                        record_auc = test_auc
                        record_f1 = test_f1
                        counter = 0
                        best_epoch = epoch
                        print(f"  >> New best epoch {best_epoch}, acc={record_acc:.4f}")
                        ckpt_name = (f"best_ckpt_layer{args.shared_layer}"
                                     f"_alpha{alpha}_mask{mask_ratio}.pt")
                        torch.save(model.state_dict(),
                                   os.path.join(args.output_dir, ckpt_name))
                    else:
                        counter += 1
                        if counter >= patience:
                            print(f"Early stopping at epoch {epoch}")
                            break

                used_time = time.time() - start_time
                print(f"Time: {used_time:.0f}s | Best epoch {best_epoch} | "
                      f"Acc={record_acc:.4f} AUC={record_auc:.4f} F1={record_f1:.4f}")
                f.write(
                    f"Head: {args.model_type}, Layer: {args.shared_layer}, "
                    f"Alpha: {alpha}, Mask: {mask_ratio}, "
                    f"Best Acc: {record_acc:.4f}, Best AUC: {record_auc:.4f}, "
                    f"Best F1: {record_f1:.4f}, Best Epoch: {best_epoch}\n"
                )

    print("Done.")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
