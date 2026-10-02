import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
import torch.optim as optim


# ── Concordance index (Harrell's C) ───────────────────────────────────────────

def concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-8):
    """
    Harrell's C-index for right-censored data.

    event_indicator : bool array, True if event occurred (not censored)
    event_time      : float array
    estimate        : float array, higher = higher risk
    Returns (c_index, concordant, discordant, tied_risk, 0)
    """
    concordant = discordant = tied_risk = 0
    n = len(event_time)
    for i in range(n):
        if not event_indicator[i]:
            continue
        for j in range(n):
            if event_time[j] <= event_time[i]:
                continue
            diff = estimate[i] - estimate[j]
            if abs(diff) <= tied_tol:
                tied_risk += 1
            elif diff > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant + tied_risk
    c = (concordant + 0.5 * tied_risk) / total if total > 0 else 0.5
    return c, concordant, discordant, tied_risk, 0


# ── Survival loss functions ────────────────────────────────────────────────────

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], dim=1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) +
        torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    return loss.mean()


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], dim=1)
    reg = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y) + eps) +
        torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    ce_l = (
        -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) -
        (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    )
    return ((1 - alpha) * ce_l + alpha * reg).mean()


class NLLSurvLoss:
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        return nll_loss(hazards, S, Y, c, alpha=self.alpha if alpha is None else alpha)


class CrossEntropySurvLoss:
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        return ce_loss(hazards, S, Y, c, alpha=self.alpha if alpha is None else alpha)


class CoxSurvLoss:
    def __call__(self, hazards, S, Y, c, **kwargs):
        device = hazards.device
        n = len(S)
        R = torch.zeros(n, n, device=device)
        for i in range(n):
            for j in range(n):
                R[i, j] = float(S[j] >= S[i])
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        c_vec = c.reshape(-1).float()
        return -torch.mean((theta - torch.log(torch.sum(exp_theta * R, dim=1))) * (1 - c_vec))


# ── Collate ────────────────────────────────────────────────────────────────────

def collate_MIL_survival(batch):
    """
    batch items: (features [N,D], coords [N,5], omic [1], label, event_time, c)
    Returns list of 6 tensors/arrays.
    """
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = torch.cat([item[1] for item in batch], dim=0)
    omic = torch.cat([item[2] for item in batch], dim=0).float()
    label = torch.LongTensor([item[3] for item in batch])
    event_time = np.array([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])
    return [img, coords, omic, label, event_time, c]


# ── Loader ─────────────────────────────────────────────────────────────────────

def get_split_loader_survival(dataset, training=False, testing=False,
                               weighted=False, batch_size=1):
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    if testing:
        sampler = SequentialSampler(dataset)
    elif training and weighted:
        weights = _make_weights(dataset)
        sampler = WeightedRandomSampler(weights, len(weights))
    elif training:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      collate_fn=collate_MIL_survival, **kwargs)


def _make_weights(dataset):
    n = float(len(dataset))
    weight_per_class = []
    for cls_ids in dataset.slide_cls_ids:
        weight_per_class.append(n / len(cls_ids) if cls_ids else 0.0)
    weights = [weight_per_class[dataset.getlabel(i)] for i in range(len(dataset))]
    return torch.DoubleTensor(weights)


# ── Optimizer ──────────────────────────────────────────────────────────────────

def get_optim(model, args):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.opt == 'adam':
        return optim.Adam(params, lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        return optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.reg)
    raise NotImplementedError(f'Unknown optimizer: {args.opt}')
