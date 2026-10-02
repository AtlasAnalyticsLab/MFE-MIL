import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import PatchEmbed, Block
import math


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    pos_embed = []
    for i in range(embed_dim // 4):
        for g in grid:
            pos_embed.append(np.sin(g * (1.0 / 10000 ** (2 * i / embed_dim))))
            pos_embed.append(np.cos(g * (1.0 / 10000 ** (2 * i / embed_dim))))
    pos_embed = np.concatenate(pos_embed, axis=0)
    pos_embed = pos_embed.reshape(embed_dim, grid_size * grid_size).T
    return pos_embed


class MAEFeatureDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features, mask, pos_embed=None):
        """
        features: [N, embed_dim]
        mask:     [N] (bool, True = masked)
        """
        x = self.decoder_embed(features)  # [N, D_dec]

        if mask.dim() == 2:
            mask = mask.squeeze(1)
        mask = mask.bool()

        if mask.any():
            x[mask] = self.mask_token.to(x.dtype)

        if pos_embed is not None:
            x = x + pos_embed.to(x.device).to(x.dtype)

        x = x.unsqueeze(0)  # [1, N, D_dec]

        for blk in self.decoder_blocks:
            x = checkpoint.checkpoint(blk, x, use_reentrant=False)

        x = x.squeeze(0)  # [N, D_dec]
        x = self.decoder_norm(x)
        x_recon = self.decoder_pred(x)

        return x_recon


class MFE_MIL(nn.Module):
    """
    MFE-MIL: a MIL model trained with an auxiliary window-masking feature
    reconstruction (MFE) objective on a shared adapter.

    MFE (alpha > 0): blocks of the packed 2D pseudo-grid are masked and a
    decoder reconstructs the raw (normalized) patch features with an MSE loss.

    Total joint loss:
        L = (1-alpha) * L_MIL + alpha * L_MFE
    """

    def __init__(self, args, feat_dim=1024, mask_ratio=0.75,
                 shared_layers=2, n_classes=1, sub_batch=8,
                 mil_head_type='mean', existing=True, window=True, survival=False):
        super().__init__()
        print("feat_dim:", feat_dim)
        self.mask_ratio = mask_ratio
        self.sub_batch = sub_batch
        self.feat_scale = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(feat_dim)
        self.mil_head_type = mil_head_type
        self.dataset = args.task
        self.existing = args.no_existing
        self.stop_grad = getattr(args, 'stop_grad', False)

        self.survival = survival
        self.shared_layers = nn.ModuleList()
        in_dim = feat_dim
        for _ in range(shared_layers):
            self.shared_layers.append(nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.ReLU(),
                nn.Dropout(0.25)
            ))

        if self.existing:
            self.decoder = MAEFeatureDecoder(embed_dim=args.in_dim)

        if mil_head_type == 'mean':
            from models.Mean_Max_MIL import MeanMIL
            self.mil_head = MeanMIL(in_dim=args.in_dim, n_classes=n_classes, dropout=True, act='relu', survival=survival)
        elif mil_head_type == 'max':
            from models.Mean_Max_MIL import MaxMIL
            self.mil_head = MaxMIL(in_dim=args.in_dim, n_classes=n_classes, dropout=True, act='relu', survival=survival)
        elif mil_head_type == 'att':
            from models.ABMIL import DAttention
            self.mil_head = DAttention(args.in_dim, n_classes, dropout=0.25, act='relu', survival=survival)
        elif mil_head_type == 'trans':
            from models.TransMIL import TransMIL
            self.mil_head = TransMIL(args.in_dim, n_classes, dropout=0.25, act='relu', survival=survival)
        elif mil_head_type == 'clam_sb':
            from models.CLAM import CLAM_SB
            self.mil_head = CLAM_SB(args, gate=True, size_arg="small", k_sample=8,
                                    instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False,
                                    test=False, act='relu', n_robust=0, survival=survival)
        elif mil_head_type == 'clam_mb':
            from models.CLAM import CLAM_MB
            self.mil_head = CLAM_MB(args, gate=True, size_arg="small", k_sample=8,
                                    instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, act='relu',
                                    survival=survival)
        else:
            raise ValueError(f"Invalid mil_head_type: {mil_head_type}")

    # ------------------------------------------------------------------
    # Masking strategies
    # ------------------------------------------------------------------

    def mask_features_window(self, data, mask_ratio=0.75, grid_size=None):
        """
        Two-phase window masking on a 2D grid:
          Phase 1: place non-overlapping 3x3 and 5x5 blocks (~80% of target)
          Phase 2: fill remaining positions individually to reach exact ratio
        """
        assert grid_size is not None
        N, D = data.shape
        H, W = grid_size
        device = data.device

        target_masked = int(mask_ratio * N)
        mask_2d = torch.zeros((H, W), device=device, dtype=torch.bool)
        masked_count = 0

        radii = [1, 2]
        max_block_limit = target_masked * 0.8
        trials = 0
        while masked_count < max_block_limit and trials < 2000:
            trials += 1
            w = radii[torch.randint(0, len(radii), (1,)).item()]
            win_area = (2 * w + 1) ** 2

            if masked_count + win_area > target_masked:
                continue

            # skip window size if grid is too small to place it
            if H <= 2 * w or W <= 2 * w:
                continue

            cx = torch.randint(w, H - w, (1,)).item()
            cy = torch.randint(w, W - w, (1,)).item()

            if not mask_2d[cx-w:cx+w+1, cy-w:cy+w+1].any():
                mask_2d[cx-w:cx+w+1, cy-w:cy+w+1] = True
                masked_count += win_area

        if masked_count < target_masked:
            needed = target_masked - masked_count
            unmasked_coords = torch.where(~mask_2d.view(-1))[0]
            perm = torch.randperm(len(unmasked_coords), device=device)[:needed]
            mask_2d.view(-1)[unmasked_coords[perm]] = True

        mask = mask_2d.view(-1)
        masked_data = data.clone()
        masked_data[mask] = 0

        return masked_data, mask

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _run_shared_layers(self, x):
        for layer in self.shared_layers:
            x = layer(x)
        return x

    def extract_shared_features(self, x):
        return self._run_shared_layers(x)

    def extract_MIL_features(self, x):
        x = self._run_shared_layers(x)
        return self.mil_head.tsne(x)

    # ------------------------------------------------------------------
    # MFE forward
    # ------------------------------------------------------------------

    def forward_mae(self, data):
        """
        Args:
            data : [N, D] patch features
        Returns:
            loss_mae : scalar MFE reconstruction loss
        """
        data = F.normalize(data, dim=1)

        num_patches = data.shape[0]
        H = math.ceil(num_patches ** 0.5)
        W = H
        N_pad = H * W - num_patches

        if N_pad > 0:
            pad = torch.zeros((N_pad, data.shape[1]), device=data.device)
            padded_data = torch.cat([data, pad], dim=0)
        else:
            padded_data = data

        pos_embed = get_2d_sincos_pos_embed(
            self.decoder.decoder_embed.out_features, H, cls_token=False)
        pos_embed = torch.from_numpy(pos_embed).to(data.device).to(data.dtype)

        masked_data, mask = self.mask_features_window(
            padded_data, mask_ratio=self.mask_ratio, grid_size=(H, W))

        if torch.isnan(masked_data).any():
            print("NaN in masked_data")

        # Shared encoder on masked data
        h_masked = masked_data.clone()
        for layer in self.shared_layers:
            h_masked = layer(h_masked)

        # Decoder + reconstruction loss on the masked positions
        reconstructed = self.decoder(h_masked, mask, pos_embed=pos_embed)[:num_patches]
        masked_idx = mask[:num_patches].bool()
        if masked_idx.sum() == 0:
            print("Warning: no masked patches for MFE loss")
            return torch.tensor(0.0, device=data.device)

        target = data[masked_idx].detach() if self.stop_grad else data[masked_idx]
        loss_mae = F.mse_loss(reconstructed[masked_idx], target)

        if torch.isnan(loss_mae):
            print("NaN in MFE loss")
            loss_mae = torch.tensor(0.0, device=data.device)

        return loss_mae

    # ------------------------------------------------------------------
    # MIL forward
    # ------------------------------------------------------------------

    def forward_mil(self, feat):
        def mil_shared_forward(x):
            return self._run_shared_layers(x)

        all_pooled = []
        for i in range(0, feat.size(0), self.sub_batch):
            f_sub = feat[i:i + self.sub_batch]          # [sub_batch, D]
            pooled = F.adaptive_avg_pool2d(
                f_sub.unsqueeze(-1).unsqueeze(-1), (1, 1)  # [sub_batch, D, 1, 1]
            ).flatten(1)
            pooled = checkpoint.checkpoint(
                mil_shared_forward, pooled, use_reentrant=False)
            all_pooled.append(pooled)

        all_pooled = torch.cat(all_pooled, dim=0)
        all_pooled = all_pooled.unsqueeze(0)  # [1, N, D]

        logits, Y_prob, Y_hat, _, _ = self.mil_head(all_pooled)
        return logits, Y_prob, Y_hat

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, feat, mode="joint"):
        if mode == "mae":
            if self.existing:
                return self.forward_mae(feat)
            return torch.tensor(0.0)

        elif mode == "mil":
            return self.forward_mil(feat)

        elif mode == "joint":
            if self.existing:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss_mae = self.forward_mae(feat)
                    logits, probs, preds = self.forward_mil(feat)
                return loss_mae, (logits, probs, preds)
            else:
                logits, probs, preds = self.forward_mil(feat)
                return torch.tensor(0.0), (logits, probs, preds)

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def heatmap(self, feat):
        feat = self._run_shared_layers(feat)
        logits, Y_prob, Y_hat, A, B = self.mil_head(feat)
        return logits, Y_prob, Y_hat, A, B
