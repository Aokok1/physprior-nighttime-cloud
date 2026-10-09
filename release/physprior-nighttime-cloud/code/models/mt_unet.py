"""MT-UNet model adapted for noise-label learning.

Adds optional EDL (Evidential Deep Learning) head for uncertainty-aware
teacher in SELFIE framework. Backward-compatible with original MT-UNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from config import model_cfg


# ── FiLM ────────────────────────────────────────────────────────────
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.SiLU(),
            nn.Linear(128, n_channels * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, cond):
        out = self.net(cond)
        n_ch = features.shape[1]
        gamma = out[:, :n_ch].view(-1, n_ch, 1, 1) + 1.0
        beta = out[:, n_ch:].view(-1, n_ch, 1, 1)
        return gamma * features + beta


# ── Dice Loss ────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, num_classes=4):
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_oh).sum(dim=(0, 2, 3))
        cardinality = (probs + targets_oh).sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


# ── Evidential Head (from Plan 1.1) ──────────────────────────────────
class EvidentialHead(nn.Module):
    """Dirichlet evidence output head. α = softplus(logits) + 1."""

    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        evidence = F.softplus(self.conv(x))
        return evidence + 1.0  # α ≥ 1


# ── MT-UNet ──────────────────────────────────────────────────────────
class MT_UNet(nn.Module):
    def __init__(self, backbone="resnet34", in_channels=3, mask_classes=4,
                 use_film=True, use_evidential=False):
        super().__init__()
        self.use_film = use_film
        self.use_evidential = use_evidential

        self.base_model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=mask_classes,
            decoder_attention_type="scse",
        )

        decoder_out = 16
        self.height_head = smp.base.SegmentationHead(
            in_channels=decoder_out, out_channels=1,
            activation=None, kernel_size=3,
        )

        if use_evidential:
            self.mask_head = EvidentialHead(decoder_out, mask_classes)
        else:
            self.mask_head = self.base_model.segmentation_head

        if use_film:
            bottleneck_ch = self.base_model.encoder.out_channels[-1]
            self.film = FiLM(cond_dim=3, n_channels=bottleneck_ch)

    def forward(self, x, moon_phase=None, solar_zenith=None):
        features = list(self.base_model.encoder(x))

        if self.use_film and moon_phase is not None:
            moon_rad = moon_phase.float() * (torch.pi / 180.0)
            sin_m = torch.sin(moon_rad).unsqueeze(1)
            cos_m = torch.cos(moon_rad).unsqueeze(1)
            sza = (solar_zenith.float() / 180.0).unsqueeze(1) if solar_zenith is not None \
                  else torch.full_like(sin_m, 0.5)
            cond = torch.cat([sin_m, cos_m, sza], dim=1)
            features[-1] = self.film(features[-1], cond)

        try:
            decoder_output = self.base_model.decoder(*features)
        except TypeError:
            decoder_output = self.base_model.decoder(features)

        mask_out = self.mask_head(decoder_output)
        height_pred = self.height_head(decoder_output)
        return mask_out, height_pred


# ── Standard Loss ────────────────────────────────────────────────────
class MT_Loss(nn.Module):
    def __init__(self, weight_mask=1.0, weight_height=1.0, weight_center=5.0,
                 dice_weight=0.5, return_per_sample=False):
        super().__init__()
        self.weight_mask = weight_mask
        self.weight_height = weight_height
        self.weight_center = weight_center
        self.dice_weight = dice_weight
        # forward() returns (total, loss_mask, loss_height, loss_center) by
        # default. Callers that need the per-sample vector for sample filtering
        # (the co-teaching runs) opt in with return_per_sample=True and unpack
        # five values. Keeping the default at four preserves the contract that
        # baseline / coteaching / mixup / loss_correction / the ablations were
        # written against; those callers were broken when the per-sample term was
        # added unconditionally.
        self.return_per_sample = return_per_sample
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_dice = DiceLoss()
        self.criterion_height = nn.MSELoss(reduction="none")
        self.criterion_center = nn.BCELoss()

    def forward(self, pred_mask, pred_height, target_mask, target_height,
                radar_locs=None, center_labels=None):
        if target_mask.dim() == 4:
            target_mask = target_mask.squeeze(1)
        if target_mask.dtype != torch.long:
            target_mask = target_mask.long()
        loss_ce = self.criterion_ce(pred_mask, target_mask)
        loss_dice = self.criterion_dice(pred_mask, target_mask)
        loss_mask = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        valid = (target_height != -1.0)
        if valid.sum() > 0:
            raw = self.criterion_height(pred_height, target_height)
            loss_height = raw[valid].mean()
        else:
            loss_height = torch.tensor(0.0, device=pred_mask.device)

        loss_center = torch.tensor(0.0, device=pred_mask.device)
        if radar_locs is not None and center_labels is not None:
            valid_idx = ~torch.isnan(center_labels)
            if valid_idx.sum() > 0:
                v_locs = radar_locs[valid_idx]
                v_labels = center_labels[valid_idx].float()
                v_preds = pred_mask[valid_idx]
                probs = torch.softmax(v_preds, dim=1)
                p_cloud = (probs[:, 2, :, :] + probs[:, 3, :, :]).clamp(1e-6, 1 - 1e-6)

                center_loss_sum = 0.0
                for i in range(len(v_locs)):
                    ry, rx = v_locs[i, 0].item(), v_locs[i, 1].item()
                    y_min = max(0, ry - 1)
                    y_max = min(p_cloud.shape[-2], ry + 2)
                    x_min = max(0, rx - 1)
                    x_max = min(p_cloud.shape[-1], rx + 2)
                    window = p_cloud[i, y_min:y_max, x_min:x_max]
                    true_label = v_labels[i]
                    val = window.max() if true_label == 1.0 else window.mean()
                    center_loss_sum += self.criterion_center(
                        val.unsqueeze(0), true_label.unsqueeze(0)
                    )
                loss_center = center_loss_sum / len(v_locs)

        total = (self.weight_mask * loss_mask +
                 self.weight_height * loss_height +
                 self.weight_center * loss_center)
        # Per-sample loss for Co-Teaching / robust learning methods.
        # Mask CE is reduced mean → use CE with reduction='none' to get [B, H, W],
        # then average over spatial dims → [B].
        per_sample_ce = torch.nn.functional.cross_entropy(
            pred_mask, target_mask, reduction='none')  # [B, H, W]
        per_sample_ce = per_sample_ce.reshape(per_sample_ce.size(0), -1).mean(1)  # [B]
        per_sample_dice = self._per_sample_dice(pred_mask, target_mask)  # [B] = [B,1,H,W] squeeze(1) before pass
        per_sample_mask = (1 - self.dice_weight) * per_sample_ce + self.dice_weight * per_sample_dice
        # Height loss per sample (zero where no valid)
        raw_h = self.criterion_height(pred_height, target_height)  # [B, 1, H, W]
        per_sample_h = raw_h.reshape(raw_h.size(0), -1).mean(1) if raw_h.dim() >= 2 else raw_h
        # valid shape: [B, 1, H, W] from (target_height != -1.0); any-pixel-valid flag
        per_sample_h = per_sample_h * valid.reshape(valid.size(0), -1).any(dim=1).float()
        # Center loss per sample (binary; broadcast to all if absent)
        per_sample_center = torch.zeros(per_sample_ce.size(0), device=pred_mask.device)
        if radar_locs is not None and center_labels is not None:
            valid_idx = ~torch.isnan(center_labels)
            if valid_idx.sum() > 0:
                # _center_loss_per_sample returns (B,) but only valid rows are non-zero;
                # assign to a per_sample_center of the same batch size
                cl = self._center_loss_per_sample(
                    pred_mask, radar_locs, center_labels)
                per_sample_center = cl  # (B,) with zeros for invalid rows already
        per_sample = (self.weight_mask * per_sample_mask
                      + self.weight_height * per_sample_h
                      + self.weight_center * per_sample_center)
        if self.return_per_sample:
            return total, loss_mask, loss_height, loss_center, per_sample
        return total, loss_mask, loss_height, loss_center

    def _per_sample_dice(self, logits, targets, num_classes=4):
        """Compute per-sample dice loss (1 - mean dice of foreground classes)."""
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        # one-hot targets: [B, H, W] -> [B, C, H, W]
        oh = torch.zeros_like(probs)
        oh.scatter_(1, targets.unsqueeze(1).long(), 1.0)
        # Per-sample dice for each class
        intersect = (probs * oh).flatten(2).sum(2)  # [B, C]
        denom = probs.flatten(2).sum(2) + oh.flatten(2).sum(2) + 1e-6  # [B, C]
        per_class_dice = 2 * intersect / denom  # [B, C]
        # Average over foreground classes (exclude index 0)
        return 1 - per_class_dice[:, 1:].mean(1)

    def _center_loss_per_sample(self, pred_mask, radar_locs, center_labels):
        """Per-sample center detection BCE."""
        probs = torch.softmax(pred_mask, dim=1)
        p_cloud = (probs[:, 2, :, :] + probs[:, 3, :, :]).clamp(1e-6, 1 - 1e-6)
        losses = torch.zeros(p_cloud.size(0), device=pred_mask.device)
        for i in range(p_cloud.size(0)):
            cl = center_labels[i]
            if cl != cl: continue  # NaN
            cl_v = cl.float() if torch.is_tensor(cl) else torch.tensor(cl, dtype=torch.float, device=pred_mask.device)
            ry, rx = int(radar_locs[i, 0]), int(radar_locs[i, 1])
            y_min = max(0, ry - 1); y_max = min(p_cloud.shape[-2], ry + 2)
            x_min = max(0, rx - 1); x_max = min(p_cloud.shape[-1], rx + 2)
            window = p_cloud[i, y_min:y_max, x_min:x_max]
            val = window.max() if cl_v.item() == 1.0 else window.mean()
            losses[i] = -(
                cl_v * torch.log(val.unsqueeze(0).clamp(1e-6, 1 - 1e-6))
                + (1 - cl_v) * torch.log((1 - val).unsqueeze(0).clamp(1e-6, 1 - 1e-6))
            ).mean()
        return losses


# ── Noise-corrected Loss (for Loss Correction method) ────────────────
class CorrectedLoss(nn.Module):
    """Cross-entropy with noise transition matrix correction.

    Forward correction: P_corrected = T @ P_predicted
    where T is the estimated noise transition matrix.

    On radar-hit pixels: use clean CE loss (no correction).
    On other pixels: use corrected CE loss.
    """

    def __init__(self, transition_matrix: torch.Tensor, weight_center=5.0,
                 dice_weight=0.5):
        super().__init__()
        self.register_buffer("T", transition_matrix)  # (4, 4)
        self.dice_weight = dice_weight
        self.weight_center = weight_center
        self.criterion_dice = DiceLoss()
        self.criterion_ce = nn.CrossEntropyLoss(reduction="none")
        self.criterion_center = nn.BCELoss()

    def forward(self, pred_mask, target_mask, radar_locs=None, center_labels=None):
        if target_mask.dim() == 4:
            target_mask = target_mask.squeeze(1)
        if target_mask.dtype != torch.long:
            target_mask = target_mask.long()
        # Forward correction: apply T to predictions before computing CE
        probs = F.softmax(pred_mask, dim=1)                     # (B, 4, H, W)
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, 4)  # (B*H*W, 4)
        corrected = (self.T @ probs_flat.T).T                   # (B*H*W, 4)
        corrected = corrected.reshape(probs.shape)              # (B, 4, H, W)

        # Numerical stability (P2-8): clamp → renormalize → log
        corrected = corrected.clamp(1e-6, 1.0)
        corrected = corrected / corrected.sum(dim=1, keepdim=True)
        corrected_log = torch.log(corrected)

        # Label smoothing on targets for additional stability
        targets_oh = F.one_hot(target_mask, 4).permute(0, 3, 1, 2).float()
        smoothing = 0.05
        targets_oh = (1 - smoothing) * targets_oh + smoothing / 4.0

        # CE on corrected probabilities
        loss_ce = -(targets_oh * corrected_log).sum(dim=1).mean()

        loss_dice = self.criterion_dice(pred_mask, target_mask)
        loss_mask = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        loss_center = torch.tensor(0.0, device=pred_mask.device)
        if radar_locs is not None and center_labels is not None:
            valid_idx = ~torch.isnan(center_labels)
            if valid_idx.sum() > 0:
                probs_cloud = (probs[:, 2, :, :] + probs[:, 3, :, :]).clamp(1e-6, 1 - 1e-6)
                loss_sum = 0.0
                v_locs = radar_locs[valid_idx]
                v_labels = center_labels[valid_idx].float()
                for i in range(len(v_locs)):
                    ry, rx = v_locs[i, 0].item(), v_locs[i, 1].item()
                    y0, y1 = max(0, ry - 1), min(probs_cloud.shape[-2], ry + 2)
                    x0, x1 = max(0, rx - 1), min(probs_cloud.shape[-1], rx + 2)
                    win = probs_cloud[valid_idx][i, y0:y1, x0:x1]
                    val = win.max() if v_labels[i] == 1.0 else win.mean()
                    loss_sum += self.criterion_center(val.unsqueeze(0),
                                                       v_labels[i].unsqueeze(0))
                loss_center = loss_sum / len(v_locs)

        return loss_mask + self.weight_center * loss_center, loss_mask, \
               torch.tensor(0.0), loss_center


# ── Physics-Guided Loss (Priority 2: soft, differentiable constraint) ─
class PhysicsGuidedLoss(nn.Module):
    """Soft physics constraint replacing hard label flipping.

    L_total = L_data + lambda_phys * L_physics

    L_data = (1-dice_w) * CE + dice_w * Dice + w_center * BCE(center)
    L_physics = (1/N) * Σ M_prior_i * P_cloud_i

    where M_prior = σ((M15 - T_min)/T_0) * [1 - σ((std - S_max)/S_0)]

    When M_prior → 1 (warm + uniform → physically clear), model is penalized
    for predicting cloud. When M_prior → 0 (cold or textured → possibly cloud),
    physics constraint is off and model learns from data.

    Key advantage: DIFFERENTIABLE — no hard threshold flipping.
    """

    def __init__(self, T_min=264.0, S_max=1.5, T0=2.0, S0=0.2,
                 lambda_phys=0.3, weight_center=5.0, dice_weight=0.5):
        super().__init__()
        self.T_min = T_min
        self.S_max = S_max
        self.T0 = T0
        self.S0 = S0
        self.lambda_phys = lambda_phys
        self.weight_center = weight_center
        self.dice_weight = dice_weight
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_center = nn.BCELoss()
        self.criterion_dice = DiceLoss()

    def compute_physics_mask(self, m15, local_std):
        """Compute continuous physics prior mask M_prior ∈ [0,1].

        m15: (B, H, W) M15 brightness temperature (not normalized!)
        local_std: (B, H, W) local spatial std of M15 (not normalized!)
        """
        # Temperature term: warm BT → high σ → M → 1
        t_term = torch.sigmoid((m15 - self.T_min) / self.T0)
        # Uniformity term: low std → 1-σ low → M → 1
        s_term = 1.0 - torch.sigmoid((local_std - self.S_max) / self.S0)
        return t_term * s_term  # (B, H, W), each ∈ [0, 1]

    def forward(self, pred_mask, target_mask, m15_bt, local_std,
                radar_locs=None, center_labels=None):
        """pred_mask: (B, 4, H, W), target_mask: (B, H, W),
           m15_bt: (B, H, W) in Kelvin, local_std: (B, H, W) in Kelvin.
        """
        if target_mask.dim() == 4:
            target_mask = target_mask.squeeze(1)
        if target_mask.dtype != torch.long:
            target_mask = target_mask.long()

        # Data loss: CE + Dice (matching MT_Loss)
        loss_ce = self.criterion_ce(pred_mask, target_mask)
        loss_dice = self.criterion_dice(pred_mask, target_mask)
        loss_data = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        # Physics prior mask
        M = self.compute_physics_mask(m15_bt, local_std)  # (B, H, W)

        # P_cloud = prob(prob_cloud) + prob(True Cloud) from model prediction
        probs = F.softmax(pred_mask, dim=1)
        P_cloud = probs[:, 2, :, :] + probs[:, 3, :, :]  # (B, H, W)

        # Physics loss: penalize predicting cloud where physics says clear
        L_physics = (M * P_cloud).mean()

        # Radar center anchor (same as MT_Loss)
        loss_center = torch.tensor(0.0, device=pred_mask.device)
        if radar_locs is not None and center_labels is not None:
            valid_idx = ~torch.isnan(center_labels)
            if valid_idx.sum() > 0:
                p_cloud_clamp = P_cloud[valid_idx].clamp(1e-6, 1 - 1e-6)
                v_locs = radar_locs[valid_idx]
                v_labels = center_labels[valid_idx].float()
                loss_sum = 0.0
                for i in range(len(v_locs)):
                    ry, rx = v_locs[i, 0].item(), v_locs[i, 1].item()
                    y0, y1 = max(0, ry - 1), min(P_cloud.shape[-2], ry + 2)
                    x0, x1 = max(0, rx - 1), min(P_cloud.shape[-1], rx + 2)
                    win = p_cloud_clamp[i, y0:y1, x0:x1]
                    val = win.max() if v_labels[i] == 1.0 else win.mean()
                    loss_sum += self.criterion_center(
                        val.unsqueeze(0), v_labels[i].unsqueeze(0))
                loss_center = loss_sum / len(v_locs)

        total = loss_data + self.lambda_phys * L_physics + \
                self.weight_center * loss_center
        return total, loss_data, L_physics, loss_center
