"""Cross-process determinism probe.

Runs a short training loop (2 epochs, seed 0) on the group-disjoint split and
prints a hash of the epoch losses plus the final weights. Running this twice in
separate processes must print the same values; if it does not, cuDNN is still
free to pick different kernels.

Run:  NL_SPLIT_TAG=grp python scripts/probe_determinism.py
"""
import hashlib
import io
import os
import sys

import numpy as np
import torch
from torch.optim import AdamW

sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config import MTUNET_DATASET, train_cfg  # noqa: E402
from data.dataset import make_dataloaders  # noqa: E402
from determinism import seed_everything  # noqa: E402
from models.mt_unet import MT_Loss, MT_UNet  # noqa: E402
from methods.physical_prior import apply_physical_correction  # noqa: E402

SEED = 0
EPOCHS = 2

seed_everything(SEED)
print(f"seed={SEED} cudnn.deterministic={torch.backends.cudnn.deterministic} "
      f"cudnn.benchmark={torch.backends.cudnn.benchmark}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, _ = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size,
                                   use_basemap=True, use_phase4=True)

model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                use_film=True, use_evidential=False).to(device)
crit = MT_Loss(weight_center=train_cfg.weight_center,
               dice_weight=train_cfg.dice_weight).to(device)
opt = AdamW(model.parameters(), lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay)

losses = []
for ep in range(1, EPOCHS + 1):
    model.train()
    tot, nb = 0.0, 0
    for batch in train_loader:
        x = batch["image"].to(device)
        y = batch["mask_noisy"].to(device)
        h = batch["height"].to(device)
        rl = batch["radar_loc"].to(device)
        cl = batch["center_label"].to(device)
        m15 = (x[:, 2] if x.shape[1] == 3 else x[:, 1]).detach().cpu().numpy()
        yc = np.stack([apply_physical_correction(y[i].cpu().numpy(), m15[i],
                                                 m15_min=264.0, std_max=1.5)[0]
                       for i in range(y.shape[0])])
        yc = torch.from_numpy(yc).long().to(device)
        opt.zero_grad()
        logits, hp = model(x)
        loss = crit(logits, hp, yc, h, rl, cl)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        tot += float(loss)
        nb += 1
    losses.append(tot / nb)
    print(f"  epoch {ep}: mean loss = {losses[-1]:.10f}")

wh = hashlib.sha256()
for k, v in sorted(model.state_dict().items()):
    wh.update(k.encode())
    wh.update(v.detach().cpu().numpy().tobytes())
print(f"loss hash  = {hashlib.sha256(np.array(losses).tobytes()).hexdigest()[:32]}")
print(f"weight hash= {wh.hexdigest()[:32]}")
