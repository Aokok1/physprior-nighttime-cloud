import torch, os
ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "output", "checkpoints", "physprior_moderate_best.pth")
print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
if isinstance(ckpt, dict):
    print(f"Keys: {list(ckpt.keys())[:10]}")
    if "epoch" in ckpt:
        print(f"Epoch: {ckpt['epoch']}")
    if "radar_acc" in ckpt:
        print(f"Radar acc: {ckpt['radar_acc']}")
else:
    print(f"Type: {type(ckpt)}")
    print(f"Total params: {sum(p.numel() for p in ckpt.values()) if hasattr(ckpt, 'values') else 'N/A'}")
