"""Refiner error vs |d| on its OWN val set, where d is drawn independently of
frame difficulty -- isolates 'large offset' from 'hard crop'."""
import sys, importlib.util
from pathlib import Path
ROOT = Path("/home/kaelin/p4p/dense deep charuco")
sys.path.insert(0, str(ROOT))
import numpy as np, torch, cv2
cv2.setNumThreads(1)
from dcc.dataset import load_config, RefinerVal
from dcc.model import Refiner
from dcc.pipeline import soft_argmax
from torch.utils.data import DataLoader
spec = importlib.util.spec_from_file_location("_tr", str(ROOT/"tools"/"train_refiner.py"))
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)

cfg = load_config(str(ROOT/"configs/rev640.yaml"))
dev = torch.device("cuda")
m = Refiner().to(dev).eval()
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m.load_state_dict(ck.get("ema", ck.get("model")))
dl = DataLoader(RefinerVal(cfg, n=1500), batch_size=64, num_workers=0, collate_fn=tr._refiner_val_collate)

D, E = [], []
with torch.no_grad():
    for crops, targets, d in dl:
        if crops is None: continue
        u = soft_argmax(m(crops.to(dev, memory_format=torch.channels_last)).sigmoid().cpu())
        pred = (np.asarray(u) - 31.5) / 8.0          # predicted offset, px
        dd = d.numpy() if hasattr(d, "numpy") else np.asarray(d)
        D.append(dd); E.append(np.linalg.norm(pred - dd, axis=1))
D = np.concatenate(D); E = np.concatenate(E)
mag = np.linalg.norm(D, axis=1)
print(f"ckpt={sys.argv[1]}  n={len(E)}\n")
print(f"{'|d| bin (px)':>14}{'n':>7}{'err_med':>10}{'err_p95':>10}{'>1px%':>9}{'resid/|d|':>11}")
for lo, hi in [(0,.25),(.25,.5),(.5,.75),(.75,1),(1,1.5),(1.5,2),(2,3),(3,4),(4,5.6)]:
    s = (mag>=lo)&(mag<hi)
    if s.sum() < 20: continue
    print(f"{f'[{lo},{hi})':>14}{s.sum():>7}{np.median(E[s]):10.4f}{np.percentile(E[s],95):10.4f}"
          f"{100*(E[s]>1).mean():9.2f}{np.median(E[s]/np.maximum(mag[s],1e-6)):11.3f}")
