"""tools/cls_form_gradient_share.py -- Gradient share of the two heads on the shared trunk, per class-head loss form, at FIXED weights.
For each checkpoint: same 128 SynthVal frames (stratified over s), bf16 autocast forward as in training,
L_hm = BCE(hm)/N, L_cls = lambda * {focal|bce}(cls)/N. Reports loss values, trunk-gradient L2 norms from each
term alone, their ratio, and the cosine between the two trunk gradients. The class form is swapped on the
same weights, so the form's effect is isolated from the training trajectory."""
import sys, json, torch, numpy as np; sys.path.insert(0, ".")
from dcc.model import DetectorNet, detector_kwargs
from dcc.board import n_corners
from dcc.dataset import SynthVal, _render_detector_targets
from dcc.losses import strict_bce, focal
torch.manual_seed(0)
CK = {"bce_bce_35k": "runs/abl_wh_ce_35k_rev6/ckpt_0035000.pt",
      "bce_focal_35k": "runs/abl_lx_wh_hmce_clsfocal_35k_rev6/ckpt_0035000.pt",
      "release_882k_100k": "runs/rel_w882_c2_100k_rev6/ckpt_0100000.pt"}
dev = "cuda"; N_FR, B = 128, 16
out = {"frames": N_FR, "batch": B, "note": __doc__.strip(), "rows": {}}
ds_cache = None
for name, path in CK.items():
    ck = torch.load(path, map_location="cpu", weights_only=False); cfg = ck["cfg"]
    W, H = cfg["input_size"]; m = DetectorNet(H, W, n_cls=n_corners(cfg.get("board")), **detector_kwargs(cfg)).to(dev)
    m.load_state_dict(ck["model"]); m.train(False)
    trunk = [(n, p) for n, p in m.named_parameters() if not (n.startswith("hm.") or n.startswith("cls."))]
    tp = [p for _, p in trunk]
    if ds_cache is None:
        ds = SynthVal(cfg, 10000); idx = [k * 10000 // N_FR for k in range(N_FR)]
        samples = [_render_detector_targets(cfg, ds[i][1]) for i in idx]; ds_cache = samples
    samples = ds_cache
    lam = float(cfg["lambda_cls"]); own = cfg.get("loss_form_cls") or cfg.get("loss_form", "focal")
    acc = {f: {"L_hm": [], "L_cls": [], "g_hm": [], "g_cls": [], "cos": [], "g_cls_blocks": [], "g_hm_blocks": []} for f in ("focal", "ce")}
    for b0 in range(0, N_FR, B):
        bt = samples[b0:b0 + B]
        x = torch.stack([s["image"] for s in bt]).to(dev); hm_t = torch.stack([s["heatmap"] for s in bt]).unsqueeze(1).to(dev)
        cls_t = torch.stack([s["classes"] for s in bt]).to(dev); n = max(float(sum(s["n_vis"] for s in bt)), 1.0)
        for form in ("focal", "ce"):
            m.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hm_l, cls_l = m(x)
            L_hm = strict_bce(hm_l, hm_t) / n
            L_cls = lam * (strict_bce(cls_l, cls_t) if form == "ce" else focal(cls_l, cls_t)) / n
            g_hm = torch.autograd.grad(L_hm, tp, retain_graph=True, allow_unused=True)
            g_cls = torch.autograd.grad(L_cls, tp, retain_graph=False, allow_unused=True)
            z = lambda g, p: (g if g is not None else torch.zeros_like(p))
            v_hm = torch.cat([z(g, p).flatten() for g, p in zip(g_hm, tp)]); v_cls = torch.cat([z(g, p).flatten() for g, p in zip(g_cls, tp)])
            bl = [i for i, (nme, _) in enumerate(trunk) if nme.startswith("blocks.")]
            vb_hm = torch.cat([z(g_hm[i], tp[i]).flatten() for i in bl]); vb_cls = torch.cat([z(g_cls[i], tp[i]).flatten() for i in bl])
            a = acc[form]; a["L_hm"].append(L_hm.item()); a["L_cls"].append(L_cls.item())
            a["g_hm"].append(v_hm.norm().item()); a["g_cls"].append(v_cls.norm().item())
            a["cos"].append(torch.nn.functional.cosine_similarity(v_hm, v_cls, dim=0).item())
            a["g_hm_blocks"].append(vb_hm.norm().item()); a["g_cls_blocks"].append(vb_cls.norm().item())
    row = {"ckpt": path, "trained_cls_form": own, "lambda_cls": lam, "trunk_params": int(sum(p.numel() for p in tp))}
    for form, a in acc.items():
        mean = lambda k: float(np.mean(a[k]))
        row[form] = {"L_hm": mean("L_hm"), "L_cls": mean("L_cls"), "loss_ratio_cls_over_hm": mean("L_cls") / mean("L_hm"),
                     "trunk_grad_norm_hm": mean("g_hm"), "trunk_grad_norm_cls": mean("g_cls"),
                     "trunk_grad_ratio_cls_over_hm": mean("g_cls") / mean("g_hm"),
                     "cls_share_of_trunk_grad": mean("g_cls") / (mean("g_cls") + mean("g_hm")),
                     "blocks_grad_ratio_cls_over_hm": mean("g_cls_blocks") / mean("g_hm_blocks"),
                     "cosine_hm_vs_cls_trunk_grad": mean("cos")}
    out["rows"][name] = row
    print(f"\n=== {name} (trained with cls={own}, lambda {lam}) ===")
    for form in ("ce", "focal"):
        r = row[form]
        print(f"  cls form {form:>5}: L_hm {r['L_hm']:.4f}  L_cls {r['L_cls']:.4f} (x{r['loss_ratio_cls_over_hm']:.2f})  "
              f"|g_trunk| hm {r['trunk_grad_norm_hm']:.3e} cls {r['trunk_grad_norm_cls']:.3e} -> cls/hm {r['trunk_grad_ratio_cls_over_hm']:.2f}, "
              f"cls share {100*r['cls_share_of_trunk_grad']:.0f}%; on blocks cls/hm {r['blocks_grad_ratio_cls_over_hm']:.2f}; cos(g_hm, g_cls) {r['cosine_hm_vs_cls_trunk_grad']:+.3f}")
    del m; torch.cuda.empty_cache()
json.dump(out, open("paper/results_rev6/12_loss/cls_form_gradient_share.json", "w"), indent=1)
print("\n-> paper/results_rev6/12_loss/cls_form_gradient_share.json")
