"""Rebuild the PyTorch networks from deployed ONNX graphs, so a deployment that ships only .onnx
files (the inference repo, a general user) can still fine-tune.

The exporter (torch.onnx, eval mode, constant folding) FOLDS every conv_bn_relu's BatchNorm into
its conv: the graph carries the fused weight and a fused bias under generated names
("onnx::Conv_NNN"), while unfused tensors keep their state_dict names. Node names keep the module
scope ("/e1/e1.0/e1.0.0/Conv"), so every tensor maps back to an exact module path. A folded conv
gets the fused weight and its BatchNorm becomes an exact identity carrying the fused bias
(weight 1, running_mean 0, running_var 1 - eps, so sqrt(var + eps) == 1).

CONSEQUENCE FOR TRAINING: the imported BatchNorms are only correct in eval mode. In train mode
they would normalise by batch statistics and discard the folded calibration, so a checkpoint
built here is marked bn_folded=True and the fine-tune entry holds every BatchNorm in eval.

The architecture is not in the graph's metadata, so it comes from a config (detector_kwargs).
Four keys change no parameter shape at all (attn_heads, rope_lambda_min_cells, xsa,
gates_enabled), so a strict state_dict load cannot validate them; verify=True therefore runs
the rebuilt module against onnxruntime and refuses a mismatch.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch


def _module_path(node_name):
    """'/e1/e1.0/e1.0.0/Conv' -> 'e1.0.0'; '/gate3/wx/Conv' -> 'gate3.wx';
    '/blocks.0/mlp/mlp.0/MatMul' -> 'blocks.0.mlp.0'. Scope components accumulate through
    containers (Sequential/ModuleList children repeat their parent's name) and reset at plain
    modules (an attribute name)."""
    parts = []
    for comp in node_name.strip("/").split("/")[:-1]:
        if parts and comp.startswith(parts[-1] + "."):
            parts[-1] = comp
        else:
            parts.append(comp)
    return ".".join(parts)


def state_dict_from_onnx(onnx_path, model):
    """ONNX initializers -> a strict-loadable state_dict for `model`, undoing BN folding."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(onnx_path)).graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    # The exporter DEDUPLICATES identical initializers (e.g. every freshly initialised LayerNorm weight is
    # all ones) and hands each duplicate back through an Identity node: n2.weight = Identity(n1.weight).
    # Resolve those aliases so every tensor is found under every name the graph uses for it.
    for node in graph.node:
        if node.op_type == "Identity" and node.input[0] in inits and node.output[0] not in inits:
            inits[node.output[0]] = inits[node.input[0]]
    ref = model.state_dict()
    modules = dict(model.named_modules())
    sd = {}
    for node in graph.node:
        mod = _module_path(node.name)
        if node.op_type == "Conv":
            w = inits[node.input[1]]
            b = inits[node.input[2]] if len(node.input) > 2 and node.input[2] else None
            if node.input[1] == f"{mod}.weight":          # unfolded conv: named initializers
                sd[f"{mod}.weight"] = w
                if b is not None:
                    sd[f"{mod}.bias"] = b
                continue
            parent, idx = mod.rsplit(".", 1)              # folded: its BN is the next Sequential slot
            bn = f"{parent}.{int(idx) + 1}"
            if f"{bn}.running_var" not in ref:
                raise ValueError(f"{onnx_path}: conv {node.name} looks BN-folded but {bn} is not a BatchNorm")
            eps = modules[bn].eps
            sd[f"{mod}.weight"] = w
            sd[f"{bn}.weight"] = np.ones_like(b)
            sd[f"{bn}.bias"] = b
            sd[f"{bn}.running_mean"] = np.zeros_like(b)
            sd[f"{bn}.running_var"] = np.full_like(b, 1.0 - eps)
            sd[f"{bn}.num_batches_tracked"] = np.array(0)
        elif node.op_type == "MatMul" and node.input[1] in inits:    # nn.Linear: x @ W^T stored as W^T
            sd[f"{mod}.weight"] = inits[node.input[1]].T
        # LayerNorm and Linear-bias tensors are read through their NODES, not their names: the exporter
        # deduplicates identical initializers, so freshly initialised LayerNorms (all ones / all zeros)
        # share ONE tensor and every name but the first vanishes from the graph.
        elif node.op_type == "LayerNormalization" and f"{mod}.weight" in ref:
            sd[f"{mod}.weight"] = inits[node.input[1]]
            if len(node.input) > 2 and node.input[2] in inits:
                sd[f"{mod}.bias"] = inits[node.input[2]]
        elif (node.op_type == "Add" and isinstance(modules.get(mod), torch.nn.Linear)
              and f"{mod}.bias" not in sd):
            b = next((inits[i] for i in node.input if i in inits), None)
            if b is not None and b.shape == tuple(ref[f"{mod}.bias"].shape):
                sd[f"{mod}.bias"] = b
    for name, arr in inits.items():                    # remaining named tensors (gate params, ...)
        if name in ref and name not in sd:
            sd[name] = arr
    missing, unexpected = sorted(set(ref) - set(sd)), sorted(set(sd) - set(ref))
    if missing or unexpected:
        raise ValueError(f"{onnx_path} does not match this architecture: missing {missing[:6]}, "
                         f"unexpected {unexpected[:6]}")
    out = {}
    for k, v in sd.items():
        # np.array, not ascontiguousarray: the latter promotes 0-d (num_batches_tracked) to shape (1,),
        # and onnx's arrays are read-only views that torch.from_numpy warns about
        t = torch.from_numpy(np.array(v, copy=True)).to(ref[k].dtype)
        if t.shape != ref[k].shape:
            raise ValueError(f"{onnx_path}: {k} has shape {tuple(t.shape)}, the config builds {tuple(ref[k].shape)}")
        out[k] = t
    return out


def onnx_parity(model, onnx_path, shape, n=2, seed=0):
    """max |ORT(graph) - torch(model)| over n random inputs, CPU fp32 on both sides."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    # the graph's own static dims win (export_parity exported a batch-1 refiner; the deployed one is dynamic)
    shape = tuple(d if isinstance(d, int) else s for d, s in zip(sess.get_inputs()[0].shape, shape))
    rng = np.random.default_rng(seed)
    worst = 0.0
    model = model.eval().cpu()
    with torch.no_grad():
        for _ in range(n):
            x = rng.random(shape, dtype=np.float32)
            outs = model(torch.from_numpy(x))
            outs = outs if isinstance(outs, tuple) else (outs,)
            for o, p in zip(sess.run(None, {name: x}), outs):
                worst = max(worst, float(np.abs(o - p.numpy()).max()))
    return worst


def detector_from_onnx(onnx_path, cfg, verify=True, tol=1e-3):
    """DetectorNet built from cfg (detector_kwargs + board) with the graph's weights."""
    from dcc.board import n_corners
    from dcc.model import DetectorNet, detector_kwargs

    W, H = cfg["input_size"]
    model = DetectorNet(H, W, n_cls=n_corners(cfg.get("board")), **detector_kwargs(cfg)).eval()
    model.load_state_dict(state_dict_from_onnx(onnx_path, model), strict=True)
    diff = onnx_parity(model, onnx_path, (1, 1, H, W)) if verify else None
    if verify and diff > tol:
        raise ValueError(f"{onnx_path}: the rebuilt network disagrees with the graph by {diff:.3g} -- the config's "
                         "shape-invisible keys (attn_heads, rope_lambda_min_cells, xsa, gates_enabled) do not match")
    return model, diff


def refiner_from_onnx(onnx_path, verify=True, tol=1e-3):
    """Refiner with its width read off the graph (body.1's out-channels, as refiner_for does)."""
    import onnx
    from onnx import numpy_helper

    from dcc.model import Refiner

    graph = onnx.load(str(onnx_path)).graph
    inits = {i.name: i for i in graph.initializer}
    conv = next(n for n in graph.node if n.op_type == "Conv" and _module_path(n.name) == "body.1.0")
    c64 = numpy_helper.to_array(inits[conv.input[1]]).shape[0]
    model = Refiner(width_mult=c64 / 64.0).eval()
    model.load_state_dict(state_dict_from_onnx(onnx_path, model), strict=True)
    diff = onnx_parity(model, onnx_path, (4, 1, 24, 24)) if verify else None
    if verify and diff > tol:
        raise ValueError(f"{onnx_path}: rebuilt refiner disagrees with the graph by {diff:.3g}")
    return model, diff


def ckpt_from_onnx(onnx_path, model, cfg, parity):
    """A dcc.trainutil.save_ckpt-shaped dict (model == ema), so train_detector's finetune: entry and
    every checkpoint loader read it unchanged. bn_folded tells the trainer to hold BatchNorm in eval."""
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    data = Path(onnx_path).read_bytes()
    return {"step": 0, "resume_count": 0, "model": sd, "ema": sd, "optim": None, "cfg": cfg,
            "git_hash": "onnx-import", "torch_rng": torch.get_rng_state(), "last_val": None,
            "bn_folded": True,
            "imported_from_onnx": {"path": str(onnx_path), "sha256": hashlib.sha256(data).hexdigest(),
                                   "max_abs_diff_vs_ort": parity}}


def load_detector(path, cfg=None, device="cpu"):
    """Detector from a trainer checkpoint (.pt: its OWN cfg, EMA preferred) or a graph (.onnx: cfg
    required). Returns (model, cfg_used)."""
    path = Path(path)
    if path.suffix == ".onnx":
        assert cfg is not None, "an .onnx detector needs --config: the graph does not carry its architecture"
        model, _ = detector_from_onnx(path, cfg)
        return model.to(device).eval(), cfg
    from dcc.board import n_corners
    from dcc.model import DetectorNet, detector_kwargs

    ck = torch.load(path, map_location="cpu", weights_only=False)
    ccfg = ck["cfg"]
    W, H = ccfg["input_size"]
    model = DetectorNet(H, W, n_cls=n_corners(ccfg.get("board")), **detector_kwargs(ccfg))
    model.load_state_dict(ck.get("ema") or ck["model"])
    return model.to(device).eval(), ccfg


def load_refiner(path, device="cpu"):
    path = Path(path)
    if path.suffix == ".onnx":
        model, _ = refiner_from_onnx(path)
        return model.to(device).eval()
    from dcc.model import refiner_for

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("ema") or ck["model"]
    model = refiner_for(sd)
    model.load_state_dict(sd)
    return model.to(device).eval()
