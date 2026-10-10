"""Deployed ONNX -> PyTorch. detector_from_onnx(path, cfg) and refiner_from_onnx(path) rebuild the networks
and verify them against onnxruntime (folded BatchNorm becomes an identity: keep BatchNorm in eval mode);
ckpt_from_onnx(path, model, cfg, diff) wraps one as a trainer checkpoint; load_detector / load_refiner
accept a .pt or an .onnx.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch


def _module_path(node_name):
    parts = []
    for comp in node_name.strip("/").split("/")[:-1]:
        if parts and comp.startswith(parts[-1] + "."):
            parts[-1] = comp
        else:
            parts.append(comp)
    return ".".join(parts)


def state_dict_from_onnx(onnx_path, model):
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(onnx_path)).graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
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
            if node.input[1] == f"{mod}.weight":
                sd[f"{mod}.weight"] = w
                if b is not None:
                    sd[f"{mod}.bias"] = b
                continue
            parent, idx = mod.rsplit(".", 1)
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
        elif node.op_type == "MatMul" and node.input[1] in inits:
            sd[f"{mod}.weight"] = inits[node.input[1]].T
        elif node.op_type == "LayerNormalization" and f"{mod}.weight" in ref:
            sd[f"{mod}.weight"] = inits[node.input[1]]
            if len(node.input) > 2 and node.input[2] in inits:
                sd[f"{mod}.bias"] = inits[node.input[2]]
        elif (node.op_type == "Add" and isinstance(modules.get(mod), torch.nn.Linear)
              and f"{mod}.bias" not in sd):
            b = next((inits[i] for i in node.input if i in inits), None)
            if b is not None and b.shape == tuple(ref[f"{mod}.bias"].shape):
                sd[f"{mod}.bias"] = b
    for name, arr in inits.items():
        if name in ref and name not in sd:
            sd[name] = arr
    missing, unexpected = sorted(set(ref) - set(sd)), sorted(set(sd) - set(ref))
    if missing or unexpected:
        raise ValueError(f"{onnx_path} does not match this architecture: missing {missing[:6]}, "
                         f"unexpected {unexpected[:6]}")
    out = {}
    for k, v in sd.items():
        t = torch.from_numpy(np.array(v, copy=True)).to(ref[k].dtype)
        if t.shape != ref[k].shape:
            raise ValueError(f"{onnx_path}: {k} has shape {tuple(t.shape)}, the config builds {tuple(ref[k].shape)}")
        out[k] = t
    return out


def onnx_parity(model, onnx_path, shape, n=2, seed=0):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
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
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    data = Path(onnx_path).read_bytes()
    return {"step": 0, "resume_count": 0, "model": sd, "ema": sd, "optim": None, "cfg": cfg,
            "git_hash": "onnx-import", "torch_rng": torch.get_rng_state(), "last_val": None,
            "bn_folded": True,
            "imported_from_onnx": {"path": str(onnx_path), "sha256": hashlib.sha256(data).hexdigest(),
                                   "max_abs_diff_vs_ort": parity}}


def load_detector(path, cfg=None, device="cpu"):
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
