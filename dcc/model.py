"""DetectorNet (Stage-1 hybrid conv encoder -> MHSA bottleneck with 2D axial
RoPE -> gated staggered-tap decoder, two factorised heads) and Refiner
(Stage-2 sub-pixel corner refiner). Architecture map:

    e1 -> e2 -> e3 -> e4 -> e5      conv encoder, H -> H/16, widths 32-64-128-256-256
                          |
                       blocks        MHSA+MLP x n_blocks, axial RoPE, over H/16 tokens
                          |
    d4 <- gate4(e4) <----+          H/16 -> H/8
    d3 <- gate3(e3) <- d4           H/8  -> H/4   --tap--> cls (H/4, n_cls ch)
    d2 <-       e2  <- d3           H/4  -> H/2, ungated
    d1 <-       e1  <- d2           H/2  -> H,   ungated  --tap--> hm  (H, 1ch)

Two-grids: "attention cells" are the H/16 bottleneck tokens (16 px each) --
the sole board-scale context mechanism. "class cells" are the H/4 grid the
class head reads out on (4 px each, identity not context). Don't conflate them.

freeze_trunk boundary: retargeting to a new board freezes everything except
cls.* -- trunk, both gates, the decoder and the heatmap head transfer to a new
board unchanged; only the class head (board-specific identity) retrains.

Three design choices are called out again at their own site below, since
they depart from what a textbook hybrid conv-transformer encoder would do:
the RoPE spectrum is wavelength- rather than base-anchored (AxialRoPE), qkv
is indexed rather than tuple-unpacked for ONNX-tracer safety (Block.qkv_heads),
and the attention gates zero their psi weight at init rather than relying on
bias alone (AttnGate). Both networks return RAW LOGITS -- no sigmoid inside
forward; losses.py / the pipeline apply it. ONNX-safe throughout: no complex
dtypes, no grid_sample, no shape-dependent control flow beyond batch size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(cin, cout, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def double_conv(cin, cout):
    return nn.Sequential(conv_bn_relu(cin, cout), conv_bn_relu(cout, cout))


def up2(x):
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class AxialRoPE(nn.Module):
    """2D axial RoPE, WAVELENGTH-ANCHORED spectrum: rather than the common single-`base`
    parameterisation (`base**(-i/(n-1))`), lambda_min and lambda_max are set directly.
    """

    def __init__(self, head_dim, h, w, lambda_min=2.5, lambda_max=None):
        super().__init__()
        n = head_dim // 4
        assert n >= 2, "head_dim // 4 must be >= 2 for geometric interpolation"
        if lambda_max is None:
            lambda_max = 2.0 * max(h, w)
        i = torch.arange(n, dtype=torch.float32)
        wavelengths = lambda_min * (lambda_max / lambda_min) ** (i / (n - 1))
        freqs = 2 * torch.pi / wavelengths
        row = torch.arange(h, dtype=torch.float32)[:, None] * freqs
        col = torch.arange(w, dtype=torch.float32)[:, None] * freqs
        ang = torch.cat([row[:, None, :].expand(h, w, n),
                         col[None, :, :].expand(h, w, n)], -1).reshape(h * w, 2 * n)
        self.register_buffer("cos", ang.cos()[None, None], persistent=False)
        self.register_buffer("sin", ang.sin()[None, None], persistent=False)

    def forward(self, x):
        x1, x2 = x.chunk(2, -1)
        return torch.cat([x1 * self.cos - x2 * self.sin,
                          x1 * self.sin + x2 * self.cos], -1)


class Block(nn.Module):
    """Pre-norm MHSA + MLP (timm vision_transformer.Block shape), RoPE on Q,K."""

    def __init__(self, d, heads, rope, mlp_ratio=4, xsa=False):
        super().__init__()
        self.heads, self.rope, self.xsa = heads, rope, xsa
        self.n1, self.n2 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(),
                                 nn.Linear(mlp_ratio * d, d))

    def qkv_heads(self, x_normed):
        """(B,T,d) -> rotated q, k and v, each (B, heads, T, head_dim)."""
        B, T, d = x_normed.shape
        qkv = (self.qkv(x_normed).reshape(B, T, 3, self.heads, d // self.heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        return self.rope(q), self.rope(k), v

    def forward(self, x):
        q, k, v = self.qkv_heads(self.n1(x))
        o = F.scaled_dot_product_attention(q, k, v)
        if self.xsa:
            vn = F.normalize(v, dim=-1)
            o = o - (o * vn).sum(-1, keepdim=True) * vn
        x = x + self.proj(o.transpose(1, 2).reshape(x.shape))
        return x + self.mlp(self.n2(x))


class AttnGate(nn.Module):
    """Oktay additive gate (Attention U-Net eq. 1-2): alpha = sigmoid(psi(relu(Wx*skip_down +
    Wg*g))), bilinearly upsampled to skip res, elementwise on the skip.
    """

    def __init__(self, skip_ch, gate_ch, inter_ch):
        super().__init__()
        self.wx = nn.Conv2d(skip_ch, inter_ch, 1, stride=2, bias=False)
        self.wg = nn.Conv2d(gate_ch, inter_ch, 1)
        self.psi = nn.Conv2d(inter_ch, 1, 1)
        nn.init.zeros_(self.psi.weight)
        nn.init.constant_(self.psi.bias, 3.0)

    def alpha(self, skip, g):
        a = torch.sigmoid(self.psi(F.relu(self.wx(skip) + self.wg(g))))
        return F.interpolate(a, size=skip.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, skip, g):
        return skip * self.alpha(skip, g)


class DetectorNet(nn.Module):
    """Stage-1 network -- see module docstring for the architecture map, the two-grids note and the
    freeze_trunk boundary.
    """

    def __init__(self, h, w, d=256, heads=8, n_blocks=2, rope_lambda_min=2.5, n_cls=16, attend_div=16,
                 gates=True, width_mult=1.0,
                xsa=False, e4_dilated=True):
        super().__init__()
        def c(n):
            return max(8, int(round(n * width_mult / 8)) * 8)
        d = c(d)
        assert d % heads == 0, f"scaled d={d} not divisible by heads={heads}"
        self.width_mult = width_mult
        assert attend_div in (8, 16), f"attend_div must be 8 or 16, got {attend_div}"
        assert h % 16 == 0 and w % 16 == 0, f"h, w must be multiples of 16, got {(h, w)}"
        self.attend_div = attend_div
        self.pool = nn.MaxPool2d(2)
        self.e1 = double_conv(1, c(32))
        self.e2 = double_conv(c(32), c(64))
        self.e3 = double_conv(c(64), c(128))
        self.e4_dilated = e4_dilated

        def _deep(cin, cout):
            core = double_conv(cin, cout)
            if not e4_dilated:
                return core
            return nn.Sequential(core,
                                 conv_bn_relu(cout, cout, dilation=2),
                                 conv_bn_relu(cout, cout, dilation=4))

        if attend_div == 16:
            self.e4 = double_conv(c(128), c(256))
            self.e5 = _deep(c(256), c(256))
            grid_h, grid_w = h // 16, w // 16
        else:
            self.e4 = _deep(c(128), c(256))
            grid_h, grid_w = h // 8, w // 8
        self.rope = AxialRoPE(d // heads, grid_h, grid_w, rope_lambda_min)
        self.blocks = nn.ModuleList(Block(d, heads, self.rope, xsa=xsa) for _ in range(n_blocks))
        self.gates_on = gates
        self.norm = nn.LayerNorm(d, eps=1e-6)
        if attend_div == 16:
            self.gate4 = AttnGate(c(256), c(256), c(128))
            self.gate3 = AttnGate(c(128), c(256), c(64))
            self.d4 = conv_bn_relu(c(256) + c(256), c(256))
            self.d3 = conv_bn_relu(c(256) + c(128), c(128))
        else:
            self.gate3 = AttnGate(c(128), c(256), c(64))
            self.d3 = conv_bn_relu(c(256) + c(128), c(128))
        self.d2 = conv_bn_relu(c(128) + c(64), c(64))
        self.d1 = conv_bn_relu(c(64) + c(32), c(32))
        self.hm = nn.Sequential(nn.Conv2d(c(32), c(32), 3, padding=1), nn.ReLU(inplace=True),
                                nn.Conv2d(c(32), 1, 1))
        self.cls = nn.Sequential(nn.Conv2d(c(128), c(128), 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(c(128), n_cls, 1))
        nn.init.constant_(self.hm[-1].bias, -2.19)
        nn.init.constant_(self.cls[-1].bias, -2.19)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        if self.attend_div == 16:
            s4 = self.e4(self.pool(s3))
            z = self.e5(self.pool(s4))
        else:
            z = self.e4(self.pool(s3))
        B, C, Hb, Wb = z.shape
        t = z.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            t = blk(t)
        z = self.norm(t).transpose(1, 2).reshape(B, C, Hb, Wb)
        if self.attend_div == 16:
            y = self.d4(torch.cat([up2(z), self.gate4(s4, z) if self.gates_on else s4], 1))
            y = self.d3(torch.cat([up2(y), self.gate3(s3, y) if self.gates_on else s3], 1))
        else:
            y = self.d3(torch.cat([up2(z), self.gate3(s3, z) if self.gates_on else s3], 1))
        cls = self.cls(y)
        y = self.d2(torch.cat([up2(y), s2], 1))
        y = self.d1(torch.cat([up2(y), s1], 1))
        return self.hm(y), cls


def detector_kwargs(cfg):
    """DetectorNet's config-tunable constructor kwargs (attend_div, n_blocks, heads,
    rope_lambda_min, xsa), read from a loaded cfg dict with the exact same defaults
    DetectorNet's own signature uses -- the one place tools/train_detector.py and
    tools/preflight.py both read these from, so the two model-construction sites can't drift
    apart. n_cls is deliberately excluded: it's derived from cfg["board"] via
    dcc.board.n_corners, not a flat cfg key.
    """
    return {"attend_div": cfg.get("attend_div", 16), "n_blocks": cfg.get("attn_blocks", 2),
            "heads": cfg.get("attn_heads", 8), "rope_lambda_min": cfg.get("rope_lambda_min_cells", 2.5),
            "xsa": cfg.get("xsa", False),
            "gates": cfg.get("gates_enabled", True),
            "width_mult": cfg.get("width_mult", 1.0),
            "e4_dilated": cfg.get("e4_dilated", True)}


class Refiner(nn.Module):
    """Sub-pixel refiner."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(conv_bn_relu(1, 32), conv_bn_relu(32, 64), conv_bn_relu(64, 64))
        self.post = conv_bn_relu(64, 64)
        self.out = nn.Conv2d(64, 64, 1)
        self.ps = nn.PixelShuffle(8)
        nn.init.constant_(self.out.bias, -2.19)

    def forward(self, x):
        f = self.body(x)[:, :, 8:16, 8:16]
        f = self.post(f)
        return self.ps(self.out(f))
