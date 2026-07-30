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
    """2D axial RoPE, WAVELENGTH-ANCHORED spectrum: rather than the common
    single-`base` parameterisation (`base**(-i/(n-1))`), lambda_min and
    lambda_max are set directly. A single `base` can't hit both ends this
    needs at once -- a short wavelength of ~2 cells for fine positional
    detail, and a long wavelength spanning the full grid for the
    no-global-alias guarantee below -- so the anchor is expressed directly in
    wavelengths instead. Rotate-half layout: pair i = (x[i], x[i+d/2]) shares
    angle i; pairs 0..n-1 carry ROW
    phases, pairs n..2n-1 COL phases (n = head_dim // 4 per axis, freqs shared
    across axes). omega_i = 2*pi/lambda_i, lambda_i geometric from lambda_min
    (fastest, i=0) to lambda_max (slowest, i=n-1; default 2*max(h,w), i.e.
    twice the grid extent).

    Per-pair aliasing of fast frequencies is BY DESIGN (a vernier: one fast
    pair alone repeats often, but no two pairs share a period). The
    no-global-alias GUARANTEE rests on the slowest pair alone: lambda_max >=
    2 * grid extent means it can't complete a half-cycle within the grid, so
    it's injective over every in-frame offset -- the full phase vector never
    repeats for two distinct positions (test_rope_no_global_alias).

    Buffers fixed to the (h, w) token grid at build time, computed fp32."""

    def __init__(self, head_dim, h, w, lambda_min=2.5, lambda_max=None):
        super().__init__()
        n = head_dim // 4
        assert n >= 2, "head_dim // 4 must be >= 2 for geometric interpolation"
        if lambda_max is None:
            lambda_max = 2.0 * max(h, w)                   # no-global-alias guarantee
        i = torch.arange(n, dtype=torch.float32)
        wavelengths = lambda_min * (lambda_max / lambda_min) ** (i / (n - 1))
        freqs = 2 * torch.pi / wavelengths                 # omega_i, fast -> slow
        row = torch.arange(h, dtype=torch.float32)[:, None] * freqs   # (h, n)
        col = torch.arange(w, dtype=torch.float32)[:, None] * freqs   # (w, n)
        ang = torch.cat([row[:, None, :].expand(h, w, n),
                         col[None, :, :].expand(h, w, n)], -1).reshape(h * w, 2 * n)
        # persistent=False: purely a function of (head_dim, h, w, lambda_min) with
        # zero learned content, so it's recomputed fresh at construction rather than
        # serialized -- keeps it out of state_dict (the freeze_trunk/checkpoint
        # contract is e1..e5/blocks/norm/gate3/gate4/d1..d4/hm/cls only) and avoids a
        # stale-grid buffer silently overriding a freshly-built one on load.
        self.register_buffer("cos", ang.cos()[None, None], persistent=False)   # (1,1,T,d/2)
        self.register_buffer("sin", ang.sin()[None, None], persistent=False)

    def forward(self, x):                                             # (B, heads, T, head_dim)
        # cos/sin are fp32 buffers; under autocast x may arrive bf16, and the
        # product's dtype follows ordinary type promotion (fp32 wins), NOT
        # autocast's op list -- so the rotation is always computed at fp32,
        # regardless of surrounding precision (verified empirically: no
        # dtype downcast happens here). F.scaled_dot_product_attention right
        # after is itself autocast-eligible and will re-homogenise q/k (now
        # fp32) against v to a common dtype before the matmul -- the
        # guarantee is on the rotation, not carried through SDPA, which is
        # fine: that's SDPA's own already-accepted precision tradeoff.
        x1, x2 = x.chunk(2, -1)
        return torch.cat([x1 * self.cos - x2 * self.sin,
                          x1 * self.sin + x2 * self.cos], -1)


class Block(nn.Module):
    """Pre-norm MHSA + MLP (timm vision_transformer.Block shape), RoPE on Q,K."""

    def __init__(self, d, heads, rope, mlp_ratio=4, xsa=False):
        super().__init__()
        self.heads, self.rope, self.xsa = heads, rope, xsa
        self.n1, self.n2 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)  # eps per timm/TransUNet refs
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(),
                                 nn.Linear(mlp_ratio * d, d))

    def qkv_heads(self, x_normed):
        """(B,T,d) -> rotated q, k and v, each (B, heads, T, head_dim)."""
        B, T, d = x_normed.shape
        qkv = (self.qkv(x_normed).reshape(B, T, 3, self.heads, d // self.heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]        # indexed, not tuple-unpacked over
        return self.rope(q), self.rope(k), v    # a tensor -- avoids the ONNX tracer warning

    def forward(self, x):
        q, k, v = self.qkv_heads(self.n1(x))
        o = F.scaled_dot_product_attention(q, k, v)
        if self.xsa:
            # Exclusive self-attention (Zhai, arXiv:2603.09078, Eq. 2 / Algorithm
            # 1): project each head's own value direction out of that head's
            # attention output, PER HEAD and BEFORE proj (W_O) -- not after, and
            # not after head concatenation (an automated summary of the paper got
            # both wrong; this is transcribed from the paper's own pseudocode).
            # No parameters, default off. Motivation/measurement:
            # paper/xsa_value_correlation.md, ablations.md A14.
            vn = F.normalize(v, dim=-1)
            o = o - (o * vn).sum(-1, keepdim=True) * vn
        x = x + self.proj(o.transpose(1, 2).reshape(x.shape))
        return x + self.mlp(self.n2(x))


class AttnGate(nn.Module):
    """Oktay additive gate (Attention U-Net eq. 1-2): alpha =
    sigmoid(psi(relu(Wx*skip_down + Wg*g))), bilinearly upsampled to skip res,
    elementwise on the skip. Pass-through init: psi.weight ZEROED and
    psi.bias = +3.0, so alpha == sigmoid(3) ~= 0.953 EXACTLY constant at step
    0 -- the zero weight kills all input-dependence outright. Biasing alone
    (e.g. +5, giving alpha ~= 0.99) would still leave alpha technically
    input-dependent at init; zeroing the weight is what makes the gate a true
    no-op at step 0, so training decides when to start gating instead of
    fighting a small residual input-dependence from the start."""

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
    """Stage-1 network -- see module docstring for the architecture map, the
    two-grids note and the freeze_trunk boundary. Module names (e1..e5,
    pool, rope, blocks, norm, gate4, gate3, d4, d3, d2, d1, hm, cls) are a
    stable contract: freeze_trunk and checkpoint transfer key off them.

    attend_div selects the encoder depth / attention grid: 16 (default) is
    this native 5-stage shape unchanged bit-for-bit (existing checkpoints
    load as before). 8 is the 4-stage variant sized for 640x480 input: e1..e4
    (widths 32-64-128-256, the dilated rate-2,4 pair moves into e4 as the
    now-deepest stage), attention over the H/8 grid, a single gate3 on the
    e3 skip fed directly by the bottleneck output, decoder d3..d1 -- e5, d4
    and gate4 simply don't exist as modules in this mode. Everything else
    (RoPE construction, gate/head shapes, inits) is identical between the
    two; the H/4 class tap and full-res heatmap tap are unaffected because
    both variants' decoders reach those rungs the same way.

    A few choices below are convention rather than hard requirements, and
    can be revisited without breaking the contract above: MLP ratio 4 (timm
    Block default); single 3x3 conv per decoder stage; trailing LayerNorm
    after the blocks; RoPE frequencies geometric per axis (wavelength- rather
    than base-anchored here, see AxialRoPE).

    xsa (default False, see Block.forward) is an optional, parameter-free
    exclusive-self-attention correction threaded into every Block; default
    off is a bit-for-bit no-op (test_xsa_default_off_is_noop)."""

    def __init__(self, h, w, d=256, heads=8, n_blocks=2, rope_lambda_min=2.5, n_cls=16, attend_div=16,
                 gates=True, width_mult=1.0,
                xsa=False, e4_dilated=True):
        super().__init__()
        # width_mult (ablation A-WIDTH): scales EVERY channel count, including the
        # attention dim d. Tests whether the network is over-parameterised for a
        # single-object, fixed-geometry task -- if 0.5x width matches 1.0x, the
        # architecture comparison (A1) is being masked by surplus capacity and no
        # architectural difference could show. Rounded to multiples of 8 to honour
        # the P10 export pin, and d stays divisible by `heads`.
        def c(n):
            return max(8, int(round(n * width_mult / 8)) * 8)
        d = c(d)
        assert d % heads == 0, f"scaled d={d} not divisible by heads={heads}"
        self.width_mult = width_mult
        assert attend_div in (8, 16), f"attend_div must be 8 or 16, got {attend_div}"
        # /16 kept for both variants, not just the native one: attend_div=8's own
        # pool/attention chain only needs h,w % 8 == 0, but every config so far
        # (default.yaml, rev640.yaml) is meant to be usable by either variant
        # interchangeably, and %16 is the stricter (hence safe-for-both) bound.
        assert h % 16 == 0 and w % 16 == 0, f"h, w must be multiples of 16, got {(h, w)}"
        self.attend_div = attend_div
        self.pool = nn.MaxPool2d(2)
        self.e1 = double_conv(1, c(32))
        self.e2 = double_conv(c(32), c(64))
        self.e3 = double_conv(c(64), c(128))
        # e4_dilated=False (ablation A-NODILATE): DROP the dilated (2, 4) pair from the
        # deepest encoder stage. Kaelin, 2026-07-29: "why do we have that dilated
        # convolution on e4 before the bottleneck? this literally directly counteracts
        # what attention is supposed to do".
        #
        # The pair is VESTIGIAL. It comes from the Rev B spec (DS-02), where the dilated
        # cascade WAS the board-scale context mechanism -- there was no transformer. When
        # Rev G added the MHSA bottleneck the design review recorded the dilation question
        # as "MOOT: the MHSA bottleneck is the board-scale context mechanism", and then
        # nobody removed it. It costs 1,180,672 parameters -- 25.1% of the detector, and
        # 0.75x the cost of the attention that superseded it.
        #
        # There is also a mechanism for it being actively unhelpful, not merely redundant:
        # RoPE attention discriminates tokens by content AND relative position, and a wide
        # dilated aggregation makes neighbouring tokens more alike, eroding exactly the
        # local distinctiveness the attention relies on. Untested until now -- conv-only
        # (attn_blocks=0) KEEPS the dilation, so "attention without dilation" is the cell
        # nobody had run.
        #
        # DELETION, not bypass. The gates ablation bypasses so parameter counts stay
        # comparable; here the parameter saving IS the hypothesis, so the modules go.
        # State dicts are therefore not cross-loadable with the reference -- same as
        # conv-only, and fine for a from-scratch arm.
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
            self.e5 = _deep(c(256), c(256))                                      # H/16
            grid_h, grid_w = h // 16, w // 16
        else:  # attend_div == 8: the deepest stage is e4 (no e5)
            self.e4 = _deep(c(128), c(256))                                      # H/8
            grid_h, grid_w = h // 8, w // 8
        self.rope = AxialRoPE(d // heads, grid_h, grid_w, rope_lambda_min)
        self.blocks = nn.ModuleList(Block(d, heads, self.rope, xsa=xsa) for _ in range(n_blocks))
        # gates=False (ablation A3): the AttnGate modules are still CONSTRUCTED so the
        # state_dict shape is unchanged and a gated checkpoint stays loadable, but the
        # skip is passed through raw. Ablating by bypass rather than by deletion keeps
        # the two arms' parameter counts comparable everywhere else.
        self.gates_on = gates
        self.norm = nn.LayerNorm(d, eps=1e-6)  # eps per timm/TransUNet refs
        if attend_div == 16:
            self.gate4 = AttnGate(c(256), c(256), c(128))     # gates enc4 (H/8) skip w/ bottleneck z
            self.gate3 = AttnGate(c(128), c(256), c(64))      # gates enc3 (H/4) skip w/ d4's output
            self.d4 = conv_bn_relu(c(256) + c(256), c(256))   # H/16 -> H/8
            self.d3 = conv_bn_relu(c(256) + c(128), c(128))   # H/8  -> H/4
        else:
            self.gate3 = AttnGate(c(128), c(256), c(64))      # gates enc3 (H/4) skip w/ bottleneck z (H/8 is the bottleneck here)
            self.d3 = conv_bn_relu(c(256) + c(128), c(128))   # H/8  -> H/4
        self.d2 = conv_bn_relu(c(128) + c(64), c(64))     # H/4  -> H/2
        self.d1 = conv_bn_relu(c(64) + c(32), c(32))      # H/2  -> H
        self.hm = nn.Sequential(nn.Conv2d(c(32), c(32), 3, padding=1), nn.ReLU(inplace=True),
                                nn.Conv2d(c(32), 1, 1))
        self.cls = nn.Sequential(nn.Conv2d(c(128), c(128), 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(c(128), n_cls, 1))     # n_cls = board's inner-corner count
        nn.init.constant_(self.hm[-1].bias, -2.19)
        nn.init.constant_(self.cls[-1].bias, -2.19)

    def forward(self, x):                       # (B,1,H,W) -> hm logits, cls logits
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        if self.attend_div == 16:
            s4 = self.e4(self.pool(s3))
            z = self.e5(self.pool(s4))          # (B, 256, H/16, W/16)
        else:
            z = self.e4(self.pool(s3))          # (B, 256, H/8, W/8)
        B, C, Hb, Wb = z.shape
        t = z.flatten(2).transpose(1, 2)        # (B, T, d) row-major tokens
        for blk in self.blocks:
            t = blk(t)
        z = self.norm(t).transpose(1, 2).reshape(B, C, Hb, Wb)
        if self.attend_div == 16:
            y = self.d4(torch.cat([up2(z), self.gate4(s4, z) if self.gates_on else s4], 1))   # H/8
            y = self.d3(torch.cat([up2(y), self.gate3(s3, y) if self.gates_on else s3], 1))   # H/4
        else:
            y = self.d3(torch.cat([up2(z), self.gate3(s3, z) if self.gates_on else s3], 1))   # H/4
        cls = self.cls(y)                       # class head taps the H/4 rung
        y = self.d2(torch.cat([up2(y), s2], 1))                  # H/2, ungated skip
        y = self.d1(torch.cat([up2(y), s1], 1))                  # H,  ungated skip
        return self.hm(y), cls


def detector_kwargs(cfg):
    """DetectorNet's config-tunable constructor kwargs (attend_div, n_blocks,
    heads, rope_lambda_min, xsa), read from a loaded cfg dict with the exact
    same defaults DetectorNet's own signature uses -- the one place
    tools/train_detector.py and tools/preflight.py both read these from, so
    the two model-construction sites can't drift apart. n_cls is
    deliberately excluded: it's derived from cfg["board"] via
    dcc.board.n_corners, not a flat cfg key."""
    return {"attend_div": cfg.get("attend_div", 16), "n_blocks": cfg.get("attn_blocks", 2),
            "heads": cfg.get("attn_heads", 8), "rope_lambda_min": cfg.get("rope_lambda_min_cells", 2.5),
            "xsa": cfg.get("xsa", False),
            "gates": cfg.get("gates_enabled", True),
            "width_mult": cfg.get("width_mult", 1.0),
            "e4_dilated": cfg.get("e4_dilated", True)}


class Refiner(nn.Module):
    """Sub-pixel refiner. Board-agnostic, size-fixed: (B,1,24,24) crop
    cut from the sensor-resolution frame -> (B,1,64,64) LOGITS over the
    central 8x8 px at 8x res (no sigmoid; loss/pipeline apply sigmoid or
    soft-argmax). Body: 3 conv stages at the full 24x24, THEN
    centre-crop to the central 8x8 (the only region the target ever
    populates -- the +/-4 px capture range sits entirely inside it), THEN one
    more conv stage at 8x8, THEN 1x1 to 64 = 8^2 channels for PixelShuffle(8).

    PixelShuffle orientation: it maps input channel c = i*r + j at coarse
    position (h, w) to output (h*r+i, w*r+j) -- upscales H, W in place, never
    transposing them. Nothing here transposes spatial axes either, so output
    dim -2/-1 stay the crop's row/col axes, matching render_refiner_target's
    (u/col <- x, v/row <- y) convention -- provided the pipeline cuts the
    sensor-frame crop in the same (row=y, col=x) order images always use;
    that crop is the pipeline's responsibility, not this module's."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(conv_bn_relu(1, 32), conv_bn_relu(32, 64), conv_bn_relu(64, 64))
        self.post = conv_bn_relu(64, 64)
        self.out = nn.Conv2d(64, 64, 1)
        self.ps = nn.PixelShuffle(8)
        nn.init.constant_(self.out.bias, -2.19)

    def forward(self, x):                       # (B,1,24,24) -> logits (B,1,64,64)
        f = self.body(x)[:, :, 8:16, 8:16]       # centre-crop to the central 8x8
        f = self.post(f)
        return self.ps(self.out(f))
