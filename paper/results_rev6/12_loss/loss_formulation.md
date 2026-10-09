---
title: "Conv-ChArT --- loss formulation"
subtitle: "Penalty-reduced focal on a Gaussian target, shared across both heads"
geometry: margin=2.2cm
fontsize: 10pt
---

# The target

Per-corner Gaussians on the full-resolution grid, **combined by max** (never sum, per
CornerNet), with the peak pixel forced to exactly 1.0:

$$\boxed{\;Y(u) \;=\; \max_{i \in \text{visible}} \exp\!\left(-\frac{\lVert u - p_i\rVert^2}{2\sigma^2}\right),
\qquad Y\big(\mathrm{rint}(p_i)\big) \equiv 1 \;}$$

- $p_i$ --- the $i$-th ground-truth corner, at **sub-pixel** float precision.
- $\sigma_{\text{hm}} = 2.0$ px at full resolution; $\sigma_{\text{cls}} = 1.0$ at H/4,
  where the cell coordinate is $x_c = (x+0.5)/4 - 0.5$.
- **Max, not sum**: two nearby corners must remain two peaks. Summing would merge them
  into one taller blob and destroy the very separation the decoder is asked to find.
- **Forcing the peak to 1.0** matters: without it the $Y=1$ branch of the loss below is
  never reachable, because a Gaussian sampled at a non-integer centre never attains 1.
- The target is **not one-hot**. A near-miss is scored by how near it is.

# The loss

Penalty-reduced focal (CornerNet), evaluated in **logit space** --- the naive
$\log(1-\hat p)$ form NaNs under bf16 near the forced peaks. With $\hat p_u = \sigma(z_u)$:

$$\boxed{\;\mathcal{L} \;=\; -\sum_{u} \begin{cases} (1-\hat p_u)^{\alpha}\,\log \hat p_u & Y_u = 1 \\[4pt]
(1-Y_u)^{\beta}\;\hat p_u^{\alpha}\,\log(1-\hat p_u) & \text{otherwise}\end{cases}\;}$$

$$\boxed{\;\mathcal{L}_{\text{total}} \;=\; \frac{\mathcal{L}_{\text{hm}} + \lambda_{\text{cls}}\,\mathcal{L}_{\text{cls}}}
{\max\big(\textstyle\sum_{\text{batch}} N_{\text{vis}},\,1\big)}\;}$$

## Parameters

- $\boldsymbol{\alpha = 2}$ --- the **focal** term. On a $640\times480$ map with 16
  positives, 0.005% of pixels are positive; without it the loss is dominated by
  already-correct background and collapses to predicting "no corner" everywhere.
- $\boldsymbol{\beta = 4}$ --- the **penalty reduction**. A negative adjacent to a
  corner has $Y \approx 0.88$, so $(1-Y)^4 \approx 2\times10^{-4}$: it is barely
  punished. **This is the only place the Gaussian's graded values enter the loss** ---
  the positive set is $Y = 1$ exactly, identical to a one-hot target's.
- $\boldsymbol{\lambda_{\text{cls}} = 1.0}$ --- both heads carry exactly 307,200
  elements at the trained variant, so equal weighting is derived, not tuned.
- $\boldsymbol{N_{\text{vis}}}$ --- **one shared normaliser** across both heads: the
  batch total of visible corners. A per-channel $N \in \{0,1\}$ would inflate
  $\mathcal{L}_{\text{cls}}$ by roughly $16\times$. Clamped at 1 so an all-negative
  batch ($N = 0$) needs no special case --- CornerNet's own convention.
- Final logit-conv biases initialise to $-2.19$ ($\pi = 0.1$), so training starts from
  "probably not a corner" rather than from an uninformative 0.5.

# The refiner's loss

Same functional form on a $64\times64$ target at $8\times$ over the crop's central
$8\times8$ px, $\sigma = 1.5$, peak forced at $\mathrm{rint}(u^\ast)$:

$$u^\ast = 31.5 + 8d, \qquad d \in [-3.9375,\ 3.9375]\ \text{px}$$

Decoding is a $5\times5$ soft-argmax around the hard argmax, border-clamped and
renormalised --- in pipeline code, deliberately outside the exported graph.

# Measured, and what it does NOT support

The A-CE ablation replaces the graded target with strict one-hot BCE. It did **not**
fail --- it **beat** the reference on aggregate (99.66% vs 98.96% m04 at matched
steps), and the per-regime sweep shows the trade:

| regime | reference (Gaussian) | A-CE (one-hot) |
|---|---|---|
| FAR ($s=16$) | 93.98 | **98.72 (+4.74)** |
| VERY CLOSE ($s=128$) | **99.20** | 95.14 (-4.06) |

The mechanism was measured directly (n=830 corners): at $\sigma = 2$, $\beta = 4$ the
trained peaks have RMS radius **1.0028 px** with only **26.1%** of local probability
mass on the peak pixel, against **0.2665 px** and **93.5%** for the one-hot arm --- and
the one-hot arm reaches that at 9k steps against a fully-trained 50k reference.

**So the loss is asking for a blurry peak and getting one.** The defensible claim is
not "the Gaussian target is better", but that **$\sigma$ and $\beta$ are
task-appropriate hyperparameters that were inherited untuned** --- $\beta = 4$ is
CornerNet's value for sparse object centres, not for a dense regular lattice. The
`abl_sigma1` arm ($\sigma_{\text{hm}} = 1.0$, one key) tests whether tightening the
target recovers the sharpness while keeping the graded target's advantages.
