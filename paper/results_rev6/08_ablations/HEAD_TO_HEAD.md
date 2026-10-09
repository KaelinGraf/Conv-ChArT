# Ablation head-to-head — training metrics only

Control: **abl_reference_50k_rev6**. Deltas are vs that control **at its own final step**;
arms still training are marked, and their deltas are provisional.
M-01 is coarse localisation error in input px (lower better); M-04 is ID accuracy.
No robustness sweeps here — this is each model's own validation performance.

| arm | steps | m01 mean | m01 median | m01 p95 | tail>4px | M-04 | ΔM-04 |
|---|---:|---:|---:|---:|---:|---:|---:|
| abl_ce_50k_rev6 | 50,000 | 0.4094 | 0.4079 | 0.6812 | 0.008% | 99.66% | +0.52 pp |
| abl_c3_wh_lam2_35k_rev6 *(running/stopped early)* | 35,000 | 0.4171 | 0.4109 | 0.7121 | 0.008% | 99.45% | +0.31 pp |
| abl_composite_s05_50k_rev6 *(running/stopped early)* | 32,288 | 0.4228 | 0.4109 | 0.7049 | 0.059% | 99.39% | +0.25 pp |
| abl_g_w50_a2_ce_nogate_35k_rev6 *(running/stopped early)* | 35,000 | 0.4137 | 0.4098 | 0.6965 | 0.008% | 99.38% | +0.23 pp |
| abl_sigma025_50k_rev6 | 50,000 | 0.4220 | 0.4098 | 0.6973 | 0.063% | 99.35% | +0.21 pp |
| abl_wh_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4143 | 0.4104 | 0.6983 | 0.008% | 99.31% | +0.16 pp |
| abl_sigma1_50k_rev6 | 50,000 | 0.4339 | 0.4148 | 0.7383 | 0.071% | 99.25% | +0.11 pp |
| abl_r502_lam2_35k_rev6 *(running/stopped early)* | 35,000 | 0.4194 | 0.4117 | 0.7230 | 0.004% | 99.21% | +0.06 pp |
| abl_sigma05_50k_rev6 *(running/stopped early)* | 36,487 | 0.4265 | 0.4115 | 0.7157 | 0.075% | 99.18% | +0.04 pp |
| abl_beta0_50k_rev6 | 50,000 | 0.4285 | 0.4121 | 0.7232 | 0.064% | 99.17% | +0.03 pp |
| abl_c2_wh_clsfocal_lam2_35k_rev6 *(running/stopped early)* | 35,000 | 0.4129 | 0.4094 | 0.6970 | 0.008% | 99.15% | +0.01 pp |
| abl_reference_50k_rev6 | 50,000 | 0.4559 | 0.4232 | 0.8189 | 0.074% | 99.14% |  |
| abl_lx_wh_hmfocal_clsce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4392 | 0.4151 | 0.7571 | 0.112% | 99.14% | -0.00 pp |
| abl_wh_attend16_35k_rev6 *(running/stopped early)* | 35,000 | 0.4345 | 0.4142 | 0.7440 | 0.084% | 99.12% | -0.02 pp |
| abl_c5_w375_g32_lam2_scls05_35k_rev6 *(running/stopped early)* | 35,000 | 0.4184 | 0.4114 | 0.7154 | 0.008% | 99.12% | -0.03 pp |
| abl_nodilate_50k_rev6 | 50,000 | 0.4588 | 0.4247 | 0.8156 | 0.158% | 99.11% | -0.03 pp |
| abl_xsa_nodilate_50k_rev6 *(running/stopped early)* | 32,051 | 0.4581 | 0.4246 | 0.8218 | 0.150% | 99.08% | -0.06 pp |
| abl_gates_off_50k_rev6 | 50,000 | 0.4572 | 0.4233 | 0.8192 | 0.129% | 99.07% | -0.08 pp |
| abl_seed2001_w375_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4171 | 0.4112 | 0.7139 | 0.008% | 99.05% | -0.09 pp |
| abl_nodilate_width_half_s05_50k_rev6 *(running/stopped early)* | 35,000 | 0.4332 | 0.4135 | 0.7399 | 0.084% | 99.04% | -0.11 pp |
| abl_g_w375_a2_ce_nogate_35k_rev6 *(running/stopped early)* | 35,000 | 0.4180 | 0.4109 | 0.7139 | 0.021% | 99.03% | -0.12 pp |
| abl_wh_rope5_35k_rev6 *(running/stopped early)* | 35,000 | 0.4336 | 0.4141 | 0.7449 | 0.072% | 99.01% | -0.14 pp |
| abl_wh_s025_35k_rev6 *(running/stopped early)* | 35,000 | 0.4316 | 0.4138 | 0.7398 | 0.076% | 98.99% | -0.15 pp |
| abl_xsa_50k_rev6 *(running/stopped early)* | 24,026 | 0.4595 | 0.4251 | 0.8277 | 0.130% | 98.99% | -0.16 pp |
| abl_width_half_50k_rev6 | 50,000 | 0.4681 | 0.4297 | 0.8620 | 0.130% | 98.96% | -0.19 pp |
| abl_lx_wh_hmce_clsfocal_35k_rev6 *(running/stopped early)* | 35,000 | 0.4124 | 0.4091 | 0.6910 | 0.016% | 98.93% | -0.21 pp |
| abl_wh_ctrl35k_rev6 *(running/stopped early)* | 35,000 | 0.4319 | 0.4132 | 0.7336 | 0.104% | 98.92% | -0.22 pp |
| abl_t_w50_a1_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4153 | 0.4101 | 0.7012 | 0.016% | 98.91% | -0.23 pp |
| abl_t_w375_a2_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4189 | 0.4115 | 0.7214 | 0.008% | 98.90% | -0.24 pp |
| abl_hp_w375_scls05_35k_rev6 *(running/stopped early)* | 35,000 | 0.4188 | 0.4113 | 0.7190 | 0.016% | 98.90% | -0.24 pp |
| abl_gs_w375_g321_35k_rev6 *(running/stopped early)* | 35,000 | 0.4167 | 0.4106 | 0.7100 | 0.008% | 98.89% | -0.25 pp |
| abl_r502_g32_35k_rev6 *(running/stopped early)* | 35,000 | 0.4172 | 0.4108 | 0.7138 | 0.012% | 98.87% | -0.27 pp |
| abl_hp_w375_scls2_35k_rev6 *(running/stopped early)* | 35,000 | 0.4181 | 0.4112 | 0.7147 | 0.008% | 98.87% | -0.28 pp |
| abl_wh_heads4_35k_rev6 *(running/stopped early)* | 35,000 | 0.4355 | 0.4142 | 0.7422 | 0.135% | 98.87% | -0.28 pp |
| abl_seed2003_w375_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4196 | 0.4115 | 0.7183 | 0.021% | 98.76% | -0.38 pp |
| abl_seed2002_w375_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4211 | 0.4119 | 0.7273 | 0.021% | 98.70% | -0.44 pp |
| abl_nodilate_width_half_xsa_s05_50k_rev6 *(running/stopped early)* | 24,866 | 0.4359 | 0.4145 | 0.7523 | 0.108% | 98.68% | -0.47 pp |
| abl_wh_attn1_35k_rev6 *(running/stopped early)* | 35,000 | 0.4339 | 0.4145 | 0.7475 | 0.064% | 98.54% | -0.60 pp |
| abl_convonly_50k_rev6 *(running/stopped early)* | 41,658 | 0.4543 | 0.4232 | 0.8141 | 0.110% | 98.46% | -0.68 pp |
| abl_t_w375_a1_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4188 | 0.4112 | 0.7195 | 0.021% | 98.11% | -1.03 pp |
| abl_hp_w25_lam4_35k_rev6 *(running/stopped early)* | 35,000 | 0.4354 | 0.4181 | 0.7840 | 0.008% | 98.11% | -1.04 pp |
| abl_hp_w25_lam2_35k_rev6 *(running/stopped early)* | 35,000 | 0.4301 | 0.4157 | 0.7661 | 0.008% | 97.82% | -1.33 pp |
| abl_c4_fast_g32_lam2_scls05_35k_rev6 *(running/stopped early)* | 35,000 | 0.4291 | 0.4156 | 0.7634 | 0.004% | 97.80% | -1.34 pp |
| abl_c1_fast_g32_lam2_35k_rev6 *(running/stopped early)* | 35,000 | 0.4321 | 0.4156 | 0.7704 | 0.029% | 97.15% | -1.99 pp |
| abl_wh_s1_35k_rev6 *(running/stopped early)* | 9,199 | 0.4546 | 0.4248 | 0.8253 | 0.093% | 96.84% | -2.31 pp |
| abl_gs_w25_g32_35k_rev6 *(running/stopped early)* | 35,000 | 0.4294 | 0.4145 | 0.7614 | 0.034% | 96.54% | -2.61 pp |
| abl_lx_fast_hmfocal_clsce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4579 | 0.4243 | 0.8358 | 0.121% | 96.49% | -2.65 pp |
| abl_hp_w25_scls05_35k_rev6 *(running/stopped early)* | 35,000 | 0.4292 | 0.4149 | 0.7582 | 0.004% | 96.35% | -2.79 pp |
| abl_gs_w25_g321_35k_rev6 *(running/stopped early)* | 35,000 | 0.4262 | 0.4148 | 0.7532 | 0.004% | 95.83% | -3.32 pp |
| abl_seed2001_fast_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4251 | 0.4136 | 0.7526 | 0.004% | 95.70% | -3.44 pp |
| abl_seed2002_fast_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4299 | 0.4160 | 0.7658 | 0.008% | 95.61% | -3.54 pp |
| abl_hp_w25_scls2_35k_rev6 *(running/stopped early)* | 35,000 | 0.4281 | 0.4147 | 0.7600 | 0.017% | 95.29% | -3.85 pp |
| abl_fast_ctrl35k_rev6 *(running/stopped early)* | 35,000 | 0.4526 | 0.4233 | 0.8210 | 0.113% | 94.63% | -4.51 pp |
| abl_fast_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4337 | 0.4180 | 0.7721 | 0.029% | 94.53% | -4.61 pp |
| abl_g_w25_a2_ce_nogate_35k_rev6 *(running/stopped early)* | 35,000 | 0.4287 | 0.4152 | 0.7587 | 0.017% | 93.71% | -5.43 pp |
| abl_hp_w25_lam05_35k_rev6 *(running/stopped early)* | 35,000 | 0.4256 | 0.4133 | 0.7437 | 0.033% | 91.88% | -7.27 pp |
| abl_nodilate_width_quarter_xsa_s05_30k_rev6 *(running/stopped early)* | 45,000 | 0.4539 | 0.4226 | 0.8207 | 0.113% | 90.91% | -8.23 pp |
| abl_nodilate_width_quarter_s05_30k_rev6 *(running/stopped early)* | 45,000 | 0.4543 | 0.4231 | 0.8224 | 0.109% | 90.73% | -8.42 pp |
| abl_seed2003_fast_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4304 | 0.4155 | 0.7631 | 0.021% | 90.10% | -9.04 pp |
| abl_t_w25_a1_xsa_ce_35k_rev6 *(running/stopped early)* | 35,000 | 0.4338 | 0.4178 | 0.7773 | 0.030% | 83.69% | -15.46 pp |
| abl_lx_fast_hmce_clsfocal_35k_rev6 *(running/stopped early)* | 35,000 | 0.4259 | 0.4147 | 0.7443 | 0.017% | 58.43% | -40.71 pp |

Rows are each arm's BEST validation by M-04, not necessarily its last.
Arms below 50,000 steps were stopped early or are still training — a
best-so-far number from a short run is not comparable to a converged one.
