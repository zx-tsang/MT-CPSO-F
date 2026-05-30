# SVD-QR baseline

Pivoted QR sensor placement on a SVD basis of the training pressure
field, with L2 (least-squares) and L1 (Lasso) reconstruction. Builds
on the Brunton/Manohar line of work on data-driven sparse sensing.

## Files

| File | Role |
|---|---|
| `svdqr_tpu.py` | Main script. Implements all 4 SVD-QR variants: baseline-L2, baseline-L1, planA, oracle. |
| `run_alpha_sweep_for_l1.py` | Lasso α-sweep driver: for each energy threshold, scans the 7-point grid `α ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}` on VALIDATION (reduced to 5 points for energy ≥ 0.99 — see script docstring), picks α\* by minimising the mean Unknown MAE over `n ∈ {2, 4, …, 20}`, then reports the TEST result at α\*. |

## Variants implemented in `svdqr_tpu.py`

| Variant | Recon | Mean centering | Basis | Sensor set | Used in paper? |
|---|---|---|---|---|---|
| `baseline` (L2) | least-squares | NO (raw concatenated) | global | 1 global set | ✅ |
| `baseline_l1` | Lasso | NO (per-snapshot y-mean only) | global | 1 global set | ✅ |
| `planA` (L2) | least-squares | per-angle | global | 1 global set | ❌ ablation |
| `oracle` (L2) | least-squares | per-angle | per-angle | per-angle (n_ang sets) | ❌ upper bound |

**Why only the two `baseline*` variants are reported.** We adopt the
no-centering "baseline" form on the SVD-QR side so that it stays
directly comparable to mrDMD-QR (single basis, single sensor set, no
per-angle mean). This also makes the placement robust to the wind
direction being unknown at deployment time — neither method needs the
AoA to be measured or inferred before reconstruction.

`planA` would require the per-angle mean to be known at reconstruction
time; `oracle` would additionally require swapping the basis and sensor
set per AoA. Both break the shared, direction-agnostic baseline setup,
so they are kept in the codebase as ablations / upper bounds only.

## Quick run

```bash
# All four variants at energy 0.95:
python svdqr_tpu.py --energy 0.95

# L1 Lasso baseline at energy 0.95 with α grid (valid-tuned):
python run_alpha_sweep_for_l1.py --energy 0.95

# Only the L2 baseline at energy 0.99 (paper-headline config):
python svdqr_tpu.py --energy 0.99 --variants baseline
```

Both scripts write under `baselines/svd_qr/mode_result/`:
- `svdqr_tpu.py` → `svdqr_l2_<variant>_<tag>/`, `svdqr_l1_baseline_<l1_tag>/`, `svdqr_summary_<tag>/`
- `run_alpha_sweep_for_l1.py` → `svdqr_l1_alpha_sweep_<pct>/`

### Files in each variant directory (`svdqr_tpu.py`)

L2 variants — example for `baseline`:

```
mode_result/svdqr_l2_baseline_<tag>/
├── mae_baseline_<tag>.xlsx            two sheets:
│                                        - "summary"   1 row per sensor count n
│                                        - "per_angle" 1 row per (n, AoA), the
│                                                      raw MAE the summary aggregates
├── sensors_baseline_<tag>.xlsx        selected tap ids per n
├── Performance.png                    MAE-vs-n curve
├── PerAngle_Heatmap.png               MAE heatmap, n × AoA
└── sensor_layout.png                  tap layout, color-coded by QR rank
```

`planA` and `oracle` follow the same pattern, in
`svdqr_l2_planA_<tag>/` and `svdqr_l2_oracle_<tag>/`. `baseline_l1`
lives at `svdqr_l1_baseline_<l1_tag>/` with files named
`mae_baseline_l1_<l1_tag>.xlsx`, `sensors_baseline_l1_<l1_tag>.xlsx`,
and the same three plots.

If more than one variant is selected, a cross-variant summary is also
written:

```
mode_result/svdqr_summary_<tag>/
├── mae_summary_all_variants_<tag>.xlsx   wide table: per-variant macro/worst/std MAE × n
├── compare_macro_MAE.png
└── compare_worst_angle.png
```

## Notes on `α` sweep

`run_alpha_sweep_for_l1.py` follows the "Plan A — global α" protocol:

```
1. SVD on training data once. Truncate to rank r at the given energy threshold.
2. Pivoted QR on U_r^T → 1 deterministic sensor ordering.
3. For each α in the grid:
       VALID MAE at n = 1..20  (uses sliding prefix of the QR pivot)
4. α* = argmin_α  mean_{n∈{2,4,…,20}}  Unknown_MAE_Mean (on VALID split)
5. Re-run α* on TEST → reported result.
```

Outputs per energy:
- `mae_test_at_alpha_star_<pct>.xlsx` — TEST result, at the chosen α\*
- `alpha_grid_summary.xlsx` — VALID grid (all α × n, wide)
- `alpha_choice.txt` — α\* and the grid used
- `mae_valid_alpha<tag>_<pct>.xlsx` × N — per-α VALID detail

## Citation

Pivoted QR for sensor selection from a POD basis:

> Manohar K., Brunton B. W., Kutz J. N., Brunton S. L. (2018).
> *Data-Driven Sparse Sensor Placement for Reconstruction: Demonstrating
> the Benefits of Exploiting Known Patterns.* IEEE Control Systems
> Magazine, 38(3), 63-86.
