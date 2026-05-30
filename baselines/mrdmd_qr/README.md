# mrDMD-QR baseline

Multi-resolution DMD basis + pivoted QR sensor selection on the TPU
benchmark. Reconstruction is L2 (least-squares).

Adapted from the AOA-unified placement strategy of Al-Chalabi et al.
(2025) MICE 70025. Since no source code was released by the original
authors, this implementation builds on their methodological ideas
rather than reproducing every detail of the paper.

## Files

| File | Role |
|---|---|
| `mrdmdqr_tpu_l2.py` | Main script. Three strategies: `baseline` (used in this paper), plus `planA` and `oracle` (provided as code-level references; see strategy table below). |
| `mrdmd_utils.py` | `build_mrdmd_basis()` — the mrDMD library: dyadic-tree time partitioning, optional Hankel embedding, exact vs FB-DMD, frequency-based mode pruning. Self-contained, depends only on numpy/scipy. |
| `grid_sweep_mrdmdqr_baseline.py` | Three-stage hyperparameter search over `(L, max_cyc, r_max_per_dmd)`. |

## Strategies in `mrdmdqr_tpu_l2.py`

| Strategy | Training data | Basis | Sensor set | Status |
|---|---|---|---|---|
| `baseline` | 11 angles, raw concatenated, no centering | 1 global mrDMD basis | 1 global set | **Used in this paper.** Adapted from the AOA-unified placement of Al-Chalabi et al. (2025). |
| `planA` | 11 angles, per-angle centered, concatenated | 1 global mrDMD basis | 1 global set | Not used in this paper. Provided as a code-level variant. |
| `oracle` | 11 angles, per-angle centered, per-angle basis | per-angle | per-angle (n_ang sets) | Not used in this paper. Upper bound; requires test-time AOA. |

`planA` and `oracle` both require the test-time wind direction to be
known a priori and still rely on a linear modal basis. The paper targets
the wind-direction-agnostic deployment scenario, for which `baseline` is
the appropriate variant. See the docstring at the top of
`mrdmdqr_tpu_l2.py` for a longer discussion.

## Quick run

```bash
# Paper baseline: textbook mrDMD (no Hankel), (L=7, max_cyc=5, r_max=5)
python mrdmdqr_tpu_l2.py --angle_strategy baseline --no_hankel \
    --L 7 --max_cyc 5 --r_max_per_dmd 5

# Quick smoke test (reduced snapshots + shallow tree, finishes in <1 min)
python mrdmdqr_tpu_l2.py --angle_strategy baseline --no_hankel \
    --n_snaps 1024 --L 2 --max_cyc 2 --r_max_per_dmd 3

# Full three-stage hyperparameter grid sweep
python grid_sweep_mrdmdqr_baseline.py
```

> **Note on `--no_hankel`.** The paper baseline is the *textbook* (no
> Hankel time-delay embedding) variant — see the VARIANT table below.
> Pass `--no_hankel` to reproduce it. Omitting the flag activates the
> Hankel-embedded variant, which is **much** slower (the time-delay
> embedding makes every per-node DMD SVD far larger) and is not the
> configuration reported in the paper.

## Output layout

Results are written under `baselines/mrdmd_qr/mode_result/` (mirrors
the `baselines/svd_qr/mode_result/` layout so the two baselines look
the same to an open-source user):

```
mode_result/
├── mrdmdqr_l2_baseline/         <- single-config run (paper baseline)
│   ├── mae_baseline_L7_mc5_r5.xlsx   two sheets:
│   │                                   - "summary"   1 row per sensor count n
│   │                                   - "per_angle" 1 row per (n, AoA)
│   ├── L2_Performance.png
│   ├── sensor_layout.png
│   └── mrdmd_basis.npz
├── mrdmdqr_l2_planA/            <- if planA is run
├── mrdmdqr_l2_oracle/           <- if oracle is run
│   └── oracle_detail_<tag>.xlsx     per-angle sensor lists (oracle only)
└── sweep/                       <- hyperparameter grid sweeps
    └── nohankel_fullsnap/       <- one folder per VARIANT x snapshot setting
        ├── grid_summary.xlsx
        ├── grid_per_angle.xlsx
        ├── heatmap_n*.png
        └── n_vs_mae_all_combos.png
```

Hyperparameters (L, max_cyc, r_max, d_embed) are encoded in the xlsx
filename's `<tag>`, so reruns with different settings coexist in the
same strategy folder without overwriting each other.

## mrDMD VARIANT switch in `grid_sweep_mrdmdqr_baseline.py`

The sweep script has a top-level `VARIANT` selector exposing three
orthogonal mrDMD configurations:

| `VARIANT` | DMD type | Hankel | Orthonormalise |
|---|---|---|---|
| `textbook_no_hankel` (paper baseline) | Exact DMD | OFF | OFF |
| `fbdmd_ortho` | FB-DMD | OFF | ON |
| `hankel` | FB-DMD | d=30 | ON |

Note: the "Orthonormalise" column refers to an SVD-based orthonormalisation
of the assembled Phi matrix, applied AFTER mrDMD modes are collected and
real-stacked, but BEFORE QR-pivoting selects the sensors. It is unrelated
to the SVDs that happen inside Exact/FB-DMD itself.

This paper uses `textbook_no_hankel` (the simplest configuration). The
other two are kept in the code as ablation knobs for readers who want
to probe how Hankel embedding and orthonormalisation affect mrDMD-QR
performance on this dataset.

## mrDMD-QR-F (finetune variant)

[`mrdmd_qr_f.sh`](mrdmd_qr_f.sh) reproduces the **mrDMD-QR-F** row
in the paper's comparison table: it takes the QR-selected sensors from
[`../idx/mrDMD-qr/`](../idx/) and feeds them to the Masked Transformer
fine-tuner in `mt-cpso-f/stepd_finetune.py`. This requires the
Stage-1 pretrain checkpoint produced by
`mt-cpso-f/scripts/driver_pretrain.sh`.

```bash
cd baselines/mrdmd_qr
bash mrdmd_qr_f.sh                  # default K = 2..20
bash mrdmd_qr_f.sh "2 10 20"        # subset
```

Outputs land in `baselines/mrdmd_qr/mrdmd_qr_ft/K{K}/seed_42/finetune/`.

## Reference

> Al-Chalabi R., Alanani M., Elshaer A., El Damatty A. (2025).
> *Data-driven optimization of wind pressure sensor placement on
> low-rise buildings using computational fluid dynamics and
> multi-resolution dynamic mode decomposition.* Computer-Aided Civil
> and Infrastructure Engineering, doi:10.1111/mice.70025.
