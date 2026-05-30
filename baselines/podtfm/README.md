# POD-Transformer baseline

This baseline is **adapted from** the POD-LSTM framework of

> Nav, F. M., Mirfakhar, S. F., & Snaiki, R. (2025). *A hybrid machine
> learning framework for wind pressure prediction on buildings with
> constrained sensor networks.* Computer-Aided Civil and Infrastructure
> Engineering 40, 2816-2832 (MICE 13488).

It follows the same POD + neural-decoder structure, but the LSTM in the
original paper is replaced by a Transformer encoder of equivalent
capacity to keep the comparison fair to MT-CPSO-F. This is an
adaptation rather than a line-for-line port — some implementation
details (e.g. the gappy-POD reconstruction and the two-SVD split below)
differ from the original.

## Pipeline

```
sparse sensors (K)  --QR-->  X_LF (500-D)
        |                       |
       SVD --> Phi_qr           U_LF_k^T
        |                       |
        v                       v
   K taps selected         c_LF (K-D)  --Transformer-->  c_HF (K-D)
                                                              |
                                                              U_HF_k
                                                              |
                                                              v
                                                        reconstructed Cp (500-D)
```

Two SVDs are used: a *centered* SVD for QR pivoting + gappy reconstruction,
and a *raw* SVD for the encoder/decoder bases (so the static mean is baked
into mode 1 and there is no `+μ` offset in the decoder).

## Files

| File | Role |
|---|---|
| `stepa_preprocess.py`     | Multi-K pre-processing (shared HF/centered SVD, one LF SVD per K) |
| `stepb_train.py`          | Single-K Transformer training (bf16, GPU-resident dataset) |
| `stepb_train_parallel.py` | Multi-K parallel training on one GPU (recommended for sweeps) |
| `stepc_evaluate.py`       | Evaluation: lift coefficients back to 500-D Cp and report MAE |
| `sweep_parallel.sh`       | Shell driver for an end-to-end parallel K sweep |

## Data layout

This baseline reads from the shared `raw_data/` at the repository root
(two levels up from this folder). Before running stepa, make sure the
following files exist at `<repo-root>/raw_data/`:

```
<repo-root>/raw_data/
    all_Data_all_place.npy    # (500, n_angles * n_t_per_angle) full Cp time series
    metadata.npz              # keys: angles, n_t_per_angle, fs
```

`stepa_preprocess.py` writes per-K artifacts back into the same shared
folder at `<repo-root>/raw_data/podtfm_p{K}_k{K}/`.

## Quick start

All commands assume the working directory is this folder
(`baselines/podtfm/`).

```bash
# 1. Prepare data for K = 2,4,...,20 in one shot
python stepa_preprocess.py --ks 2 4 6 8 10 12 14 16 18 20

# 2. Train all K in parallel on one GPU (small config: params.yaml)
python stepb_train_parallel.py --params params.yaml \
    --ks 2 4 6 8 10 12 14 16 18 20

# 3. Evaluate per K
for K in 2 4 6 8 10 12 14 16 18 20; do
  python stepc_evaluate.py --params params.yaml \
      --data-dir ../../raw_data/podtfm_p${K}_k${K} \
      --tag podtfm_p${K}_k${K} --split test --k $K
done
```

To reproduce the **large** row of the ablation table below, swap
`params.yaml` for `params_large.yaml` in steps 2 and 3, or run the bundled
driver:

```bash
bash sweep_parallel.sh small "2 4 6 8 10 12 14 16 18 20"
bash sweep_parallel.sh large "2 4 6 8 10 12 14 16 18 20"
```

## Architectural ablation — small vs. large Transformer

We swept two Transformer sizes across K = 2…20 on the T115_4 benchmark to
check whether a larger encoder helps:

| Variant | `d_model` | `n_head` | `n_layers` | Params |
|---|---:|---:|---:|---:|
| **Small** (default in `params.yaml`)       | 64   | 4  | 2 | **72 K** |
| **Large** (`params_large.yaml`; matches MT-CPSO-F encoder size) | 1024 | 16 | 3 | **18 M** |

Test MAE in physical Cp units on the held-out 10 % split (unknown 500-K
taps, i.e. directly comparable to MT-CPSO-F):

| K | **Small (d=64)** | Large (d=1024) | Δ | POD floor |
|:--:|:--:|:--:|:--:|:--:|
| 2  | **0.1450** | 0.1455 | −0.3 % | 0.1274 |
| 4  | **0.1207** | 0.1209 | −0.2 % | 0.1090 |
| 6  | 0.1072     | **0.1070** | +0.2 % | 0.0938 |
| 8  | **0.0962** | 0.0968 | −0.6 % | 0.0828 |
| 10 | **0.0897** | 0.0900 | −0.4 % | 0.0762 |
| 12 | **0.0843** | 0.0849 | −0.6 % | 0.0718 |
| 14 | **0.0801** | 0.0811 | −1.2 % | 0.0680 |
| 16 | 0.0774     | 0.0774 | ±0 %   | 0.0642 |
| 18 | **0.0740** | 0.0751 | −1.5 % | 0.0615 |
| 20 | **0.0712** | 0.0722 | −1.5 % | 0.0596 |
| **avg** | **0.0946** | 0.0951 | **−0.6 %** | 0.0816 |

**Finding — the gap between small and large is essentially zero.**

Across all K = 2…20, the small (72 K params) and the **250× larger**
variant (18 M params) produce **almost identical** test MAEs: the
average difference is only **−0.6 %**, and on a per-K basis the gap
stays within ±1.5 % (small wins 9 / 10 K, large wins K = 6 by 0.2 %,
tie at K = 16). In other words **the model size is NOT what limits
this baseline** — both encoders converge to the same accuracy.

The reason is structural: for this baseline, the *POD-truncation floor*
(`POD_floor`) dominates the error budget — it accounts for 80–85 % of
`MAE_vs_raw` at K ≥ 10 and is independent of model size. Any improvement
beyond the floor must come from the K-dim Transformer fit, and a
64-hidden encoder is already sufficient to saturate that residual
capacity.

### Why the paper reports the LARGE variant

We report the **large** variant in the main paper, not because it is
better (the ablation above shows it is not), but for **capacity
parity with MT-CPSO-F**: the `params_large.yaml` POD-Transformer uses
the same `d_model=1024, n_head=16` encoder as MT-CPSO-F, so the two
sides of the comparison have matching encoder capacity.

Reporting the large variant therefore eliminates "model capacity" as
a possible explanation for the residual MAE gap between POD-Transformer
and MT-CPSO-F. The small-variant ablation above is the supporting
evidence: it shows that **shrinking the POD-Transformer by 250×
does not improve nor hurt the baseline meaningfully**, so the gap to
MT-CPSO-F is structural (POD truncation + gappy reconstruction), not a
matter of underfitting on the baseline side.

## Three reported metrics

| Metric | Compares against | Interpretation |
|---|---|---|
| `MAE_vs_raw`        | raw Cp ground truth          | Headline number |
| `MAE_vs_truncPOD`   | POD-truncated ground truth   | Pure model error |
| `POD_floor`         | how much truncation already costs | Inherent lower bound |

`MAE_vs_raw >= POD_floor` always. The gap is what the Transformer can
realistically improve. For K ≥ 10 we observe `POD_floor` to be 80-85 %
of `MAE_vs_raw`, i.e. the baseline is *POD-limited*, not *model-limited*.
