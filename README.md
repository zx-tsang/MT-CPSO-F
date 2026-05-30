# Sensor Placement as the Bottleneck: A Masked Transformer and Combinatorial PSO Framework for Full-Field Wind Pressure Reconstruction

Reference implementation accompanying the paper. The repository ships
**four** sensor-placement / field-reconstruction methods that all train
and evaluate on the publicly available **Tokyo Polytechnic University
(TPU) Aerodynamic Database** (T115 model: square-section high-rise,
500 pressure taps × 11 wind directions × 32 768 snapshots; TPU is the
data source only):

- **MT-CPSO-F** (`mt-cpso-f/`) **— primary contribution.**
  Masked-reconstruction Transformer pre-training + Cooperative PSO
  sensor search + targeted fine-tuning.
- **SVD-QR baseline** (`baselines/svd_qr/`) — Brunton/Manohar-style
  pivoted QR on the SVD basis, with L2 (least-squares) and L1 (Lasso)
  reconstruction.
- **mrDMD-QR baseline** (`baselines/mrdmd_qr/`) — Multi-resolution DMD
  basis + pivoted QR, L2 reconstruction (Al-Chalabi et al. 2025).
- **POD-Transformer baseline** (`baselines/podtfm/`) — POD + neural-
  decoder hybrid; adapted from Nav et al. (2025) by replacing the LSTM
  with a Transformer of equivalent capacity for a fair comparison.

All four methods share the same 80 / 10 / 10 per-angle
train / valid / test split and can be run independently.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with a CUDA-compatible build
  (MT-CPSO-F and POD-Transformer require a GPU; SVD-QR and mrDMD-QR
  run on CPU)
- See [`requirements.txt`](requirements.txt) for the full list

```bash
pip install -r requirements.txt
```

## Quick start

### MT-CPSO-F — full pipeline

```bash
cd mt-cpso-f
# (a) data preprocessing: .mat -> all_Data_all_place.npy + windowed splits
python stepa_preprocess.py
# (b) masked-reconstruction pre-training (hyper-parameters in scripts/params_main.json)
bash scripts/driver_pretrain.sh
# (c) + (d) CPSO sensor search + finetune for K=2..20 (10 budgets)
bash scripts/chain_cpso_finetune.sh
```

> **Note on wall-clock times.** The numbers below are the values
> reported in the paper and were measured on the **Guangzhou Football
> Park** case study (1000 candidate taps × 36 wind directions × 9000
> samples, full sweep K = 20..200 step 2 = 91 budgets, CPSO at
> *N*<sub>p</sub>=100 and *k*<sub>max</sub>=800). This open-source
> release runs the **TPU benchmark** as a much lighter *demo* (500
> taps × 11 directions × 32 768 snapshots, only K = 2..20 = 10
> budgets) and the CPSO step in
> [`chain_cpso_finetune.sh`](mt-cpso-f/scripts/chain_cpso_finetune.sh)
> is intentionally reduced to *N*<sub>p</sub>=40, *k*<sub>max</sub>=200,
> early-stop=10, so the actual demo wall-clock is **substantially
> shorter** than the paper's per-budget numbers below — the CPSO step
> alone is roughly an order of magnitude faster per budget.

Per-budget wall-clock breakdown reported in the paper (**Guangzhou
case study**, single RTX 4090, 24 GB VRAM):

| Stage | Time per budget (paper) |
|---|---:|
| (b) Masked-Transformer pretraining (one-time, shared across all *K*) | 0.862 h |
| (c) CPSO sensor search                                                | 0.882 h |
| (d) Per-*K* targeted fine-tuning                                      | 1.078 h |
| **Single-*K* total (b)+(c)+(d)**                                      | **2.820 h** |

Stages (c) and (d) are embarrassingly parallel across *K*, so a pool
of 5 GPUs reduces the full Guangzhou *K* sweep from ~26 h serial to
~6.7 h (13-budget sub-sweep). The TPU demo here is much smaller
(10 budgets, lighter CPSO settings) and finishes in a fraction of
that time — exact figures depend on your hardware.

See [`mt-cpso-f/scripts/params_main.json`](mt-cpso-f/scripts/params_main.json)
for the exact hyper-parameters (`early_stopping_patience=120`,
`anchor_K=10`, endpoint-contact curriculum with cosine LR schedule,
pure MSE training objective).

---

**Note.** All commands below (SVD-QR, mrDMD-QR, POD-Transformer)
are configured for the **TPU demo** dataset bundled with this release,
not the Guangzhou case study above.

### SVD-QR — L2 baseline at energy = 0.95

```bash
cd baselines/svd_qr
python svdqr_tpu.py --energy 0.95
# Output → baselines/svd_qr/mode_result/svdqr_l2_baseline_95pct/  (and 3 more variant dirs)
```

### SVD-QR — L1 baseline with valid-tuned α at energy = 0.95

```bash
cd baselines/svd_qr
python run_alpha_sweep_for_l1.py --energy 0.95
# Output → baselines/svd_qr/mode_result/svdqr_l1_alpha_sweep_95pct/
```

### mrDMD-QR — single config

```bash
cd baselines/mrdmd_qr
python mrdmdqr_tpu_l2.py --strategy baseline --L 7 --max_cyc 5 --r_max 5
# Output → baselines/mrdmd_qr/mode_result/mrdmdqr_l2_baseline/
```

### mrDMD-QR — full hyperparameter grid sweep

```bash
cd baselines/mrdmd_qr
python grid_sweep_mrdmdqr_baseline.py
# Output → baselines/mrdmd_qr/mode_result/sweep/<variant>_<snap>/
```

### POD-Transformer baseline — multi-K sweep

```bash
cd baselines/podtfm
# (a) build per-K input artefacts (shared SVDs, one LF-SVD per K)
python stepa_preprocess.py --ks 2 4 6 8 10 12 14 16 18 20
# (b) train all K in parallel on one GPU
python stepb_train_parallel.py --ks 2 4 6 8 10 12 14 16 18 20
# (c) evaluate every K back to physical 500-D Cp
for K in 2 4 6 8 10 12 14 16 18 20; do
  python stepc_evaluate.py --data-dir ../../raw_data/podtfm_p${K}_k${K} \
      --tag podtfm_p${K}_k${K} --split test --k $K
done
```

## Reproducing the paper

| Method | Command | Reported hyperparameters |
|---|---|---|
| **MT-CPSO-F (ours)** | `cd mt-cpso-f && bash scripts/driver_pretrain.sh && bash scripts/chain_cpso_finetune.sh` | `params_main.json`: patience = 120, anchor_K = 10, curriculum-cosine LR |
| SVD-QR (L2 99 %) | `cd baselines/svd_qr && python svdqr_tpu.py --energy 0.99 --variants baseline` | energy = 0.99 (rank ≈ 57) |
| SVD-QR (L1 95 %, valid-tuned) | `cd baselines/svd_qr && python run_alpha_sweep_for_l1.py --energy 0.95` | α* selected on validation; ≈ 0.03 |
| mrDMD-QR | `cd baselines/mrdmd_qr && python mrdmdqr_tpu_l2.py --strategy baseline --L 7 --max_cyc 5 --r_max 5` | L = 7, max_cyc = 5, r_max = 5, no Hankel embedding[^hankel] |

[^hankel]: A Hankel time-delay embedding is sometimes added before
    DMD to enrich the modal basis. Our mrDMD-QR baseline is adapted
    from Al-Chalabi et al. (2025) and runs plain mrDMD on the
    original snapshots without a Hankel embedding.

## Pre-computed sensor placements

[`baselines/idx/`](baselines/idx/) ships the SVD-QR and mrDMD-QR
sensor indices already extracted from the QR pivots, so downstream
users can skip the QR step and load the baseline placements directly:

```
baselines/idx/svd-qr/sensors_n{N}.txt        N pivoted-QR sensor IDs from the SVD basis
baselines/idx/mrDMD-qr/sensors_n{N}.txt      N pivoted-QR sensor IDs from the mrDMD basis
```

Use [`baselines/idx/extract_idx.py`](baselines/idx/extract_idx.py) to
(re-)generate these files from the per-method `mode_result` xlsx
outputs.

## Data

This repository **does not redistribute** the raw TPU pressure-tap
data (see TPU licensing terms). The 11 `.mat` files (one per wind
direction) must be downloaded directly from the TPU Aerodynamic
Database:

> https://db.wind.arch.t-kougei.ac.jp/aerodynamic/experiment/highrise/

Model T115 (square section, B : D : H = 1 : 1 : 5), suburban exposure
(power-law α = 1/4). After download, place the 11 files at
`raw_data/T115_4_xxx_1.mat` for xxx ∈ {000, 005, 010, …, 050}.

The pre-processed `raw_data/cp_grid.npy` is ≈ 688 MB, laid out as
`(n_angles, T, 4 faces, 25 height bins, 5 width bins)` with T = 32 768.
The companion `metadata.npz` stores wind angles, B/D/H, sampling rate,
and period. The pre-processed tensor can be regenerated locally from
the raw `.mat` files via `mt-cpso-f/stepa_preprocess.py`.

See [`raw_data/README.md`](raw_data/README.md) for full details on
data placement and formats.

### Required citation when using TPU data

The TPU Aerodynamic Database license requires citing all three:

> (1) TPU (Tokyo Polytechnic University) Aerodynamic Database, YEAR,
> https://db.wind.arch.t-kougei.ac.jp/

> (2) Quan, Y., Tamura, Y., Matsui, M., Cao, S.Y., Yoshida, A. (2007).
> TPU aerodynamic database for low-rise buildings. *Proceedings of the
> 12th International Conference on Wind Engineering (ICWE12)*, Vol. 2,
> Cairns, Australia, pp. 1615–1622.

> (3) Tamura, Y. (2009). Wind and tall buildings. *Keynote Lecture,
> The 5th Europe-African Regional Conference on Wind Engineering
> (EACWE5)*, Florence, Italy, 19–23 July 2009, p. 25.

For the methodological sources of each baseline, see the per-method
READMEs under [`baselines/`](baselines/) (one per baseline:
`svd_qr/`, `mrdmd_qr/`, `podtfm/`).

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{tsang2026sensor,
  title   = {Sensor Placement as the Bottleneck: A Masked Transformer
             and Combinatorial PSO Framework for Full-Field Wind
             Pressure Reconstruction},
  author  = {Tsang, Zhixuan and others},
  journal = {Automation in Construction},
  year    = {2026},
  note    = {Under review.}
}
```

## Repository layout

<details><summary>Click to expand</summary>

```
.
├── README.md                  this file
├── requirements.txt           pip dependencies
├── LICENSE                    MIT
│
├── raw_data/                  shared input data (placeholder; see Data)
│   └── README.md
│
├── mt-cpso-f/                 primary contribution
│   ├── stepa_preprocess.py    (a) .mat → all_Data_all_place.npy + windowed splits
│   ├── stepb_train.py         (b) masked-reconstruction pre-training
│   ├── stepc_cpso_search.py   (c) CPSO sensor selection (vectorised, chunk=5)
│   ├── stepc_cpso_multi_seed.py    multi-seed driver
│   ├── stepd_finetune.py      (d) fine-tune with the K selected taps
│   ├── network/               Transformer modules
│   ├── dataloader.py          datasets + Params + EarlyStopping
│   ├── params.yaml            default hyper-parameters
│   └── scripts/               driver_pretrain.sh, chain_cpso_finetune.sh, params_main.json
│
└── baselines/                 three modal-basis baselines + sensor-index cache
    ├── svd_qr/                SVD-QR baseline (Brunton/Manohar line)
    │   ├── svdqr_tpu.py       main script: SVD + QR + L2 / L1 variants
    │   ├── run_alpha_sweep_for_l1.py   Lasso α hyper-parameter sweep
    │   └── README.md
    │
    ├── mrdmd_qr/              mrDMD-QR baseline (Al-Chalabi et al. 2025)
    │   ├── mrdmd_utils.py     build_mrdmd_basis() — dyadic tree + DMD library
    │   ├── mrdmdqr_tpu_l2.py  main script: mrDMD + QR + L2
    │   ├── grid_sweep_mrdmdqr_baseline.py   (L, max_cyc, r_max) sweep
    │   └── README.md
    │
    ├── podtfm/                POD-Transformer baseline (adapted from Nav et al. 2025)
    │   ├── stepa_preprocess.py    SVD + gappy POD + windowing (multi-K)
    │   ├── stepb_train.py         single-K Transformer training (bf16, GPU-resident)
    │   ├── stepb_train_parallel.py    multi-K parallel training
    │   ├── stepc_evaluate.py      lift coefficients back to 500-D Cp + MAE
    │   ├── sweep_parallel.sh      end-to-end K sweep
    │   └── README.md
    │
    └── idx/                   pre-computed sensor indices (SVD-QR / mrDMD-QR)
        ├── extract_idx.py     tool to (re-)extract index .txt from mode_result xlsx
        ├── svd-qr/sensors_n*.txt
        └── mrDMD-qr/sensors_n*.txt
```

</details>

## Status

This is a frozen reference implementation accompanying the paper. Bug
reports are welcome via GitHub Issues; feature requests outside the
paper's scope may not be addressed.

## License

This code is released under the **MIT License** (see
[LICENSE](LICENSE)). The TPU pressure-tap data is subject to the
Tokyo Polytechnic University Aerodynamic Database terms — see the
[Data](#data) section above for required citations.
