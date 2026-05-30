# raw_data/ — shared input data for all four methods

This folder is **empty by design** in the public release. The raw TPU
pressure-tap data is not redistributed here (≈ 688 MB, plus licensing
constraints from Tokyo Polytechnic University).

## What goes here

After you regenerate / download the data, this folder should contain:

```
raw_data/
├── cp_grid.npy            (11, 32768, 4, 25, 5)  ≈ 688 MB
├── all_Data_all_place.npy (500, 32768*11)        flat tap-major view
├── metadata.npz           locations, angles, tap_to_grid, n_t_per_angle, B, D, H, fs, period, Uh
├── data_Norm_global.npy   per-tap global mean/std
└── T115_4_xxx_1.mat       11 raw files, xxx ∈ {000,005,...,050}
```

All four methods (mt-cpso-f / baselines/svd_qr / baselines/mrdmd_qr / baselines/podtfm)
resolve their data path to this folder automatically — no env vars
needed.

## How to get the data

Download from the TPU Aerodynamic Database:

1. Visit https://db.wind.arch.t-kougei.ac.jp/aerodynamic/experiment/highrise/
2. Register / agree to the license
3. Download model **T115** (square section, B:D:H = 1:1:5), suburban
   exposure (power-law α = 1/4):
   - `T115_4_000_1.mat`, `T115_4_005_1.mat`, ..., `T115_4_050_1.mat`
     (11 files, one per wind direction 0°, 5°, ..., 50°)
4. Place the 11 `.mat` files directly in this folder.
5. Run `mt-cpso-f/stepa_preprocess.py` to generate
   `all_Data_all_place.npy`, `metadata.npz`, etc. After this, all four
   methods run out of the box.

## Required citations when using TPU data

> (1) TPU (Tokyo Polytechnic University) Aerodynamic Database, YEAR,
> https://db.wind.arch.t-kougei.ac.jp/

> (2) Quan, Y., Tamura, Y., Matsui, M., Cao, S.Y., Yoshida, A. (2007).
> TPU aerodynamic database for low-rise buildings. *Proceedings of the
> 12th International Conference on Wind Engineering (ICWE12)*, Vol.2,
> Cairns, Australia, pp.1615-1622.

> (3) Tamura, Y. (2009). Wind and tall buildings. *Keynote Lecture, The
> 5th Europe-African Regional Conference on Wind Engineering (EACWE5)*,
> Florence, Italy, 19-23 July 2009, p.25.
