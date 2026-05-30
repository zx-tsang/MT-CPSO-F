"""
Unified SVD-QR sensor placement, TPU.

Four variants in one script. Three are L2 (lstsq) reconstruction differing
in wind-awareness; the fourth is an L1 (Lasso, MATLAB-aligned) baseline
for cross-method comparison.

==================================================================
Variant differences
==================================================================

  baseline (01)
    - All angles concatenated along time as RAW Cp.
    - 1 global SVD on the raw merged matrix.
    - 1 global QR pivoting on U_r => 1 global sensor set.
    - L2 reconstruction in the global basis, wind-direction agnostic.
    - Mean centering: NO. The cross-angle mean leaks into the leading
      SVD modes and "wastes" a few modes on a constant offset.

  baseline_l1 (02)
    - Same data pipeline as baseline (raw concatenated, no centering)
      and same global SVD + QR sensor selection.
    - DIFFERENT reconstruction: per-timestep Lasso (L1) with MATLAB-
      aligned defaults (Standardize=true, Intercept=true, intercept
      dropped at recon time). Equivalent to MATLAB's
        b = lasso(zscore(Theta), y - mean(y), 'Lambda', alpha)
        x_hat = U_r * (b ./ std(Theta))
    - Basis is LOCKED at r_max(--l1_energy) for ALL n (independent of
      sensor count) because L1 penalty handles mode selection.
    - Mean centering: NO (per-snapshot y-mean is removed inside Lasso
      but not the per-tap training mean; matches MATLAB convention).
    - Slow: per-snapshot Lasso solve. Default --l1_energy 0.95 keeps
      the basis small enough for tractable runtime.

  planA (03)
    - Each angle is mean-centered with its OWN training mean, then
      centered slices are concatenated along time.
    - 1 global SVD on the centered merged matrix.
    - 1 global QR => 1 global sensor set.
    - L2 reconstruction is grouped by angle: each angle's test slice
      is centered, reconstructed, then the same per-angle mean is
      added back (so the wind-direction offset is restored exactly).
    - Mean centering: YES, per-angle. Selection global, recon
      wind-aware. Maps to "AOA-unified placement" in Al-Chalabi 2025.
    - NOT USED in this paper. To keep SVD-QR and mrDMD-QR directly
      comparable (single basis, single sensor set, no per-angle mean)
      and to make the placement robust to the wind direction being
      unknown at deployment time, we adopt the no-centering "baseline"
      form for both methods. planA's per-angle mean centering would
      require the AoA-specific mean to be available at reconstruction
      time, which breaks both the head-to-head comparison and the
      direction-agnostic deployment story.

  oracle (04)
    - Per-angle mean centering + per-angle SVD => one basis U_i per
      angle.
    - For each angle, an INDEPENDENT QR on U_i => that angle's OWN
      sensors. There are n_ang different sensor sets.
    - Each angle is reconstructed and evaluated only on its own
      sensors and own test slice; results are averaged across angles.
    - Mean centering: YES, per-angle.
    - This is the upper bound: "if you knew the wind direction in
      advance and could swap sensors with it." Not deployable, used
      as a reference to bound how much the global-sensor constraint
      costs.
    - NOT USED in this paper. Same reasoning as planA: to stay
      consistent with mrDMD-QR (single basis, no per-angle mean) and
      direction-agnostic at deployment time, the SVD-QR side also uses
      the no-centering "baseline" form. oracle would additionally
      require swapping the basis and sensor set per AoA, which is even
      further from the shared, direction-agnostic baseline setup.

==================================================================
Mean-centering summary
==================================================================

  variant     | recon | mean centered? | scope     | basis     | sensors
  ------------|-------|----------------|-----------|-----------|---------
  baseline    | L2    | no             | -         | global    | global (1)
  baseline_l1 | L1    | no (per-y only)| -         | global    | global (1)
  planA       | L2    | yes            | per angle | global    | global (1)
  oracle      | L2    | yes            | per angle | per angle | per angle (n_ang)

Answer to "do they mean-center when sensors are placed per angle?":
YES. In oracle, each angle's training slice is mean-centered with its
own training mean before SVD, and the mean is added back at evaluation
time so the metric stays on the physical Cp scale.

==================================================================
What this paper uses
==================================================================

This paper reports results for the two AoA-agnostic variants only:
  - baseline    (L2 lstsq, global SVD, global QR)
  - baseline_l1 (Lasso, global SVD, global QR)

planA and oracle are included in the codebase as ablations / upper
bounds, but they are NOT used in the reported numbers. We adopt the
no-centering "baseline" form for both methods. This keeps SVD-QR and
mrDMD-QR directly comparable (single basis, single sensor set, no
per-angle mean) and also makes the placement robust to the wind
direction being unknown at deployment time — neither method needs the
AoA to be measured or inferred before reconstruction. planA would
require the per-angle mean to be known at reconstruction time, and
oracle would additionally require swapping the basis / sensor set per
AoA — both break this shared, direction-agnostic baseline setup.

==================================================================
Outputs
==================================================================

  mode_result/svdqr_l2_baseline_{tag}/
  mode_result/svdqr_l2_planA_{tag}/
  mode_result/svdqr_l2_oracle_{tag}/
  mode_result/svdqr_l1_baseline_{l1_tag}/
  mode_result/svdqr_summary_{tag}/      <- compares all four

Each variant directory contains:
  mae_<variant>_<tag>.xlsx       two sheets:
                                   - "summary"   (1 row per sensor count)
                                   - "per_angle" (1 row per (n, AoA))
  sensors_<variant>_<tag>.xlsx   selected tap ids per n
  Performance.png, PerAngle_Heatmap.png, sensor_layout.png

The cross-variant summary directory contains:
  mae_summary_all_variants_{tag}.xlsx   wide table comparing variants
  compare_macro_MAE.png, compare_worst_angle.png

By default all four variants are run. baseline_l1 has its own energy
default (--l1_energy 0.95) and Lasso regularization (--alpha 1e-3).
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import qr
from sklearn.linear_model import Lasso

warnings.filterwarnings("ignore")

# Repository layout: this script lives at svd_qr/svdqr_tpu.py and reads
# the shared raw_data/ one level up at the repository root.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw_data"
MODE_RESULT_ROOT = Path(__file__).resolve().parent / "mode_result"

TRAIN_RATIO, VALID_RATIO = 0.80, 0.10
ENERGY_DEFAULT = "full"
SWEEP_START, SWEEP_STEP, SWEEP_END = 1, 1, 20
LAYOUT_MAX_N = 20
TEST_SUBSAMPLE = 0
L1_ENERGY_DEFAULT = 0.99  # baseline_l1 basis-lock energy.
# 0.99 ~= 57 modes for this dataset; gives L1 a proper overcomplete
# basis (n_sensors << n_modes) so the sparsity-inducing penalty has
# room to work. 0.95 (~6 modes) is too tight for compressed sensing
# behavior; "full" (~500 modes) is theoretically best but slow and
# pollutes the basis with noise modes.
L1_ALPHA_DEFAULT = 1e-2   # baseline_l1 Lasso L1 regularization
                          # (optimal from a valid-tuned α grid sweep at
                          # energy=0.97; rerun run_alpha_sweep_for_l1.py
                          # to retune for other energy thresholds.)
LASSO_TOL = 1e-4          # MATLAB lasso default RelTol
ALL_VARIANTS = ["baseline", "baseline_l1", "planA", "oracle"]

N_FACES, N_ROWS, N_COLS = 4, 25, 5
FACE_NAMES = ["Windward", "Right", "Leeward", "Left"]


def tap_to_grid(tap):
    return (tap % 20) // N_COLS, tap // 20, tap % N_COLS


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_train_test():
    """Backward-compatible loader: train_centered/test_centered + raw
    train/test concatenations + per-angle test labels. Validation slice
    is dropped silently (use load_train_valid_test for hyperparameter
    selection)."""
    (train_c, test_c, means, tr_raw, _, te_raw, _, a_te) = (
        load_train_valid_test())
    return train_c, test_c, means, tr_raw, te_raw, a_te


def load_train_valid_test():
    """Same per-angle 80/10/10 split as load_train_test() but also
    returns the validation slice for hyperparameter tuning.

    Returns
    -------
    train_centered, test_centered : list of (n_taps, n_snap) per angle,
        mean-centered with the corresponding angle's TRAINING mean.
    means : (n_ang, n_taps) per-angle training mean (physical Cp).
    train_raw : (n_taps, n_train_total) raw concatenated training data.
    valid_raw : (n_taps, n_valid_total) raw concatenated validation data.
    test_raw  : (n_taps, n_test_total)  raw concatenated test data.
    angle_va : (n_valid_total,) wind-angle id per valid snapshot.
    angle_te : (n_test_total,)  wind-angle id per test  snapshot.
    """
    grid = np.load(RAW_DIR / "cp_grid.npy")
    n_ang, T = grid.shape[:2]
    flat = grid.reshape(n_ang, T, -1).astype(np.float64)
    n_tr = int(round(T * TRAIN_RATIO))
    n_va = int(round(T * VALID_RATIO))

    train_c, test_c = [], []
    tr_raw, va_raw, te_raw = [], [], []
    a_va, a_te = [], []
    means = np.zeros((n_ang, flat.shape[2]), dtype=np.float64)
    for i in range(n_ang):
        cp_tr = flat[i, :n_tr]
        cp_va = flat[i, n_tr:n_tr + n_va]
        cp_te = flat[i, n_tr + n_va:]
        mu = cp_tr.mean(axis=0)
        means[i] = mu
        train_c.append((cp_tr - mu).T)
        test_c.append((cp_te - mu).T)
        tr_raw.append(cp_tr.T)
        va_raw.append(cp_va.T)
        te_raw.append(cp_te.T)
        a_va.append(np.full(cp_va.shape[0], i, dtype=np.int32))
        a_te.append(np.full(cp_te.shape[0], i, dtype=np.int32))
    return (train_c, test_c, means,
            np.concatenate(tr_raw, axis=1),
            np.concatenate(va_raw, axis=1),
            np.concatenate(te_raw, axis=1),
            np.concatenate(a_va),
            np.concatenate(a_te))


def rank_at_energy(cum_energy, thresh):
    if thresh is None or thresh >= 1.0:
        return len(cum_energy)
    return int(np.searchsorted(cum_energy, thresh) + 1)


def reconstruct_l2(U_r, sel, Y, rcond=None):
    A, *_ = np.linalg.lstsq(U_r[sel, :], Y, rcond=rcond)
    return U_r @ A


def reconstruct_l1_matlab(U_r, sel, Y_meas, alpha, tol=LASSO_TOL):
    """Per-snapshot Lasso with MATLAB lasso() defaults: column standardize
    + per-snapshot y-mean removal. Intercept dropped at recon time to
    match l1_analyze_and_export_v95.m. Equivalent to:
        b_std = lasso(zscore(Theta), y - mean(y), 'Lambda', alpha)
        b     = b_std ./ std(Theta, ddof=1)
        x_hat = U_r @ b
    """
    Theta = U_r[sel, :]
    n_meas, r = Theta.shape
    T = Y_meas.shape[1]
    mu_X = Theta.mean(axis=0)
    sd_X = (Theta.std(axis=0, ddof=1) if n_meas > 1
            else np.zeros(r))
    # Near-constant column handling (matches MATLAB lasso()'s internal
    # safeguard). When two picked sensor rows of U_r happen to be nearly
    # equal on a mode, that column's sd is tiny but nonzero; the
    # un-standardization step coef_/sd_X then amplifies by 1/sd_X and
    # the reconstruction explodes (n=2 r=13 with sd=1.1e-3 -> coef/sd
    # ~130 -> MAE ~8). Setting such columns' sd to 1.0 makes their
    # Theta_s nearly zero, so Lasso's L1 penalty rejects them in favor
    # of well-separated modes.
    NEAR_CONST_SD = 1e-2
    sd_X = np.where(np.isfinite(sd_X) & (sd_X > NEAR_CONST_SD), sd_X, 1.0)
    Theta_s = (Theta - mu_X) / sd_X
    A = np.zeros((r, T))
    lasso = Lasso(alpha=alpha, fit_intercept=False,
                  max_iter=50000, tol=tol,
                  warm_start=True, selection="cyclic")
    for t in range(T):
        y = Y_meas[:, t]
        lasso.fit(Theta_s, y - y.mean())
        A[:, t] = lasso.coef_ / sd_X
    return U_r @ A


# ------------------------------------------------------------------
# Variant runners
# ------------------------------------------------------------------
def run_baseline(train_raw, test_raw, angle_te, sensor_counts,
                 energy_thresh):
    """No mean centering. Global SVD + global QR + global L2 recon."""
    U, s, _ = np.linalg.svd(train_raw, full_matrices=False)
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    r_max = rank_at_energy(cum, energy_thresh)
    n_total = train_raw.shape[0]
    angle_ids = sorted(np.unique(angle_te).tolist())
    rows, per_angle_rows, sels = [], [], []

    for n in sensor_counts:
        r = max(min(int(n), r_max), 1)
        U_r = U[:, :r]
        if n > r:
            _, _, piv = qr(U_r @ U_r.T, pivoting=True, mode="economic")
        else:
            _, _, piv = qr(U_r.T, pivoting=True, mode="economic")
        sel = np.asarray(piv[:n])
        unk = np.setdiff1d(np.arange(n_total), sel)

        t0 = time.perf_counter()
        recon = reconstruct_l2(U_r, sel, test_raw[sel, :])
        recon_time = time.perf_counter() - t0

        unk_mae = float(np.mean(np.abs(test_raw[unk, :] - recon[unk, :])))
        per_ang = {}
        for ang in angle_ids:
            mask = angle_te == ang
            err = float(np.mean(
                np.abs(test_raw[unk][:, mask] - recon[unk][:, mask])))
            per_ang[ang] = err
        vals = np.array(list(per_ang.values()))
        rows.append([n, r, unk_mae, vals.mean(), vals.std(), vals.max(),
                     recon_time])
        for a in angle_ids:
            per_angle_rows.append([n, a, per_ang[a]])
        sels.append(sel)
    cols = ["Sensor_Count", "Basis_Rank", "Unknown_MAE_Mean",
            "PerAngle_MAE_Mean", "PerAngle_MAE_Std", "PerAngle_MAE_Max",
            "Recon_Time_s"]
    return (pd.DataFrame(rows, columns=cols),
            pd.DataFrame(per_angle_rows,
                         columns=["Sensor_Count", "Angle_ID", "MAE"]),
            sels)


def run_baseline_l1(train_raw, test_raw, angle_te, sensor_counts,
                    l1_energy_thresh, alpha):
    """baseline pipeline (raw, global SVD/QR, no centering) but Lasso
    reconstruction. Basis is locked at r_max(l1_energy) for all n
    because the L1 penalty handles mode selection."""
    U, s, _ = np.linalg.svd(train_raw, full_matrices=False)
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    r_locked = rank_at_energy(cum, l1_energy_thresh)
    U_r = U[:, :r_locked]
    if max(sensor_counts) > r_locked:
        _, _, piv = qr(U_r @ U_r.T, pivoting=True, mode="economic")
    else:
        _, _, piv = qr(U_r.T, pivoting=True, mode="economic")
    n_total = train_raw.shape[0]
    angle_ids = sorted(np.unique(angle_te).tolist())
    rows, per_angle_rows, sels = [], [], []

    for n in sensor_counts:
        sel = np.asarray(piv[:n])
        unk = np.setdiff1d(np.arange(n_total), sel)

        t0 = time.perf_counter()
        recon = reconstruct_l1_matlab(U_r, sel, test_raw[sel, :], alpha)
        recon_time = time.perf_counter() - t0

        unk_mae = float(np.mean(np.abs(test_raw[unk, :] - recon[unk, :])))
        per_ang = {}
        for ang in angle_ids:
            mask = angle_te == ang
            err = float(np.mean(
                np.abs(test_raw[unk][:, mask] - recon[unk][:, mask])))
            per_ang[ang] = err
        vals = np.array(list(per_ang.values()))
        rows.append([n, r_locked, unk_mae, vals.mean(), vals.std(),
                     vals.max(), recon_time])
        for a in angle_ids:
            per_angle_rows.append([n, a, per_ang[a]])
        sels.append(sel)
    cols = ["Sensor_Count", "Basis_Rank", "Unknown_MAE_Mean",
            "PerAngle_MAE_Mean", "PerAngle_MAE_Std", "PerAngle_MAE_Max",
            "Recon_Time_s"]
    return (pd.DataFrame(rows, columns=cols),
            pd.DataFrame(per_angle_rows,
                         columns=["Sensor_Count", "Angle_ID", "MAE"]),
            sels)


def run_planA(train_c, test_c, means, sensor_counts, energy_thresh):
    """Per-angle mean centering + global SVD/QR + per-angle L2 recon."""
    train_merged = np.concatenate(train_c, axis=1)
    U, s, _ = np.linalg.svd(train_merged, full_matrices=False)
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    r_max = rank_at_energy(cum, energy_thresh)
    n_total = train_merged.shape[0]
    n_ang = len(train_c)
    rows, per_angle_rows, sels = [], [], []

    for n in sensor_counts:
        r = max(min(int(n), r_max), 1)
        U_r = U[:, :r]
        if n > r:
            _, _, piv = qr(U_r @ U_r.T, pivoting=True, mode="economic")
        else:
            _, _, piv = qr(U_r.T, pivoting=True, mode="economic")
        sel = np.asarray(piv[:n])
        unk = np.setdiff1d(np.arange(n_total), sel)

        t0 = time.perf_counter()
        per_ang = {}
        total_abs, total_n = 0.0, 0
        for i in range(n_ang):
            X_te = test_c[i]
            rec = reconstruct_l2(U_r, sel, X_te[sel])
            mu = means[i][:, None]
            diff = np.abs((X_te + mu)[unk] - (rec + mu)[unk])
            per_ang[i] = float(diff.mean())
            total_abs += float(diff.sum()); total_n += diff.size
        unk_mae = total_abs / total_n
        recon_time = time.perf_counter() - t0
        vals = np.array(list(per_ang.values()))
        rows.append([n, r, unk_mae, vals.mean(), vals.std(), vals.max(),
                     recon_time])
        for a in range(n_ang):
            per_angle_rows.append([n, a, per_ang[a]])
        sels.append(sel)
    cols = ["Sensor_Count", "Basis_Rank", "Unknown_MAE_Mean",
            "PerAngle_MAE_Mean", "PerAngle_MAE_Std", "PerAngle_MAE_Max",
            "Recon_Time_s"]
    return (pd.DataFrame(rows, columns=cols),
            pd.DataFrame(per_angle_rows,
                         columns=["Sensor_Count", "Angle_ID", "MAE"]),
            sels)


def _per_angle_svd(train_c, energy_thresh):
    U_list, r_list = [], []
    for X in train_c:
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        cum = np.cumsum(s ** 2) / np.sum(s ** 2)
        r = (len(cum) if energy_thresh is None or energy_thresh >= 1.0
             else int(np.searchsorted(cum, energy_thresh) + 1))
        U_list.append(U); r_list.append(r)
    return U_list, r_list


def run_oracle(train_c, test_c, means, sensor_counts, energy_thresh):
    """Per-angle SVD + per-angle QR + per-angle sensors (upper bound)."""
    U_list, r_list = _per_angle_svd(train_c, energy_thresh)
    n_taps = train_c[0].shape[0]
    n_ang = len(train_c)
    rows, per_angle_rows, sel_per_n = [], [], []

    for n in sensor_counts:
        sels_this_n = []
        per_ang = {}
        total_abs, total_n = 0.0, 0
        t0 = time.perf_counter()
        for i in range(n_ang):
            U, r = U_list[i], r_list[i]
            r_eff = max(min(int(n), r), 1)
            U_r = U[:, :r_eff]
            if n > r_eff:
                _, _, piv = qr(U_r @ U_r.T, pivoting=True, mode="economic")
            else:
                _, _, piv = qr(U_r.T, pivoting=True, mode="economic")
            sel_i = np.asarray(piv[:n])
            unk_i = np.setdiff1d(np.arange(n_taps), sel_i)
            X_te = test_c[i]
            rec = reconstruct_l2(U_r, sel_i, X_te[sel_i])
            mu = means[i][:, None]
            diff = np.abs((X_te + mu)[unk_i] - (rec + mu)[unk_i])
            per_ang[i] = float(diff.mean())
            total_abs += float(diff.sum()); total_n += diff.size
            sels_this_n.append(sel_i)
        unk_mae = total_abs / total_n
        recon_time = time.perf_counter() - t0
        vals = np.array(list(per_ang.values()))
        avg_r = float(np.mean([min(int(n), r) for r in r_list]))
        rows.append([n, avg_r, unk_mae, vals.mean(), vals.std(),
                     vals.max(), recon_time])
        for a in range(n_ang):
            per_angle_rows.append([n, a, per_ang[a]])
        sel_per_n.append(sels_this_n)
    cols = ["Sensor_Count", "Avg_Basis_Rank", "Unknown_MAE_Mean",
            "PerAngle_MAE_Mean", "PerAngle_MAE_Std", "PerAngle_MAE_Max",
            "Recon_Time_s"]
    return (pd.DataFrame(rows, columns=cols),
            pd.DataFrame(per_angle_rows,
                         columns=["Sensor_Count", "Angle_ID", "MAE"]),
            sel_per_n)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_layout(piv_list, n_list, out_dir, title,
                fname="sensor_layout.png"):
    n_plots = len(n_list)
    if n_plots == 0:
        return
    ncol = min(5, n_plots)
    nrow = int(np.ceil(n_plots / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.6 * nrow),
                             squeeze=False)
    for k, (n, sel) in enumerate(zip(n_list, piv_list)):
        ax = axes[k // ncol, k % ncol]
        ax.imshow(np.ones((N_ROWS, N_FACES * N_COLS)),
                  cmap="Greys", vmin=0, vmax=1, alpha=0.15)
        ys, xs, ranks = [], [], []
        for rank, tap in enumerate(sel):
            f, r, c = tap_to_grid(int(tap))
            ys.append(r); xs.append(f * N_COLS + c); ranks.append(rank + 1)
        ax.scatter(xs, ys, c=ranks, cmap="viridis", s=60,
                   edgecolors="k", linewidths=0.6)
        for x in [N_COLS, 2 * N_COLS, 3 * N_COLS]:
            ax.axvline(x - 0.5, color="r", lw=0.8)
        ax.set_xticks([N_COLS / 2 - 0.5 + i * N_COLS for i in range(N_FACES)])
        ax.set_xticklabels(FACE_NAMES, fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"n={n}", fontsize=10)
        ax.set_xlim(-0.5, N_FACES * N_COLS - 0.5)
        ax.set_ylim(N_ROWS - 0.5, -0.5)
    for k in range(n_plots, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=300)
    plt.close(fig)


def plot_variant(df, df_pa, out_dir, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["Sensor_Count"], df["PerAngle_MAE_Mean"],
            "--o", lw=1.5, ms=6, label="Per-angle macro MAE")
    ax.fill_between(df["Sensor_Count"],
                    df["PerAngle_MAE_Mean"] - df["PerAngle_MAE_Std"],
                    df["PerAngle_MAE_Mean"] + df["PerAngle_MAE_Std"],
                    alpha=0.15, label="+/-1 std across angles")
    ax.plot(df["Sensor_Count"], df["PerAngle_MAE_Max"],
            ":r", lw=1.2, label="Worst-angle MAE")
    ax.set_xlabel("Number of Selected Sensors (n)")
    ax.set_ylabel("MAE (physical Cp scale)")
    ax.legend(loc="upper right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "Performance.png", dpi=600)
    plt.close(fig)

    pa_pivot = df_pa.pivot(index="Sensor_Count", columns="Angle_ID",
                           values="MAE")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pa_pivot.values, aspect="auto", cmap="viridis_r",
                   origin="lower")
    ax.set_xticks(range(len(pa_pivot.columns)))
    ax.set_xticklabels(pa_pivot.columns)
    ax.set_yticks(range(len(pa_pivot.index)))
    ax.set_yticklabels(pa_pivot.index)
    ax.set_xlabel("Wind-angle ID")
    ax.set_ylabel("Number of sensors")
    ax.set_title(title + " - per-angle MAE")
    plt.colorbar(im, ax=ax, label="MAE")
    fig.tight_layout()
    fig.savefig(out_dir / "PerAngle_Heatmap.png", dpi=300)
    plt.close(fig)


def plot_oracle_overlap(sel_per_n, n_list, n_ang, out_dir, tag):
    """For oracle: one heatmap per n showing how many of the n_ang
    angles selected each tap. Bright = consensus tap; dim = idiosyncratic."""
    n_plots = len(n_list)
    if n_plots == 0:
        return
    ncol = min(5, n_plots)
    nrow = int(np.ceil(n_plots / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.6 * nrow),
                             squeeze=False)
    n_taps_total = N_FACES * N_ROWS * N_COLS
    for k, (n, sels_at_n) in enumerate(zip(n_list, sel_per_n)):
        ax = axes[k // ncol, k % ncol]
        count = np.zeros(n_taps_total, dtype=int)
        for sel_i in sels_at_n:
            count[sel_i] += 1
        grid = np.zeros((N_ROWS, N_FACES * N_COLS), dtype=int)
        for tap in range(n_taps_total):
            f, r, c = tap_to_grid(tap)
            grid[r, f * N_COLS + c] = count[tap]
        im = ax.imshow(grid, cmap="hot", vmin=0, vmax=n_ang,
                       aspect="auto")
        for x in [N_COLS, 2 * N_COLS, 3 * N_COLS]:
            ax.axvline(x - 0.5, color="cyan", lw=0.8)
        ax.set_xticks([N_COLS / 2 - 0.5 + i * N_COLS for i in range(N_FACES)])
        ax.set_xticklabels(FACE_NAMES, fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"n={n}", fontsize=10)
    for k in range(n_plots, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle(f"oracle - per-tap selection count across "
                 f"{n_ang} wind angles ({tag})\n"
                 f"bright = consensus tap; dim = picked by few angles",
                 fontsize=11)
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        fraction=0.025, pad=0.02)
    cbar.set_label(f"# angles selecting tap (max = {n_ang})")
    fig.savefig(out_dir / "sensor_overlap_heatmap.png", dpi=300)
    plt.close(fig)


def plot_summary(all_results, summary_dir, tag):
    colors = {"baseline": "k", "planA": "g",
              "oracle": "b", "baseline_l1": "r"}
    markers = {"baseline": "s", "planA": "^",
               "oracle": "o", "baseline_l1": "v"}
    for ycol, fname, ylabel in [
        ("PerAngle_MAE_Mean", "compare_macro_MAE.png",
         "Per-angle macro MAE (physical Cp)"),
        ("PerAngle_MAE_Max", "compare_worst_angle.png",
         "Worst-angle MAE (physical Cp)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for v, df in all_results.items():
            ax.plot(df["Sensor_Count"], df[ycol],
                    "-" + markers.get(v, "o"),
                    color=colors.get(v), lw=1.6, ms=6, label=v)
        ax.set_xlabel("Number of Selected Sensors (n)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Variant comparison ({tag})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(summary_dir / fname, dpi=600)
        plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_variants(spec):
    if spec.lower() == "all":
        return list(ALL_VARIANTS)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in ALL_VARIANTS:
            raise ValueError(f"unknown variant: {tok!r} (valid: "
                             f"{ALL_VARIANTS})")
        out.append(tok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", default=ENERGY_DEFAULT)
    ap.add_argument("--s_start", type=int, default=SWEEP_START)
    ap.add_argument("--s_step", type=int, default=SWEEP_STEP)
    ap.add_argument("--s_end", type=int, default=SWEEP_END)
    ap.add_argument("--layout_max_n", type=int, default=LAYOUT_MAX_N)
    ap.add_argument("--test_subsample", type=int, default=TEST_SUBSAMPLE)
    ap.add_argument("--l1_energy", type=float, default=L1_ENERGY_DEFAULT,
                    help="energy threshold for baseline_l1 basis lock "
                         "(default 0.99 ~ 57 modes for TPU data)")
    ap.add_argument("--alpha", type=float, default=L1_ALPHA_DEFAULT,
                    help="Lasso L1 regularization for baseline_l1 "
                         "(default 1e-2; see run_alpha_sweep_for_l1.py "
                         "for valid-tuned selection)")
    ap.add_argument("--variants", default="all",
                    help='subset, e.g. "baseline,planA"; default "all"')
    args = ap.parse_args()

    if str(args.energy).lower() in ("full", "1", "1.0"):
        energy_thresh, tag = None, "full"
    else:
        energy_thresh = float(args.energy)
        tag = f"{int(round(energy_thresh * 100))}pct"

    variants = parse_variants(args.variants)

    print("[1/4] Loading TPU data ...")
    (train_c, test_c, means,
     train_raw, test_raw, angle_te) = load_train_test()
    n_ang = len(train_c); n_taps = train_c[0].shape[0]
    n_te_total = test_raw.shape[1]
    print(f"  n_angles={n_ang}, n_taps={n_taps}, "
          f"n_test_total={n_te_total}")

    if 0 < args.test_subsample < n_te_total:
        new_test_c, new_te_raw, new_a_te = [], [], []
        for i, X in enumerate(test_c):
            T_i = X.shape[1]
            keep = max(1, int(round(T_i * args.test_subsample / n_te_total)))
            idx = np.linspace(0, T_i - 1, keep).astype(int)
            new_test_c.append(X[:, idx])
            new_te_raw.append(X[:, idx] + means[i][:, None])
            new_a_te.append(np.full(keep, i, dtype=np.int32))
        test_c = new_test_c
        test_raw = np.concatenate(new_te_raw, axis=1)
        angle_te = np.concatenate(new_a_te)

    sensor_counts = np.arange(args.s_start, args.s_end + 1, args.s_step)

    l1_tag = (f"{int(round(args.l1_energy * 100))}pct"
              if 0 < args.l1_energy < 1 else "full")
    runners = {
        "baseline": lambda: run_baseline(train_raw, test_raw, angle_te,
                                          sensor_counts, energy_thresh),
        "planA":    lambda: run_planA(train_c, test_c, means,
                                      sensor_counts, energy_thresh),
        "oracle":   lambda: run_oracle(train_c, test_c, means,
                                       sensor_counts, energy_thresh),
        "baseline_l1": lambda: run_baseline_l1(
            train_raw, test_raw, angle_te, sensor_counts,
            args.l1_energy, args.alpha),
    }

    print(f"[2/4] Running {len(variants)} variant(s): {variants}")
    summaries = {}
    for v in variants:
        print(f"\n  --- variant: {v} ---")
        df, df_pa, sels = runners[v]()
        for _, row in df.iterrows():
            print(f"    n={int(row['Sensor_Count']):3d}  "
                  f"macroMAE={row['PerAngle_MAE_Mean']:.6f}  "
                  f"worst={row['PerAngle_MAE_Max']:.6f}  "
                  f"std={row['PerAngle_MAE_Std']:.6f}  "
                  f"t={row['Recon_Time_s']:5.2f}s")
        if v == "baseline_l1":
            out_dir = MODE_RESULT_ROOT / f"svdqr_l1_baseline_{l1_tag}"
            tag_for_files = l1_tag
        else:
            out_dir = MODE_RESULT_ROOT / f"svdqr_l2_{v}_{tag}"
            tag_for_files = tag
        out_dir.mkdir(parents=True, exist_ok=True)
        # Single workbook, two sheets: a 1-row-per-n "summary" with all
        # aggregate stats, and a 1-row-per-(n, angle) "per_angle" with the
        # raw MAE the summary aggregates over.
        with pd.ExcelWriter(out_dir / f"mae_{v}_{tag_for_files}.xlsx",
                            engine="openpyxl") as w:
            df.to_excel(w, sheet_name="summary", index=False)
            df_pa.to_excel(w, sheet_name="per_angle", index=False)

        sensor_rows = []
        if v == "oracle":
            for n, sels_at_n in zip(sensor_counts, sels):
                for ang_id, sel_i in enumerate(sels_at_n):
                    for rank, tap in enumerate(sel_i):
                        f, r, c = tap_to_grid(int(tap))
                        sensor_rows.append([int(n), int(ang_id), rank + 1,
                                            int(tap), FACE_NAMES[f], r, c])
            sensor_cols = ["Sensor_Count", "Angle_ID", "Rank",
                           "Tap_ID", "Face", "Row", "Col"]
        else:
            for n, sel in zip(sensor_counts, sels):
                for rank, tap in enumerate(sel):
                    f, r, c = tap_to_grid(int(tap))
                    sensor_rows.append([int(n), rank + 1, int(tap),
                                        FACE_NAMES[f], r, c])
            sensor_cols = ["Sensor_Count", "Rank", "Tap_ID",
                           "Face", "Row", "Col"]
        pd.DataFrame(sensor_rows, columns=sensor_cols).to_excel(
            out_dir / f"sensors_{v}_{tag_for_files}.xlsx", index=False)
        plot_variant(df, df_pa, out_dir, f"{v} ({tag_for_files})")

        layout_n = [int(n) for n in sensor_counts if n <= args.layout_max_n]
        if v == "oracle":
            # angle-0 reference layout (only one of the n_ang configurations)
            sels_layout = [sels[k][0] for k, n in enumerate(sensor_counts)
                           if n <= args.layout_max_n]
            plot_layout(
                sels_layout, layout_n, out_dir,
                f"oracle - reference layout for angle 0 ONLY "
                f"(each angle has its own set, {tag_for_files})",
                fname="sensor_layout_angle0.png")
            # cross-angle overlap: shows how much consensus exists
            sels_per_n_layout = [sels[k] for k, n in enumerate(sensor_counts)
                                 if n <= args.layout_max_n]
            plot_oracle_overlap(sels_per_n_layout, layout_n, n_ang,
                                out_dir, tag_for_files)
        else:
            sels_layout = [sels[k] for k, n in enumerate(sensor_counts)
                           if n <= args.layout_max_n]
            plot_layout(sels_layout, layout_n, out_dir,
                        f"{v} ({tag_for_files})")

        summaries[v] = df
        print(f"    saved -> {out_dir}")

    if len(summaries) >= 2:
        print("\n[3/4] Building cross-variant summary ...")
        summary_dir = MODE_RESULT_ROOT / f"svdqr_summary_{tag}"
        summary_dir.mkdir(parents=True, exist_ok=True)
        macro = pd.DataFrame({"Sensor_Count": sensor_counts})
        for v, df in summaries.items():
            macro[f"{v}_macroMAE"] = df["PerAngle_MAE_Mean"].values
            macro[f"{v}_worstMAE"] = df["PerAngle_MAE_Max"].values
            macro[f"{v}_stdMAE"]   = df["PerAngle_MAE_Std"].values
        macro.to_excel(summary_dir / f"mae_summary_all_variants_{tag}.xlsx",
                       index=False)
        plot_summary(summaries, summary_dir, tag)
        print(f"    saved -> {summary_dir}")

    print("\n[4/4] Done.")


if __name__ == "__main__":
    main()
