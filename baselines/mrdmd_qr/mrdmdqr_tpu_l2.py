"""mrDMD-QR sensor placement, TPU.

Three variants matching svdqr_tpu.py. The SVD basis U_r is replaced by
the multi-resolution DMD modal library Phi; pivoted-QR sensor selection
and L2 reconstruction are otherwise unchanged.

Reference
---------
  Al-Chalabi R., Alanani M., Elshaer A., El Damatty A. (2025).
  Data-driven optimization of wind pressure sensor placement on
  low-rise buildings using CFD and multi-resolution DMD. CACAIE.
  doi:10.1111/mice.70025

  The implementation below is BASED ON the methodological ideas of the
  above paper (no source code was released by the original authors);
  the `baseline` variant is our re-implementation following its
  AOA-unified placement strategy.

Variants
--------
  baseline     : raw Cp concat -> 1 global mrDMD -> 1 global QR -> L2 recon.
                 Wind-direction agnostic. Adapted from the AOA-unified
                 placement strategy of Al-Chalabi et al. (2025).
  planA        : per-AOA mean centering -> centered concat -> 1 global mrDMD
                 -> 1 global QR -> per-AOA L2 recon (means added back).
                 Variant that removes each angle's temporal-mean Cp
                 BEFORE the mrDMD fit.
                 NOT USED IN THIS PAPER (see note below).
  oracle       : per-AOA mrDMD + per-AOA QR + per-AOA sensors. Each angle
                 reconstructed with its own sensors and basis. Upper bound;
                 not deployable.
                 NOT USED IN THIS PAPER (see note below).

Why this paper uses ONLY `baseline`
-----------------------------------
Both `planA` and `oracle` are provided here as code-level references for
readers wishing to explore wind-direction-aware extensions of the
mrDMD-QR framework. They are NOT part of the comparison reported in the
paper, for two reasons that apply to both variants:

  (i)  Both require the test-time wind direction (AOA) to be known a
       priori — planA needs the per-AOA mean field to add back at
       reconstruction time, and oracle needs the per-AOA basis and the
       per-AOA sensor set. Al-Chalabi et al. (2025) themselves compare
       an AOA-specific placement against an AOA-unified placement (their
       Table 1) and report that, although AOA-specific placements are
       marginally better at their targeted direction, the common sensor
       layout still maintains R^2 > 0.9 across all flow metrics, which
       they take as "supporting its use in practical implementations
       requiring robustness to changing wind directions". We adopt the
       same conclusion: the deployable configuration is the AOA-unified
       one (`baseline`).
  (ii) Both still rely on a LINEAR modal basis (Phi multiplied by a
       coefficient vector) for reconstruction, inheriting the same
       linear-subspace limitation that motivates our proposed nonlinear
       MT-CPSO-F method. Including them would not change the
       linear-vs-nonlinear dichotomy that frames the paper's argument.

In short: `baseline` is the only variant that is both deployable under
the paper's wind-direction-agnostic setting AND a fair linear-baseline
target for our nonlinear method to outperform.

Note: the SVD-QR script sweeps r=min(n, r_max) so the basis "matches" the
sensor count. mrDMD does not produce an energy-ordered basis, so the fixed
mrDMD basis is used at every n.

Hyperparameter sweep history (textbook_no_hankel, full-snap, no centering)
-------------------------------------------------------------------------
Three progressively-finer stages; each starts from the previous winner.
Selection = lowest mean Valid_Unknown_MAE across the stage's n grid (one
fixed (L,C,R) shared by all n, NOT per-n oracle).

  n  = sensor count (how many taps selected out of 500).
  Each "run" = one full reconstruction at a fixed (L, C, R, n) -> one MAE.
  runs = (L,C,R) combos x n values evaluated in that stage.

  Stage | (L,C,R) combos | (L, C, r) grid                                | n values             | runs | winner (L, C, R) | mean MAE
  ------+----------------+-----------------------------------------------+----------------------+------+------------------+---------
   1    |       36       | L {3,5,7}; C {3,5,7}; r {30,50,70,100}        | 5  of [4..20]        | 180  | (7, 5, 30)       | 0.1319
   2    |       21       | L {6,7,8}; C = 5; r {10,15,20,25,30,35,40}    | 10 of [2,4,..,20]    | 210  | (7, 5, 10)       | 0.1276
   3    |        9       | L {6,7,8}; C = 5; r {4, 5, 6}                 | 10 of [2,4,..,20]    |  90  | (7, 5,  5)       | 0.1216  <- paper

Totals: 66 (L,C,R) combos, 480 runs. Monotone improvement
0.1319 -> 0.1276 -> 0.1216 motivated each zoom; Stage 3 R=4/R=6 are within
noise of R=5, so the search stops. Paper baseline: (L=7, max_cyc=5, r_max=5).
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import qr

# mrdmd_utils.py sits in the same directory as this script (mrdmd_qr/).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrdmd_utils import build_mrdmd_basis

warnings.filterwarnings("ignore")

# Textbook mrDMD (Kutz/Fu/Brunton 2016): no Hankel embedding.
# Activated by --no_hankel; bypasses build_mrdmd_basis.
MR_MIN_WINDOW = 64


def _dmd_exact(X, r=None):
    X1, X2 = X[:, :-1], X[:, 1:]
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    if r is None or r > len(s):
        r = len(s)
    tol = max(s[0] * 1e-10, 1e-14) if len(s) else 0.0
    keep = s[:r] > tol
    if not keep.any():
        return np.zeros((X.shape[0], 0), dtype=complex), \
               np.zeros(0, dtype=complex)
    U_r = U[:, :r][:, keep]
    s_r = s[:r][keep]
    V_r = Vh[:r, :].conj().T[:, keep]
    A_tilde = U_r.conj().T @ X2 @ V_r / s_r
    eigs, W = np.linalg.eig(A_tilde)
    Phi = X2 @ V_r @ (W / s_r[:, None])
    return Phi.astype(complex), eigs.astype(complex)


def _mrdmd_library(X, dt, max_levels, max_cyc, r_dmd,
                   min_window=MR_MIN_WINDOW):
    library = []

    def _recurse(Xs, level):
        m = Xs.shape[1]
        if m < min_window or level > max_levels:
            return
        T_level = m * dt
        f_thresh = max_cyc / T_level
        Phi, eigs = _dmd_exact(Xs, r=r_dmd)
        if Phi.shape[1] == 0:
            return
        omega = np.log(eigs) / dt
        f = np.abs(omega.imag) / (2 * np.pi)
        slow = f < f_thresh
        if slow.any():
            Phi_s = Phi[:, slow]; eigs_s = eigs[slow]
            library.append(Phi_s)
            b, *_ = np.linalg.lstsq(Phi_s, Xs[:, 0].astype(complex),
                                    rcond=None)
            Vand = eigs_s[:, None] ** np.arange(m)[None, :]
            slow_recon = (Phi_s * b[None, :]) @ Vand
            residual = Xs - slow_recon.real
        else:
            residual = Xs
        half = m // 2
        _recurse(residual[:, :half], level + 1)
        _recurse(residual[:, half:], level + 1)

    _recurse(X, 1)
    if not library:
        return np.zeros((X.shape[0], 0), dtype=complex)
    return np.hstack(library)


def _complex_to_real_cols(Psi):
    if Psi.shape[1] == 0:
        return np.zeros((Psi.shape[0], 0))
    col_norms = np.linalg.norm(Psi, axis=0).max()
    keep_imag = np.linalg.norm(Psi.imag, axis=0) > col_norms * 1e-8
    blocks = [Psi.real]
    if keep_imag.any():
        blocks.append(Psi.imag[:, keep_imag])
    return np.hstack(blocks)


def _build_mrdmd_basis_nohankel(train_list, max_levels, max_cyc, r_dmd, dt):
    """Drop-in replacement for build_mrdmd_basis with d_embed=1 enforced.
    Concatenates train_list along time, runs textbook mrDMD, returns
    (Phi_real, info_dict) matching the surrounding code's expectations."""
    X = np.concatenate(train_list, axis=1)
    Psi = _mrdmd_library(X, dt=dt, max_levels=max_levels,
                         max_cyc=max_cyc, r_dmd=r_dmd)
    Phi_real = _complex_to_real_cols(Psi)
    info = {
        "basis_rank": Phi_real.shape[1],
        "singular_values": np.array([]),
        "n_modes_per_angle": [Psi.shape[1]],
    }
    return Phi_real, info

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent       # repository root (two levels up from baselines/mrdmd_qr/)
RAW_DIR = PROJECT_ROOT / "raw_data"
# Results live next to the code at baselines/mrdmd_qr/mode_result/,
# matching the baselines/svd_qr/ layout for consistency.
MODE_RESULT_ROOT = ROOT / "mode_result"

# =====================================================================
# CONFIG
# =====================================================================
TRAIN_RATIO, VALID_RATIO = 0.80, 0.10

SENSOR_COUNTS_SPEC = "1:20"
LAYOUT_MAX_N       = 20
TEST_SUBSAMPLE     = 0

# mrDMD basis hyperparameters
DT_TPU                = 1e-3
N_SNAPS_DEFAULT       = 26214  # full 80% training portion per AOA
D_EMBED_DEFAULT       = 30
L_LEVEL_DEFAULT       = 5
MAX_CYC_DEFAULT       = 5
R_MAX_PER_DMD_DEFAULT = 50
AMP_QUANTILE_DEFAULT  = 0.0

ANGLE_STRATEGY     = "baseline"

ALL_VARIANTS = ["baseline", "planA", "oracle"]

N_FACES, N_ROWS, N_COLS = 4, 25, 5
FACE_NAMES = ["Windward", "Right", "Leeward", "Left"]
# =====================================================================


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def parse_sensor_counts(spec):
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            parts = [int(p) for p in tok.split(":")]
            if len(parts) == 2:
                a, b = parts; step = 1
            elif len(parts) == 3:
                a, b, step = parts
            else:
                raise ValueError(f"bad range token: {tok!r}")
            if step <= 0:
                raise ValueError(f"step must be >0 in {tok!r}")
            out.extend(range(a, b + 1, step))
        else:
            out.append(int(tok))
    return sorted(set(out))


def tap_to_grid(tap):
    return (tap % 20) // N_COLS, tap // 20, tap % N_COLS


def reconstruct_l2(U_r, sel, Y, rcond=None):
    A, *_ = np.linalg.lstsq(U_r[sel, :], Y, rcond=rcond)
    return U_r @ A


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------
def load_data(n_snaps_train):
    """Returns a dict with raw + per-angle-centered training/test data.

    Shape convention: n_taps along rows, time along cols.
    """
    grid = np.load(RAW_DIR / "cp_grid.npy")
    n_ang, T = grid.shape[:2]
    flat = grid.reshape(n_ang, T, -1).astype(np.float64)
    n_tr = int(round(T * TRAIN_RATIO))
    n_va = int(round(T * VALID_RATIO))
    n_use = min(n_snaps_train, n_tr)

    n_taps = flat.shape[2]
    means = np.zeros((n_ang, n_taps))
    pa_train_raw, pa_test_raw = [], []
    pa_train_ctr, pa_test_ctr = [], []
    angle_te = []
    for i in range(n_ang):
        cp_tr_full = flat[i, :n_tr]      # full train for mean estimation
        cp_tr_use  = flat[i, :n_use]     # what mrDMD actually consumes
        cp_te      = flat[i, n_tr + n_va:]
        mu = cp_tr_full.mean(axis=0)
        means[i] = mu
        pa_train_raw.append(cp_tr_use.T.copy())
        pa_test_raw.append(cp_te.T.copy())
        pa_train_ctr.append((cp_tr_use - mu).T.copy())
        pa_test_ctr.append((cp_te - mu).T.copy())
        angle_te.append(np.full(cp_te.shape[0], i, dtype=np.int32))

    return {
        "n_ang": n_ang,
        "n_taps": n_taps,
        "means": means,
        "pa_train_raw": pa_train_raw,
        "pa_test_raw": pa_test_raw,
        "pa_train_ctr": pa_train_ctr,
        "pa_test_ctr": pa_test_ctr,
        "test_concat_raw": np.concatenate(pa_test_raw, axis=1),
        "train_concat_raw": np.concatenate(pa_train_raw, axis=1),
        "angle_te": np.concatenate(angle_te),
    }


def subsample_test(data, k):
    """Subsample each AOA's test slice to roughly k snapshots total."""
    if not (0 < k < data["test_concat_raw"].shape[1]):
        return data
    n_te_total = data["test_concat_raw"].shape[1]
    new_pa_test_raw, new_pa_test_ctr, new_a_te = [], [], []
    for i, X in enumerate(data["pa_test_raw"]):
        T_i = X.shape[1]
        keep = max(1, int(round(T_i * k / n_te_total)))
        idx = np.linspace(0, T_i - 1, keep).astype(int)
        new_pa_test_raw.append(X[:, idx])
        new_pa_test_ctr.append(data["pa_test_ctr"][i][:, idx])
        new_a_te.append(np.full(keep, i, dtype=np.int32))
    data["pa_test_raw"] = new_pa_test_raw
    data["pa_test_ctr"] = new_pa_test_ctr
    data["test_concat_raw"] = np.concatenate(new_pa_test_raw, axis=1)
    data["angle_te"] = np.concatenate(new_a_te)
    return data


def build_basis(args, train_list, label, n_space):
    print(f"  [{label}] mrDMD on {len(train_list)} trajectory(ies), "
          f"sizes={[t.shape for t in train_list]}"
          f"{' [NO HANKEL]' if args.no_hankel else ''}")
    if args.no_hankel:
        Phi, info = _build_mrdmd_basis_nohankel(
            train_list, max_levels=args.L, max_cyc=args.max_cyc,
            r_dmd=args.r_max_per_dmd, dt=DT_TPU)
    else:
        Phi, info = build_mrdmd_basis(
            train_list, n_space=n_space,
            d_embed=args.d_embed, L=args.L, max_cyc=args.max_cyc,
            dt=DT_TPU, r_max_per_dmd=args.r_max_per_dmd,
            amp_quantile=args.amp_quantile,
            orthonormalize=getattr(args, "orthonormalize", True),
            use_fb=getattr(args, "use_fb", True),
            verbose=True)
    print(f"  [{label}] basis rank = {info['basis_rank']}")
    return Phi, info


# ---------------------------------------------------------------------
# Per-AOA evaluation given a global sensor set + per-angle (basis, X_te,
# optional mean). Returns per-AOA total/unknown MAE on physical Cp scale.
# ---------------------------------------------------------------------
def _eval_per_angle_global_sensors(sel, n_taps, per_angle_basis,
                                   per_angle_test_ctr, means,
                                   recon_fn, recon_kwargs):
    """For each AOA i: rec_ctr = recon_fn(Phi_i, sel, X_te_ctr_i[sel]);
    add mu_i back; MAE against (X_te_ctr_i + mu_i) i.e. the physical Cp.
    Mathematically MAE on centered space == MAE on physical (mu cancels)."""
    unk = np.setdiff1d(np.arange(n_taps), sel)
    per_ang_total, per_ang_unk = [], []
    for i, (Phi_i, X_te_ctr) in enumerate(
            zip(per_angle_basis, per_angle_test_ctr)):
        rec = recon_fn(Phi_i, sel, X_te_ctr[sel, :], **recon_kwargs)
        # mu cancels in |X - rec|, but materialize Cp scale for clarity.
        mu = means[i][:, None]
        diff = np.abs((X_te_ctr + mu) - (rec + mu))
        per_ang_total.append(float(diff.mean()))
        per_ang_unk.append(float(diff[unk].mean()) if unk.size else 0.0)
    return per_ang_total, per_ang_unk


def _eval_global_sensors_no_centering(sel, n_taps, Phi,
                                      test_concat_raw, angle_te,
                                      recon_fn, recon_kwargs):
    """Single global basis, no centering. Compute per-angle MAE by slicing
    test_concat_raw via angle_te."""
    unk = np.setdiff1d(np.arange(n_taps), sel)
    rec = recon_fn(Phi, sel, test_concat_raw[sel, :], **recon_kwargs)
    diff = np.abs(test_concat_raw - rec)
    angles = np.unique(angle_te)
    per_ang_total, per_ang_unk = [], []
    for a in angles:
        m = angle_te == a
        per_ang_total.append(float(diff[:, m].mean()))
        per_ang_unk.append(float(diff[unk][:, m].mean())
                           if unk.size else 0.0)
    return per_ang_total, per_ang_unk, rec


# ---------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------
def run_global_basis_variant(args, data, sensor_counts, train_list,
                             label, recon_fn, recon_kwargs,
                             save_per_n_npz=False, out_dir=None):
    """Shared path for variants using ONE global basis + ONE global sensor
    set. Used by baseline (raw, L2).
    NO per-AOA centering. Per-angle MAE is via angle_te slicing."""
    n_taps = data["n_taps"]
    Phi, info = build_basis(args, train_list, label, n_taps)
    r = info["basis_rank"]
    _, _, piv = qr(Phi.T, pivoting=True, mode="economic")

    rows, per_angle_rows, sels = [], [], []
    angles = sorted(np.unique(data["angle_te"]).tolist())
    for n in sensor_counts:
        sel = np.sort(piv[:n])
        t0 = time.perf_counter()
        per_t, per_u, rec = _eval_global_sensors_no_centering(
            sel, n_taps, Phi, data["test_concat_raw"],
            data["angle_te"], recon_fn, recon_kwargs)
        recon_time = time.perf_counter() - t0
        total = np.array(per_t); unk = np.array(per_u)
        rows.append([n, r, total.mean(), total.std(),
                     unk.mean(), unk.std(), recon_time])
        for a, t_a, u_a in zip(angles, per_t, per_u):
            per_angle_rows.append([n, a, t_a, u_a])
        sels.append(sel)
        print(f"  n={n:3d}  r={r:3d}  unkMAE={unk.mean():.6f}  "
              f"worst={unk.max():.6f}  t={recon_time:6.2f}s")

        if save_per_n_npz and out_dir is not None:
            sub = out_dir / f"save_{n}"
            sub.mkdir(exist_ok=True)
            np.savez(sub / "data.npz",
                     all_labels=data["test_concat_raw"].T,
                     all_preds=rec.T,
                     current_sensor_indices=sel)

    cols = ["Sensor_Count", "Basis_Rank",
            "Total_MAE_Mean", "Total_MAE_Std",
            "Unknown_MAE_Mean", "Unknown_MAE_Std", "Recon_Time_s"]
    df = pd.DataFrame(rows, columns=cols)
    df_pa = pd.DataFrame(per_angle_rows, columns=[
        "Sensor_Count", "Angle_ID", "Total_MAE", "Unknown_MAE"])
    extras = {"Phi": Phi, "info": info, "piv_global": piv}
    return df, df_pa, sels, extras


def run_baseline_l2(args, data, sensor_counts, out_dir):
    train_list = [data["train_concat_raw"]]
    return run_global_basis_variant(
        args, data, sensor_counts, train_list, "baseline",
        reconstruct_l2, {}, save_per_n_npz=False, out_dir=out_dir)


def run_planA_l2(args, data, sensor_counts):
    """Per-AOA centered concat -> 1 global mrDMD -> global QR -> per-AOA recon."""
    n_taps = data["n_taps"]
    train_list = [np.concatenate(data["pa_train_ctr"], axis=1)]
    Phi, info = build_basis(args, train_list, "planA", n_taps)
    r = info["basis_rank"]
    _, _, piv = qr(Phi.T, pivoting=True, mode="economic")

    rows, per_angle_rows, sels = [], [], []
    n_ang = data["n_ang"]
    for n in sensor_counts:
        sel = np.sort(piv[:n])
        t0 = time.perf_counter()
        per_t, per_u = _eval_per_angle_global_sensors(
            sel, n_taps,
            per_angle_basis=[Phi] * n_ang,
            per_angle_test_ctr=data["pa_test_ctr"],
            means=data["means"],
            recon_fn=reconstruct_l2, recon_kwargs={})
        recon_time = time.perf_counter() - t0
        total = np.array(per_t); unk = np.array(per_u)
        rows.append([n, r, total.mean(), total.std(),
                     unk.mean(), unk.std(), recon_time])
        for a in range(n_ang):
            per_angle_rows.append([n, a, per_t[a], per_u[a]])
        sels.append(sel)
        print(f"  n={n:3d}  r={r:3d}  unkMAE={unk.mean():.6f}  "
              f"worst={unk.max():.6f}  t={recon_time:6.2f}s")

    df = pd.DataFrame(rows, columns=[
        "Sensor_Count", "Basis_Rank",
        "Total_MAE_Mean", "Total_MAE_Std",
        "Unknown_MAE_Mean", "Unknown_MAE_Std", "Recon_Time_s"])
    df_pa = pd.DataFrame(per_angle_rows, columns=[
        "Sensor_Count", "Angle_ID", "Total_MAE", "Unknown_MAE"])
    return df, df_pa, sels, {"Phi": Phi, "info": info, "piv_global": piv}


def run_oracle_l2(args, data, sensor_counts):
    """Per-AOA mrDMD + per-AOA QR + per-AOA sensors. Each angle evaluated
    only on its own slice with its own sensors."""
    n_taps = data["n_taps"]
    n_ang = data["n_ang"]
    print(f"  [oracle] building {n_ang} per-angle mrDMD bases ...")

    detail_rows, per_angle_rows = [], []
    per_n_total, per_n_unk = (
        {int(n): [] for n in sensor_counts},
        {int(n): [] for n in sensor_counts})
    sel_per_n = [[] for _ in sensor_counts]

    for ai in range(n_ang):
        Phi_a, info_a = build_basis(
            args, [data["pa_train_ctr"][ai]], f"oracle[AOA={ai}]", n_taps)
        r_a = info_a["basis_rank"]
        _, _, piv_a = qr(Phi_a.T, pivoting=True, mode="economic")
        X_te_ctr = data["pa_test_ctr"][ai]

        for k, n in enumerate(sensor_counts):
            sel = np.sort(piv_a[:n])
            unk = np.setdiff1d(np.arange(n_taps), sel)
            rec = reconstruct_l2(Phi_a, sel, X_te_ctr[sel, :])
            diff = np.abs(X_te_ctr - rec)
            t_m = float(diff.mean())
            u_m = float(diff[unk].mean()) if unk.size else 0.0
            detail_rows.append([ai, int(n), r_a, t_m, u_m])
            per_angle_rows.append([int(n), ai, t_m, u_m])
            per_n_total[int(n)].append(t_m)
            per_n_unk[int(n)].append(u_m)
            sel_per_n[k].append(sel)

    rows = []
    for n in sensor_counts:
        ts = np.array(per_n_total[int(n)])
        us = np.array(per_n_unk[int(n)])
        rows.append([int(n), np.nan, ts.mean(), ts.std(),
                     us.mean(), us.std(), np.nan])

    df = pd.DataFrame(rows, columns=[
        "Sensor_Count", "Avg_Basis_Rank",
        "Total_MAE_Mean", "Total_MAE_Std",
        "Unknown_MAE_Mean", "Unknown_MAE_Std", "Recon_Time_s"])
    df_pa = pd.DataFrame(per_angle_rows, columns=[
        "Sensor_Count", "Angle_ID", "Total_MAE", "Unknown_MAE"])
    df_detail = pd.DataFrame(detail_rows, columns=[
        "AOA_Index", "Sensor_Count", "Basis_Rank", "Total_MAE", "Unknown_MAE"])
    return df, df_pa, sel_per_n, {"detail": df_detail}


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_sensor_layout(piv_list, n_list, out_dir, tag, fname="sensor_layout.png"):
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
    fig.suptitle(f"mrDMD-QR sensor layout ({tag})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=300)
    plt.close(fig)


def plot_oracle_overlap(sel_per_n, n_list, n_ang, out_dir, tag):
    n_plots = len(n_list)
    if n_plots == 0:
        return
    ncol = min(5, n_plots)
    nrow = int(np.ceil(n_plots / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.6 * nrow),
                             squeeze=False)
    n_taps_total = N_FACES * N_ROWS * N_COLS
    im = None
    for k, (n, sels_at_n) in enumerate(zip(n_list, sel_per_n)):
        ax = axes[k // ncol, k % ncol]
        count = np.zeros(n_taps_total, dtype=int)
        for sel_i in sels_at_n:
            count[sel_i] += 1
        grid = np.zeros((N_ROWS, N_FACES * N_COLS), dtype=int)
        for tap in range(n_taps_total):
            f, r, c = tap_to_grid(tap)
            grid[r, f * N_COLS + c] = count[tap]
        im = ax.imshow(grid, cmap="hot", vmin=0, vmax=n_ang, aspect="auto")
        for x in [N_COLS, 2 * N_COLS, 3 * N_COLS]:
            ax.axvline(x - 0.5, color="cyan", lw=0.8)
        ax.set_xticks([N_COLS / 2 - 0.5 + i * N_COLS for i in range(N_FACES)])
        ax.set_xticklabels(FACE_NAMES, fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"n={n}", fontsize=10)
    for k in range(n_plots, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle(f"oracle - per-tap selection count across {n_ang} AOAs "
                 f"({tag})", fontsize=11)
    fig.tight_layout()
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.02)
        cbar.set_label(f"# angles selecting tap (max = {n_ang})")
    fig.savefig(out_dir / "sensor_overlap_heatmap.png", dpi=300)
    plt.close(fig)


def plot_variant_curve(df, out_dir, title, ylabel="Unknown Sensor MAE"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["Sensor_Count"], df["Unknown_MAE_Mean"],
            "--c^", lw=1.5, ms=6, mfc="c", label="Unknown MAE (mean over AOAs)")
    ax.fill_between(df["Sensor_Count"],
                    df["Unknown_MAE_Mean"] - df["Unknown_MAE_Std"],
                    df["Unknown_MAE_Mean"] + df["Unknown_MAE_Std"],
                    alpha=0.15, label="+/-1 std across AOAs")
    ax.set_xlabel("Number of Selected Sensors (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "L2_Performance.png", dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors", default=SENSOR_COUNTS_SPEC)
    ap.add_argument("--layout_max_n", type=int, default=LAYOUT_MAX_N)
    ap.add_argument("--test_subsample", type=int, default=TEST_SUBSAMPLE)
    ap.add_argument("--n_snaps", type=int, default=N_SNAPS_DEFAULT)
    ap.add_argument("--d_embed", type=int, default=D_EMBED_DEFAULT)
    ap.add_argument("--L", type=int, default=L_LEVEL_DEFAULT)
    ap.add_argument("--max_cyc", type=int, default=MAX_CYC_DEFAULT)
    ap.add_argument("--r_max_per_dmd", type=int, default=R_MAX_PER_DMD_DEFAULT)
    ap.add_argument("--amp_quantile", type=float, default=AMP_QUANTILE_DEFAULT)
    ap.add_argument("--angle_strategy", default=ANGLE_STRATEGY,
                    choices=ALL_VARIANTS)
    ap.add_argument("--no_hankel", action="store_true",
                    help="Use textbook mrDMD (no Hankel time-delay "
                         "embedding). Ignores --d_embed/--amp_quantile.")
    args = ap.parse_args()

    strategy = args.angle_strategy
    # Folder = variant only. Hyperparameters live in filenames so reruns
    # with different L/d/mc/ns coexist in the same folder.
    if args.no_hankel:
        hp_tag = f"L{args.L}_mc{args.max_cyc}_r{args.r_max_per_dmd}"
    else:
        hp_tag = f"L{args.L}_d{args.d_embed}_mc{args.max_cyc}"
    out_dir = MODE_RESULT_ROOT / f"mrdmdqr_l2_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{strategy}_{hp_tag}"  # used in plot titles + filenames

    print("[1/4] Loading TPU data ...")
    data = load_data(args.n_snaps)
    print(f"  n_angles={data['n_ang']}, n_taps={data['n_taps']}, "
          f"n_test_total={data['test_concat_raw'].shape[1]}")
    data = subsample_test(data, args.test_subsample)

    sensor_counts = np.array(parse_sensor_counts(args.sensors), dtype=int)
    sensor_counts = sensor_counts[(sensor_counts >= 1) &
                                  (sensor_counts <= data["n_taps"])]
    if sensor_counts.size == 0:
        raise ValueError(f"no valid sensor counts in spec {args.sensors!r}")

    print(f"[2/4] Running variant: {strategy}")
    if strategy == "baseline":
        df, df_pa, sels, extras = run_baseline_l2(
            args, data, sensor_counts, out_dir)
    elif strategy == "planA":
        df, df_pa, sels, extras = run_planA_l2(args, data, sensor_counts)
    elif strategy == "oracle":
        df, df_pa, sels, extras = run_oracle_l2(args, data, sensor_counts)
    else:
        raise ValueError(strategy)

    print("[3/4] Saving results ...")
    # Single workbook, two sheets: a 1-row-per-n "summary" and a
    # 1-row-per-(n, angle) "per_angle". Same convention as svd_qr.
    with pd.ExcelWriter(out_dir / f"mae_{tag}.xlsx",
                        engine="openpyxl") as w:
        df.to_excel(w, sheet_name="summary", index=False)
        df_pa.to_excel(w, sheet_name="per_angle", index=False)
    if "detail" in extras:
        extras["detail"].to_excel(out_dir / f"oracle_detail_{tag}.xlsx",
                                  index=False)
    if "Phi" in extras:
        np.savez(out_dir / "mrdmd_basis.npz",
                 Phi=extras["Phi"],
                 piv_global=extras["piv_global"],
                 singular_values=extras["info"]["singular_values"],
                 n_modes_per_angle=np.array(
                     extras["info"]["n_modes_per_angle"]))

    print("[4/4] Plotting ...")
    plot_variant_curve(df, out_dir,
                       f"mrDMD-QR {strategy} ({tag})")

    layout_n = [int(n) for n in sensor_counts if n <= args.layout_max_n]
    if strategy == "oracle":
        # angle-0 reference layout + cross-angle overlap heatmap
        sels_layout_a0 = [sels[k][0] for k, n in enumerate(sensor_counts)
                          if n <= args.layout_max_n]
        plot_sensor_layout(sels_layout_a0, layout_n, out_dir,
                           f"oracle angle-0 reference ({tag})",
                           fname="sensor_layout_angle0.png")
        sels_per_n_layout = [sels[k] for k, n in enumerate(sensor_counts)
                             if n <= args.layout_max_n]
        plot_oracle_overlap(sels_per_n_layout, layout_n,
                            data["n_ang"], out_dir, tag)
    else:
        sels_layout = [sels[k] for k, n in enumerate(sensor_counts)
                       if n <= args.layout_max_n]
        plot_sensor_layout(sels_layout, layout_n, out_dir,
                           f"{strategy} ({tag})")
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
