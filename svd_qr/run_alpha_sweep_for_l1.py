"""Valid-tuned alpha sweep for the SVD-QR-L1 (Lasso) baseline.

The per-angle 80/10/10 split (see svdqr_tpu.load_train_valid_test) gives
three disjoint slices: TRAIN builds the SVD basis and QR sensor ordering;
VALID is used here to choose the Lasso regularization alpha; TEST is held
out and only touched once with the chosen alpha*. Tuning on VALID instead
of TEST is what keeps the reported TEST MAE an honest generalization
estimate rather than an optimistic in-sample fit.

Protocol (Plan A — global alpha):
    1. Build basis (SVD + QR) on TRAIN.
    2. For each alpha in the grid, evaluate Unknown_MAE on the VALID
       slice at sensor counts n = 1..20. Save per-alpha valid results.
    3. Pick alpha* = argmin over alphas of valid step2 mean (n=2..20 step 2).
    4. Run the chosen alpha* on the TEST slice → main reported result.
    5. Save alpha_grid_summary.xlsx (VALID MAEs) plus the test result.

Alpha grids per energy threshold:
    The default 7-point grid {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}
    is used at energy = 0.90 and 0.95 (basis rank r ≈ 2 and 6).

    At energy = 0.99 the basis rank jumps to r ≈ 57, which makes Lasso
    non-convergent (and unreasonably slow) at the two smallest alphas.
    For that case we drop 1e-4 and 3e-4 and sweep only 5 points
    {1e-3, 3e-3, 1e-2, 3e-2, 1e-1} — pass them explicitly via --alphas.
    The picked alpha* lives well inside the surviving range so the
    truncation does not affect the reported result.

Usage:
    python run_alpha_sweep_for_l1.py --energy 0.95
    python run_alpha_sweep_for_l1.py --energy 0.99 \
           --alphas 1e-3,3e-3,1e-2,3e-2,1e-1

Output:
    svd_qr/mode_result/svdqr_l1_alpha_sweep_<pct>/
        mae_test_at_alpha_star_<pct>.xlsx           (TEST, at alpha*)
        mae_valid_alpha<tag>_<pct>.xlsx             (per-alpha VALID, one per alpha)
        alpha_grid_summary.xlsx                     (all alphas on VALID, wide)
        alpha_choice.txt                            (records alpha* and the grid used)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr

# svdqr_tpu.py lives next to this script in the same svd_qr/ directory.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from svdqr_tpu import (
    load_train_valid_test, rank_at_energy, reconstruct_l1_matlab,
)

DEFAULT_ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SENSOR_COUNTS = list(range(1, 21))
STEP2_NS = list(range(2, 21, 2))


def alpha_tag(a):
    if a >= 1:
        return "1e0"
    s = f"{a:.0e}"
    return s.replace("e-0", "e-").lstrip("0")


def eval_one(U_r, piv, eval_raw, eval_angles, n_total, alpha):
    """Sweep sensor counts at one alpha on one evaluation split.
    Returns dataframe with one row per n."""
    angle_ids = sorted(np.unique(eval_angles).tolist())
    rows = []
    for n in SENSOR_COUNTS:
        sel = np.asarray(piv[:n])
        unk = np.setdiff1d(np.arange(n_total), sel)
        t0 = time.perf_counter()
        recon = reconstruct_l1_matlab(U_r, sel, eval_raw[sel, :], alpha)
        recon_time = time.perf_counter() - t0
        unk_mae = float(np.mean(np.abs(eval_raw[unk, :] - recon[unk, :])))
        per_ang = []
        for ang in angle_ids:
            mask = eval_angles == ang
            err = float(np.mean(
                np.abs(eval_raw[unk][:, mask] - recon[unk][:, mask])))
            per_ang.append(err)
        per_ang = np.array(per_ang)
        rows.append({
            "Sensor_Count": n,
            "Basis_Rank": U_r.shape[1],
            "Unknown_MAE_Mean": unk_mae,
            "PerAngle_MAE_Mean": float(per_ang.mean()),
            "PerAngle_MAE_Std":  float(per_ang.std()),
            "PerAngle_MAE_Max":  float(per_ang.max()),
            "Recon_Time_s": recon_time,
        })
        print(f"  n={n:>3}  MAE={unk_mae:.4f}  t={recon_time:.1f}s", flush=True)
    return pd.DataFrame(rows)


def step2_mean(df):
    """Plan A criterion: mean of Unknown_MAE_Mean across n in [2, 4, ..., 20]."""
    sub = df[df["Sensor_Count"].isin(STEP2_NS)]
    return float(sub["Unknown_MAE_Mean"].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", type=float, required=True)
    ap.add_argument("--alphas", type=str, default=None,
                    help="Comma-separated alpha list. Defaults to "
                         "1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1. For high-rank "
                         "energies (e.g. 99pct rank=57), drop the smallest "
                         "alphas to avoid slow non-converging Lasso solves.")
    args = ap.parse_args()
    alphas = (DEFAULT_ALPHAS if args.alphas is None
              else [float(x) for x in args.alphas.split(",")])

    pct = f"{int(round(args.energy * 100))}pct"
    out_dir = (Path(__file__).resolve().parent
               / "mode_result" / f"svdqr_l1_alpha_sweep_{pct}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] TPU data ...", flush=True)
    (_, _, _, train_raw, valid_raw, test_raw,
     angle_va, angle_te) = load_train_valid_test()
    n_total = train_raw.shape[0]
    print(f"  train: {train_raw.shape}, valid: {valid_raw.shape}, "
          f"test: {test_raw.shape}", flush=True)

    print(f"[SVD + QR] (once on training data)", flush=True)
    U, s, _ = np.linalg.svd(train_raw, full_matrices=False)
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    r_locked = rank_at_energy(cum, args.energy)
    U_r = U[:, :r_locked]
    if max(SENSOR_COUNTS) > r_locked:
        _, _, piv = qr(U_r @ U_r.T, pivoting=True, mode="economic")
    else:
        _, _, piv = qr(U_r.T, pivoting=True, mode="economic")
    cum_E = float(cum[r_locked - 1])
    print(f"  rank @ {pct} = {r_locked}, actual cum energy = {cum_E:.4f}",
          flush=True)

    # ---------- Phase 1: VALID sweep ----------
    print(f"\n[VALID sweep] {len(alphas)} alphas x {len(SENSOR_COUNTS)} n", flush=True)
    valid_results = {}
    t_phase1 = time.perf_counter()
    for a in alphas:
        tag = alpha_tag(a)
        print(f"\n[valid alpha={a}] ({tag})", flush=True)
        t0 = time.perf_counter()
        df_v = eval_one(U_r, piv, valid_raw, angle_va, n_total, a)
        valid_results[tag] = df_v
        out_file = out_dir / f"mae_valid_alpha{tag}_{pct}.xlsx"
        df_v.to_excel(out_file, index=False)
        s2 = step2_mean(df_v)
        print(f"  [save] {out_file.name}  step2 mean (n=2..20) = {s2:.4f}  "
              f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    print(f"\n[valid] phase done in {(time.perf_counter()-t_phase1)/60:.1f} min",
          flush=True)

    # ---------- Phase 2: pick alpha* on VALID step2 mean ----------
    scores = {tag: step2_mean(df) for tag, df in valid_results.items()}
    best_tag = min(scores, key=scores.get)
    best_alpha = alphas[[alpha_tag(a) for a in alphas].index(best_tag)]
    print(f"\n[pick] alpha* = {best_alpha} (tag {best_tag})  "
          f"valid step2 mean = {scores[best_tag]:.4f}", flush=True)

    # Save grid summary on VALID (rows = n, cols = alpha)
    summary = {"n": SENSOR_COUNTS}
    for tag, df in valid_results.items():
        summary[f"a={tag}"] = df["Unknown_MAE_Mean"].values
    df_grid = pd.DataFrame(summary)
    cols = [c for c in df_grid.columns if c.startswith("a=")]
    df_grid["best_alpha_valid"] = df_grid[cols].idxmin(axis=1)
    df_grid["best_MAE_valid"]   = df_grid[cols].min(axis=1)
    df_grid.to_excel(out_dir / "alpha_grid_summary.xlsx", index=False)
    print(f"[save] alpha_grid_summary.xlsx (VALID, all alphas)", flush=True)

    # ---------- Phase 3: chosen alpha on TEST ----------
    print(f"\n[TEST] running alpha* = {best_alpha} on test split", flush=True)
    t0 = time.perf_counter()
    df_test = eval_one(U_r, piv, test_raw, angle_te, n_total, best_alpha)
    main_file = out_dir / f"mae_test_at_alpha_star_{pct}.xlsx"
    df_test.to_excel(main_file, index=False)
    s2_test = step2_mean(df_test)
    print(f"\n[save] {main_file.name}  TEST step2 mean = {s2_test:.4f}  "
          f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

    # ---------- Record the choice ----------
    note_file = out_dir / "alpha_choice.txt"
    note_file.write_text(
        f"energy_threshold = {args.energy}\n"
        f"basis_rank       = {r_locked}\n"
        f"cum_energy       = {cum_E:.6f}\n"
        f"alpha_grid       = {alphas}\n"
        f"selection        = Plan A: argmin over alphas of step2 mean "
        f"(n=2,4,...,20) on VALID\n"
        f"alpha_star       = {best_alpha} (tag {best_tag})\n"
        f"valid_step2_mean = {scores[best_tag]:.6f}\n"
        f"test_step2_mean  = {s2_test:.6f}\n"
    )
    print(f"[save] {note_file.name}", flush=True)


if __name__ == "__main__":
    main()
