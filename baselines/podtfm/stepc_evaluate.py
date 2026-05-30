"""
stepc_evaluate.py — POD-Transformer evaluation.

Lifts the model's K-D coefficient predictions back to the full 500-D Cp field
and reports MAE in physical Cp units.

Pairs with:
    stepa_preprocess.py    (raw_data/podtfm_p{K}_k{K}/)
    stepb_train.py         (output_sensor/model/<tag>/checkpoint.pt)

Inputs
------
    raw_data/podtfm_p{K}_k{K}/{train,valid,test}.npz
    raw_data/podtfm_p{K}_k{K}/U_HF_k.npy        (500, K)  decoder
    raw_data/podtfm_p{K}_k{K}/mu.npy            (500, 1)  static training mean
    raw_data/podtfm_p{K}_k{K}/z_stats.npz       per-mode z-score stats
    raw_data/podtfm_p{K}_k{K}/sensors.npy       (K,)      sensor tap indices
    raw_data/podtfm_p{K}_k{K}/config.npz        n_train / n_valid / n_test / angles ...
    raw_data/all_Data_all_place.npy             (500, 11*T)  raw Cp ground truth
    raw_data/metadata.npz                       angles, n_t_per_angle, fs ...
    output_sensor/model/<tag>/checkpoint.pt

Outputs (output_sensor/<tag>/eval/)
    metrics.json / metrics.txt        global + per-angle MAE/RMSE/R
    per_angle.csv                     dataframe-style per-angle table
    per_angle_MAE.png                 bar chart
    per_tap_MAE_heatmap.png           per-tap MAE on the unfolded 4-face grid
    pred_q*_tap*.png                  quantile-spaced time-series examples
    eval.log                          stdout copy

Metrics
-------
  MAE_vs_raw      | predicted Cp vs. ORIGINAL raw Cp           <-- the headline
  MAE_vs_truncPOD | predicted Cp vs. POD-truncated truth       <-- model-only error
  POD_floor       | POD-truncated truth vs. raw Cp             <-- inherent floor
                    (MAE_vs_raw is bounded below by POD_floor)
"""
from __future__ import absolute_import, division, print_function

import argparse
import json
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parent          # baselines/podtfm/ (params.yaml + utils.py here)
PROJECT_ROOT = ROOT.parent.parent                # repository root (raw_data here)
sys.path.insert(0, str(ROOT))

from utils import Params
from stepb_train import PODTransformer, PODCoeffDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_TAG = "baseline_nav_podtfm"

N_FACES, N_ROWS, N_COLS = 4, 25, 5
FACE_NAMES = ["Windward", "Right", "Leeward", "Left"]

parser = argparse.ArgumentParser()
parser.add_argument("--params",   default=str(ROOT / "params.yaml"))
parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "raw_data" / "podtfm_p6_k6"))
parser.add_argument("--raw-dir",  default=str(PROJECT_ROOT / "raw_data"))
parser.add_argument("--out-dir",  default=str(ROOT / "output_sensor"))
parser.add_argument("--tag",      default=DEFAULT_TAG,
                    help="output subdir; must match the --tag used in steprb")
parser.add_argument("--ckpt",     default=None,
                    help="override default checkpoint path")
parser.add_argument("--split",    default="test",
                    choices=["train", "valid", "test"],
                    help="which split to evaluate (default: test)")
parser.add_argument("--n-plots",  type=int, default=4,
                    help="number of example time-series plots")
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--k",          type=int, default=None,
                    help="override pod_input_dim AND pod_output_dim (= POD modes); "
                         "must match the value used in steprb")


# =======================================================================
# helpers
# =======================================================================
def tap_to_grid(tap):
    """Original .mat tap_id -> (face, row, col)."""
    return (tap % 20) // N_COLS, tap // 20, tap % N_COLS


def get_raw_split_slice(n_per, n_train, n_valid, split):
    """Return (start, stop) indices within each angle's time series."""
    if split == "train":
        return 0, n_train
    if split == "valid":
        return n_train, n_train + n_valid
    return n_train + n_valid, n_per


def build_windows_from_raw(flat_raw, angles, n_per, n_train, n_valid,
                            split, window, stride):
    """Mirror stepra's per-angle window slicing on the RAW (500-d) signal.

    Returns: X_raw_windows  (n_total, W, 500)   raw Cp ground truth
             angle_per_win  (n_total,)
    """
    s, e = get_raw_split_slice(n_per, n_train, n_valid, split)
    chunks_X, chunks_a = [], []
    for i, ang in enumerate(angles):
        block = flat_raw[:, i*n_per + s : i*n_per + e].T          # (T_split, 500)
        T = block.shape[0]
        if T < window:
            continue
        n_w = (T - window) // stride + 1
        for w in range(n_w):
            chunks_X.append(block[w*stride : w*stride + window])
        chunks_a.append(np.full(n_w, int(ang), dtype=np.int32))
    X = np.stack(chunks_X, axis=0).astype(np.float32)             # (n, W, 500)
    a = np.concatenate(chunks_a)
    return X, a


def per_tap_mae_grid(per_tap_mae_500):
    """Project (500,) MAE onto the 4-face unfolded 25x20 grid."""
    grid = np.full((N_ROWS, N_FACES * N_COLS), np.nan, dtype=np.float64)
    for tap in range(500):
        f, r, c = tap_to_grid(tap)
        grid[r, f * N_COLS + c] = per_tap_mae_500[tap]
    return grid


# =======================================================================
# core eval
# =======================================================================
def evaluate(args, params, device):
    data_dir = Path(args.data_dir)
    raw_dir  = Path(args.raw_dir)
    out_dir  = Path(args.out_dir) / args.tag / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- artifacts -------------------------------------------------------
    U_HF_k  = np.load(data_dir / "U_HF_k.npy").astype(np.float32)        # (500, 6)
    mu      = np.load(data_dir / "mu.npy").astype(np.float32)             # (500, 1)
    sensors = np.load(data_dir / "sensors.npy").astype(np.int64)          # (6,)
    z       = np.load(data_dir / "z_stats.npz")
    cfg     = np.load(data_dir / "config.npz")
    z_mean  = z["z_mean_HF"].astype(np.float32)                            # (6,)
    z_std   = z["z_std_HF"].astype(np.float32)

    meta    = np.load(raw_dir / "metadata.npz")
    angles  = meta["angles"]
    n_per   = int(meta["n_t_per_angle"])
    n_train = int(cfg["n_train"]); n_valid = int(cfg["n_valid"])
    window  = int(cfg["window_size"])
    stride  = int(params.stride_eval)

    # ---- dataset (z-scored coefficients) ---------------------------------
    ds = PODCoeffDataset(data_dir / f"{args.split}.npz")
    loader = DataLoader(ds, batch_size=(args.batch_size or params.predict_batch),
                        shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    print(f"  split={args.split}  n_windows={len(ds)}  "
          f"x.shape={ds.x.shape}  y.shape={ds.y.shape}")

    # ---- raw Cp ground truth aligned to the same windows ------------------
    flat_raw = np.load(raw_dir / "all_Data_all_place.npy")                # (500, 11*T)
    X_raw, angle_per_win = build_windows_from_raw(
        flat_raw, angles, n_per, n_train, n_valid, args.split,
        window, stride)
    assert X_raw.shape[0] == len(ds), \
        f"raw windows {X_raw.shape[0]} != coeff windows {len(ds)}"
    assert np.array_equal(angle_per_win, ds.angle), \
        "angle ordering mismatch between coeff npz and raw windows"
    print(f"  raw ground truth: {X_raw.shape}   sensors={sensors.tolist()}")

    # ---- model -----------------------------------------------------------
    model = PODTransformer(
        in_dim   = int(params.pod_input_dim),
        out_dim  = int(params.pod_output_dim),
        d_model  = int(params.pod_d_model),
        n_head   = int(params.pod_n_head),
        ffn_hidden = int(params.pod_ffn_hidden),
        n_layers = int(params.pod_n_layers),
        dropout  = float(params.pod_dropout),
        max_len  = int(params.pod_max_len),
    ).to(device)
    ckpt_path = Path(args.ckpt) if args.ckpt else (
        Path(args.out_dir) / "model" / args.tag / "checkpoint.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"  loaded checkpoint: {ckpt_path}")

    # ---- inference (bf16 autocast where available) -----------------------
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available()
                                   and torch.cuda.is_bf16_supported()) else torch.float16
    pred_chunks, lbl_chunks = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"infer {args.split}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype) if torch.cuda.is_available() \
                 else torch.no_grad():
                out = model(x)
            pred_chunks.append(out["mean"].float().cpu())
            lbl_chunks .append(y.cpu())
    cHF_pred_z = torch.cat(pred_chunks, 0).numpy().astype(np.float32)     # (n, W, k)
    cHF_true_z = torch.cat(lbl_chunks,  0).numpy().astype(np.float32)

    # ---- reverse z-score -------------------------------------------------
    cHF_pred = cHF_pred_z * z_std + z_mean                                 # (n, W, 6)
    cHF_true = cHF_true_z * z_std + z_mean

    # ---- lift 6 -> 500 via U_HF_k (mean already baked into the modes) ----
    # X = c @ U_HF_k^T,  shape (n, W, 500)
    X_pred = cHF_pred @ U_HF_k.T                                           # predicted Cp
    X_trc  = cHF_true @ U_HF_k.T                                           # POD-truncated truth

    # ---- metrics ---------------------------------------------------------
    def mae(a, b):  return float(np.mean(np.abs(a - b)))
    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    def pearson_r(a, b):
        af = a.ravel(); bf = b.ravel()
        am = af - af.mean(); bm = bf - bf.mean()
        d = (np.sqrt((am**2).sum()) * np.sqrt((bm**2).sum())) + 1e-12
        return float((am * bm).sum() / d)

    # "unknown" = the (500 - p) taps that the model has to reconstruct.
    # CPSO eval and baseline_nopod report this metric, so we mirror it here
    # for direct cross-method comparison.
    unknown = np.setdiff1d(np.arange(500), sensors)
    Xp_unk = X_pred[:, :, unknown]
    Xr_unk = X_raw [:, :, unknown]
    Xt_unk = X_trc [:, :, unknown]

    metrics = {
        "split": args.split,
        "n_windows": int(X_pred.shape[0]),
        "window_size": int(X_pred.shape[1]),
        "n_taps": int(X_pred.shape[2]),
        "n_sensors": int(sensors.size),
        "n_unknown": int(unknown.size),
        # full-field (all 500 taps)
        "MAE_vs_raw":              mae (X_pred, X_raw),
        "RMSE_vs_raw":             rmse(X_pred, X_raw),
        "Pearson_R_vs_raw":        pearson_r(X_pred, X_raw),
        # unknown-only (same convention as CPSO eval and baseline_nopod)
        "MAE_vs_raw_unknown":      mae (Xp_unk, Xr_unk),
        "RMSE_vs_raw_unknown":     rmse(Xp_unk, Xr_unk),
        "Pearson_R_unknown":       pearson_r(Xp_unk, Xr_unk),
        # model-only error (excludes POD truncation)
        "MAE_vs_truncPOD":         mae (X_pred, X_trc),
        "MAE_vs_truncPOD_unknown": mae (Xp_unk, Xt_unk),
        "RMSE_vs_truncPOD":        rmse(X_pred, X_trc),
        # POD inherent floor
        "POD_floor_MAE":           mae (X_trc, X_raw),
        "POD_floor_MAE_unknown":   mae (Xt_unk, Xr_unk),
        "POD_floor_RMSE":          rmse(X_trc, X_raw),
        "ckpt": str(ckpt_path),
    }

    print(f"\n  MAE(Cp) vs raw      (all 500)  : {metrics['MAE_vs_raw']:.5f}")
    print(f"  MAE(Cp) vs raw      (unknown)  : {metrics['MAE_vs_raw_unknown']:.5f}   <- comparable to CPSO eval")
    print(f"  MAE(Cp) vs truncPOD (unknown)  : {metrics['MAE_vs_truncPOD_unknown']:.5f}   (model-only)")
    print(f"  POD floor MAE       (unknown)  : {metrics['POD_floor_MAE_unknown']:.5f}   (inherent)")
    print(f"  Pearson R vs raw    (all 500)  : {metrics['Pearson_R_vs_raw']:.4f}")

    # ---- per-angle breakdown ---------------------------------------------
    per_ang_rows = []
    for ang in sorted(set(int(a) for a in angle_per_win)):
        mask = (angle_per_win == ang)
        per_ang_rows.append({
            "angle":               ang,
            "n_windows":           int(mask.sum()),
            "MAE_vs_raw":          mae (X_pred[mask], X_raw[mask]),
            "MAE_vs_raw_unknown":  mae (Xp_unk[mask], Xr_unk[mask]),
            "RMSE_vs_raw":         rmse(X_pred[mask], X_raw[mask]),
            "Pearson_R":           pearson_r(X_pred[mask], X_raw[mask]),
            "MAE_vs_truncPOD":     mae (X_pred[mask], X_trc[mask]),
            "POD_floor_MAE":       mae (X_trc [mask], X_raw[mask]),
        })
    metrics["per_angle"] = per_ang_rows

    # ---- per-tap breakdown (500,) ----------------------------------------
    per_tap_mae = np.abs(X_pred - X_raw).mean(axis=(0, 1))                 # (500,)

    # ---- save metrics ----------------------------------------------------
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        for k in ["split", "n_windows", "window_size", "n_taps",
                  "n_sensors", "n_unknown",
                  "MAE_vs_raw", "MAE_vs_raw_unknown",
                  "RMSE_vs_raw", "RMSE_vs_raw_unknown",
                  "Pearson_R_vs_raw", "Pearson_R_unknown",
                  "MAE_vs_truncPOD", "MAE_vs_truncPOD_unknown",
                  "RMSE_vs_truncPOD",
                  "POD_floor_MAE", "POD_floor_MAE_unknown",
                  "POD_floor_RMSE", "ckpt"]:
            f.write(f"{k:24s}: {metrics[k]}\n")
        f.write("\nper-angle:\n")
        for r in per_ang_rows:
            f.write(f"  ang {r['angle']:3d}  n={r['n_windows']:3d}  "
                    f"MAE_all={r['MAE_vs_raw']:.5f}  "
                    f"MAE_unknown={r['MAE_vs_raw_unknown']:.5f}  "
                    f"R={r['Pearson_R']:.4f}  "
                    f"POD_floor={r['POD_floor_MAE']:.5f}\n")

    # CSV-style per-angle table
    import csv
    with open(out_dir / "per_angle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ang_rows[0].keys()))
        w.writeheader()
        for r in per_ang_rows:
            w.writerow(r)

    # ---- plots -----------------------------------------------------------
    plot_per_angle_bar(per_ang_rows, out_dir)
    plot_per_tap_heatmap(per_tap_mae, out_dir, sensors)
    plot_quantile_examples(X_pred, X_raw, per_tap_mae, angle_per_win,
                            out_dir, args.n_plots, fs=int(meta["fs"]))

    print(f"\n  artifacts -> {out_dir}")
    return metrics


# =======================================================================
# plotting
# =======================================================================
def plot_per_angle_bar(rows, out_dir):
    angs = [r["angle"] for r in rows]
    mae_raw   = [r["MAE_vs_raw"]      for r in rows]
    mae_trunc = [r["MAE_vs_truncPOD"] for r in rows]
    mae_floor = [r["POD_floor_MAE"]   for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(angs))
    w = 0.28
    ax.bar(x - w, mae_raw,   width=w, color="#e41a1c", label="MAE vs raw Cp")
    ax.bar(x,     mae_trunc, width=w, color="#377eb8", label="MAE vs trunc POD (model-only)")
    ax.bar(x + w, mae_floor, width=w, color="#999999", label="POD truncation floor")
    ax.set_xticks(x); ax.set_xticklabels([str(a) for a in angs])
    ax.set_xlabel("wind direction (deg)")
    ax.set_ylabel("MAE (Cp)")
    ax.set_title("Per-angle reconstruction error")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "per_angle_MAE.png", dpi=200)
    plt.close(fig)


def plot_per_tap_heatmap(per_tap_mae_500, out_dir, sensors):
    grid = per_tap_mae_grid(per_tap_mae_500)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(grid, cmap="hot_r", aspect="auto")
    for x in [N_COLS, 2 * N_COLS, 3 * N_COLS]:
        ax.axvline(x - 0.5, color="cyan", lw=0.8)
    # mark the sensor taps
    sx, sy = [], []
    for tap in sensors:
        f, r, c = tap_to_grid(int(tap))
        sx.append(f * N_COLS + c); sy.append(r)
    ax.scatter(sx, sy, marker="o", s=110, facecolors="none",
               edgecolors="#2ca02c", linewidths=2.0, label="sensors")
    ax.set_xticks([N_COLS / 2 - 0.5 + i * N_COLS for i in range(N_FACES)])
    ax.set_xticklabels(FACE_NAMES)
    ax.set_ylabel("row (top -> bottom)")
    ax.set_title("Per-tap reconstruction MAE (Cp).  green = sensor")
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="MAE (Cp)")
    fig.tight_layout()
    fig.savefig(out_dir / "per_tap_MAE_heatmap.png", dpi=200)
    plt.close(fig)


def plot_quantile_examples(X_pred, X_raw, per_tap_mae, angle_per_win,
                            out_dir, n_plots, fs):
    """Pick n_plots taps at MAE quantiles (1/8, 3/8, 5/8, 7/8 for n=4).
    For each, plot the full time series concatenated across all wind
    directions (matches the style of stepre_evaluate.py)."""
    order = np.argsort(per_tap_mae)                                # asc
    n = order.size
    qs = [(2 * i - 1) / (2 * n_plots) for i in range(1, n_plots + 1)]
    picks = [int(min(n - 1, max(0, int(round(q * n - 0.5))))) for q in qs]
    feat_ids = order[picks]

    angles_sorted = sorted(set(int(a) for a in angle_per_win))
    npa = sum(angle_per_win == angles_sorted[0])                   # windows per angle
    W = X_pred.shape[1]
    seg_len = npa * W
    t_full = np.arange(len(angles_sorted) * seg_len) / fs

    for q, feat in zip(qs, feat_ids):
        feat = int(feat)
        gt_segs, pr_segs, per_ang_mae = [], [], []
        for ang in angles_sorted:
            mask = (angle_per_win == ang)
            gt = X_raw [mask][:, :, feat].reshape(-1)
            pr = X_pred[mask][:, :, feat].reshape(-1)
            gt_segs.append(gt); pr_segs.append(pr)
            per_ang_mae.append(np.abs(gt - pr).mean())
        gt = np.concatenate(gt_segs); pr = np.concatenate(pr_segs)
        r = float(np.corrcoef(gt, pr)[0, 1]) if gt.size > 1 else 0.0

        fig, ax = plt.subplots(figsize=(16, 4.2))
        ax.plot(t_full, gt, label="Ground Truth", color="#377eb8", linewidth=1.0)
        ax.plot(t_full, pr, label="Prediction",   color="#e41a1c",
                linewidth=1.0, linestyle=(0, (5, 2)))
        ymin, ymax = ax.get_ylim()
        head_y = ymax + 0.06 * (ymax - ymin)
        ax.set_ylim(ymin, ymax + 0.18 * (ymax - ymin))
        for i, ang in enumerate(angles_sorted):
            t0 = i * seg_len / fs; t1 = (i + 1) * seg_len / fs
            if i > 0: ax.axvline(t0, color="0.4", linestyle=":", linewidth=0.8)
            ax.hlines(per_ang_mae[i], t0, t1, colors="#2ca02c",
                      linestyles="--", linewidth=1.2)
            ax.text((t0 + t1) / 2, head_y,
                    f"{int(ang)}°\nMAE={per_ang_mae[i]:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#2ca02c")
        ax.set_xlabel("time (s)"); ax.set_ylabel("Cp")
        ax.set_title(f"tap={feat}  q={q:.3f}  "
                     f"overall MAE={per_tap_mae[feat]:.4f}  R={r:.3f}")
        ax.legend(loc="upper right"); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"pred_q{int(q*1000):03d}_tap{feat}.png", dpi=150)
        plt.close(fig)


def main():
    args = parser.parse_args()
    params = Params(args.params)
    if args.k is not None:
        params.pod_input_dim  = args.k
        params.pod_output_dim = args.k
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluate(args, params, device)


if __name__ == "__main__":
    main()
