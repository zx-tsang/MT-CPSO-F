"""
stepa_preprocess.py — multi-K POD preprocessing for the POD-Transformer baseline.

Runs all expensive SVDs (centered HF for QR, raw HF/LF for decoders) ONCE
and emits per-K artifact folders raw_data/podtfm_p{K}_k{K}/ for K in $KS.

For each K:
  * Phi_qr <- centered SVD basis truncated to r=p=K  --> QR placement
  * Gappy reconstruction with mean removed/added back
  * U_HF_k / U_LF_k <- raw SVDs truncated to k=K
  * Per-mode z-score using TRAIN portion only
  * Sliding-window train/valid/test

Sensor placement uses MEAN-REMOVED data (Nav 2025 §3.1), as required.

NOTE on LF SVD reuse: U_LF depends on the gappy reconstruction X_LF, which
depends on the K sensors picked at that K. So LF SVDs are NOT shareable
across K and we run one LF SVD per K. The HF raw SVD however IS shareable
(same matrix every time) and we run it exactly once.

Usage:
    python stepa_preprocess.py --ks 2 4 6 8 10 12 14 16 18 20
"""
import argparse
from pathlib import Path
import numpy as np
import scipy.linalg as sla
import yaml

ROOT = Path(__file__).resolve().parent          # baselines/podtfm/ (params.yaml here)
PROJECT_ROOT = ROOT.parent.parent                # repository root (raw_data here)
RAW = PROJECT_ROOT / "raw_data"
N_TAPS = 500


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ks", type=int, nargs="+",
                   default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                   help="K values to emit (uses p=k=K for each)")
    return p.parse_args()


def split_idx(n, tr, va):
    nt = int(round(n * tr)); nv = int(round(n * va))
    return nt, nv, n - nt - nv


def windows(series, W, stride):
    T, F = series.shape
    if T < W:
        return np.empty((0, W, F), dtype=series.dtype)
    n = (T - W) // stride + 1
    return np.stack([series[i*stride:i*stride + W] for i in range(n)])


def svd_qr_placement(Phi, p):
    m, r = Phi.shape
    assert p <= r, f"need p<=r; got p={p} r={r}"
    _, _, piv = sla.qr(Phi.T, pivoting=True, mode="economic")
    return np.sort(piv[:p])


def main():
    args = parse_args()
    KS = sorted(set(args.ks))
    K_MAX = max(KS)

    with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
        PRM = yaml.safe_load(f)
    WINDOW = int(PRM["window_size"])
    STR_TR = int(PRM["stride_train"])
    STR_EV = int(PRM["stride_eval"])
    TR_R = float(PRM["train_ratio"])
    VA_R = float(PRM["valid_ratio"])

    meta = np.load(RAW / "metadata.npz")
    angles = meta["angles"]
    n_per = int(meta["n_t_per_angle"])
    n_ang = len(angles)
    flat = np.load(RAW / "all_Data_all_place.npy")
    assert flat.shape == (N_TAPS, n_ang * n_per), flat.shape
    nt, nv, ne = split_idx(n_per, TR_R, VA_R)
    print(f"[stepra-shared] KS={KS}  W={WINDOW}  n_train={nt} n_valid={nv} n_test={ne}")

    # ============ ONE-TIME EXPENSIVE WORK ============
    HF_tr = np.concatenate(
        [flat[:, i*n_per : i*n_per + nt] for i in range(n_ang)], axis=1
    )                                              # (500, nt*n_ang)
    mu = HF_tr.mean(axis=1, keepdims=True).astype(np.float32)

    print(f"[shared] SVD #1 (centered, for QR + gappy)  matrix {HF_tr.shape} ...")
    HF_tr_c = (HF_tr - mu).astype(np.float64)
    U_qr_full, S_qr, _ = np.linalg.svd(HF_tr_c, full_matrices=False)
    print(f"[shared]   done.  variance(K_MAX={K_MAX}): "
          f"{(S_qr**2).cumsum()[K_MAX-1] / (S_qr**2).sum():.4f}")

    print(f"[shared] SVD #2 (raw HF, for U_HF_k decoder)  matrix {HF_tr.shape} ...")
    U_HF_full, S_HF, _ = np.linalg.svd(HF_tr.astype(np.float64),
                                       full_matrices=False)
    cum_HF = (S_HF**2).cumsum() / (S_HF**2).sum()
    print(f"[shared]   done.  variance(K_MAX={K_MAX}): {cum_HF[K_MAX-1]:.4f}")

    # ============ PER-K LOOP ============
    HF_per_angle = [flat[:, i*n_per : (i+1)*n_per].astype(np.float32)
                    for i in range(n_ang)]

    for K in KS:
        TAG = f"podtfm_p{K}_k{K}"
        OUT = RAW / TAG
        OUT.mkdir(parents=True, exist_ok=True)
        print(f"\n[K={K}] -> {OUT}")

        # Phi_qr (centered, used ONLY for sensor placement + gappy recon)
        Phi_qr = U_qr_full[:, :K].astype(np.float32)
        sens = svd_qr_placement(Phi_qr, K)
        print(f"[K={K}] sensors (0-based): {sens.tolist()}")

        # gappy reconstruction (mean removed -> solve LSQ -> mean added back)
        mu_sens = mu[sens]
        A = Phi_qr[sens, :]                                # (K, K)
        X_LF_list = []
        for i in range(n_ang):
            Y_raw = HF_per_angle[i][sens, :]               # (K, n_per)
            a, *_ = np.linalg.lstsq(A, Y_raw - mu_sens, rcond=None)  # (K, n_per)
            X_LF_list.append((Phi_qr @ a + mu).astype(np.float32))   # (500, n_per)

        # U_HF_k = shared (top-K columns of full HF SVD)
        U_HF_k = U_HF_full[:, :K].astype(np.float32)

        # U_LF_k: LF SVD is K-specific (X_LF depends on the K sensors picked)
        LF_tr = np.concatenate([X_LF_list[i][:, :nt] for i in range(n_ang)], axis=1)
        U_LF_full, _, _ = np.linalg.svd(LF_tr.astype(np.float64),
                                        full_matrices=False)
        U_LF_k = U_LF_full[:, :K].astype(np.float32)

        # project to coefficients
        cLF_per_angle, cHF_per_angle = [], []
        for i in range(n_ang):
            cLF_per_angle.append((U_LF_k.T @ X_LF_list[i]).T.astype(np.float32))
            cHF_per_angle.append((U_HF_k.T @ HF_per_angle[i]).T.astype(np.float32))

        # z-score (train portion only)
        cLF_tr = np.concatenate([cLF_per_angle[i][:nt] for i in range(n_ang)], axis=0)
        cHF_tr = np.concatenate([cHF_per_angle[i][:nt] for i in range(n_ang)], axis=0)
        z_mean_LF = cLF_tr.mean(axis=0).astype(np.float32)
        z_std_LF  = cLF_tr.std (axis=0).astype(np.float32) + 1e-8
        z_mean_HF = cHF_tr.mean(axis=0).astype(np.float32)
        z_std_HF  = cHF_tr.std (axis=0).astype(np.float32) + 1e-8

        # windowing per split
        bucket = {s: {"cLF": [], "cHF": [], "ang": []} for s in ("tr", "va", "te")}
        for i, ang in enumerate(angles):
            cLF = (cLF_per_angle[i] - z_mean_LF) / z_std_LF
            cHF = (cHF_per_angle[i] - z_mean_HF) / z_std_HF
            slices = {"tr": (slice(0, nt),         STR_TR),
                      "va": (slice(nt, nt + nv),   STR_EV),
                      "te": (slice(nt + nv, None), STR_EV)}
            for s, (sl, stride) in slices.items():
                wL = windows(cLF[sl], WINDOW, stride)
                wH = windows(cHF[sl], WINDOW, stride)
                bucket[s]["cLF"].append(wL)
                bucket[s]["cHF"].append(wH)
                bucket[s]["ang"].append(np.full(wL.shape[0], ang, dtype=np.int32))

        # save artifacts (same filenames as the original stepra)
        np.save(OUT / "sensors.npy",    sens.astype(np.int32))
        np.save(OUT / "Phi_qr.npy",     Phi_qr)
        np.save(OUT / "U_HF_k.npy",     U_HF_k)
        np.save(OUT / "U_LF_k.npy",     U_LF_k)
        np.save(OUT / "mu.npy",         mu)
        np.save(OUT / "pod_energy.npy", cum_HF.astype(np.float32))
        np.savez(OUT / "z_stats.npz",
                 z_mean_LF=z_mean_LF, z_std_LF=z_std_LF,
                 z_mean_HF=z_mean_HF, z_std_HF=z_std_HF)
        np.savez(OUT / "config.npz",
                 window_size=WINDOW, k=K, p=K, r=K,
                 angles=angles, n_train=nt, n_valid=nv, n_test=ne)
        for s, fname in [("tr", "train.npz"), ("va", "valid.npz"), ("te", "test.npz")]:
            np.savez(OUT / fname,
                     c_LF=np.concatenate(bucket[s]["cLF"]),
                     c_HF=np.concatenate(bucket[s]["cHF"]),
                     angle=np.concatenate(bucket[s]["ang"]))
            n_tot = sum(b.shape[0] for b in bucket[s]["cLF"])
            sz = (OUT / fname).stat().st_size / 1e6
            print(f"[K={K}]   {fname}: {n_tot} windows  ({sz:.1f} MB)")

    print(f"\n[stepra-shared] DONE — all K artifacts in {RAW}/podtfm_p*_k*/")


if __name__ == "__main__":
    main()
