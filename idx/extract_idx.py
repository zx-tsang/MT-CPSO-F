"""Extract pivoted-QR sensor indices for SVD-QR and mrDMD-QR baselines.

For each sensor count n in {2, 4, ..., 20}, picks the n tap-ids that
the QR pivoting would deploy. Writes:

    idx/svd-qr/sensors_n{n}.txt           (1 tap-id per line)
    idx/svd-qr/svd-qr_idx_all.xlsx        (long table)
    idx/mrDMD-qr/sensors_n{n}.txt
    idx/mrDMD-qr/mrDMD-qr_idx_all.xlsx

Tap IDs are 0-based (consistent with numpy / cp_grid indexing
convention) — feed them straight into U_r[sel, :] or cp_grid[sel].
Pass `one_based=True` to save_sensors() if you need MATLAB-style
1-based indices instead.

SVD-QR uses the L2 99% energy threshold (the paper's "average best"
configuration; SVD is computed on RAW concatenated training data, no
mean subtraction). For each n, the basis rank is r = min(n, r_max=57),
so the gappy-POD system Phi[sens] @ a = y is SQUARE (n equations,
n unknowns). The QR pivot is re-run on the truncated U[:, :r] for
every n, which is why the n=4 sensors are not in general a superset
of the n=2 sensors.

mrDMD-QR uses the paper baseline:
    L = 7, max_cyc = 5, r_max_per_dmd = 5
    variant = textbook_no_hankel (exact DMD, no Hankel, no orthonormalisation)
    N_SNAPS = 30000 (full training portion of every angle)

Run from anywhere:
    python idx/extract_idx.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.linalg import qr

HERE = Path(__file__).resolve().parent       # idx/
ROOT = HERE.parent                            # repository root
sys.path.insert(0, str(ROOT / "baselines" / "svd_qr"))
sys.path.insert(0, str(ROOT / "baselines" / "mrdmd_qr"))

NS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def extract_svdqr_idx(energy=0.99):
    """SVD-QR sensors at the given energy threshold. Mirrors
    svdqr_tpu.run_baseline() exactly: for each n, the basis rank is
    r = min(n, r_max_energy), and a fresh QR pivot is done on U[:, :r].
    This per-n re-pivoting is what the paper's
    `sensors_baseline_99pct.xlsx` actually contains."""
    from svdqr_tpu import load_train_test, rank_at_energy

    print(f"[svd-qr] L2 energy={energy:.2f}")
    print("  loading TPU training data...")
    _, _, _, train_raw, _, _ = load_train_test()
    print(f"  train_raw shape: {train_raw.shape}")

    print("  SVD on raw training matrix...")
    U, s, _ = np.linalg.svd(train_raw, full_matrices=False)
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    r_max = rank_at_energy(cum, energy)
    print(f"  r_max @ energy {energy:.2f} = {r_max}  (actual cum energy = {cum[r_max-1]:.4f})")

    out = {}
    for n in NS:
        # Per-n basis truncation: r = min(n, r_max). This matches
        # svdqr_tpu.run_baseline() L267.
        r = max(min(int(n), r_max), 1)
        U_r = U[:, :r]
        if n > r:
            _, _, piv = qr(U_r @ U_r.T, pivoting=True, mode="economic")
        else:
            _, _, piv = qr(U_r.T, pivoting=True, mode="economic")
        sensors = np.sort(piv[:n]).astype(int).tolist()
        out[n] = sensors
        print(f"  n={n:2d}  r={r:2d}  sensors={sensors}")
    return out


def extract_mrdmdqr_idx():
    """Stage-3 best (L=7, C=5, R=5) mrDMD-QR sensors.

    Reuses the same build_basis / load_data implementations that the
    grid sweep uses, so the basis and QR ordering are identical to
    what the grid sweep would have produced at this (L, C, R)."""
    from mrdmdqr_tpu_l2 import load_data, build_basis

    print("[mrDMD-qr] Stage-3 best: L=7, C=5, R=5, no Hankel")
    print("  loading TPU data (N_SNAPS=30000)...")
    N_SNAPS = 30000
    data = load_data(N_SNAPS)
    n_taps = data["n_taps"]

    args = SimpleNamespace(
        L=7, max_cyc=5, r_max_per_dmd=5,
        d_embed=30, amp_quantile=0.0,
        no_hankel=True,
        use_fb=False,
        orthonormalize=False,
    )
    train_list = [data["train_concat_raw"]]

    print("  building mrDMD basis...")
    Phi, info = build_basis(args, train_list, "stage3_L7_C5_R5", n_taps)
    print(f"  basis rank = {info['basis_rank']}")

    print("  pivoted QR on Phi.T...")
    _, _, piv = qr(Phi.T, pivoting=True, mode="economic")
    out = {n: np.sort(piv[:n]).astype(int).tolist() for n in NS}
    for n in NS:
        print(f"  n={n:2d}  sensors={out[n]}")
    return out


def save_sensors(out_dir, name, sensors_per_n, one_based=False):
    """Save sensor indices to txt + xlsx.

    Internally the QR pivots return 0-based numpy indices into the 500-tap
    cp_grid; that is what we save by default so the files can be fed
    straight back into U_r[sel, :] / cp_grid[sel] without an off-by-one
    correction. Pass one_based=True to add +1 to every Tap_ID for MATLAB
    / paper-style indexing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shift = 1 if one_based else 0
    rows = []
    for n, taps in sensors_per_n.items():
        taps_out = [t + shift for t in taps]
        (out_dir / f"sensors_n{n}.txt").write_text(
            "\n".join(str(t) for t in taps_out) + "\n", encoding="utf-8")
        for rank, tap in enumerate(taps_out, start=1):
            rows.append({"Sensor_Count": n, "Rank": rank, "Tap_ID": tap})
    pd.DataFrame(rows).to_excel(
        out_dir / f"{name}_idx_all.xlsx", index=False)
    base_tag = "1-based" if one_based else "0-based"
    print(f"  saved ({base_tag}) -> {out_dir}")


def main():
    print("=" * 60)
    print("Extracting SVD-QR sensor indices (L2 99% energy)")
    print("=" * 60)
    svdqr = extract_svdqr_idx(energy=0.99)
    save_sensors(HERE / "svd-qr", "svd-qr", svdqr)

    print()
    print("=" * 60)
    print("Extracting mrDMD-QR sensor indices (L=7, C=5, R=5)")
    print("=" * 60)
    mrdmd = extract_mrdmdqr_idx()
    save_sensors(HERE / "mrDMD-qr", "mrDMD-qr", mrdmd)

    print()
    print("Done. Outputs in", HERE)


if __name__ == "__main__":
    main()
