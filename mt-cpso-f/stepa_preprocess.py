"""
TPU high-rise (Quan/Tamura 2007) full preprocessing pipeline.

Steps:
  1. Move 11 raw .mat files into raw_data/
  2. Load + assemble:
       - all_Data_all_place.npy   (500, T*11)              flat tap-major layout
       - cp_grid.npy              (11, T, 4, 25, 5)        face-aware grid
       - metadata.npz             locations, angles, B/D/H/fs/Uh, tap_to_grid
  3. Per-angle 80/10/10 chronological split (mirrors Guangzhou pipeline)
  4. Global StandardScaler fit on (T*11, 500); save (500, 2) [mean, std]
  5. Per-angle sliding windows -> stack across angles
       - train.npy  (n_train_windows, window_size, 500)
       - valid.npy  (n_valid_windows, window_size, 500)
       - test.npy   (n_test_windows,  window_size, 500)
All outputs land in raw_data/.

Tap layout (T115_4_channels.jpg):
  4 faces (1=windward, 2=right, 3=leeward, 4=left), each 5 cols x 25 rows,
  spacing 0.02 m. Numbering is row-major across the 4 faces:
     row r (0=top..24=bottom):  y = 0.49 - 0.02*r
     face f (0..3) = ((tap-1) % 20) // 5
     col  c (0..4) = (tap-1) % 5
"""
import shutil
import time
from pathlib import Path
import numpy as np
import scipy.io as sio
import yaml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw_data"
# Output dataset folder, matched to what stepb_train.py expects.
DATASET_TAG = "T115_4_all_place"

ANGLES = list(range(0, 55, 5))
FILE_TPL = "T115_4_{:03d}_1.mat"
N_TAPS, N_T_PER_ANGLE = 500, 32768
N_FACES, N_ROWS, N_COLS = 4, 25, 5

with open(ROOT / "params.yaml", "r", encoding="utf-8") as _f:
    _P = yaml.safe_load(_f)
WINDOW_SIZE = int(_P["window_size"])
STRIDE_TRAIN = int(_P["stride_train"])
STRIDE_EVAL = int(_P["stride_eval"])
TRAIN_RATIO = float(_P["train_ratio"])
VALID_RATIO = float(_P["valid_ratio"])
TEST_RATIO = float(_P["test_ratio"])
assert abs(TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0) < 1e-6, "ratios must sum to 1"
# Auto-tagged folder, e.g. T115_4_all_place_ws50_ss10_pss50
DATA_TAG = f"{DATASET_TAG}_ws{WINDOW_SIZE}_ss{STRIDE_TRAIN}_pss{STRIDE_EVAL}"
DATA_DIR = ROOT / "data" / DATA_TAG



def ensure_raw_dir_and_move_mats():
    RAW_DIR.mkdir(exist_ok=True)
    moved = 0
    for ang in ANGLES:
        name = FILE_TPL.format(ang)
        src = ROOT / name
        dst = RAW_DIR / name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            moved += 1
        elif src.exists() and dst.exists():
            src.unlink()
    if moved:
        print(f"Moved {moved} .mat files into {RAW_DIR.name}/")


def tap_to_grid_index():
    idx = np.arange(N_TAPS)
    return np.stack([(idx % 20) // N_COLS, idx // 20, idx % N_COLS], axis=1)


def quality_check(cp_per_angle, angles, g_idx):
    print("\n--- Quality check (per-face mean Cp), no auto-fix ---")
    print(f"{'angle':>6} | {'face1(WW)':>10} {'face2(R)':>10} "
          f"{'face3(LW)':>10} {'face4(L)':>10}  flags")
    face_id = g_idx[:, 0]
    for ang, cp in zip(angles, cp_per_angle):
        means = [cp[:, face_id == f].mean() for f in range(N_FACES)]
        flags = []
        if ang == 0:
            if means[0] != max(means): flags.append("WW not max@0deg")
            if means[2] != min(means): flags.append("LW not min@0deg")
        if any(np.isnan(means)) or any(abs(m) > 5 for m in means):
            flags.append("Cp out of plausible range")
        print(f"{ang:>6} | {means[0]:>10.3f} {means[1]:>10.3f} "
              f"{means[2]:>10.3f} {means[3]:>10.3f}  {'; '.join(flags)}")
    print("--- end check ---\n")


def load_all_angles():
    n_angles = len(ANGLES)
    flat = np.zeros((N_TAPS, N_T_PER_ANGLE * n_angles), dtype=np.float32)
    grid = np.zeros((n_angles, N_T_PER_ANGLE, N_FACES, N_ROWS, N_COLS), dtype=np.float32)
    cp_per_angle = []
    locations, meta = None, {}
    g_idx = tap_to_grid_index()

    print(f"Loading {n_angles} angles ...")
    t0 = time.time()
    for i, ang in enumerate(ANGLES):
        fp = RAW_DIR / FILE_TPL.format(ang)
        if not fp.exists():
            raise FileNotFoundError(fp)
        d = sio.loadmat(str(fp))
        cp = d["Wind_pressure_coefficients"].astype(np.float32)
        if cp.shape != (N_T_PER_ANGLE, N_TAPS):
            raise ValueError(f"{fp.name}: got {cp.shape}, expected ({N_T_PER_ANGLE},{N_TAPS})")

        flat[:, i * N_T_PER_ANGLE:(i + 1) * N_T_PER_ANGLE] = cp.T
        for tap in range(N_TAPS):
            f, r, c = g_idx[tap]
            grid[i, :, f, r, c] = cp[:, tap]
        cp_per_angle.append(cp)

        if locations is None:
            locations = d["Location_of_measured_points"].astype(np.float32)
            meta = dict(
                B=float(d["Building_breadth"][0, 0]),
                D=float(d["Building_depth"][0, 0]),
                H=float(d["Building_height"][0, 0]),
                fs=int(d["Sample_frequency"][0, 0]),
                period=float(d["Sample_period"][0, 0]),
                Uh=float(str(d["Uh_AverageWindSpeed"][0])),
            )
        print(f"  [{i+1:2d}/{n_angles}] {fp.name}  angle={ang:3d} deg")

    print(f"Load done in {time.time()-t0:.1f}s.")
    return flat, grid, cp_per_angle, locations, meta, g_idx


def split_indices(n):
    n_train = int(round(n * TRAIN_RATIO))
    n_valid = int(round(n * VALID_RATIO))
    n_test = n - n_train - n_valid
    return n_train, n_valid, n_test


def make_windows(series, window_size, stride):
    """series: (T, F) -> (n_windows, window_size, F)."""
    T, F = series.shape
    if T < window_size:
        return np.empty((0, window_size, F), dtype=series.dtype)
    n = (T - window_size) // stride + 1
    out = np.zeros((n, window_size, F), dtype=series.dtype)
    for i in range(n):
        s = i * stride
        out[i] = series[s:s + window_size]
    return out


def main():
    ensure_raw_dir_and_move_mats()
    flat, grid, cp_per_angle, locations, meta, g_idx = load_all_angles()

    quality_check(cp_per_angle, ANGLES, g_idx)

    np.save(RAW_DIR / "all_Data_all_place.npy", flat)
    np.save(RAW_DIR / "cp_grid.npy", grid)
    np.savez(RAW_DIR / "metadata.npz",
             locations=locations,
             angles=np.array(ANGLES, dtype=np.int32),
             tap_to_grid=g_idx,
             n_t_per_angle=N_T_PER_ANGLE,
             **meta)
    print(f"Saved flat {flat.shape}, grid {grid.shape}, metadata.")

    full = flat.T.astype(np.float32)
    scaler = StandardScaler().fit(full)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _norm = np.stack([scaler.mean_, scaler.scale_], axis=1).astype(np.float32)
    np.save(DATA_DIR / "data_Norm_global.npy", _norm)
    np.save(RAW_DIR / "data_Norm_global.npy", _norm)
    print(f"Global scaler fit on {full.shape}; mean/std saved.")

    n_train, n_valid, n_test = split_indices(N_T_PER_ANGLE)
    print(f"Per-angle split: train={n_train}, valid={n_valid}, test={n_test} "
          f"(sum={n_train+n_valid+n_test}, expect {N_T_PER_ANGLE})")

    train_chunks, valid_chunks, test_chunks = [], [], []
    for i, ang in enumerate(ANGLES):
        cp = cp_per_angle[i]
        cp_norm = scaler.transform(cp).astype(np.float32)
        tr = cp_norm[:n_train]
        va = cp_norm[n_train:n_train + n_valid]
        te = cp_norm[n_train + n_valid:]

        train_chunks.append(make_windows(tr, WINDOW_SIZE, STRIDE_TRAIN))
        valid_chunks.append(make_windows(va, WINDOW_SIZE, STRIDE_EVAL))
        test_chunks.append(make_windows(te, WINDOW_SIZE, STRIDE_EVAL))

        print(f"  angle {ang:3d}: train {train_chunks[-1].shape} "
              f"valid {valid_chunks[-1].shape} test {test_chunks[-1].shape}")

    train = np.concatenate(train_chunks, axis=0)
    valid = np.concatenate(valid_chunks, axis=0)
    test = np.concatenate(test_chunks, axis=0)

    np.save(DATA_DIR / f"train_data_{DATA_TAG}.npy", train)
    np.save(DATA_DIR / f"valid_data_{DATA_TAG}.npy", valid)
    np.save(DATA_DIR / f"test_data_{DATA_TAG}.npy", test)
    print(f"\nFinal:")
    print(f"  train.npy {train.shape}  ({train.nbytes/1e6:.0f} MB)")
    print(f"  valid.npy {valid.shape}  ({valid.nbytes/1e6:.0f} MB)")
    print(f"  test.npy  {test.shape}  ({test.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
