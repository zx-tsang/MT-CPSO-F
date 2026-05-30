"""
Multi-seed fine-tuning of the pretrained mask-Transformer using fixed
sensor subsets produced by f_steprc_run_pso_multi_seed.py.

For each seed N:
    - reads  <ms-out>/seed_<N>/optimized_sensor_indices.txt
    - loads  <model-name>/model/mrm_awa_Transformer_add_sensor_mae/checkpoint.pt
      (the original pretrained checkpoint, NOT re-trained from scratch)
    - masks the *unselected* sensors over the whole window (fixed mask,
      no random drop ratio), computes MSE only on the masked positions
    - saves the fine-tuned checkpoint to
          <ms-out>/seed_<N>/finetune/model/checkpoint.pt
      with loss curves under
          <ms-out>/seed_<N>/finetune/DeFigs/

Examples
--------
# Sequential (default, fine-tune all seeds one at a time)
python f_steprd.py

# Parallel: 3 seeds at a time on the same GPU (each child holds its own
# copy of the model -- VRAM ~3x).
python f_steprd.py --max-workers 3

# Spread 4 seeds across 2 GPUs
python f_steprd.py --seeds 42 123 456 789 --max-workers 2 --gpu-ids 0 1

# Run just one seed in this process (no subprocess)
python f_steprd.py --worker --seed 42

Design notes
------------
Borrows from steprd_finetune.py the parts the existing project scripts
don't have:
    * loads pretrained ckpt and fine-tunes from there
    * deterministic fixed mask (unlike f_pso_a_fix_Transformer_100-time.py
      which uses random drop ratios)
    * subprocess-per-seed launcher with GPU round-robin
    * AMP (autocast + GradScaler)
    * pin_memory + cudnn.benchmark
    * tracks both normalized loss and de-normalized (real-scale) loss
    * per-run finetune.log + finetune_summary.txt
"""
from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# AMP shim: torch.amp.* is PyTorch >= 2.0; older versions only have
# torch.cuda.amp.*. Pick the right pair at import time.
# ---------------------------------------------------------------------------
if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    def _make_autocast(enabled, dtype):
        return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)
    def _make_scaler(enabled):
        return torch.amp.GradScaler("cuda", enabled=enabled)
else:
    def _make_autocast(enabled, dtype):
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
    def _make_scaler(enabled):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ===========================================================================
# CLI
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser()
    # data / model -- defaults match f_steprc_run_pso_multi_seed.py
    p.add_argument("--dataset", default="awa_sensor_random_mask_all_place")
    p.add_argument("--data-folder", default="data")
    p.add_argument("--model-name", default="output_sensor")
    p.add_argument("--data-root-prefix", default="",
                   help="Optional path prefix prepended to <data-folder>/<dataset> "
                        "(useful if datasets live outside the repo root). Empty by default.")
    p.add_argument("--ckpt", default=None,
                   help="Override pretrained checkpoint path. "
                        "Default: <model-name>/model/"
                        "mrm_awa_Transformer_add_sensor_mae/checkpoint.pt")
    p.add_argument("--norm-file", default=None,
                   help="Override path to data_Norm_global.npy. "
                        "Default: data/<dataset>_global_standard_deviation/"
                        "data_Norm_global.npy")
    p.add_argument("--params-json", default=None,
                   help="Override path to params json. Default: "
                        "<model-name>/params_Transformer_all_place.json")
    # multi-seed input
    p.add_argument("--ms-out", default="f_pso_multiseed_select100",
                   help="Top-level dir produced by f_steprc_run_pso_multi_seed.")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 2024],
                   help="List of seeds to fine-tune. One job per seed.")
    p.add_argument("--total-sensors", type=int, default=1014)
    # training overrides
    p.add_argument("--epochs", type=int, default=None,
                   help="Override params.epochs for fine-tuning.")
    # Default fine-tune lr is 1e-5 (small, since we're fine-tuning a
    # pretrained model with a fixed mask). Override with --lr 1e-4 (or
    # any value) if you want the original behavior.
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Learning rate (default 1e-5 for fine-tuning).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--predict-batch", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--scheduler-patience", type=int, default=5)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable AMP (fp16) -- runs in fp32.")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader num_workers per job.")
    # launcher control
    p.add_argument("--max-workers", type=int, default=1,
                   help="Concurrent fine-tunes (1 = strictly sequential).")
    p.add_argument("--gpu-ids", type=int, nargs="*", default=None,
                   help="Optional list of GPU ids for CUDA_VISIBLE_DEVICES "
                        "round-robin across workers.")
    # internal worker-mode args
    p.add_argument("--worker", action="store_true",
                   help="(internal) run as a single-seed worker.")
    p.add_argument("--seed", type=int, default=None,
                   help="(worker mode) the seed to fine-tune.")
    return p


# ===========================================================================
# Utilities
# ===========================================================================
class Params:
    def __init__(self, json_path):
        with open(json_path) as f:
            self.__dict__.update(json.load(f))


def read_selected_indices(path):
    """Read 0-based sensor indices, one per line."""
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def fixed_mask(batch, unselected_idx):
    """Whole-channel mask: keep only sensors NOT in unselected_idx."""
    mask = torch.ones_like(batch)
    mask[:, :, unselected_idx] = 0
    return batch * mask, mask


def plot_loss(history, out_folder, name):
    out_folder.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(name)
    plt.grid(True)
    plt.savefig(out_folder / f"{name}.png")
    plt.close()


class EarlyStop:
    """Minimal early stopper — keeps best val loss and saves checkpoint."""
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best = float("inf")
        self.stop = False

    def __call__(self, val_loss, state_dict, save_dir, save_name="checkpoint"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            torch.save(state_dict, save_dir / f"{save_name}.pt")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def _force_dropout_eval(model):
    """Disable dropout for eval (mirrors f_steprc_CPSO)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()


# ===========================================================================
# Train / eval one epoch (AMP-aware, fixed-mask masked-MSE)
# ===========================================================================
def run_epoch(model, loader, device, unselected, optimizer=None,
              desc="", scale=None, scaler=None, use_amp=True,
              amp_dtype=torch.float16):
    """
    Returns (mse_norm, mse_denorm, mae_norm, mae_denorm).

    - mse_norm  : MSE in normalized space, masked to unselected positions.
                  This is what backprop optimizes.
    - mse_denorm: MSE after multiplying residuals by per-sensor std (real
                  physical units). Only computed if `scale` is given.
    - mae_norm  : MAE in normalized space (reporting only, NOT used for
                  backprop). Directly comparable to CPSO's fitness.
    - mae_denorm: MAE in real physical units. Only if `scale` is given.

    AMP path: forward + loss in `amp_dtype`; backward uses fp32 master
    grads via `scaler`. No-op on CPU.
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    if not train_mode:
        _force_dropout_eval(model)

    total_mse_n, total_mse_d = 0.0, 0.0
    total_mae_n, total_mae_d = 0.0, 0.0
    n = 0
    amp_on = use_amp and (device.type == "cuda")
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch.to(torch.float32).to(device, non_blocking=True)
            labels = batch.clone()
            masked, mask = fixed_mask(batch, unselected)
            x_in = torch.cat([masked, mask], dim=-1)
            with _make_autocast(amp_on, amp_dtype):
                results = model(x_in)
                mask_w = 1 - mask
                resid = results["mean"].float() - labels
                loss_n = (resid.pow(2) * mask_w).sum() / (mask_w.sum() + 1e-8)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_on:
                    scaler.scale(loss_n).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_n.backward()
                    optimizer.step()
            total_mse_n += loss_n.item()
            # MAE tracked for reporting only (does NOT affect backprop)
            with torch.no_grad():
                denom = mask_w.sum() + 1e-8
                mae_n = (resid.detach().abs() * mask_w).sum() / denom
                total_mae_n += mae_n.item()
                if scale is not None:
                    resid_d = resid.detach() * scale
                    mse_d = (resid_d.pow(2) * mask_w).sum() / denom
                    mae_d = (resid_d.abs() * mask_w).sum() / denom
                    total_mse_d += mse_d.item()
                    total_mae_d += mae_d.item()
            n += 1
    avg_mse_n = total_mse_n / max(n, 1)
    avg_mse_d = total_mse_d / max(n, 1) if scale is not None else float("nan")
    avg_mae_n = total_mae_n / max(n, 1)
    avg_mae_d = total_mae_d / max(n, 1) if scale is not None else float("nan")
    return avg_mse_n, avg_mse_d, avg_mae_n, avg_mae_d


# ===========================================================================
# Worker: fine-tune one seed
# ===========================================================================
def finetune_one_seed(seed, args):
    # ---- resolve paths ---------------------------------------------------
    model_dir = Path(args.model_name)
    params_json = Path(args.params_json) if args.params_json else \
        model_dir / "params_Transformer_all_place.json"
    assert params_json.is_file(), f"missing params json: {params_json}"
    params = Params(params_json)
    # overrides
    if args.epochs is not None:
        params.epochs = args.epochs
    if args.batch_size is not None:
        params.batch_size = args.batch_size
    if args.predict_batch is not None:
        params.predict_batch = args.predict_batch
    lr = args.lr if args.lr is not None else getattr(params, "lr", 1e-4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ms_out = Path(args.ms_out)
    seed_dir = ms_out / f"seed_{seed}"
    idx_file = seed_dir / "optimized_sensor_indices.txt"
    if not idx_file.exists():
        raise FileNotFoundError(f"missing PSO indices for seed {seed}: {idx_file}")

    ft_dir = seed_dir / "finetune"
    ckpt_out_dir = ft_dir / "model"
    plot_dir = ft_dir / "DeFigs"
    ckpt_out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- sensor selection -> unselected ----------------------------------
    selected = read_selected_indices(idx_file)
    total = int(args.total_sensors)
    unselected = sorted(set(range(total)) - set(selected))
    print(f"[seed={seed}] selected={len(selected)} unselected={len(unselected)} "
          f"total={total}")

    # ---- denorm scale (std per sensor) -----------------------------------
    norm_path = Path(args.norm_file) if args.norm_file else \
        Path("data") / f"{args.dataset}_global_standard_deviation" / "data_Norm_global.npy"
    norm = np.load(norm_path)
    scale = torch.from_numpy(norm[:, 1].astype(np.float32)).to(device)
    # scale shape: (total_sensors,); broadcast over (B, T, C) for residuals
    scale = scale.view(1, 1, -1)

    # ---- data ------------------------------------------------------------
    # Local import: keeps this script importable from machines without
    # the project's dataloader module on the path.
    from dataloader import (
        TrainDataset_X_and_label,
        ValidDataset_X_and_label,
        TestDataset_X_and_label,
    )
    data_dir = os.path.join(args.data_root_prefix, args.data_folder, args.dataset)
    pin = torch.cuda.is_available()
    train_set = TrainDataset_X_and_label(data_dir, args.dataset)
    valid_set = ValidDataset_X_and_label(data_dir, args.dataset)
    test_set = TestDataset_X_and_label(data_dir, args.dataset)
    train_loader = DataLoader(train_set, batch_size=params.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin)
    valid_loader = DataLoader(valid_set, batch_size=params.predict_batch,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch,
                             shuffle=False, num_workers=1, pin_memory=pin)

    # ---- model + pretrained ckpt -----------------------------------------
    params.device = device
    params.model_dir = str(model_dir)
    params.dataset = args.dataset
    from network.Transformer import Transformer
    model = Transformer(params).to(device)

    ckpt_path = Path(args.ckpt) if args.ckpt else \
        model_dir / "model" / "mrm_awa_Transformer_add_sensor_mae" / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"pretrained checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[seed={seed}] loaded pretrained ckpt: {ckpt_path}")

    # ---- optim, scheduler, AMP -------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=args.scheduler_patience)
    early = EarlyStop(patience=args.early_stop_patience)
    use_amp = (not args.no_amp) and (device.type == "cuda")
    scaler = _make_scaler(use_amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- train loop ------------------------------------------------------
    # MSE histories (norm/denorm) -- MSE is what backprop optimizes
    tr_mse_n_hist, tr_mse_d_hist = [], []
    va_mse_n_hist, va_mse_d_hist = [], []
    te_mse_n_hist, te_mse_d_hist = [], []
    # MAE histories (norm/denorm) -- reporting only, comparable to CPSO MAE
    tr_mae_n_hist, tr_mae_d_hist = [], []
    va_mae_n_hist, va_mae_d_hist = [], []
    te_mae_n_hist, te_mae_d_hist = [], []
    t0 = time.time()
    for epoch in range(params.epochs):
        print(f"[seed={seed}] epoch {epoch+1}/{params.epochs} "
              f"(lr={optimizer.param_groups[0]['lr']:.2e})")
        tr_mse_n, tr_mse_d, tr_mae_n, tr_mae_d = run_epoch(
            model, train_loader, device, unselected,
            optimizer, desc="train", scale=scale,
            scaler=scaler, use_amp=use_amp)
        va_mse_n, va_mse_d, va_mae_n, va_mae_d = run_epoch(
            model, valid_loader, device, unselected,
            None, desc="valid", scale=scale, use_amp=use_amp)
        te_mse_n, te_mse_d, te_mae_n, te_mae_d = run_epoch(
            model, test_loader, device, unselected,
            None, desc="test", scale=scale, use_amp=use_amp)
        print(f"  MSE  train={tr_mse_n:.5f}/{tr_mse_d:.5f}  "
              f"valid={va_mse_n:.5f}/{va_mse_d:.5f}  "
              f"test={te_mse_n:.5f}/{te_mse_d:.5f}  (norm/denorm)")
        print(f"  MAE  train={tr_mae_n:.5f}/{tr_mae_d:.5f}  "
              f"valid={va_mae_n:.5f}/{va_mae_d:.5f}  "
              f"test={te_mae_n:.5f}/{te_mae_d:.5f}  (norm/denorm)")

        tr_mse_n_hist.append(tr_mse_n); tr_mse_d_hist.append(tr_mse_d)
        va_mse_n_hist.append(va_mse_n); va_mse_d_hist.append(va_mse_d)
        te_mse_n_hist.append(te_mse_n); te_mse_d_hist.append(te_mse_d)
        tr_mae_n_hist.append(tr_mae_n); tr_mae_d_hist.append(tr_mae_d)
        va_mae_n_hist.append(va_mae_n); va_mae_d_hist.append(va_mae_d)
        te_mae_n_hist.append(te_mae_n); te_mae_d_hist.append(te_mae_d)
        plot_loss(tr_mse_n_hist, plot_dir, "TrainMSE_norm")
        plot_loss(tr_mse_d_hist, plot_dir, "TrainMSE_denorm")
        plot_loss(va_mse_n_hist, plot_dir, "ValidMSE_norm")
        plot_loss(va_mse_d_hist, plot_dir, "ValidMSE_denorm")
        plot_loss(te_mse_n_hist, plot_dir, "TestMSE_norm")
        plot_loss(te_mse_d_hist, plot_dir, "TestMSE_denorm")
        plot_loss(tr_mae_n_hist, plot_dir, "TrainMAE_norm")
        plot_loss(tr_mae_d_hist, plot_dir, "TrainMAE_denorm")
        plot_loss(va_mae_n_hist, plot_dir, "ValidMAE_norm")
        plot_loss(va_mae_d_hist, plot_dir, "ValidMAE_denorm")
        plot_loss(te_mae_n_hist, plot_dir, "TestMAE_norm")
        plot_loss(te_mae_d_hist, plot_dir, "TestMAE_denorm")

        scheduler.step(va_mse_n)
        early(va_mse_n, model.state_dict(), str(ckpt_out_dir),
              save_name="checkpoint")
        if early.stop:
            print(f"[seed={seed}] early stop at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    with open(ft_dir / "finetune_summary.txt", "w") as f:
        f.write(f"seed                  : {seed}\n")
        f.write(f"pretrained_ckpt       : {ckpt_path}\n")
        f.write(f"selected_count        : {len(selected)}\n")
        f.write(f"total_sensors         : {total}\n")
        f.write(f"epochs_run            : {len(tr_mse_n_hist)}\n")
        # MSE (training objective)
        f.write(f"best_valid_MSE_norm   : {min(va_mse_n_hist):.6f}\n")
        f.write(f"final_test_MSE_norm   : {te_mse_n_hist[-1]:.6f}\n")
        # MAE (comparable to CPSO fitness)
        f.write(f"best_valid_MAE_norm   : {min(va_mae_n_hist):.6f}\n")
        f.write(f"final_test_MAE_norm   : {te_mae_n_hist[-1]:.6f}\n")
        if not np.isnan(va_mse_d_hist[-1]):
            f.write(f"best_valid_MSE_denorm : {min(va_mse_d_hist):.6f}\n")
            f.write(f"final_test_MSE_denorm : {te_mse_d_hist[-1]:.6f}\n")
            f.write(f"best_valid_MAE_denorm : {min(va_mae_d_hist):.6f}\n")
            f.write(f"final_test_MAE_denorm : {te_mae_d_hist[-1]:.6f}\n")
        f.write(f"elapsed_sec           : {elapsed:.1f}\n")
        f.write(f"amp                   : {use_amp}\n")
        f.write(f"lr                    : {lr}\n")
        f.write(f"batch_size            : {params.batch_size}\n")
    print(f"[seed={seed}] done in {elapsed:.0f}s -> {ckpt_out_dir}/checkpoint.pt")


# ===========================================================================
# Launcher: subprocess per seed (each gets its own CUDA context)
# ===========================================================================
def launch_one(seed, args, gpu_id):
    seed_dir = Path(args.ms_out) / f"seed_{seed}" / "finetune"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_path = seed_dir / "finetune.log"

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
           "--worker", "--seed", str(seed),
           "--dataset", args.dataset,
           "--data-folder", args.data_folder,
           "--model-name", args.model_name,
           "--data-root-prefix", args.data_root_prefix,
           "--ms-out", args.ms_out,
           "--total-sensors", str(args.total_sensors),
           "--early-stop-patience", str(args.early_stop_patience),
           "--scheduler-patience", str(args.scheduler_patience),
           "--num-workers", str(args.num_workers)]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    if args.norm_file:
        cmd += ["--norm-file", args.norm_file]
    if args.params_json:
        cmd += ["--params-json", args.params_json]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.predict_batch is not None:
        cmd += ["--predict-batch", str(args.predict_batch)]
    if args.no_amp:
        cmd += ["--no-amp"]

    t0 = time.time()
    print(f"[start] seed={seed} "
          f"gpu={gpu_id if gpu_id is not None else 'inherit'} -> {log_path}")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            env=env).returncode
    return seed, rc, time.time() - t0, log_path


def aggregate(args, ok_seeds):
    """Write a small aggregate summary across seeds, mirroring the multi-seed
    PSO aggregator so downstream analysis can pair them up."""
    rows = []
    for s in ok_seeds:
        summ = Path(args.ms_out) / f"seed_{s}" / "finetune" / "finetune_summary.txt"
        if not summ.exists():
            continue
        kv = {}
        for line in summ.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip()] = v.strip()
        rows.append((s, kv))

    if not rows:
        return

    # Keys collected for the aggregate. We include BOTH norm and denorm
    # so the summary mirrors what CPSO reports (CPSO's `final_MAE` is
    # actually denorm). The per-seed `finetune_summary.txt` already
    # stores all of these.
    NORM_KEYS = [
        "best_valid_MSE_norm", "best_valid_MAE_norm",
        "final_test_MSE_norm", "final_test_MAE_norm",
    ]
    DENORM_KEYS = [
        "best_valid_MSE_denorm", "best_valid_MAE_denorm",
        "final_test_MSE_denorm", "final_test_MAE_denorm",
    ]

    def _collect(key):
        vs = []
        for _, kv in rows:
            try:
                vs.append(float(kv.get(key, "nan")))
            except ValueError:
                pass
        return np.array([v for v in vs if not np.isnan(v)], dtype=float)

    has_denorm = _collect("final_test_MAE_denorm").size > 0

    out_path = Path(args.ms_out) / "finetune_aggregate_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Multi-seed fine-tune aggregate summary\n")
        f.write("=" * 78 + "\n")
        f.write(f"dataset      : {args.dataset}\n")
        f.write(f"ms_out       : {args.ms_out}\n")
        f.write(f"seeds        : {[s for s, _ in rows]}\n")
        f.write(f"has_denorm   : {has_denorm}\n")
        f.write("-" * 78 + "\n")
        # Per-seed tables: one for NORM, one for DENORM. Each shows
        # every seed's MSE + MAE on valid and test, so the denorm
        # metrics for any single seed are readable at a glance.
        def _per_seed_table(title, key_v_mse, key_v_mae, key_t_mse, key_t_mae):
            f.write(f"\n{title}\n")
            f.write("-" * 78 + "\n")
            f.write(f"{'seed':>8} | "
                    f"{'valid_MSE':>12} {'valid_MAE':>12} | "
                    f"{'test_MSE':>12} {'test_MAE':>12} | "
                    f"{'elapsed_s':>10}\n")
            for s, kv in rows:
                f.write(f"{s:>8d} | "
                        f"{kv.get(key_v_mse,'?'):>12} "
                        f"{kv.get(key_v_mae,'?'):>12} | "
                        f"{kv.get(key_t_mse,'?'):>12} "
                        f"{kv.get(key_t_mae,'?'):>12} | "
                        f"{kv.get('elapsed_sec','?'):>10}\n")

        _per_seed_table(
            "Per-seed metrics  [NORM space]",
            "best_valid_MSE_norm", "best_valid_MAE_norm",
            "final_test_MSE_norm", "final_test_MAE_norm",
        )
        if has_denorm:
            _per_seed_table(
                "Per-seed metrics  [DENORM / Cp space, physical units]",
                "best_valid_MSE_denorm", "best_valid_MAE_denorm",
                "final_test_MSE_denorm", "final_test_MAE_denorm",
            )

        def _emit_block(title, keys):
            f.write("\n" + title + "\n")
            f.write("-" * 78 + "\n")
            for key in keys:
                vals = _collect(key)
                if vals.size == 0:
                    continue
                # std(n) matches numpy default / Excel STDEV.P
                # std(n-1) matches sample std / Excel STDEV.S
                std_s = vals.std(ddof=1) if vals.size > 1 else float("nan")
                f.write(
                    f"{key:<28}  mean={vals.mean():.6f}  "
                    f"std(n)={vals.std(ddof=0):.6f}  "
                    f"std(n-1)={std_s:.6f}  "
                    f"min={vals.min():.6f}  max={vals.max():.6f}  "
                    f"(n={vals.size})\n"
                )

        _emit_block("Statistics across seeds  [NORM space]", NORM_KEYS)
        if has_denorm:
            _emit_block("Statistics across seeds  [DENORM / Cp space]",
                        DENORM_KEYS)
        _emit_block("Statistics across seeds  [misc]", ["elapsed_sec"])

        f.write("\n" + "=" * 78 + "\n")
        f.write("std(n)   = population std,  matches Excel STDEV.P / "
                "numpy default std(ddof=0)\n")
        f.write("std(n-1) = sample std,      matches Excel STDEV.S / "
                "numpy std(ddof=1)\n")
        f.write("=" * 78 + "\n")
    print(f"[aggregate] -> {out_path}")


def main():
    args = build_parser().parse_args()

    # ---- worker mode: single-seed, no subprocess -----------------------
    if args.worker:
        if args.seed is None:
            sys.exit("--worker requires --seed")
        finetune_one_seed(args.seed, args)
        return

    # ---- launcher mode -------------------------------------------------
    seeds = list(dict.fromkeys(args.seeds))
    workers = max(1, args.max_workers)
    print(f"fine-tuning {len(seeds)} seed(s) with up to {workers} in "
          f"parallel: {seeds}")
    if workers > 1:
        print(f"WARNING: each worker loads its own model on the GPU; "
              f"VRAM ~{workers}x single-run.")

    t_global = time.time()
    failed, ok = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for i, s in enumerate(seeds):
            gpu = (args.gpu_ids[i % len(args.gpu_ids)]
                   if args.gpu_ids else None)
            futures.append(pool.submit(launch_one, s, args, gpu))
        for fut in as_completed(futures):
            s, rc, dt, log = fut.result()
            tag = "ok" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[done ] seed={s}  {tag}  {dt:.0f}s  log={log}")
            (ok if rc == 0 else failed).append(s)

    aggregate(args, ok)
    print(f"\n[all done] {len(ok)} ok / {len(failed)} failed "
          f"in {time.time()-t_global:.1f}s")
    if failed:
        sys.exit(f"failed seeds: {failed}")


if __name__ == "__main__":
    main()
