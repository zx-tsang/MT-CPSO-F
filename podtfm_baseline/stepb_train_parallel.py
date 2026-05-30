"""
stepb_train_parallel.py — train POD-Transformer for multiple K values concurrently
on a single GPU.

Each K gets its own model + optimizer + GPU-resident dataset. Per epoch we
loop over Ks and do forward/backward sequentially, but with no CPU<->GPU
traffic between Ks, and the GPU stays busy nearly the whole time.

Outputs (per K, same layout as stepb_train.py):
    output_sensor/model/podtfm_p{K}_k{K}/checkpoint.pt
    output_sensor/DeFigs/podtfm_p{K}_k{K}/TrainLoss.png
    output_sensor/podtfm_p{K}_k{K}/eval/    (produced by stepc_evaluate.py)
"""
from __future__ import absolute_import, division, print_function

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import sys
ROOT = Path(__file__).resolve().parent
if not (ROOT / "params.yaml").exists() and (ROOT.parent / "params.yaml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from utils import EarlyStopping, Params
from stepb_train import PODTransformer, xavier_init

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


parser = argparse.ArgumentParser()
parser.add_argument("--params",   default=str(ROOT / "params.yaml"))
parser.add_argument("--ks",       type=int, nargs="+", required=True,
                    help="K values to train concurrently")
parser.add_argument("--raw-dir",  default=str(ROOT / "raw_data"))
parser.add_argument("--out-dir",  default=str(ROOT / "output_sensor"))
parser.add_argument("--lr",        type=float, default=None)
parser.add_argument("--epochs",    type=int,   default=None)
parser.add_argument("--batch-size", type=int,  default=None)


def load_split_gpu(data_dir, name, device):
    """Load a split (.npz) entirely onto GPU as fp32 tensors."""
    d = np.load(data_dir / name)
    X = torch.from_numpy(d["c_LF"].astype(np.float32)).to(device, non_blocking=True)
    Y = torch.from_numpy(d["c_HF"].astype(np.float32)).to(device, non_blocking=True)
    return X, Y


def run_one_K_epoch(model, X, Y, batch_size, optimizer, amp_dtype,
                    device, shuffle=True):
    """One epoch over (X, Y) GPU tensors. Returns mean MSE (fp32 reduction)."""
    train_mode = optimizer is not None
    model.train(train_mode)
    N = X.shape[0]
    idx = torch.randperm(N, device=device) if shuffle else torch.arange(N, device=device)
    total = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for s in range(0, N, batch_size):
            b = idx[s : s + batch_size]
            x = X.index_select(0, b)
            y = Y.index_select(0, b)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)
                pred = out["mean"]
            loss = torch.mean((pred.float() - y) ** 2)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item()
            n_batches += 1
    return total / max(n_batches, 1)


def main():
    args = parser.parse_args()
    params = Params(args.params)
    if args.lr is not None:        params.lr = args.lr
    if args.epochs is not None:    params.epochs = args.epochs
    if args.batch_size is not None: params.batch_size = args.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available()
                                   and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"device={device}  amp_dtype={amp_dtype}  KS={args.ks}")

    # ---- build per-K state -------------------------------------------------
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    bs_tr = int(params.batch_size)
    bs_ev = int(params.predict_batch)

    states = {}     # K -> dict of model/opt/dataset/early/etc.
    for K in args.ks:
        TAG = f"podtfm_p{K}_k{K}"
        data_dir = raw_dir / TAG
        if not (data_dir / "train.npz").exists():
            print(f"[K={K}] SKIP — no train.npz at {data_dir}")
            continue
        X_tr, Y_tr = load_split_gpu(data_dir, "train.npz", device)
        X_va, Y_va = load_split_gpu(data_dir, "valid.npz", device)
        X_te, Y_te = load_split_gpu(data_dir, "test.npz",  device)

        model = PODTransformer(
            in_dim     = K,
            out_dim    = K,
            d_model    = int(params.pod_d_model),
            n_head     = int(params.pod_n_head),
            ffn_hidden = int(params.pod_ffn_hidden),
            n_layers   = int(params.pod_n_layers),
            dropout    = float(params.pod_dropout),
            max_len    = int(params.pod_max_len),
        ).to(device)
        model.apply(xavier_init)
        n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr,
                                      betas=(0.9, 0.999), eps=1e-8,
                                      weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=6)
        early = EarlyStopping(patience=int(params.early_stop_patience))

        model_dir = out_dir / "model" / TAG
        plot_dir  = out_dir / "DeFigs" / TAG
        model_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        states[K] = dict(
            tag=TAG, model=model, opt=optimizer, sched=scheduler,
            early=early, X_tr=X_tr, Y_tr=Y_tr, X_va=X_va, Y_va=Y_va,
            X_te=X_te, Y_te=Y_te, model_dir=str(model_dir), plot_dir=plot_dir,
            train_hist=[], valid_hist=[], test_hist=[], stopped=False,
            best_va=float("inf"),
        )
        print(f"[K={K}] params={n_param:,}  train={X_tr.shape[0]}  "
              f"valid={X_va.shape[0]}  test={X_te.shape[0]}")

    if not states:
        print("nothing to train"); return

    EPOCHS = int(params.epochs)
    print(f"\n========== training {len(states)} models in parallel, "
          f"epochs<={EPOCHS} ==========\n")

    t_global = time.time()
    for epoch in range(EPOCHS):
        t0 = time.time()
        alive_lines = []
        for K, st in states.items():
            if st["stopped"]:
                continue
            tr = run_one_K_epoch(st["model"], st["X_tr"], st["Y_tr"],
                                 bs_tr, st["opt"], amp_dtype, device,
                                 shuffle=True)
            va = run_one_K_epoch(st["model"], st["X_va"], st["Y_va"],
                                 bs_ev, None,     amp_dtype, device,
                                 shuffle=False)
            te = run_one_K_epoch(st["model"], st["X_te"], st["Y_te"],
                                 bs_ev, None,     amp_dtype, device,
                                 shuffle=False)
            st["train_hist"].append(tr)
            st["valid_hist"].append(va)
            st["test_hist"].append(te)
            st["sched"].step(va)
            st["early"](va, st["model"].state_dict(), st["model_dir"],
                        save_name="checkpoint")
            if va < st["best_va"]:
                st["best_va"] = va
            if st["early"].early_stop:
                st["stopped"] = True
                alive_lines.append(f"K={K}: STOPPED (best_va={st['best_va']:.5f})")
            else:
                alive_lines.append(f"K={K} tr={tr:.5f} va={va:.5f}")

        n_alive = sum(1 for s in states.values() if not s["stopped"])
        dt = time.time() - t0
        print(f"ep {epoch+1:3d}/{EPOCHS}  alive={n_alive}/{len(states)}  "
              f"({dt:.1f}s)  | " + "  ".join(alive_lines))

        if n_alive == 0:
            print(f"\nall K stopped at epoch {epoch+1}")
            break

    print(f"\n========== DONE in {(time.time()-t_global)/60:.1f} min ==========")

    # ---- final per-K summary + loss plots ----------------------------------
    for K, st in states.items():
        plot_dir = st["plot_dir"]
        for hist, name in [(st["train_hist"], "TrainLoss"),
                           (st["valid_hist"], "ValidLoss"),
                           (st["test_hist"],  "TestLoss")]:
            if not hist: continue
            plt.figure()
            plt.plot(range(1, len(hist)+1), hist)
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"{name} K={K}")
            plt.grid(True)
            plt.savefig(plot_dir / f"{name}.png")
            plt.close()
        print(f"[K={K}]  best_valid={st['best_va']:.6f}  "
              f"ckpt -> {st['model_dir']}/checkpoint.pt")


if __name__ == "__main__":
    main()
