"""
stepb_train.py — POD-Transformer training (single K).

Pairs with:
    stepa_preprocess.py    (produces raw_data/podtfm_p{K}_k{K}/)
    stepc_evaluate.py      (evaluation + 500-d Cp MAE)

Inputs
------
    raw_data/podtfm_p{K}_k{K}/train.npz    keys: c_LF, c_HF, angle
    raw_data/podtfm_p{K}_k{K}/valid.npz
    raw_data/podtfm_p{K}_k{K}/test.npz
    raw_data/podtfm_p{K}_k{K}/U_HF_k.npy   (only used by stepc)
    raw_data/podtfm_p{K}_k{K}/mu.npy       (only used by stepc)
    params.yaml                            (pod_* hyperparameters)

Training objective
------------------
    Single Transformer encoder maps  c_LF (B, W, K)  ->  c_HF (B, W, K)
    using MSE loss in POD-coefficient space. No masking, no angle
    conditioning, all 11 wind directions trained jointly.

Outputs
-------
    output_sensor/model/<tag>/checkpoint.pt
    output_sensor/DeFigs/<tag>/{TrainLoss,ValidLoss,TestLoss}.png
    output_sensor/DeFigs/<tag>/pred_{train,valid,test}.png
"""
from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import sys
ROOT = Path(__file__).resolve().parent          # baselines/podtfm/ (params.yaml + utils.py here)
PROJECT_ROOT = ROOT.parent.parent                # repository root (raw_data here)
sys.path.insert(0, str(ROOT))      # so `from utils import ...` works

from utils import EarlyStopping, Params

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_TAG = "baseline_nav_podtfm"

parser = argparse.ArgumentParser()
parser.add_argument("--params",   default=str(ROOT / "params.yaml"))
parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "raw_data" / "podtfm_p6_k6"))
parser.add_argument("--out-dir",  default=str(ROOT / "output_sensor"))
parser.add_argument("--tag",      default=DEFAULT_TAG,
                    help="output subdir under output_sensor/{model,DeFigs}/; "
                         "use a different tag to avoid overwriting prior runs")
parser.add_argument("--lr",        type=float, default=None,
                    help="override params.lr")
parser.add_argument("--epochs",    type=int,   default=None,
                    help="override params.epochs")
parser.add_argument("--batch-size", type=int,  default=None,
                    help="override params.batch_size")
parser.add_argument("--k",          type=int,  default=None,
                    help="override pod_input_dim AND pod_output_dim (= POD modes)")


# =======================================================================
# Dataset
# =======================================================================
class PODCoeffDataset(Dataset):
    """Loads c_LF / c_HF coefficient windows from a .npz file."""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.x = d["c_LF"].astype(np.float32)        # (n, W, k)
        self.y = d["c_HF"].astype(np.float32)        # (n, W, k)
        self.angle = d["angle"].astype(np.int32)     # (n,) — for diagnostics only
        assert self.x.shape == self.y.shape, (self.x.shape, self.y.shape)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])


# =======================================================================
# Model
# =======================================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):                            # x: (B, W, d_model)
        return x + self.pe[:, : x.size(1), :]


class PODTransformer(nn.Module):
    """Encoder-only Transformer for k-dim coefficient sequences.

    Input  : (B, W, in_dim)
    Output : {'mean': (B, W, out_dim)}
    """

    def __init__(self, in_dim, out_dim, d_model, n_head, ffn_hidden,
                 n_layers, dropout, max_len):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc    = SinusoidalPositionalEncoding(d_model, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=ffn_hidden, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x):
        h = self.pos_enc(self.input_proj(x))
        h = self.encoder(h)
        return {"mean": self.head(h)}


# =======================================================================
# Plotting helpers
# =======================================================================
def plot_loss_curve(losses, out_folder, name):
    out_folder.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(name)
    plt.grid(True)
    plt.savefig(out_folder / f"{name}.png")
    plt.close()


def plot_prediction(preds, labels, out_folder, tag):
    """Single random window, single random POD mode."""
    out_folder.mkdir(parents=True, exist_ok=True)
    B, W, K = preds.shape
    b = random.randint(0, B - 1)
    k = random.randint(0, K - 1)
    pr = preds[b, :, k].detach().cpu().numpy()
    lb = labels[b, :, k].detach().cpu().numpy()
    r = float(np.corrcoef(pr, lb)[0, 1]) if W > 1 else 0.0
    plt.figure(figsize=(12, 4))
    plt.plot(lb, label="Label",   color="#377eb8", linewidth=1.4)
    plt.plot(pr, label="Pred",    color="#e41a1c", linewidth=1.4,
             linestyle=(0, (5, 2)))
    plt.title(f"mode={k}  window={b}  R={r:.3f}")
    plt.xlabel("time step")
    plt.ylabel("POD coefficient")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_folder / f"pred_{tag}.png", dpi=150)
    plt.close()


# =======================================================================
# Train / eval epoch — GPU-resident tensors, bf16 autocast, batch slicing
# =======================================================================
def run_epoch_gpu(model, X, Y, batch_size, device, optimizer=None,
                  amp_dtype=torch.bfloat16, shuffle=False,
                  plot_dir=None, tag=""):
    """Run one epoch using GPU-resident tensors X, Y of shape (N, W, k).

    Loss is computed in fp32 (mean of (pred-y)^2) for numerical accuracy.
    Forward pass uses bf16 autocast for speed.
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    N = X.shape[0]
    idx = torch.randperm(N, device=device) if shuffle else torch.arange(N, device=device)
    total = 0.0
    n_batches = 0
    plotted = False
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for s in range(0, N, batch_size):
            b = idx[s : s + batch_size]
            x = X.index_select(0, b)
            y = Y.index_select(0, b)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)
                pred = out["mean"]
            # cast back to fp32 for the loss reduction
            loss = torch.mean((pred.float() - y) ** 2)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item()
            n_batches += 1
            if plot_dir is not None and not plotted:
                plot_prediction(pred.float(), y, plot_dir, tag)
                plotted = True
    return total / max(n_batches, 1)


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =======================================================================
# Main
# =======================================================================
def main():
    args = parser.parse_args()
    params = Params(args.params)
    if args.lr is not None:        params.lr = args.lr
    if args.epochs is not None:    params.epochs = args.epochs
    if args.batch_size is not None: params.batch_size = args.batch_size
    if args.k is not None:
        params.pod_input_dim  = args.k
        params.pod_output_dim = args.k
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir   = Path(args.out_dir)
    plot_dir  = out_dir / "DeFigs" / args.tag
    model_dir = out_dir / "model"  / args.tag
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_ds = PODCoeffDataset(data_dir / "train.npz")
    valid_ds = PODCoeffDataset(data_dir / "valid.npz")
    test_ds  = PODCoeffDataset(data_dir / "test.npz")
    print(f"train {len(train_ds)}  valid {len(valid_ds)}  test {len(test_ds)}")
    print(f"input shape  per window: {train_ds.x.shape[1:]}  "
          f"target shape: {train_ds.y.shape[1:]}")

    # Move whole splits to GPU once (datasets are small: K=20, W=50 -> ~120 MB)
    X_train = torch.from_numpy(train_ds.x).to(device, non_blocking=True)
    Y_train = torch.from_numpy(train_ds.y).to(device, non_blocking=True)
    X_valid = torch.from_numpy(valid_ds.x).to(device, non_blocking=True)
    Y_valid = torch.from_numpy(valid_ds.y).to(device, non_blocking=True)
    X_test  = torch.from_numpy(test_ds.x ).to(device, non_blocking=True)
    Y_test  = torch.from_numpy(test_ds.y ).to(device, non_blocking=True)
    bs_tr = int(params.batch_size)
    bs_ev = int(params.predict_batch)
    n_tr_batches = (X_train.shape[0] + bs_tr - 1) // bs_tr
    print(f"GPU-resident  train_batches={n_tr_batches}  bs_tr={bs_tr}  bs_ev={bs_ev}")
    # bf16 if supported (Ampere+), else fp16 (older GPU); cpu falls back to fp32
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available()
                                   and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"AMP dtype: {amp_dtype}")

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
    model.apply(xavier_init)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model parameters: {n_param:,}  "
          f"d_model={params.pod_d_model}  n_head={params.pod_n_head}  "
          f"n_layers={params.pod_n_layers}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6)
    early = EarlyStopping(patience=params.early_stop_patience)

    import time
    train_hist, valid_hist, test_hist = [], [], []
    for epoch in range(params.epochs):
        t0 = time.time()
        tr = run_epoch_gpu(model, X_train, Y_train, bs_tr, device,
                           optimizer=optimizer, amp_dtype=amp_dtype,
                           shuffle=True, plot_dir=plot_dir, tag="train")
        va = run_epoch_gpu(model, X_valid, Y_valid, bs_ev, device,
                           optimizer=None,      amp_dtype=amp_dtype,
                           shuffle=False, plot_dir=plot_dir, tag="valid")
        te = run_epoch_gpu(model, X_test,  Y_test,  bs_ev, device,
                           optimizer=None,      amp_dtype=amp_dtype,
                           shuffle=False, plot_dir=plot_dir, tag="test")
        dt = time.time() - t0
        print(f"epoch {epoch+1:3d}/{params.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"train={tr:.5f}  valid={va:.5f}  test={te:.5f}  ({dt:.2f}s)")

        train_hist.append(tr); valid_hist.append(va); test_hist.append(te)
        plot_loss_curve(train_hist, plot_dir, "TrainLoss")
        plot_loss_curve(valid_hist, plot_dir, "ValidLoss")
        plot_loss_curve(test_hist,  plot_dir, "TestLoss")

        scheduler.step(va)
        early(va, model.state_dict(), str(model_dir), save_name="checkpoint")
        if early.early_stop:
            print("Early stopping.")
            break

    print(f"\nbest valid MSE: {min(valid_hist):.6f}")
    print(f"checkpoint -> {model_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
