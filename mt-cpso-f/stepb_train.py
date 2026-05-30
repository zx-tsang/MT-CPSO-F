#################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Explicit imports (was "from dataloader import *" before)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from dataloader import (
    TrainDataset_X_and_label,
    ValidDataset_X_and_label,
    TestDataset_X_and_label,
    Params,           # merged from utils.py
    EarlyStopping,    # merged from utils.py
)
import json
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="awa_sensor_random_mask_all_place", help="Name of the dataset")
parser.add_argument("--data-folder", default="data", help="Parent dir of the dataset")
parser.add_argument("--model-name", default="output_sensor", help="Directory containing params_new.json")
parser.add_argument("--relative-metrics",action="store_true",help="Whether to normalize the metrics by label scales",)
parser.add_argument("--save-best", action="store_true", help="Whether to save best ND to param_search.txt")
parser.add_argument("--restore-file",default=None,help="Optional, name of the file in --model_dir containing weights to reload before training",)
parser.add_argument("--beta", type=float, default=10, help="hyperparameter of loss function")
parser.add_argument("--gamma", type=float, default=10, help="hyperparameter of loss function")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--params-json", default=None,
                    help="Path to params json. Default: "
                         "<model-name>/params_new.json. Pass a different "
                         "file (e.g. output_sensor/params_new_ws300.json) "
                         "to train with another window-size config.")
parser.add_argument("--seed", type=int, default=None,
                    help="Override the JSON 'train_seed' field. If neither is "
                         "set, defaults to 42.")


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    @property
    def dict(self):
        """Gives dict-like access to Params instance by params.dict['learning_rate']"""
        return self.__dict__



def _set_all_seeds(seed):
    """Fix all RNG sources so the run is reproducible.

    Mirrors set_seed() in f_steprc_CPSO.py. Covers Python random, NumPy,
    PyTorch (CPU + all CUDA devices), cudnn determinism, and PYTHONHASHSEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # PATCH-ACCEL-1: TF32 enabled on matmul (safe with deterministic; ~10-20% speedup on 4090/A100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[set_seed] seed={seed}, cudnn.deterministic=True, TF32=True")
    return seed


def mask_random_sequences(batch, min_drop_ratio=0.2,max_drop_ratio=0.9, fixed_unselect_idx=None):

    if fixed_unselect_idx is not None:

        batch_size, timesteps, features = batch.shape
        
        # 1. 随机选择一条序列
        unknown_idx = torch.tensor(fixed_unselect_idx)
        
        # 2. 创建全1掩码
        mask = torch.ones_like(batch)
        
        # 3. 将选中的序列全部掩码（设为0）
        mask[:,:, unknown_idx] = 0
        
        # 4. 应用掩码
        masked_batch = batch * mask
        
        drop_ratio=len(fixed_unselect_idx)/features
    
        return masked_batch, mask, unknown_idx, drop_ratio

    else:
        """
        随机选择多条序列进行掩码（丢弃50%-80%的序列）
        Args:
            batch: 输入批次 (batch_size, timesteps, features)
            min_drop_ratio: 最少丢弃比例 (默认50%)
            max_drop_ratio: 最多丢弃比例 (默认80%)
        Returns:
            masked_batch: 掩码后的批次
            mask: 掩码矩阵
            dropped_indices: 被丢弃的序列索引列表
            drop_ratio: 实际丢弃比例
        """
        batch_size, timesteps, features = batch.shape
        # 1. 随机确定丢弃比例（50%-80%）
        drop_ratio = torch.FloatTensor(1).uniform_(min_drop_ratio, max_drop_ratio).item()
        # 2. 计算要丢弃的序列数量（至少1条）
        num_to_drop = max(1, int(features * drop_ratio))
        num_to_drop = min(num_to_drop, features - 1)  # 确保至少保留1条序列
        # 3. 随机选择要丢弃的序列索引
        dropped_indices = torch.randperm(features)[:num_to_drop].tolist()
        # 4. 创建全1掩码
        mask = torch.ones_like(batch)
        # 5. 将选中的序列全部掩码（设为0）
        for idx in dropped_indices:
            mask[:, :, idx] = 0
        # 6. 应用掩码
        masked_batch = batch * mask
        return masked_batch, mask, dropped_indices, drop_ratio


# ---------------------------------------------------------------------------
# Curriculum + Anchored validation helpers (gated by params.curriculum_enabled)
# Default OFF -- all existing params files (no `curriculum_enabled` field)
# fall through to original behavior via getattr(..., False) guard.
# ---------------------------------------------------------------------------
def get_train_mask_range(epoch, params):
    """Return (rmin, rmax) for training mask sampling.

    Priority:
      1) If curriculum disabled -> static [min_drop, max_drop].
      2) PATCH-CURRMASK: If mixture-transition window active
         (params._transition_remaining_epochs > 0), return old_range with
         probability (1 - alpha) else new_range. alpha = (M - left) / M.
      3) If params has `_current_stage_idx` (plateau-triggered mode) -> use it
         to index curriculum_ranges directly.
      4) Otherwise fall back to milestone-based selection by epoch.
    """
    if not getattr(params, "curriculum_enabled", False):
        return params.mask_min_drop_ratio, params.mask_max_drop_ratio
    ranges = list(params.curriculum_ranges)
    # PATCH-CURRMASK: mixture-transition window (NeurIPS 2024 CurrMask)
    _trans_left = int(getattr(params, "_transition_remaining_epochs", 0))
    _old_rng = getattr(params, "_old_stage_range", None)
    if _trans_left > 0 and _old_rng is not None:
        _M = int(getattr(params, "curriculum_transition_window", 8))
        alpha = max(0.0, min(1.0, (_M - _trans_left) / float(max(1, _M))))
        if np.random.random() < (1.0 - alpha):
            return float(_old_rng[0]), float(_old_rng[1])
        # else fall through to current-stage range below
    # PATCH-PLATEAU: stage-idx-driven path (overrides milestones)
    if hasattr(params, "_current_stage_idx"):
        idx = int(getattr(params, "_current_stage_idx", 0))
        idx = max(0, min(idx, len(ranges) - 1))
        rng = ranges[idx]
        return float(rng[0]), float(rng[1])
    # Legacy milestone path
    milestones = list(getattr(params, "curriculum_milestones", []) or [])
    if len(ranges) == len(milestones) + 1:
        for ms, rng in zip(milestones, ranges[:-1]):
            if epoch <= ms:
                return float(rng[0]), float(rng[1])
        return float(ranges[-1][0]), float(ranges[-1][1])
    # Degenerate fallback
    return float(ranges[0][0]), float(ranges[0][1])


def build_anchor_indices(params):
    """Generate K fixed groups of `dropped_indices` for anchored validation.

    Two modes:
      A) Single-K (legacy): anchor_K groups, all with n_drop = total*anchor_ratio.
         Returns 2D int64 array of shape (K, n_drop), backward-compatible.
      B) Multi-K (PATCH-MULTIK): if params.anchor_multi_K_config.enabled=True,
         generate variable-length groups, each with different n_drop.
         Returns a Python list of 1D int64 arrays (length K_total).
         validate() must handle list form.
    """
    if not getattr(params, "curriculum_enabled", False):
        return None

    total = int(params.cov_dim)
    seed  = int(getattr(params, "valid_anchor_seed", 0))

    # PATCH-MULTIK: check multi-K mode
    _mk = getattr(params, "anchor_multi_K_config", None) or {}
    _mk_enabled = bool(_mk.get("enabled", False)) if isinstance(_mk, dict) else False

    if _mk_enabled:
        K_list = list(_mk.get("K_list", [200, 150, 100, 50, 20]))
        groups_per_K = list(_mk.get("groups_per_K", [10, 10, 10, 10, 10]))
        assert len(K_list) == len(groups_per_K), "K_list and groups_per_K length mismatch"
        anchor_groups = []  # list of 1D int64 arrays
        rng = np.random.RandomState(seed)
        for K_keep, n_groups in zip(K_list, groups_per_K):
            n_drop = total - int(K_keep)
            for _ in range(int(n_groups)):
                drop_idx = rng.permutation(total)[:n_drop].astype(np.int64)
                anchor_groups.append(drop_idx)
        # Save as object array (variable-length)
        save_path = os.path.join(params.exp_dir, "anchor_indices.npy")
        os.makedirs(params.exp_dir, exist_ok=True)
        np.save(save_path, np.array(anchor_groups, dtype=object), allow_pickle=True)
        print(f"[ANCHOR-MULTIK] K_list={K_list} groups_per_K={groups_per_K} "
              f"total_groups={len(anchor_groups)} seed={seed}  saved -> {save_path}")
        return anchor_groups

    # Single-K legacy path
    K     = int(getattr(params, "anchor_K", 50))
    ratio = float(getattr(params, "anchor_ratio", 0.97))
    n_drop = int(total * ratio)
    rng = np.random.RandomState(seed)
    anchor_indices_arr = np.stack([
        rng.permutation(total)[:n_drop] for _ in range(K)
    ]).astype(np.int64)  # shape (K, n_drop)
    save_path = os.path.join(params.exp_dir, "anchor_indices.npy")
    os.makedirs(params.exp_dir, exist_ok=True)
    np.save(save_path, anchor_indices_arr)
    print(f"[ANCHOR] {K} groups x drop_ratio={ratio} "
          f"(n_drop={n_drop}, seed={seed})  saved -> {save_path}")
    return anchor_indices_arr



def nanmean_ignore_zeros(tensor, dim=None, keepdim=False):
    """
    将零值替换为NaN，然后使用nanmean计算均值
    Args:
        tensor: 输入张量
        dim: 沿哪个维度计算均值
        keepdim: 是否保持维度
    Returns:
        忽略零值和NaN的均值
    """
    # 创建副本并将零值替换为NaN
    tensor_with_nan = tensor.clone()
    tensor_with_nan[tensor_with_nan == 0] = float('nan')
    # 使用nanmean计算均值（自动忽略NaN）
    result_feature = torch.nanmean(tensor_with_nan, dim=dim, keepdim=keepdim)
    result = torch.nanmean(tensor_with_nan)
    
    # 处理所有值都是NaN的情况
    if torch.isnan(result).all():
        # 返回与输入相同形状的零张量
        result = torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
        result_feature = torch.zeros(result_feature.shape)
    return result,result_feature


def kl_loss(mu, logvar):
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KL/mu.shape[0]



def gaussian_nll_loss(mean, logvar, target):
    """高斯负对数似然损失(论文公式8)"""
    return 0.5 * (logvar + (target - mean).pow(2) / logvar.exp()).mean()
    
def mse_loss(mean, logvar, target):
    """均方误差损失函数"""
    return (target - mean).pow(2)




# ---------------------------------------------------------------------------
# MAE helpers (norm + denorm)
# ---------------------------------------------------------------------------
def _mae_pair_accum(resid, mask_w, scale):
    """Per-batch accumulator for MAE in BOTH normalized and de-normalized
    (Cp / physical-unit) space, restricted to masked positions.

    The key identity: if labels and preds are in normalized space,
        residual_denorm = (pred * std + mean) - (label * std + mean)
                        = (pred - label) * std
    so |residual_denorm| at sensor c equals |residual_norm| * std_c.
    The per-sensor mean cancels out and we don't need it.

    Args:
      resid : (B, T, C) tensor of (pred_norm - label_norm) -- expects float.
      mask_w: (B, T, C) tensor with 1 on positions to score, 0 elsewhere.
      scale : (C,) tensor of per-sensor std (the denorm factor).

    Returns:
      (sum_abs_norm, sum_abs_denorm, sum_w) as Python floats.
      Caller divides by sum_w at end of epoch to get the average MAE.
    """
    abs_n = resid.detach().abs().float()
    sum_n = (abs_n * mask_w).sum().item()
    sum_d = (abs_n * scale.view(1, 1, -1) * mask_w).sum().item()
    sum_w = mask_w.sum().item()
    return sum_n, sum_d, sum_w


def plot_curve(values, output_folder, filename, ylabel="value"):
    """Plot a single 1D curve (one value per epoch) and save to disk."""
    os.makedirs(output_folder, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(values) + 1), values)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(filename)
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"{filename}.png"))
    plt.close()


def plot_and_save_loss(epoch_losses, output_folder, filename="Loss",epistemics=None, milestones=None):
    """绘制损失曲线并保存为图片文件"""
    plt.figure()
    plt.plot( range(1, epoch_losses.size(0)+1) , epoch_losses[:,0].numpy() , label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    # PATCH-VLINE: dashed lines at curriculum stage transitions
    if milestones:
        for ms in milestones:
            if 0 < ms <= epoch_losses.size(0):
                plt.axvline(x=ms + 0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    # 创建保存路径
    os.makedirs(output_folder, exist_ok=True)
    loss_plot_path = os.path.join(output_folder, f"{filename}.png")
    # 保存图像
    plt.savefig(loss_plot_path)
    plt.close()


# PATCH-LP     plt.figure(figsize=(20, 10))
# PATCH-LP     plt.plot( range(1, epoch_losses.size(0)+1) , epoch_losses[:, 1]/ epoch_losses[:, 0] , color='red', label='Recon')
# PATCH-LP     plt.plot( range(1, epoch_losses.size(0)+1) , epoch_losses[:, 2]/ epoch_losses[:, 0], color='green', label='KL')
# PATCH-LP     plt.xlabel("Epoch")
# PATCH-LP     plt.ylabel("LossPercent")
# PATCH-LP     plt.legend()
# PATCH-LP     plt.grid(True)
# PATCH-LP     losspercent_plot_path = os.path.join(output_folder, f"{filename}Percent.png")
# PATCH-LP     plt.savefig(losspercent_plot_path)
    plt.close()

    # if epistemics is not None:
        
    #     # 第一个子图 - 模型不确定性 (epistemics[:,0])
    #     plt.subplot(1, 2, 1)  # 1行2列的第1个子图
    #     plt.plot(range(1, epistemics.shape[0]+1), epistemics[:, 0], 
    #             color='royalblue', label="Model Uncertainty")
    #     plt.xlabel("Epoch", fontsize=10)
    #     plt.ylabel("Uncertainty Value", fontsize=10)
    #     plt.title("Epistemic Uncertainty (Model)", fontsize=12)
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
        
    #     # 第二个子图 - 数据不确定性 (epistemics[:,1])
    #     plt.subplot(1, 2, 2)  # 1行2列的第2个子图
    #     plt.plot(range(1, epistemics.shape[0]+1), epistemics[:, 1], 
    #             color='crimson', label="Data Uncertainty")
    #     plt.xlabel("Epoch", fontsize=10)
    #     plt.ylabel("Uncertainty Value", fontsize=10)
    #     plt.title("Aleatoric Uncertainty (Data)", fontsize=12)
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     # 创建保存路径
    #     os.makedirs(output_folder, exist_ok=True)
    #     plot_path = os.path.join(output_folder, f"{filename}_Uncertainties.png")
        
    #     # 保存图像
    #     plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    #     plt.close()






def train_and_evaluate(model,train_iter,valid_iter, test_iter,params,optimizer,scheduler):
    loss =mse_loss
    # loss =gaussian_nll_loss
    output_folder = os.path.join(params.model_dir, "DeFigs", params.ckpt_name)
    os.makedirs(output_folder, exist_ok=True)
    # PATCH-EVENTS: concise event log (curriculum + LR decay) lives in output_folder
    events_path = os.path.join(output_folder, "events.log")
    def _ev(msg):
        from datetime import datetime as _dt
        line = f"[{_dt.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            with open(events_path, "a", encoding="utf-8") as _f:
                _f.write(line + "\n")
        except Exception as _e:
            print(f"[events_log_write_err] {_e}", flush=True)
    _ev(f"RUN_START ckpt={params.ckpt_name} epochs={params.epochs} lr_init={args.lr} curriculum_enabled={getattr(params, 'curriculum_enabled', False)} milestones={list(getattr(params, 'curriculum_milestones', []))} ranges={list(getattr(params, 'curriculum_ranges', []))} stage_lrs={list(getattr(params, 'curriculum_stage_lrs', []) or [])} max_stage_ep={getattr(params, 'curriculum_max_stage_epochs', 50)} decay_to_switch={getattr(params, 'curriculum_decay_to_switch', 2)}")
    _prev_lr_for_log = float(args.lr)
    # PATCH-VLINE2: collect actual epochs where stage transitions happen
    _stage_switch_epochs = []
    # PATCH-PLATEAU: init runtime stage tracker (used by get_train_mask_range)
    if getattr(params, "curriculum_enabled", False) and len(list(getattr(params, "curriculum_stage_lrs", []) or [])) > 0:
        params._current_stage_idx = 0
        # PATCH-CURRMASK: init runtime mixture-transition state
        params._transition_remaining_epochs = 0
        params._old_stage_range = None
        # apply stage-0 lr at start (in case args.lr differs from stage_lrs[0])
        _stage0_lr = float(params.curriculum_stage_lrs[0])
        for pg in optimizer.param_groups:
            pg["lr"] = _stage0_lr
        _prev_lr_for_log = _stage0_lr
        # PATCH-COSINE: remember stage-0 init lr for cosine schedule
        params._cosine_stage_init_lr = _stage0_lr
        _ev(f"PLATEAU-MODE-INIT stage=0 lr={_stage0_lr}")

    epoch_losses = torch.tensor([]).reshape(0, 3)
    valid_losses = torch.tensor([]).reshape(0, 3)
    test_losses= torch.tensor([]).reshape(0, 3)
    valid_epistemics=torch.tensor([]).reshape(0,2)
    test_epistemics=torch.tensor([]).reshape(0,2)
    # Per-epoch MAE histories: norm space + denorm (Cp / physical units).
    train_mae_n_hist, train_mae_d_hist = [], []
    valid_mae_n_hist, valid_mae_d_hist = [], []
    test_mae_n_hist,  test_mae_d_hist  = [], []
    early_stopping = EarlyStopping(patience=int(getattr(params, "early_stopping_patience", 15)), verbose=True)

    # Build anchored validation mask groups once.
    # Returns None when curriculum_enabled is absent/False (preserves old behavior).
    anchor_indices_arr = build_anchor_indices(params)

    for epoch in range(params.epochs):


        params.beta=0

        print('param.beta is ',params.beta)
        # print('param.latent_dim is ',params.latent_dim)
        print('param.dropout is ',params.dropout)

        epoch_1based = epoch + 1
        if getattr(params, "curriculum_enabled", False):
            # PATCH-PLATEAU: init runtime stage state on first epoch
            _stage_lrs = list(getattr(params, "curriculum_stage_lrs", []) or [])
            _n_stages = len(list(getattr(params, "curriculum_ranges", []) or []))
            _plateau_mode = len(_stage_lrs) > 0
            if not hasattr(params, "_current_stage_idx"):
                params._current_stage_idx = 0
            if not hasattr(train_and_evaluate, "_n_decay_in_stage"):
                train_and_evaluate._n_decay_in_stage = 0
            if not hasattr(train_and_evaluate, "_epoch_at_stage_start"):
                train_and_evaluate._epoch_at_stage_start = 1
            rmin_cur, rmax_cur = get_train_mask_range(epoch_1based, params)
            print(f"[CURRICULUM] epoch={epoch_1based} stage={params._current_stage_idx} train_mask=[{rmin_cur}, {rmax_cur}]")

            if _plateau_mode:
                # Plateau-triggered switching is handled AFTER scheduler.step
                # (i.e. at end of this epoch); only handle legacy milestone
                # fallback here when stage_lrs is absent.
                pass
            else:
                # PATCH-LR-RESET (legacy milestone path)
                milestones = list(getattr(params, "curriculum_milestones", []))
                current_stage_idx = sum(1 for ms in milestones if epoch_1based > ms)
                if not hasattr(train_and_evaluate, "_prev_stage_idx"):
                    train_and_evaluate._prev_stage_idx = current_stage_idx
                if current_stage_idx != train_and_evaluate._prev_stage_idx:
                    # PATCH-NO-LR-RESET: skip LR reset if curriculum_no_lr_reset=True
                    _skip_lr_reset = bool(getattr(params, "curriculum_no_lr_reset", False))
                    if _skip_lr_reset:
                        target_lr = float(optimizer.param_groups[0]["lr"])  # keep current
                        print(f"[CURRICULUM-NO-LR-RESET] stage {train_and_evaluate._prev_stage_idx} -> {current_stage_idx}, lr stays at {target_lr}")
                    else:
                        target_lr = args.lr
                        for pg in optimizer.param_groups:
                            pg["lr"] = target_lr
                        from torch.optim.lr_scheduler import ReduceLROnPlateau
                        # Re-init resets num_bad_epochs to 0 so PATCH-PLATEAU restarts
                        # plateau detection in the new stage.  The LR field reset
                        # here is overwritten by PATCH-COSINE at the next epoch.
                        scheduler.__class__.__init__(scheduler, optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-6)
                        print(f"[CURRICULUM-LR-RESET] stage {train_and_evaluate._prev_stage_idx} -> {current_stage_idx}, lr reset to {target_lr}")
                    _ev(f"CURRICULUM stage {train_and_evaluate._prev_stage_idx}->{current_stage_idx} epoch={epoch_1based} range=[{rmin_cur},{rmax_cur}] lr_reset_to={target_lr} train_loss={float(epoch_losses[-1,0]):.4f} valid_loss={float(valid_losses[-1,0]):.4f} test_loss={float(test_losses[-1,0]):.4f} train_mae_denorm={(train_mae_d_hist[-1] if train_mae_d_hist else float('nan')):.5f} valid_mae_denorm={(valid_mae_d_hist[-1] if valid_mae_d_hist else float('nan')):.5f} test_mae_denorm={(test_mae_d_hist[-1] if test_mae_d_hist else float('nan')):.5f}")
                    _prev_lr_for_log = float(target_lr)
                    train_and_evaluate._prev_stage_idx = current_stage_idx
                    _stage_switch_epochs.append(int(epoch_1based))


        # PATCH-COSINE: apply per-stage cosine LR before training step
        _cos_mode = getattr(params, "curriculum_cosine_mode", None) or {}
        _cos_enabled = bool(_cos_mode.get("enabled", False)) if isinstance(_cos_mode, dict) else False
        if _cos_enabled and getattr(params, "curriculum_enabled", False):
            import math as _math
            _cos_min_ratio = float(_cos_mode.get("cosine_min_ratio", 0.4))
            _stage_idx_now = int(getattr(params, "_current_stage_idx", 0))
            _stage_init_lr = float(getattr(params, "_cosine_stage_init_lr", _stage_lrs[0] if _stage_lrs else 5e-5))
            _stage_start_ep = int(getattr(train_and_evaluate, "_epoch_at_stage_start", 1))
            _epoch_in_stage_cos = epoch_1based - _stage_start_ep + 1
            _stage_lrs_now_b = list(getattr(params, "curriculum_stage_lrs", []) or [])
            _is_last = _stage_idx_now >= (len(_stage_lrs_now_b) - 1)
            if _is_last:
                # full cosine to final_min_lr over remaining epochs
                _final_min_lr = float(_cos_mode.get("final_min_lr", 1e-6))
                _max_remaining = max(1, int(params.epochs) - _stage_start_ep + 1)
                _progress = min(1.0, (_epoch_in_stage_cos - 1) / float(_max_remaining))
                _lr_now = _final_min_lr + (_stage_init_lr - _final_min_lr) * 0.5 * (1.0 + _math.cos(_math.pi * _progress))
            else:
                # per-stage cosine to _cos_min_ratio * init within expected stage length
                _stage_len = int(_cos_mode.get("epochs_per_stage_first", 30))
                _progress = min(1.0, (_epoch_in_stage_cos - 1) / float(max(1, _stage_len)))
                _lr_min_this = _cos_min_ratio * _stage_init_lr
                _lr_now = _lr_min_this + (_stage_init_lr - _lr_min_this) * 0.5 * (1.0 + _math.cos(_math.pi * _progress))
            for pg in optimizer.param_groups:
                pg["lr"] = float(_lr_now)

#==============================训练===============================
        epoch_loss, train_mae_n, train_mae_d = train_seq2seq(
            model, train_iter, params, loss, optimizer, epoch_1based=epoch_1based)
        # 记录当前 epoch 的损失
        epoch_losses=torch.cat((epoch_losses,epoch_loss/len(train_iter)),dim=0)
        train_mae_n_hist.append(train_mae_n)
        train_mae_d_hist.append(train_mae_d)

        print(f"Epoch [{epoch + 1}/{params.epochs}], Loss: {epoch_losses[-1,0]:.4f}")
        # 绘制损失曲线并保存
        plot_and_save_loss(epoch_losses, output_folder, filename="TrainLoss", milestones=list(_stage_switch_epochs))
        # PATCH-SKIP: plot_curve(train_mae_n_hist, output_folder, "TrainMAE_norm",   ylabel="MAE (norm)")
        plot_curve(train_mae_d_hist, output_folder, "TrainMAE_denorm", ylabel="MAE (Cp)")

#==============================验证===============================
        valid_loss, valid_epistemic, valid_mae_n, valid_mae_d = validate(
            model, valid_iter, params, loss, anchor_indices_arr=anchor_indices_arr)

        valid_losses=torch.cat((valid_losses,valid_loss/len(valid_iter)),dim=0)
        valid_epistemics=torch.cat((valid_epistemics,valid_epistemic),dim=0)
        valid_mae_n_hist.append(valid_mae_n)
        valid_mae_d_hist.append(valid_mae_d)

        print(f"Epoch [{epoch + 1}/{params.epochs}], Valid_Loss: {valid_losses[-1,0]:.4f}")

        plot_and_save_loss(valid_losses, output_folder, filename="ValidLoss",epistemics=valid_epistemics, milestones=list(_stage_switch_epochs))
        # PATCH-SKIP: plot_curve(valid_mae_n_hist, output_folder, "ValidMAE_norm",   ylabel="MAE (norm)")
        plot_curve(valid_mae_d_hist, output_folder, "ValidMAE_denorm", ylabel="MAE (Cp)")

#==============================测试===============================
        test_loss, test_epistemic, test_mae_n, test_mae_d = evaluate(
            model, test_iter, params, loss)

        test_losses=torch.cat((test_losses,test_loss/len(test_iter)),dim=0)
        test_epistemics=torch.cat((test_epistemics,test_epistemic),dim=0)
        test_mae_n_hist.append(test_mae_n)
        test_mae_d_hist.append(test_mae_d)

        print(f"Epoch [{epoch + 1}/{params.epochs}], Test_Loss: {test_losses[-1,0]:.4f}")

        plot_and_save_loss(test_losses, output_folder, filename="TestLoss",epistemics=test_epistemics, milestones=list(_stage_switch_epochs))
        # PATCH-SKIP: plot_curve(test_mae_n_hist, output_folder, "TestMAE_norm",   ylabel="MAE (norm)")
        plot_curve(test_mae_d_hist, output_folder, "TestMAE_denorm", ylabel="MAE (Cp)")




        # scheduler.step() keeps RLROP's num_bad_epochs current so that
        # PATCH-PLATEAU below can detect validation plateau.  The LR
        # change step() may issue is irrelevant: PATCH-COSINE re-writes
        # optimizer.lr at the top of the next epoch.
        scheduler.step(valid_losses[-1,0])
        # PATCH-EVENTS: detect LR decay event after scheduler.step()
        _cur_lr_now = float(optimizer.param_groups[0]['lr'])
        _lr_decayed_this_epoch = abs(_cur_lr_now - _prev_lr_for_log) > 1e-12
        if _lr_decayed_this_epoch:
            _ev(f"LR_DECAY epoch={epoch_1based} lr {_prev_lr_for_log:.2e}->{_cur_lr_now:.2e} train_loss={float(epoch_losses[-1,0]):.4f} valid_loss={float(valid_losses[-1,0]):.4f} test_loss={float(test_losses[-1,0]):.4f} train_mae_denorm={(train_mae_d_hist[-1] if train_mae_d_hist else float('nan')):.5f} valid_mae_denorm={(valid_mae_d_hist[-1] if valid_mae_d_hist else float('nan')):.5f} test_mae_denorm={(test_mae_d_hist[-1] if test_mae_d_hist else float('nan')):.5f}")
            _prev_lr_for_log = _cur_lr_now
            # PATCH-PLATEAU: count decay events within current stage
            if getattr(params, "curriculum_enabled", False) and len(list(getattr(params, "curriculum_stage_lrs", []) or [])) > 0:
                train_and_evaluate._n_decay_in_stage += 1
        # PATCH-PLATEAU: stage switching (after scheduler.step, end of epoch)
        # PATCH-CURRMASK: decrement transition window counter (mixture sampling)
        if int(getattr(params, "_transition_remaining_epochs", 0)) > 0:
            params._transition_remaining_epochs -= 1
        if getattr(params, "curriculum_enabled", False):
            _stage_lrs_now = list(getattr(params, "curriculum_stage_lrs", []) or [])
            _ranges_now = list(getattr(params, "curriculum_ranges", []) or [])
            if len(_stage_lrs_now) > 0:
                _ft = getattr(params, "curriculum_fast_transition", None) or {}
                _ft_enabled = bool(_ft.get("enabled", False)) if isinstance(_ft, dict) else False
                _decay_to_switch = int(getattr(params, "curriculum_decay_to_switch", 2))
                _max_stage_ep = int(getattr(params, "curriculum_max_stage_epochs", 50))
                _patience_cur = int(getattr(scheduler, "patience", 6))
                _bad_now = int(getattr(scheduler, "num_bad_epochs", 0))
                _epoch_in_stage = epoch_1based - int(getattr(train_and_evaluate, "_epoch_at_stage_start", 1)) + 1
                _is_last_stage = params._current_stage_idx >= (len(_stage_lrs_now) - 1)

                if _ft_enabled and not _is_last_stage:
                    # PATCH-CURRMASK: fast-transition mode (CurrMask, NeurIPS 2024)
                    _ft_patience = int(_ft.get("patience", 3))
                    _ft_min_ep = int(_ft.get("min_epochs_per_stage", 5))
                    _ft_max_ep = int(_ft.get("max_epochs_per_stage", 25))
                    _trigger_plateau = (_bad_now >= _ft_patience) and (_epoch_in_stage >= _ft_min_ep)
                    _trigger_hardcap = (_epoch_in_stage >= _ft_max_ep)
                    _switch_keep_lr = True
                    _trigger_mixture = True
                else:
                    # legacy plateau-mode (expand_curr cumulative)
                    _trigger_plateau = (
                        (not _is_last_stage)
                        and (train_and_evaluate._n_decay_in_stage >= _decay_to_switch)
                        and (_bad_now >= _patience_cur)
                    )
                    _trigger_hardcap = (not _is_last_stage) and (_epoch_in_stage >= _max_stage_ep)
                    _switch_keep_lr = False
                    _trigger_mixture = False

                if _trigger_plateau or _trigger_hardcap:
                    _old_stage = params._current_stage_idx
                    _old_range_snapshot = list(_ranges_now[_old_stage]) if _old_stage < len(_ranges_now) else None
                    params._current_stage_idx = _old_stage + 1

                    # PATCH-COSINE: partial restart for next stage (cosine mode)
                    _cos_mode_sw = getattr(params, "curriculum_cosine_mode", None) or {}
                    _cos_enabled_sw = bool(_cos_mode_sw.get("enabled", False)) if isinstance(_cos_mode_sw, dict) else False
                    if _cos_enabled_sw:
                        _partial_ratio = float(_cos_mode_sw.get("partial_restart_ratio", 0.8))
                        _prev_stage_init = float(getattr(params, "_cosine_stage_init_lr", _stage_lrs_now[_old_stage]))
                        _new_lr = _prev_stage_init * _partial_ratio
                        params._cosine_stage_init_lr = _new_lr
                    elif _switch_keep_lr:
                        _new_lr = float(optimizer.param_groups[0]['lr'])
                    else:
                        _new_lr = float(_stage_lrs_now[params._current_stage_idx])
                    for pg in optimizer.param_groups:
                        pg["lr"] = _new_lr

                    from torch.optim.lr_scheduler import ReduceLROnPlateau
                    # Re-init resets num_bad_epochs to 0 so PATCH-PLATEAU restarts
                    # plateau detection in the new stage.  The LR field reset
                    # here is overwritten by PATCH-COSINE at the next epoch.
                    scheduler.__class__.__init__(scheduler, optimizer, mode="min", factor=0.5, patience=_patience_cur, min_lr=float(getattr(params, "min_lr", 1e-6)))
                    train_and_evaluate._n_decay_in_stage = 0
                    train_and_evaluate._epoch_at_stage_start = epoch_1based + 1

                    # PATCH-CURRMASK: arm mixture-transition window
                    if _trigger_mixture and _old_range_snapshot is not None:
                        _M = int(getattr(params, "curriculum_transition_window", 8))
                        params._transition_remaining_epochs = _M
                        params._old_stage_range = _old_range_snapshot
                    else:
                        params._transition_remaining_epochs = 0
                        params._old_stage_range = None

                    _reason = "plateau" if _trigger_plateau else "hardcap"
                    _new_range = _ranges_now[params._current_stage_idx] if params._current_stage_idx < len(_ranges_now) else None
                    print(f"[CURRICULUM-PLATEAU-SWITCH] stage {_old_stage}->{params._current_stage_idx} epoch={epoch_1based} reason={_reason} lr_reset_to={_new_lr} new_range={_new_range} mixture={_trigger_mixture}")
                    _ev(f"CURRICULUM-SWITCH stage {_old_stage}->{params._current_stage_idx} epoch={epoch_1based} reason={_reason} new_range={_new_range} lr_reset_to={_new_lr} keep_lr={_switch_keep_lr} mixture={_trigger_mixture} mixture_M={int(getattr(params, '_transition_remaining_epochs', 0))} bad_epochs_at_switch={_bad_now} decay_count_in_stage={train_and_evaluate._n_decay_in_stage} train_loss={float(epoch_losses[-1,0]):.4f} valid_loss={float(valid_losses[-1,0]):.4f} test_loss={float(test_losses[-1,0]):.4f} train_mae_denorm={(train_mae_d_hist[-1] if train_mae_d_hist else float('nan')):.5f} valid_mae_denorm={(valid_mae_d_hist[-1] if valid_mae_d_hist else float('nan')):.5f} test_mae_denorm={(test_mae_d_hist[-1] if test_mae_d_hist else float('nan')):.5f}")
                    _prev_lr_for_log = _new_lr
                    _stage_switch_epochs.append(int(epoch_1based))

                    # PATCH-CURRMASK: entered final stage -> ES reset + big-patience scheduler
                    if params._current_stage_idx == (len(_stage_lrs_now) - 1):
                        early_stopping.counter = 0
                        early_stopping.best_score = None
                        _final_patience = int(getattr(params, "curriculum_final_stage_patience", 12))
                        # Re-init resets num_bad_epochs to 0 so PATCH-PLATEAU restarts
                        # plateau detection in the new stage.  The LR field reset
                        # here is overwritten by PATCH-COSINE at the next epoch.
                        scheduler.__class__.__init__(scheduler, optimizer, mode="min", factor=0.5, patience=_final_patience, min_lr=float(getattr(params, "min_lr", 1e-6)))
                        # Disable mixture in final stage
                        params._transition_remaining_epochs = 0
                        params._old_stage_range = None
                        _ev(f"FINAL_STAGE_INIT epoch={epoch_1based} es_counter_reset=True es_best_score_reset=True final_patience={_final_patience}")


        # current_lr = scheduler.get_lr()[0]
        current_lr = optimizer.param_groups[0]['lr'] 
        print(f"================Epoch {epoch+1}, Learning Rate: {current_lr}==================")

        model_folder=os.path.join(params.model_dir, 'model', params.ckpt_name)

        os.makedirs(model_folder,exist_ok=True)

        # early_stopping(valid_losses[-1,0], model.state_dict(), model_folder)
        early_stopping(valid_losses[-1,0], model.state_dict(), model_folder, save_name='checkpoint')

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # ----------------------------------------------------------------------
    # Save final_metrics.json: capture the valid/test MAE (both norm and
    # denorm Cp space) at the epoch where EarlyStopping saved the model
    # (i.e. the epoch with lowest valid loss). Lets a separate aggregate
    # script collect mean/std across seeds without re-running anything.
    # ----------------------------------------------------------------------
    if len(valid_mae_d_hist) > 0:
        valid_losses_per_epoch = [float(valid_losses[i, 0]) for i in range(len(valid_mae_d_hist))]
        best_epoch_idx = int(np.argmin(valid_losses_per_epoch))
        final_metrics = {
            "best_valid_epoch_1based":     best_epoch_idx + 1,
            "best_valid_loss":             valid_losses_per_epoch[best_epoch_idx],
            "valid_mae_norm_at_best":      float(valid_mae_n_hist[best_epoch_idx]),
            "valid_mae_denorm_Cp_at_best": float(valid_mae_d_hist[best_epoch_idx]),
            "test_mae_norm_at_best":       float(test_mae_n_hist[best_epoch_idx]),
            "test_mae_denorm_Cp_at_best":  float(test_mae_d_hist[best_epoch_idx]),
            "total_epochs_run":            len(valid_mae_d_hist),
            "seed":                        int(params.train_seed),
            "window_size":                 int(params.window_size),
            "mask_min_drop_ratio":         float(params.mask_min_drop_ratio),
            "mask_max_drop_ratio":         float(params.mask_max_drop_ratio),
            "ckpt_name":                   params.ckpt_name,
            "curriculum_enabled":          bool(getattr(params, "curriculum_enabled", False)),
            "curriculum_milestones":       getattr(params, "curriculum_milestones", None),
            "curriculum_ranges":           getattr(params, "curriculum_ranges", None),
            "anchor_ratio":                getattr(params, "anchor_ratio", None),
            "anchor_K":                    getattr(params, "anchor_K", None),
            "valid_anchor_seed":           getattr(params, "valid_anchor_seed", None),
        }
        metrics_path = os.path.join(params.exp_dir, "final_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)
        print(f"[run] final_metrics saved -> {metrics_path}")
        print(f"[run] best_valid_epoch={best_epoch_idx+1}/{len(valid_mae_d_hist)} "
              f"valid_MAE(Cp)={final_metrics['valid_mae_denorm_Cp_at_best']:.5f} "
              f"test_MAE(Cp)={final_metrics['test_mae_denorm_Cp_at_best']:.5f}")



def plot(results, labels, dropped_indices, params, task):
    import random
    import pandas as pd
    import matplotlib.pyplot as plt
    
    unknown_idx= random.choice(dropped_indices)

    rd = random.randint(0, results['mean'].shape[0]-1)
    pred= results['mean'][rd, :,unknown_idx].to('cpu').detach().numpy()
    label = labels[rd, :, unknown_idx].to('cpu').detach().numpy()
    
    T = np.array(range(labels[rd, :, unknown_idx].__len__()))
    plt.figure()
    plt.figure(figsize=(20, 10))
    plt.plot(T,label, label='Label')
    plt.plot(T[-pred.shape[0]:], pred, label='Pred (mu)', color='orange')  # 预测均值线

    os.makedirs(os.path.join(params.model_dir, 'DeFigs', params.ckpt_name), exist_ok=True)

    x, y = np.array(pred).ravel(), np.array(label).ravel()
    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
    # 初始化标题和绘图
    title_parts = [f"Idx={unknown_idx}, Unknown_Condition={len(dropped_indices)},R={r:.3f}"]  # 始终显示相关系数

    # if 'epistemic_std' in results or 'aleatoric_std' in results:
    #     # 初始化标题和置信区间绘制
    #     # 处理 epistemic_std
    #     if 'epistemic_std' in results:
    #         sigma = results['epistemic_std'][rd, :, unknown_idx].to('cpu').detach().numpy()
    #         pred_upper = pred + 1.96 * sigma
    #         pred_lower = pred - 1.96 * sigma
    #         plt.fill_between(
    #             T[-pred.shape[0]:], 
    #             pred_lower[-pred.shape[0]:], 
    #             pred_upper[-pred.shape[0]:], 
    #             color='gray', alpha=0.3, 
    #             label='Epistemic ±1.96σ'
    #         )
    #         title_parts.append(f"Epistemic={results['epistemic_uncertainty'][rd,unknown_idx].item():.4f}")
    #         print(f"Epistemic={results['epistemic_uncertainty'][rd,unknown_idx].item():.4f}")

    #     # 处理 aleatoric_std
    #     if 'aleatoric_std' in results:
    #         sigma = results['aleatoric_std'][rd, :, unknown_idx].to('cpu').detach().numpy()
    #         pred_upper = pred + 1.96 * sigma
    #         pred_lower = pred - 1.96 * sigma
    #         plt.fill_between(
    #             T[-pred.shape[0]:], 
    #             pred_lower[-pred.shape[0]:], 
    #             pred_upper[-pred.shape[0]:], 
    #             color='gray', alpha=0.3, 
    #             label='Aleatoric ±1.96σ'
    #         )
    #         title_parts.append(f"Aleatoric={results['aleatoric_uncertainty'][rd,unknown_idx].item():.4f}")
        
    #     # 设置标题和标签
    #     plt.xlabel('Time')
    #     plt.legend(loc='best')
    #     plt.title(
    #         " | ".join(title_parts),  # 格式示例: "R=0.85 | Epistemic=0.12 | Aleatoric=0.05"
    #         fontsize=12, 
    #         pad=20
    #     )
    # else:
    #     # 若无不确定性数据，仅显示相关系数
    #     plt.title(f"R={r:.3f}", fontsize=12, pad=20)

    plt.title(f"R={r:.3f}", fontsize=12, pad=20)
    # 保存图像
    plt.savefig(os.path.join(params.model_dir, 'DeFigs', params.ckpt_name, f'pred_{task}.png'))



def train_seq2seq(model, data_iter, params:Params ,loss ,optimizer, epoch_1based=None):

    """训练序列到序列模型。

    `epoch_1based` 用于 curriculum mask schedule;当 curriculum disabled 时该参数
    被 get_train_mask_range 忽略,保持原行为。
    """

    model.train()

    loss_matrix = torch.zeros((1,3))
    # MAE accumulators (sums of |resid| over masked positions); divided by
    # the total mask weight at the end of the epoch to get average MAE
    # in normalized space and de-normalized (Cp) space.
    mae_n_sum, mae_d_sum, mae_w_sum = 0.0, 0.0, 0.0

    for i, (train_batch) in enumerate(tqdm(data_iter)):

        train_batch= train_batch.to(torch.float32).to(params.device)

        # ABLATION-SHUFFLE-TIME-TRAIN: per-sample time-axis permutation (input and target share same perm via clone after shuffle)
        if getattr(params, 'shuffle_time_train', False):
            _B, _T, _C = train_batch.shape
            if _T > 1:
                _perms = torch.stack([torch.randperm(_T, device=params.device) for _ in range(_B)])
                _idx_b = _perms.unsqueeze(-1).expand(-1, -1, _C)
                train_batch = torch.gather(train_batch, 1, _idx_b)

        labels_batch = train_batch.clone()

        rmin_cur, rmax_cur = get_train_mask_range(epoch_1based, params)
        masked_batch, mask, dropped_indices, drop_ratio = mask_random_sequences(
            train_batch,
            min_drop_ratio=rmin_cur,
            max_drop_ratio=rmax_cur,
            # fixed_unselect_idx=unselected_indices
        )

        # 1. 创建掩码权重矩阵（掩码位置为1，非掩码位置为0）
        mask_weights = 1 - mask  # 因为mask中掩码位置是0，非掩码位置是1

        # 将掩码标识拼接为额外特征维度
        extended_batch = torch.cat([masked_batch, mask], dim=-1)

        # PATCH-ACCEL-5: AMP-BF16 forward+loss (4090/A100/H100 native bf16, no GradScaler needed)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            results  = model(extended_batch)   
  
            l1= loss(results['mean'], results['aleatoric_logvar'], labels_batch)

            # 3. 只对掩码位置计算损失
            l1 = (l1 * mask_weights).sum() / (mask_weights.sum() + 1e-8)        

            l2 = params.beta*model.kl_loss() /len(train_batch)

            l = l1 + l2

        optimizer.zero_grad(set_to_none=True)
        l.backward()

        optimizer.step()

        loss_matrix = loss_matrix+torch.tensor([[l.item(), l1.item(), l2.item()]])

        # MAE bookkeeping (no_grad to keep the graph small).
        with torch.no_grad():
            resid = results['mean'] - labels_batch
            sn, sd, sw = _mae_pair_accum(resid, mask_weights, params.scale)
            mae_n_sum += sn
            mae_d_sum += sd
            mae_w_sum += sw

        if i==0:

            pass  # PATCH-SKIP-PLOT: plot(results , labels_batch, dropped_indices, params, 'train')

    mae_n_epoch = mae_n_sum / max(mae_w_sum, 1e-8)
    mae_d_epoch = mae_d_sum / max(mae_w_sum, 1e-8)
    print(f"Train_Loss={loss_matrix[-1]/len(data_iter)}  "
          f"Train_MAE_norm={mae_n_epoch:.6f}  Train_MAE_denorm={mae_d_epoch:.6f}")

    return loss_matrix, mae_n_epoch, mae_d_epoch







def validate(model,data_iter,params: Params,loss, anchor_indices_arr=None):
    """Validate.

    If `anchor_indices_arr` is None: original behavior (random mask sampled
    per batch from [params.mask_min_drop_ratio, params.mask_max_drop_ratio]).

    Otherwise: anchored validation. For each batch we evaluate K=anchor_indices_arr.shape[0]
    fixed mask configurations and average the per-batch loss/MAE across K groups.
    This makes valid_loss comparable across epochs (deterministic mask configs).
    """
    model.eval()
    loss_matrix = torch.zeros((1,3))
    all_epistemic_uncertainty =torch.tensor([]).to(params.device)
    all_aleatoric_uncertainty =torch.tensor([]).to(params.device)
    # MAE accumulators (norm + denorm) -- averaged at the end of validation.
    mae_n_sum, mae_d_sum, mae_w_sum = 0.0, 0.0, 0.0

    use_anchor = anchor_indices_arr is not None
    # PATCH-MULTIK: support both ndarray (single-K) and list (multi-K)
    if use_anchor:
        if isinstance(anchor_indices_arr, list):
            K_groups = len(anchor_indices_arr)
        else:
            K_groups = int(anchor_indices_arr.shape[0])
    else:
        K_groups = 1

    with torch.no_grad():

        for i, (test_batch ) in enumerate(tqdm(data_iter)):

            test_batch= test_batch.to(torch.float32).to(params.device)

            # ABLATION-SHUFFLE-TIME-EVAL: match train-time perturbation if enabled
            if getattr(params, 'shuffle_time_eval', False):
                _B, _T, _C = test_batch.shape
                if _T > 1:
                    _perms = torch.stack([torch.randperm(_T, device=params.device) for _ in range(_B)])
                    _idx_b = _perms.unsqueeze(-1).expand(-1, -1, _C)
                    test_batch = torch.gather(test_batch, 1, _idx_b)

            labels = test_batch.clone()

            for k in range(K_groups):
                if use_anchor:
                    masked_batch, mask, dropped_indices, drop_ratio = mask_random_sequences(
                        test_batch,
                        fixed_unselect_idx=anchor_indices_arr[k].tolist(),
                    )
                else:
                    masked_batch, mask, dropped_indices, drop_ratio = mask_random_sequences(
                        test_batch,
                        min_drop_ratio=params.mask_min_drop_ratio,
                        max_drop_ratio=params.mask_max_drop_ratio,
                    )

                # 只计算掩码部分的损失
                # 1. 创建掩码权重矩阵（掩码位置为1，非掩码位置为0）
                mask_weights = 1 - mask  # 因为mask中掩码位置是0，非掩码位置是1

                # 将掩码标识拼接为额外特征维度
                extended_batch = torch.cat([masked_batch, mask], dim=-1)

                # MC-dropout switch driven by params.mc_predict (set in JSON):
                #   - mc_predict == 1  -> plain deterministic forward
                #   - mc_predict != 1  -> if model exposes .mc_predict, use it
                #                         with n_samples=params.mc_predict;
                #                         otherwise fall back to plain forward.
                if params.mc_predict == 1 or not hasattr(model, 'mc_predict'):
                    # PATCH-ACCEL-5: AMP-BF16 also for eval forward
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                        results = model(extended_batch)
                else:
                    results = model.mc_predict(extended_batch, n_samples=params.mc_predict)

                l1= loss(results['mean'], results['aleatoric_logvar'], labels)

                # 3. 只对掩码位置计算损失
                l1 = (l1 * mask_weights).sum() / (mask_weights.sum() + 1e-8)

                l2 = params.beta*model.kl_loss()/len(test_batch)

                l = l1+ l2

                # Average across K anchor groups so totals stay on the same scale
                # as the original (non-anchored) path.
                loss_matrix = loss_matrix + torch.tensor(
                    [[l.item(), l1.item(), l2.item()]]) / K_groups

                # MAE bookkeeping (no_grad is already in effect above).
                resid = results['mean'] - labels
                sn, sd, sw = _mae_pair_accum(resid, mask_weights, params.scale)
                mae_n_sum += sn / K_groups
                mae_d_sum += sd / K_groups
                mae_w_sum += sw / K_groups

                # Only plot once per validation pass (first batch, first anchor group).
                if i==0 and k==0:
                    pass  # PATCH-SKIP-PLOT: plot(results, labels, dropped_indices, params, 'valid')

                if 'epistemic_uncertainty' in results:
                    pass  # PATCH-SKIP-CAT: all_epistemic_uncertainty=torch.cat(...)

                # PATCH-SKIP-CAT: all_aleatoric_uncertainty=torch.cat(...)

        aleatoric_mean ,aleatoric_mean_feature = nanmean_ignore_zeros(all_aleatoric_uncertainty, dim=0)
        epistemic_mean, epistemic_mean_feature = nanmean_ignore_zeros(all_epistemic_uncertainty, dim=0)
        mae_n_epoch = mae_n_sum / max(mae_w_sum, 1e-8)
        mae_d_epoch = mae_d_sum / max(mae_w_sum, 1e-8)
        print(f"Valid_Loss={loss_matrix[-1]/len(data_iter)}  "
              f"all_epistemic_uncertainty={epistemic_mean}  "
              f"all_aleatoric_uncertainty={aleatoric_mean}  "
              f"Valid_MAE_norm={mae_n_epoch:.6f}  Valid_MAE_denorm={mae_d_epoch:.6f}")



    return (loss_matrix,
            torch.tensor([[epistemic_mean.item(), aleatoric_mean.item()]]),
            mae_n_epoch, mae_d_epoch)






def evaluate(model,data_iter,params: Params,loss):

    model.eval()

    loss_matrix = torch.zeros((1,3))
    # MAE accumulators (norm + denorm) -- averaged at the end of evaluation.
    mae_n_sum, mae_d_sum, mae_w_sum = 0.0, 0.0, 0.0

    with torch.no_grad():

        all_labels = torch.tensor([]).to(params.device)
        all_preds= torch.tensor([]).to(params.device)  # 每个样本的均值、
        all_epistemic_uncertainty =torch.tensor([]).to(params.device)
        all_aleatoric_uncertainty =torch.tensor([]).to(params.device)
        all_mask_weights=torch.tensor([]).to(params.device)
        # all_pred_stds = torch.tensor([]).to(params.device)   # 每个样本的标准差

        for i, (test_batch) in enumerate(tqdm(data_iter)):

            test_batch= test_batch.to(torch.float32).to(params.device)

            # ABLATION-SHUFFLE-TIME-EVAL: match train-time perturbation if enabled
            if getattr(params, 'shuffle_time_eval', False):
                _B, _T, _C = test_batch.shape
                if _T > 1:
                    _perms = torch.stack([torch.randperm(_T, device=params.device) for _ in range(_B)])
                    _idx_b = _perms.unsqueeze(-1).expand(-1, -1, _C)
                    test_batch = torch.gather(test_batch, 1, _idx_b)

            labels = test_batch.clone()

            # Drop-ratio range now read from params (mask_min_drop_ratio / mask_max_drop_ratio).
            masked_batch, mask, dropped_indices, drop_ratio = mask_random_sequences(
                test_batch,
                min_drop_ratio=params.mask_min_drop_ratio,
                max_drop_ratio=params.mask_max_drop_ratio,
                # fixed_unselect_idx=unselected_indices
            )
            # 只计算掩码部分的损失
            # 1. 创建掩码权重矩阵（掩码位置为1，非掩码位置为0）
            mask_weights = 1 - mask  # 因为mask中掩码位置是0，非掩码位置是1

            # 将掩码标识拼接为额外特征维度
            extended_batch = torch.cat([masked_batch, mask], dim=-1)

            # MC-dropout switch driven by params.mc_predict (set in JSON).
            # Unified with the validate() path: both follow the same rule.
            #   - mc_predict == 1  -> plain deterministic forward
            #   - mc_predict != 1  -> if model exposes .mc_predict, use it
            #                         with n_samples=params.mc_predict;
            #                         otherwise fall back to plain forward.
            if params.mc_predict == 1 or not hasattr(model, 'mc_predict'):
                # PATCH-ACCEL-5: AMP-BF16 also for predict forward
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    results = model(extended_batch)
            else:
                results = model.mc_predict(extended_batch, n_samples=params.mc_predict)

            l1= loss(results['mean'], results['aleatoric_logvar'], labels)

            # 3. 只对掩码位置计算损失
            l1 = (l1 * mask_weights).sum() / (mask_weights.sum() + 1e-8)

            l2 = params.beta*model.kl_loss() /len(test_batch)

            l = l1+ l2

            loss_matrix = loss_matrix+torch.tensor([[l.item(), l1.item(), l2.item()]])

            # MAE bookkeeping (no_grad already in effect).
            resid = results['mean'] - labels
            sn, sd, sw = _mae_pair_accum(resid, mask_weights, params.scale)
            mae_n_sum += sn
            mae_d_sum += sd
            mae_w_sum += sw

            if i==0:

                pass  # PATCH-SKIP-PLOT: plot(results, labels, dropped_indices, params, 'test')

            all_labels = torch.cat((all_labels, labels[:,:,]), dim=0)
            all_preds = torch.cat((all_preds, results['mean'][:,:,]), dim=0)
            all_mask_weights = torch.cat((all_mask_weights , mask_weights), dim=0)

            if 'epistemic_uncertainty' in results:

                pass  # PATCH-SKIP-CAT: all_epistemic_uncertainty=torch.cat(...)

            # PATCH-SKIP-CAT: all_aleatoric_uncertainty=torch.cat(...)

        aleatoric_mean ,aleatoric_mean_feature = nanmean_ignore_zeros(all_aleatoric_uncertainty, dim=0)
        epistemic_mean, epistemic_mean_feature = nanmean_ignore_zeros(all_epistemic_uncertainty, dim=0)
        mae_n_epoch = mae_n_sum / max(mae_w_sum, 1e-8)
        mae_d_epoch = mae_d_sum / max(mae_w_sum, 1e-8)
        print(f"Test_Loss={loss_matrix[-1]/len(data_iter)}  "
              f"all_epistemic_uncertainty={epistemic_mean}  "
              f"all_aleatoric_uncertainty={aleatoric_mean}  "
              f"Test_MAE_norm={mae_n_epoch:.6f}  Test_MAE_denorm={mae_d_epoch:.6f}")


    return (loss_matrix,
            torch.tensor([[epistemic_mean.item(), aleatoric_mean.item()]]),
            mae_n_epoch, mae_d_epoch)




def save_model_weights(state, model_dir, epoch, name_prefix="checkpoint"):
    """保存模型的权重。Caller should pass params.ckpt_name (or any tag)
    as name_prefix so different runs don't clobber each other.
    This helper is not currently invoked in train_and_evaluate, but
    is kept for ad-hoc snapshots."""
    model_path = os.path.join(model_dir, f"{name_prefix}_epoch_{epoch}.pt")
    torch.save(state, model_path)
    print(f"Model weights saved at {model_path}")



def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])  

# 1. 读取选中的节点索引
def read_selected_indices(file_path):
    """从文件中读取选中的节点索引，并将索引减1（1-based转0-based）"""
    with open(file_path, 'r') as f:
        # 读取文件内容并转换为整数列表，然后每个索引减1
        indices = [int(line.strip()) - 1 for line in f.readlines() if line.strip()]
    return indices
    # 读取选中的节点索引

if __name__ == "__main__":

    # global selected_indices,unselected_indices
    # selected_indices = read_selected_indices("data/select_idx.txt")
    # unselected_indices = read_selected_indices("data/unselect_idx.txt")

    model_dir = "./output_sensor"
    args = parser.parse_args()
    model_dir = args.model_name
    # Switched to the unified params_new.json so both preprocess
    # (f_stepra_data_preprocess.py) and training read the same file.
    # Fields used here:
    #   - directly:   batch_size, predict_batch, epochs, beta, dropout
    #   - via model:  cov_dim, d_model, dropout, ffn_hidden, n_head,
    #                 n_layers, max_len, noise_std, alpha, gamma, latent_dim
    # Fields used only by preprocessing (window_size, stride_size,
    # predict_stride_size, idx) are present but ignored at train time.
    # JSON resolution:
    #   1. --params-json <path>   (highest priority)
    #   2. <model-name>/params_new.json   (default)
    json_path = args.params_json if args.params_json else \
        os.path.join(model_dir, "params_new.json")
    print(f"[run] params_json = {json_path}")
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    params = Params(json_path)

    # ------------------------------------------------------------------
    # Seed resolution (CLI > JSON 'train_seed' > default 42).
    # Must happen BEFORE model init / DataLoader construction so weight
    # init and shuffle order are reproducible.
    # ------------------------------------------------------------------
    if args.seed is not None:
        params.train_seed = args.seed
    elif not hasattr(params, "train_seed"):
        params.train_seed = 42
    _set_all_seeds(params.train_seed)

    # ------------------------------------------------------------------
    # Append a preprocess-tag suffix to the dataset folder name so that
    # different (window_size, stride_size, predict_stride_size)
    # combinations live in different folders and never overwrite each
    # other. This mirrors what f_stepra_data_preprocess.py writes.
    #
    # Special case for backward compatibility (matches preprocess):
    #   window_size == 25 -> use the legacy un-tagged folder
    #   (the baseline data already on disk).
    # All other window sizes load from the tagged folder:
    #   data/<base>_ws<W>_ss<S>_pss<PS>/...
    # ------------------------------------------------------------------
    base_dataset = args.dataset
    # PATCH-DATASET-FIX: always auto-tag based on (ws, stride, predict_stride).
    # Previously ws==25 was special-cased to legacy untagged folder, which had
    # stride=5 (NOT stride=10), causing ws=25 runs to silently train on 2x samples
    # vs. other ws, making cross-ws comparisons unfair.
    args.dataset = (
        f"{base_dataset}"
        f"_ws{params.window_size}"
        f"_ss{params.stride_size}"
        f"_pss{params.predict_stride_size}"
    )
    print(f"[run] dataset folder = {args.dataset}  (base={base_dataset}, auto-tagged uniformly)")

    data_dir = os.path.join(args.data_folder, args.dataset)
    params.relative_metrics = args.relative_metrics
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, "figures")
    params.dataset = args.dataset

    # ------------------------------------------------------------------
    # Auto-build the per-run folder name. The base prefix comes from
    # params.checkpoint_base (set in params_new.json); key hyperparameters
    # are appended so different settings save to different directories
    # and never overwrite each other.
    #
    # Encoded in the name:
    #   d   = dropout
    #   rmin= mask_min_drop_ratio
    #   rmax= mask_max_drop_ratio
    #   ws  = window_size
    #   nl  = n_layers
    #
    # Example:
    #   checkpoint_base = "mrm_awa_Transformer_add_sensor_mae"
    #   -> ckpt_name = "mrm_awa_Transformer_add_sensor_mae_d0.1_rmin0.0_rmax0.9_ws25_nl2"
    # ------------------------------------------------------------------
    params.ckpt_name = (
        f"{params.checkpoint_base}"
        f"_d{params.dropout}"
        f"_rmin{params.mask_min_drop_ratio}"
        f"_rmax{params.mask_max_drop_ratio}"
        f"_ws{params.window_size}"
        f"_nl{params.n_layers}"
        f"_seed{params.train_seed}"
    )

    # ------------------------------------------------------------------
    # Resolve the experiment output root.
    #   exp_dir = <output_base>/ws<W>/pretrain/seed_<N>/
    # All artefacts (model/, figures/, DeFigs/) go under exp_dir so that
    # multi-seed runs don't overwrite each other and results are
    # organized by window-size + stage + seed.
    # output_base resolution: JSON 'output_base' > legacy --model-name dir
    # ------------------------------------------------------------------
    _output_base = getattr(params, "output_base", model_dir)
    params.exp_dir = os.path.join(
        _output_base,
        f"ws{params.window_size}",
        "pretrain",
        f"seed_{params.train_seed}",
    )
    os.makedirs(params.exp_dir, exist_ok=True)
    params.model_dir = params.exp_dir
    params.plot_dir = os.path.join(params.exp_dir, "figures")
    print(f"[run] exp_dir = {params.exp_dir}")

    # ------------------------------------------------------------------
    # Load per-sensor (mean, std) so we can report MAE in physical units
    # (Cp / de-normalized space) in addition to normalized space.
    # Preprocess saves this at one of two conventional paths -- try both
    # so the script works regardless of which convention is in use.
    # ------------------------------------------------------------------
    _norm_candidates = [
        os.path.join("data", f"{args.dataset}_global_standard_deviation",
                     "data_Norm_global.npy"),
        os.path.join("data", args.dataset, "data_Norm_global.npy"),
    ]
    _norm_path = next((p for p in _norm_candidates if os.path.isfile(p)), None)
    assert _norm_path is not None, (
        f"Cannot locate data_Norm_global.npy. Looked in: {_norm_candidates}. "
        f"Run the preprocess script first."
    )
    _norm_arr = np.load(_norm_path)            # shape (cov_dim, 2)  -> [:, 0]=mean, [:, 1]=std
    params.scale = torch.from_numpy(
        _norm_arr[:, 1].astype(np.float32)
    )                                          # per-sensor std for denorm MAE
    print(f"[run] norm_file = {_norm_path}  (scale shape = {tuple(params.scale.shape)})")
    print(f"[run] ckpt_name = {params.ckpt_name}")
    
    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass

    # use GPU if available
    params.ngpu = torch.cuda.device_count()
    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Move the per-sensor denorm scale onto the same device as forward passes,
    # so _mae_pair_accum can broadcast it without a CPU<->GPU sync per batch.
    params.scale = params.scale.to(params.device)

    from network.Transformer import Transformer

    model = Transformer(params).to(params.device)
    model.apply(xavier_init_weights)

    train_set = TrainDataset_X_and_label(data_dir, args.dataset)
    valid_set = ValidDataset_X_and_label(data_dir, args.dataset)
    test_set = TestDataset_X_and_label(data_dir, args.dataset)


    train_loader = DataLoader(train_set, batch_size=params.batch_size,shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=params.predict_batch, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, shuffle=False, num_workers=1)


    print('train_loader=',len(train_loader))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    # --------------------------------------------------------------------
    # NOTE on scheduler:
    #   The actual LR controller is PATCH-COSINE inside train_and_evaluate
    #   (around line 510): it rewrites optimizer.param_groups[lr] at the
    #   start of every epoch with a per-stage cosine value.
    #   The ReduceLROnPlateau object below is kept only as the
    #   *plateau detector* for PATCH-PLATEAU stage switching, which reads
    #   `scheduler.num_bad_epochs` to know when validation has plateaued.
    #   Any LR change scheduler.step() would issue is overwritten by
    #   PATCH-COSINE at the next epoch boundary.
    # --------------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',    # tracks val loss (for plateau detection)
    factor=0.5,    # halving factor (no-op for LR; PATCH-COSINE overrides)
    patience=6,    # plateau threshold used by PATCH-PLATEAU via num_bad_epochs
    min_lr=1e-6,   # PATCH-MIN-LR
    verbose=True   # prints LR adjustments (their effect is overwritten by PATCH-COSINE)
    )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = count_parameters(model) 
    print("总参数量:", total_params)

    train_and_evaluate(model,train_loader,valid_loader, test_loader ,params,optimizer,scheduler)

    
