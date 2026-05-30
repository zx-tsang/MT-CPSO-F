"""
Multi-seed runner for the enhanced PSO sensor selection.

For each seed in --seeds:
    1. set all RNGs to that seed
    2. run PSO from stepc_cpso_search
    3. save everything to <base-out>/seed_<N>/
After all seeds complete:
    write <base-out>/aggregate_summary.txt with mean/std of final MAE,
    and <base-out>/overlay_convergence.png showing best-MAE curves side-by-side.

Concurrency
-----------
--max-parallel N   (default 1 = serial, backward compatible)
    When N > 1, up to N seeds run simultaneously in separate processes
    via ProcessPoolExecutor with the 'spawn' start method (required for
    CUDA). Each worker loads its OWN copy of the model onto the GPU, so
    GPU memory usage is roughly N x single-run memory. Likewise, the
    forward passes from N workers still serialise on the GPU driver,
    so realistic speedup is sub-linear (typically 1.5x-2.2x for N=3).
    For true linear speedup, use multiple GPUs.

Examples
--------
# Serial (default, same as before)
python run_pso_multi_seed.py --seeds 42 123 456 789 2024

# 3 seeds in parallel on the same GPU
python run_pso_multi_seed.py --seeds 42 123 456 789 2024 --max-parallel 3
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import the refactored module
from stepc_cpso_search import (
    Params,
    set_seed,
    count_parameters,
    _force_dropout_eval,
    run_enhanced_pso_optimization,
)


def parse_args():
    p = argparse.ArgumentParser()
    # data / model
    p.add_argument("--dataset", default="awa_sensor_random_mask_all_place")
    p.add_argument("--data-folder", default="data")
    p.add_argument("--model-name", default="output_sensor")
    p.add_argument("--data-root-prefix", default="",
                   help="Optional path prefix prepended to <data-folder>/<dataset> "
                        "(useful if datasets live outside the repo root). Empty by default.")
    p.add_argument("--ckpt", default=None,
                   help="Override path to checkpoint.pt; default mirrors original.")
    p.add_argument("--norm-file", default=None,
                   help="Override path to data_Norm_global.npy.")
    p.add_argument("--params-json", default=None,
                   help="Override path to the params JSON. Default: "
                        "<model-name>/params_Transformer_all_place.json. "
                        "Use this to point at params_new.json when running "
                        "PSO against a model trained with the new config.")
    # PSO config
    p.add_argument("--total-sensors", type=int, default=1014)
    p.add_argument("--select-num", type=int, default=100)
    p.add_argument("--n-particles", type=int, default=60)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--early-stop", type=int, default=15)
    p.add_argument(
        "--stagnation-criterion",
        choices=["mae", "indices"],
        default="mae",
        help="'mae': global-best MAE strictly decreased; "
             "'indices': selected sensor index set changed.",
    )
    # multi-seed control
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 2024],
                   help="List of seeds. One PSO run per seed.")
    p.add_argument("--max-parallel", type=int, default=1,
                   help="Max number of seeds to run concurrently. 1 = serial "
                        "(default). >1 spawns separate processes; each worker "
                        "loads its OWN model on the GPU, so memory ~N x. "
                        "Realistic speedup is sub-linear on a single GPU.")
    p.add_argument("--base-out", default="f_pso_multiseed_select100",
                   help="Top-level output directory. Each seed gets a subdir.")
    return p.parse_args()


def build_model_and_loader(cfg):
    """Build the model and the validation DataLoader."""
    # Project-specific imports (kept local so this script can be inspected
    # without the project being installed).
    from dataloader import ValidDataset_X_and_label
    from network.Transformer import Transformer

    # JSON resolution: --params-json overrides; default = legacy path.
    json_path = (cfg.params_json if cfg.params_json
                 else os.path.join(cfg.model_name, "params_Transformer_all_place.json"))
    assert os.path.isfile(json_path), f"missing params json: {json_path}"
    print(f"[setup] params_json = {json_path}")
    params = Params(json_path)
    params.model_dir = cfg.model_name
    params.plot_dir = os.path.join(cfg.model_name, "figures",
                                   "mrm_awa_Transformer_add_sensor_mae")
    params.dataset = cfg.dataset
    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(cfg.data_root_prefix, cfg.data_folder, cfg.dataset)
    valid_set = ValidDataset_X_and_label(data_dir, cfg.dataset)
    valid_loader = DataLoader(valid_set, batch_size=params.predict_batch,
                              shuffle=False, num_workers=0)

    model = Transformer(params).to(params.device)
    ckpt_path = cfg.ckpt or os.path.join(
        cfg.model_name, "model", "mrm_awa_Transformer_add_sensor_mae", "checkpoint.pt"
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=params.device))
    _force_dropout_eval(model)
    print(f"[setup] model: {count_parameters(model):,} params  |  ckpt: {ckpt_path}")
    return model, valid_loader, params


def run_one_seed(seed, cfg, model=None, valid_loader=None, params=None):
    """Run PSO for a single seed.

    If model/valid_loader/params are passed in (serial path), reuse them.
    Otherwise (worker process in parallel mode), build a fresh copy locally.
    Returns a summary dict; this function is the unit of work for both
    serial and ProcessPoolExecutor paths.
    """
    set_seed(seed)
    if model is None or valid_loader is None or params is None:
        # Worker process: build its own model + loader on its own CUDA context.
        model, valid_loader, params = build_model_and_loader(cfg)

    out_dir = Path(cfg.base_out) / f"seed_{seed}"
    t0 = time.time()
    best_indices, best_mae = run_enhanced_pso_optimization(
        model=model,
        eval_loader=valid_loader,
        params=params,
        total_sensors=cfg.total_sensors,
        select_num=cfg.select_num,
        n_particles=cfg.n_particles,
        max_iter=cfg.max_iter,
        out_dir=str(out_dir),
        seed=seed,
        stagnation_criterion=cfg.stagnation_criterion,
        early_stop=cfg.early_stop,
        norm_file_path=cfg.norm_file,
    )
    elapsed = time.time() - t0
    return {
        "seed": int(seed),
        "out_dir": str(out_dir),
        "final_mae": float(best_mae),
        "elapsed_sec": float(elapsed),
        "n_indices": int(len(best_indices)),
    }


def overlay_plot(base_out, runs):
    """Plot best-MAE curves from every seed on a single figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for r in runs:
        hist_path = Path(r["out_dir"]) / "convergence_history.txt"
        if not hist_path.exists():
            continue
        data = np.loadtxt(hist_path, skiprows=1)
        if data.ndim == 1:
            data = data[None, :]
        ax.plot(data[:, 0], data[:, 1], lw=1.5,
                label=f"seed={r['seed']}  finalMAE={r['final_mae']:.4f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best MAE")
    ax.set_title("PSO best-MAE across seeds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(base_out) / "overlay_convergence.png", dpi=200, bbox_inches="tight")
    plt.close()


def write_aggregate(base_out, runs, cfg):
    """Save an aggregate summary across all seeds."""
    base_out = Path(base_out)
    finals = np.array([r["final_mae"] for r in runs], dtype=float)
    elapsed = np.array([r["elapsed_sec"] for r in runs], dtype=float)

    with open(base_out / "aggregate_summary.txt", "w") as f:
        f.write("Multi-seed PSO aggregate summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"dataset             : {cfg.dataset}\n")
        f.write(f"select_num          : {cfg.select_num}\n")
        f.write(f"n_particles         : {cfg.n_particles}\n")
        f.write(f"max_iter            : {cfg.max_iter}\n")
        f.write(f"early_stop          : {cfg.early_stop}\n")
        f.write(f"stagnation_criterion: {cfg.stagnation_criterion}\n")
        f.write(f"max_parallel        : {cfg.max_parallel}\n")
        f.write(f"seeds               : {[r['seed'] for r in runs]}\n")
        f.write("-" * 60 + "\n")
        for r in runs:
            f.write(f"  seed={r['seed']:>6d}  finalMAE={r['final_mae']:.6f}  "
                    f"elapsed={r['elapsed_sec']:.1f}s  -> {r['out_dir']}\n")
        f.write("-" * 60 + "\n")
        f.write(f"final_MAE mean      : {finals.mean():.6f}\n")
        f.write(f"final_MAE std       : {finals.std(ddof=0):.6f}\n")
        f.write(f"final_MAE min       : {finals.min():.6f}\n")
        f.write(f"final_MAE max       : {finals.max():.6f}\n")
        f.write(f"elapsed_sec total   : {elapsed.sum():.1f}\n")

    # also dump machine-readable JSON
    with open(base_out / "aggregate_summary.json", "w") as f:
        json.dump({
            "config": {
                "dataset": cfg.dataset,
                "select_num": cfg.select_num,
                "n_particles": cfg.n_particles,
                "max_iter": cfg.max_iter,
                "early_stop": cfg.early_stop,
                "stagnation_criterion": cfg.stagnation_criterion,
                "max_parallel": cfg.max_parallel,
                "seeds": [r["seed"] for r in runs],
            },
            "runs": runs,
            "stats": {
                "mean": float(finals.mean()),
                "std": float(finals.std(ddof=0)),
                "min": float(finals.min()),
                "max": float(finals.max()),
            },
        }, f, indent=2)
    print(f"\n[aggregate] -> {base_out}/aggregate_summary.txt")


def main():
    cfg = parse_args()
    base_out = Path(cfg.base_out)
    base_out.mkdir(parents=True, exist_ok=True)

    t_global = time.time()
    runs = []

    if cfg.max_parallel <= 1:
        # ---------- Serial path (build model once, reuse across seeds) ----
        model, valid_loader, params = build_model_and_loader(cfg)
        for k, seed in enumerate(cfg.seeds):
            print("\n" + "#" * 70)
            print(f"# run {k+1}/{len(cfg.seeds)}  seed={seed}  [serial]")
            print("#" * 70)
            runs.append(run_one_seed(seed, cfg, model, valid_loader, params))

    else:
        # ---------- Parallel path -----------------------------------------
        # 'spawn' context is required for CUDA in subprocesses; 'fork' would
        # share the parent's CUDA context and immediately corrupt it.
        # Each worker calls run_one_seed(seed, cfg) WITHOUT preloaded model,
        # so it builds its own copy in its own process / CUDA context.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")
        max_workers = min(cfg.max_parallel, len(cfg.seeds))

        print("\n" + "#" * 70)
        print(f"# Parallel: {max_workers} workers for {len(cfg.seeds)} seeds")
        print(f"# WARNING: each worker loads its OWN model copy on the GPU.")
        print(f"#          GPU memory needed ~ {max_workers} x single-run memory.")
        print(f"#          If you hit OOM, reduce --max-parallel.")
        print("#" * 70)

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            future_to_seed = {
                ex.submit(run_one_seed, seed, cfg): seed
                for seed in cfg.seeds
            }
            for fut in as_completed(future_to_seed):
                seed = future_to_seed[fut]
                try:
                    result = fut.result()
                    runs.append(result)
                    print(f"[completed] seed={seed}  "
                          f"finalMAE={result['final_mae']:.5f}  "
                          f"elapsed={result['elapsed_sec']:.1f}s")
                except Exception as e:
                    print(f"[FAILED] seed={seed}: {type(e).__name__}: {e}")

        # Sort results by the original seed order so the aggregate is stable
        # regardless of which worker finishes first.
        seed_order = {s: i for i, s in enumerate(cfg.seeds)}
        runs.sort(key=lambda r: seed_order[r["seed"]])

    # Aggregate
    if not runs:
        print("[error] no successful runs to aggregate")
        return
    write_aggregate(base_out, runs, cfg)
    overlay_plot(base_out, runs)
    print(f"\n[all done] {len(runs)} runs in {time.time()-t_global:.1f}s")


if __name__ == "__main__":
    main()
