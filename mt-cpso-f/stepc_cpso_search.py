"""
Enhanced PSO sensor selection on top of a trained Transformer.

Reads
-----
data/<dataset>/data_Norm_global.npy     (mean/std per sensor)
<model-name>/params_Transformer_all_place.json
<model-name>/model/mrm_awa_Transformer_add_sensor_mae/checkpoint.pt

Writes (per run)
----------------
<out-dir>/
    optimized_sensor_indices.txt
    optimized_sensor_indices_1based.txt
    convergence.png                       (ONE image, overwritten each iter)
    convergence_history.txt
    best_indices_matrix_{max_iter}x{select_num}.npy
    best_indices_matrix_{max_iter}x{select_num}.txt
    summary.txt

Key changes vs the original:
1) Fitness evaluation NEVER uses mc_predict; goes straight through
   `self.model(x)` with `@torch.no_grad()` AND `model.eval()` AND explicit
   Dropout-submodule .eval(). This makes the per-(sensor-set) MAE
   deterministic, since dropout is fully disabled.
   ── note: `torch.no_grad()` alone does NOT disable dropout; dropout is
   controlled by train/eval mode. Both must be set.
2) Fitness is computed on the VALIDATION set, not the test set, to avoid
   test leakage during sensor search.
3) After a restart, the swarm is immediately re-evaluated and a SECOND
   convergence point is recorded for the same epoch, so mean_MAE / Dk
   never contain inf and the convergence plot shows the restart kick.
4) Stagnation criterion is a hyperparameter:
       - 'mae'      : strict MAE decrease (default)
       - 'indices'  : change of selected-sensor index set
5) Only ONE convergence image is kept (overwritten each iter).
6) Designed to be imported: call `run_enhanced_pso_optimization(...)`
   from another script. A `__main__` block is kept for standalone use.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# CLI (only used when run as a script; runner scripts can ignore this)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dataset", default="awa_sensor_random_mask_all_place")
parser.add_argument("--data-folder", default="data")
parser.add_argument("--model-name", default="output_sensor")
parser.add_argument("--relative-metrics", action="store_true")
parser.add_argument("--save-best", action="store_true")
parser.add_argument("--restore-file", default=None)
parser.add_argument("--beta", type=float, default=10)
parser.add_argument("--gamma", type=float, default=10)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--select-num", type=int, default=100)
parser.add_argument("--n-particles", type=int, default=60)
parser.add_argument("--max-iter", type=int, default=200)
parser.add_argument("--early-stop", type=int, default=15)
parser.add_argument(
    "--stagnation-criterion",
    choices=["mae", "indices"],
    default="mae",
    help="How to decide 'no improvement': 'mae' = global_best MAE strictly "
         "decreased; 'indices' = the selected sensor index set changed.",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--out-dir", default="f_pso_multiseed_select100")
args, _unknown = parser.parse_known_args()


# ===========================================================================
# Utilities
# ===========================================================================
class Params:
    """Loads hyperparameters from a JSON file."""

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def set_seed(seed=42):
    """Set all RNG seeds for reproducibility.
    NOTE: returns the seed for caller convenience.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[set_seed] seed={seed}, cudnn.deterministic=True")
    return seed


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _force_dropout_eval(model):
    """Belt-and-suspenders: explicitly put every Dropout submodule in eval.
    Needed because mc_predict-style code can leave Dropout in train mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()


# ===========================================================================
# Enhanced PSO
# ===========================================================================
class EnhancedSensorPSO:
    """PSO-based sensor selection.

    Parameters
    ----------
    model : nn.Module
        The trained Transformer; must return a dict with key 'mean' from forward.
    eval_loader : DataLoader
        Validation-set loader used for fitness evaluation.
    params : object
        Must expose `params.device`.
    total_sensors : int
    select_num : int
    n_particles, max_iter : int
    out_dir : str | Path
        Output directory for this run.
    seed : int | None
        If given, drives a private np.random.Generator that controls every
        swarm-side random choice. Allows reproducible per-run results
        independent of any external RNG state.
    stagnation_criterion : {'mae', 'indices'}
        How to decide "no improvement" for the early-stop / restart counter.
        Default 'mae'.
    early_stop : int
    norm_file_path : str | Path | None
        Path to data_Norm_global.npy. If None, falls back to the original
        hard-coded location.
    """

    def __init__(
        self,
        model,
        eval_loader,
        params,
        total_sensors=1014,
        select_num=100,
        n_particles=20,
        max_iter=50,
        out_dir="f_pso_results_add_sensor_denorm_100",
        seed=None,
        stagnation_criterion="mae",
        early_stop=15,
        norm_file_path=None,
        forbidden_indices=None,   # PATCH-CPSO-FORBIDDEN
    ):
        self.model = model
        self.eval_loader = eval_loader
        self.params = params
        self.total_sensors = total_sensors
        self.select_num = select_num
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.early_stop = early_stop
        assert stagnation_criterion in ("mae", "indices")
        self.stagnation_criterion = stagnation_criterion

        # PATCH-CPSO-FORBIDDEN: sensor indices that PSO must never select.
        if forbidden_indices is None:
            self.forbidden_idx = np.array([], dtype=np.int64)
        else:
            self.forbidden_idx = np.asarray(forbidden_indices, dtype=np.int64).ravel()
        if len(self.forbidden_idx) > 0:
            print(f"[CPSO-FORBIDDEN] {len(self.forbidden_idx)} sensors forbidden: "
                  f"{self.forbidden_idx.tolist()}")

        # mean/std for de-normalisation
        if norm_file_path is None:
            # ws-aware default: pick per-sensor norm matching training window size.
            # ws=25 falls back to legacy un-tagged folder (per-sensor std, same numerics).
            _ws = getattr(self.params, "window_size", 100)
            if _ws == 25:
                _norm_dir = "awa_sensor_random_mask_all_place"
            elif _ws == 100:
                _norm_dir = "awa_sensor_random_mask_all_place_ws100_ss20_pss100"
            elif _ws == 200:
                _norm_dir = "awa_sensor_random_mask_all_place_ws200_ss40_pss200"
            elif _ws == 300:
                _norm_dir = "awa_sensor_random_mask_all_place_ws300_ss60_pss300"
            else:
                raise ValueError(f"No default norm path for window_size={_ws}; pass --norm-file explicitly.")
            norm_file_path = f"data/{_norm_dir}/data_Norm_global.npy"
        global_mean_std = torch.from_numpy(np.load(norm_file_path)).to(self.params.device)
        self.global_mean = global_mean_std[:, 0]
        self.global_std = global_mean_std[:, 1]

        # PSO hyperparameters
        self.w_start = 0.9
        self.w_end = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.v_max = 2.0
        self.restart_threshold = 10
        self.diversity_threshold = 0.1
        self.elite_ratio = 0.2

        # Private RNG: governs swarm-side random ops only. Independent of any
        # global numpy seed manipulations elsewhere.
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # History (best_hist / mean_hist / div_hist share `epoch_hist` x-axis).
        # On restart epochs we append twice so a single x value gets two y
        # values -- captured as a vertical "kick" in the plot.
        self.best_hist = []
        self.mean_hist = []
        self.div_hist = []
        self.epoch_hist = []
        self.restart_pts = []
        self.indices_hist = []
        self.global_best_position = None

        # Output directory
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # PATCH-CPSO-ACCEL-2: preload entire valid set onto GPU once.
        # Avoids re-iterating the dataloader (and CPU->GPU copies) on every
        # particle evaluation. The valid set is small enough (<0.5 GB BF16).
        _force_dropout_eval(self.model)
        _valid_chunks = []
        for _b in self.eval_loader:
            _valid_chunks.append(_b.to(torch.float32).to(self.params.device, non_blocking=True))
        self.valid_gpu = torch.cat(_valid_chunks, dim=0) if _valid_chunks else None
        del _valid_chunks
        print(f"[CPSO-ACCEL] preloaded valid set onto GPU: shape={tuple(self.valid_gpu.shape)}")

    # ---------- particle <-> sensor set ------------------------------------
    def particle_to_sensor_indices(self, particle):
        """Map a continuous particle to a sorted list of `select_num` sensor IDs.

        PATCH-CPSO-FORBIDDEN: forbidden indices have prob = -inf so argpartition
        never picks them, regardless of the particle's continuous value there.
        """
        prob = 1.0 / (1.0 + np.exp(-particle))
        if len(self.forbidden_idx) > 0:
            prob = prob.copy()
            prob[self.forbidden_idx] = -np.inf
        idx = np.argpartition(prob, -self.select_num)[-self.select_num:]
        return sorted(idx.tolist())

    # ---------- deterministic fitness --------------------------------------
    @torch.no_grad()
    def evaluate_sensor_selection(self, selected_indices):
        """Compute MAE on the validation set for one sensor selection.

        PATCH-CPSO-ACCEL-1+2:
        - Reuses preloaded GPU valid set (self.valid_gpu) instead of iterating
          the dataloader on every call.
        - Forward pass under BF16 AMP autocast (matches training precision and
          gives 1.5-2x speedup on RTX 4090).
        """
        sel = torch.as_tensor(selected_indices, device=self.params.device, dtype=torch.long)

        # Use preloaded GPU tensor; chunk by predict_batch to control mem.
        chunk = int(getattr(self.params, "predict_batch", 256))
        total_err = 0.0
        total_w = 0.0
        N = self.valid_gpu.shape[0]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, N, chunk):
                batch = self.valid_gpu[start:start + chunk]
                labels = batch
                mask = torch.zeros_like(batch)
                mask[:, :, sel] = 1
                x_in = torch.cat([batch * mask, mask], dim=-1)

                out = self.model(x_in)
                preds = out["mean"].to(torch.float32)  # cast back for denorm math

                mask_w = 1 - mask
                B, T, _ = labels.shape
                mean_b = self.global_mean.view(1, 1, -1).expand(B, T, -1)
                std_b = self.global_std.view(1, 1, -1).expand(B, T, -1)
                preds_denorm = preds * std_b + mean_b
                lab_denorm = labels * std_b + mean_b

                diff = (preds_denorm - lab_denorm).abs() * mask_w
                total_err += diff.sum().item()
                total_w += mask_w.sum().item()
        return total_err / max(total_w, 1e-8)

    def evaluate_particle(self, particle):
        idx = self.particle_to_sensor_indices(particle)
        if len(set(idx)) != self.select_num:
            return float("inf")
        return self.evaluate_sensor_selection(idx)

    @torch.no_grad()
    def evaluate_particles_batch(self, positions_list, particles_per_chunk=5):
        """Evaluate N particles in chunks of `particles_per_chunk`.

        For each chunk, we tile the GPU valid set along the batch axis
        with the chunk's masks side-by-side, then do ONE forward per
        valid-chunk per particle-chunk. With particles_per_chunk=5 and
        chunk=256, the model sees a fat batch of 5*256=1280 windows.

        @torch.no_grad() prevents autograd-graph accumulation across
        the particle loop (which was the OOM cause in earlier drafts).

        Returns: list[float] of MAEs in the same order as positions_list.
        Invalid (duplicate-sensor) particles get inf.
        """
        N_p = len(positions_list)
        sels = []
        valid = []
        for pos in positions_list:
            idx = self.particle_to_sensor_indices(pos)
            if len(set(idx)) != self.select_num:
                sels.append(None)
                valid.append(False)
            else:
                sels.append(torch.as_tensor(idx, device=self.params.device, dtype=torch.long))
                valid.append(True)
        scores = [float("inf")] * N_p
        good = [i for i, ok in enumerate(valid) if ok]
        if not good:
            return scores

        chunk = int(getattr(self.params, "predict_batch", 256))
        N = self.valid_gpu.shape[0]
        F = self.valid_gpu.shape[-1]
        T = self.valid_gpu.shape[1]
        mean_b = self.global_mean.view(1, 1, -1)
        std_b  = self.global_std.view(1, 1, -1)

        for p_start in range(0, len(good), particles_per_chunk):
            pi_batch = good[p_start : p_start + particles_per_chunk]
            P = len(pi_batch)
            tot_err = [0.0] * P
            tot_w   = [0.0] * P

            for s in range(0, N, chunk):
                batch = self.valid_gpu[s : s + chunk]    # (B, T, F)
                B = batch.shape[0]

                # Build (P*B, T, F) by tiling batch P times, then apply
                # each particle's mask.
                tiled = batch.unsqueeze(0).expand(P, -1, -1, -1).reshape(P * B, T, F)
                masks = torch.zeros_like(tiled)
                for k, pi in enumerate(pi_batch):
                    masks[k * B : (k + 1) * B, :, sels[pi]] = 1
                x_in = torch.cat([tiled * masks, masks], dim=-1)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self.model(x_in)
                preds = out["mean"].float()              # (P*B, T, F)
                mask_w = 1 - masks
                # Denormalise
                preds_d = preds * std_b + mean_b
                lab_d   = tiled * std_b + mean_b
                diff = (preds_d - lab_d).abs() * mask_w
                # Sum into per-particle accumulators
                for k in range(P):
                    e = diff[k * B : (k + 1) * B].sum().item()
                    w = mask_w[k * B : (k + 1) * B].sum().item()
                    tot_err[k] += e
                    tot_w[k]   += w

            for k, pi in enumerate(pi_batch):
                scores[pi] = tot_err[k] / max(tot_w[k], 1e-12)
        return scores


    # ---------- swarm ops --------------------------------------------------
    def calculate_swarm_diversity(self, positions):
        n = len(positions)
        if n < 2:
            return 0.0
        s, k = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                s += float(np.linalg.norm(positions[i] - positions[j]))
                k += 1
        return s / k

    def adaptive_inertia_weight(self, iteration):
        return self.w_start - (self.w_start - self.w_end) * (iteration / self.max_iter)

    def initialize_swarm(self):
        """Uniform-in-[-2,2] init for every particle, in `total_sensors` dims.
        Using a single uniform call keeps the swarm bit-for-bit reproducible
        across different select_num values for the same seed (same RNG draws).
        """
        return self.rng.uniform(-2, 2, size=(self.n_particles, self.total_sensors))

    def maybe_restart(self, positions, velocities, pbest_positions, pbest_scores, stagnation_count):
        """Returns (new_stagnation, restarted_flag).

        PATCH-CPSO-N1-RESTART: with n_particles <= 1 there is no diversity to
        restart (and argpartition would fail), so we simply reset the counter.
        """
        if stagnation_count < self.restart_threshold:
            return stagnation_count, False
        if self.n_particles <= 1:
            # Reset counter and skip restart logic (no swarm to diversify)
            return 0, False

        print("\n[restart] stagnation detected -> reseeding non-elite particles")
        elite_n = max(1, int(self.n_particles * self.elite_ratio))
        elite = np.argpartition(pbest_scores, elite_n)[:elite_n]
        for i in range(self.n_particles):
            if i in elite:
                continue
            if self.rng.random() < 0.8:
                positions[i] = self.rng.uniform(-2, 2, self.total_sensors)
            else:
                e = self.rng.choice(elite)
                positions[i] = pbest_positions[e] + self.rng.normal(0, 1, self.total_sensors)
            velocities[i] = self.rng.uniform(-1, 1, self.total_sensors)
            # Mark for re-evaluation; this is what we explicitly fix below.
            pbest_scores[i] = float("inf")

        # Mark restart at the current epoch in epoch_hist coords.
        if len(self.epoch_hist) > 0:
            self.restart_pts.append(self.epoch_hist[-1])
        return 0, True

    # ---------- main loop --------------------------------------------------
    def optimize(self):
        print("=" * 60)
        print(f"PSO start: n_particles={self.n_particles}  max_iter={self.max_iter}")
        print(f"select_num={self.select_num}/{self.total_sensors}  "
              f"seed={self.seed}  criterion={self.stagnation_criterion}")
        print(f"out_dir={self.out_dir}")
        print("=" * 60)
        t0 = time.time()

        positions = self.initialize_swarm()
        velocities = self.rng.uniform(-1, 1, (self.n_particles, self.total_sensors))

        pbest_positions = positions.copy()
        pbest_scores = np.array(self.evaluate_particles_batch(list(positions)))

        gbest_idx = int(np.argmin(pbest_scores))
        self.global_best_position = pbest_positions[gbest_idx].copy()
        gbest_score = float(pbest_scores[gbest_idx])

        print(f"[init] best MAE = {gbest_score:.5f}")

        # Initial history point
        self.best_hist = [gbest_score]
        self.mean_hist = [float(np.mean(pbest_scores))]
        self.div_hist = [self.calculate_swarm_diversity(positions)]
        self.epoch_hist = [0]
        prev_best_indices = self.particle_to_sensor_indices(self.global_best_position)
        self.indices_hist = [prev_best_indices]
        self.restart_pts = []

        stagnation_count = 0

        for it in range(self.max_iter):
            it_t0 = time.time()
            w = self.adaptive_inertia_weight(it)
            prev_gbest_score = gbest_score

            # Update positions (r1,r2 drawn in particle order to preserve seed determinism)
            for i in range(self.n_particles):
                r1, r2 = self.rng.random(2)
                cog = self.c1 * r1 * (pbest_positions[i] - positions[i])
                soc = self.c2 * r2 * (self.global_best_position - positions[i])
                velocities[i] = np.clip(w * velocities[i] + cog + soc,
                                        -self.v_max, self.v_max)
                positions[i] = positions[i] + velocities[i]

            # Batched evaluation (chunk=5)
            scores = self.evaluate_particles_batch(list(positions))
            for i, s in enumerate(scores):
                if s < pbest_scores[i]:
                    pbest_scores[i] = s
                    pbest_positions[i] = positions[i].copy()
                    if s < gbest_score:
                        gbest_score = s
                        self.global_best_position = positions[i].copy()

            # Stagnation decision -- chosen by hyperparameter.
            curr_best_indices = self.particle_to_sensor_indices(self.global_best_position)
            if self.stagnation_criterion == "mae":
                improved = gbest_score < prev_gbest_score
            else:  # 'indices'
                improved = set(curr_best_indices) != set(prev_best_indices)
            stagnation_count = 0 if improved else stagnation_count + 1
            prev_best_indices = curr_best_indices

            # --- Record convergence BEFORE restart ------------------------
            # If we restart first and then record, mean(pbest_scores) gets
            # polluted by the just-injected inf values. Record-then-restart
            # preserves the true state at the end of this iteration.
            self.best_hist.append(gbest_score)
            self.mean_hist.append(float(np.mean(pbest_scores)))
            self.div_hist.append(self.calculate_swarm_diversity(positions))
            self.epoch_hist.append(it + 1)
            self.indices_hist.append(curr_best_indices)

            tag = "improved" if improved else "stale"
            print(f"  iter {it+1:3d}/{self.max_iter}  best={gbest_score:.5f}  "
                  f"mean={self.mean_hist[-1]:.5f}  "
                  f"div={self.div_hist[-1]:.3f}  [{tag}]  "
                  f"({time.time()-it_t0:.1f}s)")

            # --- Restart AFTER recording ----------------------------------
            stagnation_count, restarted = self.maybe_restart(
                positions, velocities, pbest_positions, pbest_scores, stagnation_count
            )

            if restarted:
                # Immediately re-evaluate the inf'ed particles, so the next
                # iteration starts from a well-defined state AND we can
                # record a second data point for this same epoch.
                for i in range(self.n_particles):
                    if not np.isfinite(pbest_scores[i]):
                        s = self.evaluate_particle(positions[i])
                        pbest_scores[i] = s
                        pbest_positions[i] = positions[i].copy()
                        if s < gbest_score:
                            gbest_score = s
                            self.global_best_position = positions[i].copy()

                # Second history point for the SAME epoch index.
                self.best_hist.append(gbest_score)
                self.mean_hist.append(float(np.mean(pbest_scores)))
                self.div_hist.append(self.calculate_swarm_diversity(positions))
                self.epoch_hist.append(it + 1)  # duplicate epoch -> vertical kick
                post_restart_indices = self.particle_to_sensor_indices(self.global_best_position)
                self.indices_hist.append(post_restart_indices)
                prev_best_indices = post_restart_indices
                print(f"  post-restart      best={gbest_score:.5f}  "
                      f"mean={self.mean_hist[-1]:.5f}  "
                      f"div={self.div_hist[-1]:.3f}")

            # ONE convergence image, overwritten each iter.
            self._save_convergence_plot()

            if stagnation_count >= self.early_stop:
                print(f"[early stop] no improvement for {stagnation_count} iters "
                      f"(criterion={self.stagnation_criterion})")
                break

        total_time = time.time() - t0
        best_indices = self.particle_to_sensor_indices(self.global_best_position)
        final_mae = self.evaluate_sensor_selection(best_indices)

        print(f"\n[done] total time {total_time:.1f}s | final MAE {final_mae:.5f} "
              f"| selected {len(best_indices)} sensors")

        self._save_detailed_results(best_indices, final_mae, total_time)
        return best_indices, final_mae, self.global_best_position

    # ---------- IO ---------------------------------------------------------
    def _save_convergence_plot(self):
        x = self.epoch_hist
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

        ax1.plot(x, self.best_hist, "b-", lw=2, label="Best MAE")
        ax1.plot(x, self.mean_hist, "g-", lw=1, alpha=0.7, label="Mean MAE")
        for r in self.restart_pts:
            ax1.axvline(r, color="r", ls="--", alpha=0.4)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("MAE")
        ax1.set_title("PSO Convergence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(x, self.div_hist, "r-", lw=2, label="Diversity Dk")
        for r in self.restart_pts:
            ax2.axvline(r, color="r", ls="--", alpha=0.4)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Diversity")
        ax2.set_title("Swarm diversity")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.out_dir / "convergence.png", dpi=200, bbox_inches="tight")
        plt.close()

    def _save_detailed_results(self, best_indices, final_mae, elapsed):
        np.savetxt(self.out_dir / "optimized_sensor_indices.txt",
                   np.asarray(best_indices, dtype=int), fmt="%d")
        np.savetxt(self.out_dir / "optimized_sensor_indices_1based.txt",
                   np.asarray(best_indices, dtype=int) + 1, fmt="%d")
        np.savetxt(self.out_dir / "convergence_history.txt",
                   np.column_stack([self.epoch_hist, self.best_hist,
                                    self.mean_hist, self.div_hist]),
                   fmt="%.6f",
                   header="epoch best_MAE mean_MAE diversity")
        self._save_best_indices_matrix()
        with open(self.out_dir / "summary.txt", "w") as f:
            f.write("PSO sensor-selection summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"seed                : {self.seed}\n")
            f.write(f"stagnation_criterion: {self.stagnation_criterion}\n")
            f.write(f"select_num          : {self.select_num}\n")
            f.write(f"n_particles         : {self.n_particles}\n")
            f.write(f"max_iter            : {self.max_iter}\n")
            f.write(f"iterations_recorded : {len(self.best_hist) - 1}\n")
            f.write(f"restarts            : {len(self.restart_pts)}\n")
            f.write(f"final_MAE           : {final_mae:.6f}\n")
            f.write(f"elapsed_sec         : {elapsed:.1f}\n")
            f.write(f"best_indices        : {best_indices}\n")
        print(f"[save] results -> {self.out_dir}")

    def _save_best_indices_matrix(self):
        """Save the best-particle index history.

        Produces two artefacts per run:
          - best_indices_matrix_full_{N}x{K}.{txt,npy}   FULL untruncated trace.
              N = exact number of recorded points (init + every iter + restart).
              Use this for any post-hoc analysis (e.g. plotting how the chosen
              taps evolved). No information is lost.
          - best_indices_matrix_{max_iter}x{K}.{txt,npy} LEGACY truncated/padded
              version kept for backward compatibility with older downstream
              scripts that expect a fixed (max_iter, K) shape.
        """
        if not self.indices_hist:
            print("[warn] no indices history to save")
            return
        rows_all = np.asarray(list(self.indices_hist), dtype=int)
        n_full = rows_all.shape[0]

        # NEW: full untruncated trace
        np.savetxt(self.out_dir / f"best_indices_matrix_full_{n_full}x{self.select_num}.txt",
                   rows_all, fmt="%d")
        np.save(self.out_dir / f"best_indices_matrix_full_{n_full}x{self.select_num}.npy", rows_all)
        print(f"[save] FULL trace: {n_full} rows (no truncation)")

        # LEGACY: also save truncated/padded fixed-shape version for backward compat
        matrix_size = self.max_iter
        rows = rows_all.copy()
        if rows.shape[0] >= matrix_size:
            rows = rows[-matrix_size:]
        else:
            pad_n = matrix_size - rows.shape[0]
            rows = np.vstack([rows, np.tile(rows[-1:], (pad_n, 1))])
        np.savetxt(self.out_dir / f"best_indices_matrix_{matrix_size}x{self.select_num}.txt",
                   rows, fmt="%d")
        np.save(self.out_dir / f"best_indices_matrix_{matrix_size}x{self.select_num}.npy", rows)


# ===========================================================================
# Public entry point
# ===========================================================================
def run_enhanced_pso_optimization(
    model,
    eval_loader,
    params,
    total_sensors=1014,
    select_num=100,
    n_particles=60,
    max_iter=200,
    out_dir="f_pso_results_add_sensor_denorm_100",
    seed=None,
    stagnation_criterion="mae",
    early_stop=15,
    norm_file_path=None,
):
    """External callable entry point. Returns (best_indices, best_mae)."""
    print("=" * 50)
    print("Enhanced PSO sensor-selection")
    print("=" * 50)
    optimizer = EnhancedSensorPSO(
        model=model,
        eval_loader=eval_loader,
        params=params,
        total_sensors=total_sensors,
        select_num=select_num,
        n_particles=n_particles,
        max_iter=max_iter,
        out_dir=out_dir,
        seed=seed,
        stagnation_criterion=stagnation_criterion,
        early_stop=early_stop,
        norm_file_path=norm_file_path,
    )
    best_indices, best_mae, _ = optimizer.optimize()
    return best_indices, best_mae


# ===========================================================================
# Single-run script entry (kept for backward compatibility)
# ===========================================================================
if __name__ == "__main__":
    # Local import: only resolved when this file is run as a script
    # (so external importers don't need the project's dataloader module).
    from dataloader import (
        TrainDataset_X_and_label,
        ValidDataset_X_and_label,
        TestDataset_X_and_label,
    )

    if args.seed is not None:
        set_seed(args.seed)

    model_dir = args.model_name
    json_path = os.path.join(model_dir, "params_Transformer_all_place.json")
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    params = Params(json_path)
    params.relative_metrics = args.relative_metrics
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, "figures", "mrm_awa_Transformer_add_sensor_mae")
    params.dataset = args.dataset

    valid_set = ValidDataset_X_and_label(data_dir, args.dataset)
    # use valid for fitness; test only for final independent evaluation if needed
    valid_loader = DataLoader(valid_set, batch_size=params.predict_batch,
                              shuffle=False, num_workers=4)

    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from network.Transformer import Transformer
    model = Transformer(params).to(params.device)
    model_path = os.path.join(model_dir, "model", "mrm_awa_Transformer_add_sensor_mae",
                              "checkpoint.pt")
    model.load_state_dict(torch.load(model_path, map_location=params.device))
    _force_dropout_eval(model)
    print(f"[main] model loaded ({count_parameters(model):,} params)")

    best_indices, best_mae = run_enhanced_pso_optimization(
        model=model,
        eval_loader=valid_loader,
        params=params,
        total_sensors=getattr(params, "cov_dim", 1014),
        select_num=args.select_num,
        n_particles=args.n_particles,
        max_iter=args.max_iter,
        out_dir=args.out_dir,
        seed=args.seed,
        stagnation_criterion=args.stagnation_criterion,
        early_stop=args.early_stop,
    )
    print(f"[main] best MAE = {best_mae:.5f}, indices count = {len(best_indices)}")
