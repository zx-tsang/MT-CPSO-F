"""
Shared utilities for the POD-Transformer baseline.

Exposes:
    Params         - lightweight YAML/JSON hyperparameter container.
    EarlyStopping  - standard single-metric early stopping with checkpointing.
"""
import json
import logging
import os

import numpy as np
import torch
import yaml

logger = logging.getLogger("Transformer.Utils")


class Params:
    """Hyperparameter container, supports both .yaml and .json."""

    def __init__(self, path):
        path = str(path)
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) if path.endswith((".yaml", ".yml")) else json.load(f)
        self.__dict__.update(d)

    def save(self, path):
        path = str(path)
        with open(path, "w", encoding="utf-8") as f:
            if path.endswith((".yaml", ".yml")):
                yaml.safe_dump(self.__dict__, f, sort_keys=False, allow_unicode=True)
            else:
                json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    @property
    def dict(self):
        return self.__dict__


class EarlyStopping:
    """Stop training when validation loss has not improved for `patience` epochs."""

    def __init__(self, patience=15, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, state, model_folder, save_name="checkpoint"):
        score = val_loss
        if self.best_score is None or score < self.best_score - self.delta:
            self.best_score = score
            self._save(val_loss, state, model_folder, save_name)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, val_loss, state, model_folder, save_name):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). "
                  f"Saving model to {model_folder}/{save_name}.pt")
        os.makedirs(model_folder, exist_ok=True)
        torch.save(state, os.path.join(model_folder, f"{save_name}.pt"))
        self.val_loss_min = val_loss
