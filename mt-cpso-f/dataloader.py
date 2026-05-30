"""
Dataset wrappers for the MT-CPSO-F pipeline.

Each split is stored as a single ``.npy`` file shaped (N, T, F) at
``<data_path>/<split>_data_<data_name>.npy``. Datasets yield one window
per index.
"""
import logging
import os

import numpy as np
import torch
import json
import yaml

from torch.utils.data import Dataset

logger = logging.getLogger("Transformer.Data")


class TrainDataset_X_and_label(Dataset):
    def __init__(self, data_path, data_name):
        self.data = np.load(os.path.join(data_path, f"train_data_{data_name}.npy"))
        self.train_len = self.data.shape[0]
        logger.info(f"train_len: {self.train_len}")
        logger.info(f"building datasets from {data_path}...")

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return self.data[index, :, :]


class ValidDataset_X_and_label(Dataset):
    def __init__(self, data_path, data_name):
        self.data = np.load(os.path.join(data_path, f"valid_data_{data_name}.npy"))
        self.valid_len = self.data.shape[0]
        logger.info(f"valid_len: {self.valid_len}")
        logger.info(f"building datasets from {data_path}...")

    def __len__(self):
        return self.valid_len

    def __getitem__(self, index):
        return self.data[index, :, :]


class TestDataset_X_and_label(Dataset):
    def __init__(self, data_path, data_name):
        self.data = np.load(os.path.join(data_path, f"test_data_{data_name}.npy"))
        self.test_len = self.data.shape[0]
        logger.info(f"test_len: {self.test_len}")
        logger.info(f"building datasets from {data_path}...")

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return self.data[index, :, :]


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

