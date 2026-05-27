from __future__ import annotations

import os
import pickle
import tarfile
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Cifar10Data:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _load_batch_from_tar(tar: tarfile.TarFile, member_name: str) -> Tuple[np.ndarray, np.ndarray]:
    member = tar.getmember(member_name)
    with tar.extractfile(member) as f:
        if f is None:
            raise FileNotFoundError(f"Could not extract {member_name} from tar archive.")
        batch = pickle.load(f, encoding="bytes")

    data = batch[b"data"]
    labels = batch[b"labels"]

    x = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
    return x, y


def load_cifar10_from_tar(tar_gz_path: str = ".keras/datasets/cifar-10-python.tar.gz") -> Cifar10Data:
    """
    Load CIFAR-10 from a local tar.gz archive (no network).

    This mirrors `tf.keras.datasets.cifar10.load_data()` output shapes:
    - x_train: (50000, 32, 32, 3) uint8
    - y_train: (50000, 1) int64
    - x_test : (10000, 32, 32, 3) uint8
    - y_test : (10000, 1) int64
    """
    if not os.path.exists(tar_gz_path):
        raise FileNotFoundError(
            f"Missing CIFAR-10 archive at {tar_gz_path}. "
            "Download it once and place it there."
        )

    with tarfile.open(tar_gz_path, "r:gz") as tar:
        train_parts = []
        train_labels = []
        for i in range(1, 6):
            x_part, y_part = _load_batch_from_tar(tar, f"cifar-10-batches-py/data_batch_{i}")
            train_parts.append(x_part)
            train_labels.append(y_part)

        x_train = np.concatenate(train_parts, axis=0).astype(np.uint8, copy=False)
        y_train = np.concatenate(train_labels, axis=0).astype(np.int64, copy=False)

        x_test, y_test = _load_batch_from_tar(tar, "cifar-10-batches-py/test_batch")
        x_test = x_test.astype(np.uint8, copy=False)
        y_test = y_test.astype(np.int64, copy=False)

    return Cifar10Data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

