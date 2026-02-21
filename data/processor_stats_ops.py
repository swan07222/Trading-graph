from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_LABEL_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

def prepare_single_sequence(
    self, df: pd.DataFrame, feature_cols: list[str]
) -> np.ndarray:
    """Alias for prepare_inference_sequence."""
    return self.prepare_inference_sequence(df, feature_cols)


def get_class_distribution(self, y: np.ndarray) -> dict[str, int]:
    """
    Get class distribution for logging.
    Dynamically handles any NUM_CLASSES value.

    FIX BINCOUNT: Guards against negative labels which would crash bincount.
    """
    num_classes = int(CONFIG.NUM_CLASSES)

    if len(y) == 0:
        dist: dict[str, int] = {}
        for i in range(num_classes):
            label_name = _DEFAULT_LABEL_NAMES.get(i, f"CLASS_{i}")
            dist[label_name] = 0
        dist["total"] = 0
        return dist

    y_int = y.astype(int)

    # FIX BINCOUNT: Filter out invalid labels before bincount
    valid_mask = (y_int >= 0) & (y_int < num_classes)
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum()
        log.warning(
            f"get_class_distribution: {invalid_count} labels outside "
            f"[0, {num_classes}) — excluding from distribution"
        )
        y_int = y_int[valid_mask]

    if len(y_int) == 0:
        dist = {}
        for i in range(num_classes):
            label_name = _DEFAULT_LABEL_NAMES.get(i, f"CLASS_{i}")
            dist[label_name] = 0
        dist["total"] = 0
        return dist

    counts = np.bincount(y_int, minlength=num_classes)

    dist = {}
    for i in range(num_classes):
        label_name = _DEFAULT_LABEL_NAMES.get(i, f"CLASS_{i}")
        dist[label_name] = int(counts[i]) if i < len(counts) else 0
    dist["total"] = int(len(y))  # Original length including invalid

    return dist


def get_scaler_info(self) -> dict[str, Any]:
    """Get scaler metadata."""
    with self._lock:
        return {
            "fitted": self._fitted,
            "n_features": self._n_features,
            "fit_samples": self._fit_samples,
            "interval": self._interval,
            "horizon": self._horizon,
            "version": self._scaler_version,
        }
