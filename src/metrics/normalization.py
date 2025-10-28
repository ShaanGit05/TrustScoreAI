# src/metrics/normalization.py
import numpy as np
import pandas as pd
from typing import Union, Tuple
import logging

logger = logging.getLogger("bias_detection.normalization")

class Normalizer:
    """Handles normalization and percentile clipping for bias metrics."""
    
    def __init__(self, p_min: float = 5.0, p_max: float = 95.0):
        self.p_min = p_min
        self.p_max = p_max
        self.fitted = False
        self.actual_min = None
        self.actual_max = None
    
    def fit(self, values: np.ndarray) -> None:
        """Fit the normalizer to data."""
        if len(values) == 0:
            raise ValueError("Cannot fit normalizer with empty data")
        
        self.actual_min = np.percentile(values, self.p_min)
        self.actual_max = np.percentile(values, self.p_max)
        self.fitted = True
        
        logger.debug(f"Fitted normalizer: min={self.actual_min:.4f}, max={self.actual_max:.4f}")
    
    def transform(self, values: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Transform values using fitted normalization."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transformation")
        
        # Clip values to percentile range
        clipped = np.clip(values, self.actual_min, self.actual_max)
        
        # Normalize to [0, 1]
        normalized = (clipped - self.actual_min) / (self.actual_max - self.actual_min)
        
        return normalized
    
    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(values)
        return self.transform(values)

def clip_to_percentile(values: np.ndarray, p_min: float = 5.0, p_max: float = 95.0) -> np.ndarray:
    """Clip values to specified percentiles."""
    v_min = np.percentile(values, p_min)
    v_max = np.percentile(values, p_max)
    return np.clip(values, v_min, v_max)

def normalize_to_range(values: np.ndarray, target_min: float = 0.0, target_max: float = 1.0) -> np.ndarray:
    """Normalize values to target range."""
    if len(values) == 0:
        return values
    
    v_min, v_max = np.min(values), np.max(values)
    
    if v_max - v_min == 0:
        # All values are the same
        return np.full_like(values, target_min)
    
    normalized = (values - v_min) / (v_max - v_min)
    return normalized * (target_max - target_min) + target_min