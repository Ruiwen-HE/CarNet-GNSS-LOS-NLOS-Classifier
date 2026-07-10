"""
Data loading and preprocessing utilities for CarNet
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from typing import Tuple, List, Optional


def load_data(
    train_path: str,
    val_path: str,
    feature_names: List[str],
    target_col: str = 'LosNLos'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training and validation data from CSV files.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        feature_names: List of feature column names
        target_col: Target column name
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    
    X_train = train_data[feature_names].values
    y_train = train_data[target_col].values.astype('float32')
    X_val = val_data[feature_names].values
    y_val = val_data[target_col].values.astype('float32')
    
    return X_train, y_train, X_val, y_val


def prepare_features(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    fit: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], QuantileTransformer]:
    """
    ⚠️ IMPORTANT: Apply Quantile Transformation with Gaussian distribution.
    This is REQUIRED for CarNet to work properly.
    
    Args:
        X_train: Training features
        X_val: Validation features (optional)
        X_test: Test features (optional)
        fit: Whether to fit the transformer
    
    Returns:
        Tuple of (X_train_transformed, X_val_transformed, X_test_transformed, qt)
    """
    # ⚠️ CRITICAL: Must use QuantileTransformer with normal distribution
    qt = QuantileTransformer(output_distribution='normal')
    
    if fit:
        X_train_transformed = qt.fit_transform(X_train)
    else:
        X_train_transformed = qt.transform(X_train)
    
    X_val_transformed = None
    if X_val is not None:
        X_val_transformed = qt.transform(X_val)
    
    X_test_transformed = None
    if X_test is not None:
        X_test_transformed = qt.transform(X_test)
    
    return X_train_transformed, X_val_transformed, X_test_transformed, qt


def get_feature_names() -> List[str]:
    """
    Get the 10 core feature names used by CarNet.
    
    Returns:
        List of 10 feature names
    """
    return ['f4', 'f7', 'f9', 'f10', 'f11', 'f13', 'f17', 'f22', 'f23', 'f24']


def get_feature_descriptions() -> dict:
    """
    Get descriptions for the 10 core features.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return {
        'SNR': 'Signal-to-Noise Ratio',
        'φ': 'Carrier phase first derivative',
        'NPR': 'Normalized Pseudorange Residual',
        'CMC': 'Code-Minus-Carrier',
        'CPRRC': 'Carrier Phase Range Rate Consistency',
        'LT': 'Local Test statistic',
        'Nsat': 'Number of visible satellites',
        'Var(SNR)': 'SNR variance (rolling window)',
        'Var(P)': 'Pseudorange variance (rolling window)',
        'Var(Pr)': 'Pseudorange residual variance (rolling window)'
    }
