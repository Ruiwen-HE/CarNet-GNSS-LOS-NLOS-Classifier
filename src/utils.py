"""
Utility functions for CarNet
"""

import numpy as np
import joblib
import os
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def preprocess_data(X, qt=None, fit=False):
    """
    Preprocess data using Quantile Transformer.
    
    ⚠️ IMPORTANT: This function MUST use QuantileTransformer with 
                   output_distribution='normal'.
    
    Args:
        X: Input data (numpy array)
        qt: Existing QuantileTransformer (if None, a new one is created)
        fit: Whether to fit the transformer (True for training, False for inference)
    
    Returns:
        X_transformed: Transformed data
        qt: QuantileTransformer object (if fit=True)
    """
    if qt is None:
        qt = QuantileTransformer(output_distribution='normal')
    
    if fit:
        X_transformed = qt.fit_transform(X)
        return X_transformed, qt
    else:
        X_transformed = qt.transform(X)
        return X_transformed


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities or labels
        threshold: Classification threshold
    
    Returns:
        dict: Dictionary of metrics
    """
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    
    if y_pred.dtype == float:
        y_pred_binary = (y_pred > threshold).astype(int)
    else:
        y_pred_binary = y_pred
    
    return {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision_los': precision_score(y_true, y_pred_binary, pos_label=0, zero_division=0),
        'precision_nlos': precision_score(y_true, y_pred_binary, pos_label=1, zero_division=0),
        'recall_los': recall_score(y_true, y_pred_binary, pos_label=0, zero_division=0),
        'recall_nlos': recall_score(y_true, y_pred_binary, pos_label=1, zero_division=0),
        'f1_los': f1_score(y_true, y_pred_binary, pos_label=0, zero_division=0),
        'f1_nlos': f1_score(y_true, y_pred_binary, pos_label=1, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred_binary)
    }


def save_quantile_transformer(qt, path='models/quantile_transformer.pkl'):
    """Save Quantile Transformer to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(qt, path)
    print(f"✅ Quantile transformer saved to {path}")


def load_quantile_transformer(path='models/quantile_transformer.pkl'):
    """Load Quantile Transformer from disk."""
    return joblib.load(path)


def print_metrics(metrics):
    """Pretty print classification metrics."""
    print("\n" + "="*60)
    print("📊 Classification Results")
    print("="*60)
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"LOS Precision:      {metrics['precision_los']:.4f}")
    print(f"NLOS Precision:     {metrics['precision_nlos']:.4f}")
    print(f"LOS Recall:         {metrics['recall_los']:.4f}")
    print(f"NLOS Recall:        {metrics['recall_nlos']:.4f}")
    print(f"LOS F1-Score:       {metrics['f1_los']:.4f}")
    print(f"NLOS F1-Score:      {metrics['f1_nlos']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("="*60)
