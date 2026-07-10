"""
Evaluation script for CarNet
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from src.model import CarNet
from src.utils import load_quantile_transformer, calculate_metrics, print_metrics
from src.data_loader import get_feature_names


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CarNet model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to test data CSV')
    parser.add_argument('--weights', type=str, default='models/best_model.hdf5',
                        help='Path to model weights')
    parser.add_argument('--qt_path', type=str, default='models/quantile_transformer.pkl',
                        help='Path to quantile transformer')
    parser.add_argument('--feature_names', nargs='+', default=None,
                        help='Feature names (default: 10 core features)')
    parser.add_argument('--target', type=str, default='LosNLos',
                        help='Target column name')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for inference')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get feature names
    if args.feature_names is None:
        feature_names = get_feature_names()
    else:
        feature_names = args.feature_names
    
    print("="*60)
    print("📊 CarNet Evaluation")
    print("="*60)
    
    # 1. Load data
    print(f"\n📂 Loading data from {args.data}...")
    data = pd.read_csv(args.data)
    X = data[feature_names].values
    y = data[args.target].values.astype('float32')
    print(f"Samples: {X.shape[0]}")
    print(f"LOS: {1 - y.mean():.2%}, NLOS: {y.mean():.2%}")
    
    # 2. Load quantile transformer
    print(f"\n📂 Loading quantile transformer from {args.qt_path}...")
    qt = load_quantile_transformer(args.qt_path)
    
    # 3. ⚠️ IMPORTANT: Apply quantile transformation
    print("\n" + "="*60)
    print("⚠️ Applying quantile transformation...")
    print("="*60)
    X_transformed = qt.transform(X)
    
    # 4. Load model
    print(f"\n📂 Loading model weights from {args.weights}...")
    model = CarNet(tf.keras.layers.Input(shape=(10,)))
    model.load_weights(args.weights)
    
    # 5. Evaluate
    print("\n🎯 Running evaluation...")
    predictions = model.predict(X_transformed, batch_size=args.batch_size, verbose=1)
    
    # 6. Calculate metrics
    metrics = calculate_metrics(y, predictions, args.threshold)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
