"""
Inference script for CarNet
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from src.model import CarNet
from src.utils import load_quantile_transformer
from src.data_loader import get_feature_names


def parse_args():
    parser = argparse.ArgumentParser(description='Run CarNet inference')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data CSV')
    parser.add_argument('--weights', type=str, default='models/best_model.hdf5',
                        help='Path to model weights')
    parser.add_argument('--qt_path', type=str, default='models/quantile_transformer.pkl',
                        help='Path to quantile transformer')
    parser.add_argument('--feature_names', nargs='+', default=None,
                        help='Feature names (default: 10 core features)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV path')
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
    print("🔮 CarNet Inference")
    print("="*60)
    
    # 1. Load data
    print(f"\n📂 Loading data from {args.data}...")
    data = pd.read_csv(args.data)
    X = data[feature_names].values
    print(f"Samples: {X.shape[0]}")
    
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
    
    # 5. Predict
    print("\n🎯 Running inference...")
    predictions = model.predict(X_transformed, batch_size=args.batch_size, verbose=1)
    labels = (predictions > args.threshold).astype(int)
    
    # 6. Save results
    results = pd.DataFrame({
        'prediction_probability': predictions.flatten(),
        'predicted_label': labels.flatten()
    })
    results.to_csv(args.output, index=False)
    
    print(f"\n✅ Predictions saved to {args.output}")
    print(f"📊 LOS: {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.2f}%)")
    print(f"📊 NLOS: {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.2f}%)")


if __name__ == "__main__":
    main()
