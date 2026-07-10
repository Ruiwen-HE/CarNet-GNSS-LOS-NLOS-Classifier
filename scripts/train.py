"""
Training script for CarNet
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from src.model import CarNet
from src.data_loader import load_data, prepare_features, get_feature_names
from src.trainer import train_model
from src.utils import save_quantile_transformer


def parse_args():
    parser = argparse.ArgumentParser(description='Train CarNet model')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data CSV')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data CSV')
    parser.add_argument('--feature_names', nargs='+', default=None,
                        help='Feature names (default: 10 core features)')
    parser.add_argument('--target', type=str, default='LosNLos',
                        help='Target column name')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models/best_model.hdf5',
                        help='Path to save model weights')
    parser.add_argument('--qt_path', type=str, default='models/quantile_transformer.pkl',
                        help='Path to save quantile transformer')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get feature names
    if args.feature_names is None:
        feature_names = get_feature_names()
    else:
        feature_names = args.feature_names
    
    print("="*60)
    print("🚀 CarNet Training")
    print("="*60)
    print(f"Features: {feature_names}")
    print(f"Target: {args.target}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*60)
    
    # 1. Load data
    print("\n📂 Loading data...")
    X_train, y_train, X_val, y_val = load_data(
        args.train_data, args.val_data, feature_names, args.target
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Training - LOS: {1 - y_train.mean():.2%}, NLOS: {y_train.mean():.2%}")
    
    # 2. ⚠️ IMPORTANT: Quantile Transformation
    print("\n" + "="*60)
    print("⚠️ Applying Quantile Transformer (Gaussian distribution)...")
    print("="*60)
    
    X_train_transformed, X_val_transformed, _, qt = prepare_features(
        X_train, X_val, fit=True
    )
    
    # Save quantile transformer
    save_quantile_transformer(qt, args.qt_path)
    
    # 3. Create model
    print("\n🏗️ Creating CarNet model...")
    model = CarNet(tf.keras.layers.Input(shape=(10,)))
    model.summary()
    
    # 4. Train
    print("\n" + "="*60)
    print("🎯 Starting training...")
    print("="*60)
    
    history = train_model(
        model=model,
        X_train=X_train_transformed,
        y_train=y_train,
        X_val=X_val_transformed,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )
    
    # 5. Print results
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n✅ Training completed!")
    print(f"
