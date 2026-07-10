"""
Training utilities for CarNet
"""

import os
import tensorflow as tf
import numpy as np
from typing import List


def create_callbacks(
    save_path: str = 'models/best_model.hdf5',
    monitor: str = 'val_accuracy',
    patience: int = 10,
    verbose: int = 1
) -> List[tf.keras.callbacks.Callback]:
    """
    Create training callbacks for CarNet.
    
    Args:
        save_path: Path to save best model
        monitor: Metric to monitor
        patience: Early stopping patience
        verbose: Verbosity level
    
    Returns:
        List of callbacks
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_path,
        monitor=monitor,
        save_best_only=True,
        verbose=verbose
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=verbose
    )
    
    return [checkpoint, early_stop, reduce_lr]


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    save_path: str = 'models/best_model.hdf5',
    verbose: int = 1
) -> tf.keras.callbacks.History:
    """
    Train CarNet model.
    
    Args:
        model: CarNet model instance
        X_train: Training features (already transformed)
        y_train: Training labels
        X_val: Validation features (already transformed)
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Path to save best model
        verbose: Verbosity level
    
    Returns:
        Training history
    """
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = create_callbacks(save_path=save_path, verbose=verbose)
    
    # Train
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose
    )
    
    # Save final model
    model.save('models/carnet_final.keras')
    
    return history


def get_optimizer(optimizer_name: str = 'sgd', learning_rate: float = 0.01):
    """
    Get optimizer by name.
    
    Args:
        optimizer_name: Name of optimizer
        learning_rate: Learning rate
    
    Returns:
        Optimizer instance
    """
    optimizers = {
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate),
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        'adamw': tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        'adadelta': tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
        'adamax': tf.keras.optimizers.Adamax(learning_rate=learning_rate),
        'adagrad': tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
    }
    
    return optimizers.get(optimizer_name.lower(), optimizers['sgd'])
