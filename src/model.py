"""
CarNet: Generative CNN for GNSS LOS/NLOS Classification
Paper: CarNet: A generative convolutional neural network-based line-of-sight/
        non-line-of-sight classifier for global navigation satellite systems
Authors: Ni Zhu, Ruiwen He, Zhiqiang Wang
"""

import tensorflow as tf


def inception_block(x, n_map):
    """
    Inception module with 3x3, 5x5, 7x7 convolutions and residual connection.
    
    Args:
        x: Input tensor
        n_map: Number of output feature maps
    
    Returns:
        Output tensor with residual addition
    """
    # 1x1 convolution for dimensionality reduction
    x = tf.keras.layers.Conv2D(n_map, (1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    
    # Path 1: 3x3 convolution (1 stacked 3x3)
    x1 = tf.keras.layers.Conv2D(n_map, (3, 3), strides=(1, 1), padding='same')(x)
    x1 = tf.keras.layers.LeakyReLU(alpha=0.01)(x1)
    
    # Path 2: 5x5 convolution (2 stacked 3x3)
    x2 = tf.keras.layers.Conv2D(n_map, (5, 5), strides=(1, 1), padding='same')(x)
    x2 = tf.keras.layers.LeakyReLU(alpha=0.01)(x2)
    
    # Path 3: 7x7 convolution (3 stacked 3x3)
    x3 = tf.keras.layers.Conv2D(n_map, (7, 7), strides=(1, 1), padding='same')(x)
    x3 = tf.keras.layers.LeakyReLU(alpha=0.01)(x3)
    
    # Residual addition (not concatenation)
    return tf.keras.layers.Add()([x, x1, x2, x3])


def CarNet(raw_features):
    """
    CarNet: Generative CNN for GNSS LOS/NLOS classification.
    
    Args:
        raw_features: Input tensor of shape (batch_size, 10)
    
    Returns:
        Keras Model instance
    """
    # Input: 10 GNSS features
    x = raw_features
    
    # ==================== Image Generator ====================
    # FC Layer: 1×10 → 16 × 7 × 7 (Data augmentation)
    x = tf.keras.layers.Dense(16 * 7 * 7, use_bias=False)(x)
    x = tf.keras.layers.Reshape((7, 7, 16))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    
    # Upsample to 14×14×10
    x = tf.keras.layers.Conv2DTranspose(10, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    
    # Upsample to 28×28×5
    x = tf.keras.layers.Conv2DTranspose(5, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    
    # Final output: 28×28×1 with Tanh
    x = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)
    
    # ==================== Image Classifier ====================
    # Inception Block 1 (8 filters) + MaxPool
    x = inception_block(x, 8)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)
    
    # Inception Block 2 (16 filters) - no MaxPool
    x = inception_block(x, 16)
    
    # Global Average Pooling → (16)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layers
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=raw_features, outputs=x, name='CarNet')
    return model


def create_model():
    """Create CarNet model."""
    inputs = tf.keras.layers.Input(shape=(10,))
    model = CarNet(inputs)
    return model


if __name__ == "__main__":
    model = create_model()
    model.summary()
    print(f"Total parameters: {model.count_params():,}")
