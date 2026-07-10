# CarNet: GNSS LOS/NLOS Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)

> A Generative Convolutional Neural Network (GCNN)-based GNSS Line-of-Sight (LOS) / Non-Line-of-Sight (NLOS) classifier that transforms multivariate time-series data into images.

---

## 📖 Overview

**CarNet** is a deep learning framework for classifying GNSS signal receptions as Line-of-Sight (LOS) or Non-Line-of-Sight (NLOS) in challenging urban environments. The model uses a generative CNN to convert 1D GNSS feature vectors into 2D images, followed by an Inception-based CNN for multi-scale feature extraction and classification.

### Key Advantages

- ✅ **Generative Image Encoding**: Automatically transforms 10 core GNSS features into 28×28 images
- ✅ **Inception-Based Classifier**: Multi-scale feature extraction with 3×3, 5×5, 7×7 parallel convolutions + residual connections
- ✅ **Quantile Transformation**: Robust preprocessing for skewed GNSS data distributions and outliers
- ✅ **Lightweight Design**: Only **27,964** trainable parameters
- ✅ **Real-Time Ready**: 4.68 ms/sample on CPU, 2.91 ms/sample on GPU
- ✅ **Cross-City Generalization**: Validated on 6 different urban datasets (Nantes, Paris, Toulouse, La Défense, etc.)

### Performance Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | **81.47%** |
| LOS Precision | **83.30%** |
| NLOS Precision | **70.99%** |
| Model Parameters | 27,964 |
| GPU Inference | 2.91 ms/sample |
| CPU Inference | 4.68 ms/sample |

### Positioning Accuracy Improvement (Horizontal Position Error - HPE)

| Weighting Method | Median HPE | 75% HPE |
|------------------|------------|---------|
| OLS | 13.58 m | 25.87 m |
| WLS (Elevation) | 6.88 m | 13.86 m |
| WLS (C/N₀) | 4.41 m | 8.37 m |
| **WLS (CarNet)** | **3.54 m** | **6.34 m** |

---

## ⚠️ CRITICAL: Data Preprocessing Requirements

### MUST USE Quantile Transformer

CarNet **REQUIRES** **Quantile Transformer** for data preprocessing with **Gaussian (normal) distribution** as the target.

```python
from sklearn.preprocessing import QuantileTransformer

# ⚠️ REQUIRED: Quantile Transformer with normal distribution
qt = QuantileTransformer(output_distribution='normal')
X_transformed = qt.fit_transform(X)
