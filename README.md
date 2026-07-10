# CarNet: GNSS LOS/NLOS Classifier

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
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
```

### ❌ DO NOT USE These Methods

| Method | Issue |
|--------|-------|
| ❌ Min-Max Scaling | Accuracy drops to 77.32% |
| ❌ StandardScaler / Z-score | Accuracy only 78.21% |

### ✅ Recommended Methods Comparison

| Preprocessing Method | Target Distribution | Cross-Validation Accuracy |
|----------------------|---------------------|---------------------------|
| None | - | 78.00% |
| Min-Max Scaling | [0, 1] | 77.32% |
| Z-score Standardization | Mean 0, Std 1 | 78.21% |
| **Quantile Transformation** | **Normal** | **81.01%** |
| Quantile Transformation | Uniform | 80.30% |

### Why Quantile Transformation?

1. **Robustness**: GNSS data has skewed distributions and many outliers
2. **Non-linear**: Handles non-Gaussian characteristics of GNSS signals
3. **Data Consistency**: Makes features of different scales comparable
4. **Model Stability**: Reduces the impact of extreme values on training

### Complete Data Preprocessing Pipeline

```python
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# 1. Load data
train_data = pd.concat([Nantes, Paris, Toulouse, FullNantes, Boulevard])
val_data = Defense

# 2. Extract features and labels
feature_names = ['f4', 'f7', 'f9', 'f10', 'f11', 'f13', 'f17', 'f22', 'f23', 'f24']
X_train = train_data[feature_names].values
y_train = train_data['LosNLos'].values.astype('float32')
X_val = val_data[feature_names].values
y_val = val_data['LosNLos'].values.astype('float32')

# 3. ⚠️ REQUIRED: Quantile Transformation
qt = QuantileTransformer(output_distribution='normal')
X_train_transformed = qt.fit_transform(X_train)
X_val_transformed = qt.transform(X_val)

# 4. Train model with transformed data
# model.fit(X_train_transformed, y_train, validation_data=(X_val_transformed, y_val))
```

---

## 🏗️ Architecture

```
Input (10 features)
    ↓
Quantile Transformer → Normal Distribution
    ↓
Reshape → (1, 1, 10)
    ↓
┌─────────────────────────────────────┐
│         Image Generator             │
│  Conv2DTranspose (1×1, stride=3)    │
│  Conv2DTranspose (3×3, stride=2)    │
│  Conv2DTranspose (3×3, stride=2)    │
│  Output: (12, 12, 1)               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│      Inception Module (×2)          │
│  1×1 Conv → 3×3 Conv (LeakyReLU)   │
│  1×1 Conv → 5×5 Conv (LeakyReLU)   │
│  1×1 Conv → 7×7 Conv (LeakyReLU)   │
│  Residual Connection (Add)          │
│  MaxPooling 2×2                    │
└─────────────────────────────────────┘
    ↓
Global Average Pooling (GAP)
    ↓
Dense(32) → LeakyReLU → Dropout(0.4)
    ↓
Dense(1) → Sigmoid
    ↓
LOS/NLOS Prediction
```

---

## 🚀 Quick Start

### Requirements

```bash
Python 3.9+
TensorFlow 2.15+
NumPy 1.24+
Pandas 2.0+
Scikit-learn 1.3+
```

### Installation

```bash
git clone https://github.com/yourusername/CarNet-GNSS-LOS-NLOS-Classifier.git
cd CarNet-GNSS-LOS-NLOS-Classifier
pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py --train_data /path/to/train.csv --val_data /path/to/val.csv --epochs 50 --batch_size 256 --lr 0.01
```

### Inference

```bash
python scripts/predict.py --data /path/to/test.csv --weights models/best_model.hdf5 --output predictions.csv
```

### Evaluation

```bash
python scripts/evaluate.py --data /path/to/test.csv --weights models/best_model.hdf5
```

---

## 📁 10 Core Features

| Feature | Description |
|---------|-------------|
| SNR | Signal-to-Noise Ratio |
| φ | Carrier phase first derivative |
| NPR | Normalized Pseudorange Residual |
| CMC | Code-Minus-Carrier |
| CPRRC | Carrier Phase Range Rate Consistency |
| LT | Local Test statistic |
| Nsat | Number of visible satellites |
| Var(SNR) | SNR variance (rolling window) |
| Var(P) | Pseudorange variance (rolling window) |
| Var(Pr) | Pseudorange residual variance (rolling window) |

---

## 📊 Dataset

The dataset consists of 6 tracks with over 9 hours of real vehicle data (~1.56 million samples):

| Track | Environment | Samples | LOS Percentage |
|-------|-------------|---------|----------------|
| Nantes | Deep Urban | 42,560 | 79.83% |
| Full Nantes | Deep Urban / Suburban | 166,033 | 70.31% |
| Paris XII | Deep Urban | 357,762 | 72.51% |
| Boulevards | Highly Deep Urban | 710,400 | 69.86% |
| La Défense | Highly Deep Urban | 166,998 | 79.22% |
| Toulouse | Deep Urban | 117,576 | 46.56% |

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{zhu2025carnet,
  title={CarNet: A generative convolutional neural network-based line-of-sight/non-line-of-sight classifier for global navigation satellite systems},
  author={Zhu, Ni and He, Ruiwen and Wang, Zhiqiang},
  journal={Engineering Applications of Artificial Intelligence},
  volume={146},
  pages={110160},
  year={2025},
  doi={10.1016/j.engappai.2025.110160}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Ni Zhu** - *AME-GEOLOC, University Gustave Eiffel*
- **Ruiwen He** - *Léonard de Vinci Pôle Universitaire, Research Center*
- **Zhiqiang Wang** - *Léonard de Vinci Pôle Universitaire, Research Center*

---

## 🙏 Acknowledgments

- This work was supported by the French National Research Agency (ANR) under project ReSilientGAIA (ANR-23-CE22-0004-01)
- Data collected using the VERT (Vehicle for Experimental Research on Trajectories) platform at GEOLOC Laboratory

---

## 📁 Repository Structure

```
CarNet-GNSS-LOS-NLOS-Classifier/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── data_loader.py
│   ├── trainer.py
│   └── utils.py
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── configs/
│   └── default_config.yaml
├── notebooks/
├── weights/
├── models/
└── docs/
```

---

## 📄 License

```
Creative Commons Attribution-NonCommercial 4.0 International License

Copyright (c) 2025 Ni Zhu, Ruiwen He, Zhiqiang Wang

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
International License. To view a copy of this license, visit:
http://creativecommons.org/licenses/by-nc/4.0/

You are free to:

- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:

- Attribution — You must give appropriate credit, provide a link to the license,
  and indicate if changes were made. You may do so in any reasonable manner,
  but not in any way that suggests the licensor endorses you or your use.

- NonCommercial — You may not use the material for commercial purposes.

- No additional restrictions — You may not apply legal terms or technological
  measures that legally restrict others from doing anything the license permits.

Notices:

You do not have to comply with the license for elements of the material in the
public domain or where your use is permitted by an applicable exception or
limitation.

No warranties are given. The license may not give you all of the permissions
necessary for your intended use. For example, other rights such as publicity,
privacy, or moral rights may limit how you use the material.

---

⚠️ This software is provided for ACADEMIC RESEARCH, LEARNING, and EDUCATIONAL
purposes only. COMMERCIAL USE is strictly prohibited without explicit written
permission from the authors.

✅ Permitted Uses:
  - Academic research and publications
  - Classroom teaching and learning
  - Personal and non-commercial projects
  - Scientific benchmarking and comparisons
  - Open-source contributions (non-commercial)

❌ Prohibited Uses:
  - Commercial products or services
  - Selling or licensing for profit
  - Integration into commercial software
  - Any for-profit application or deployment

For commercial licensing inquiries, please contact: ni.zhu@univ-eiffel.fr
```

---
