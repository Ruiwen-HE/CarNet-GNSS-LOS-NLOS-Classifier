
---

## 📄 File 4: setup.py

```python
"""
Setup script for CarNet package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="carnet-gnss-classifier",
    version="1.0.0",
    author="Ni Zhu, Ruiwen He, Zhiqiang Wang",
    author_email="ni.zhu@univ-eiffel.fr",
    description="CarNet: Generative CNN for GNSS LOS/NLOS Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CarNet-GNSS-LOS-NLOS-Classifier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "joblib>=1.2.0",
    ],
)
