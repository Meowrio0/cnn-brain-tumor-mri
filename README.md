# CNN Brain Tumor MRI Classification (PyTorch)

Course assignment for *Deep Learning for Image Analysis* (Department of Electrical Engineering, National Taipei University).
A convolutional neural network built from scratch in PyTorch classifies brain MRI slices into four classes: **glioma, meningioma, no tumor, pituitary**.

## Files

| File | Contents |
|---|---|
| `CNN_Assignment_BrainTumor_MRI.ipynb` | Executed notebook: data download, preprocessing, two CNN models, layer-by-layer visualization, training, testing, comparison experiments, explanations |
| `CNN_Assignment_Report.pdf` | 5-page report with the key figures and results |

## Dataset

*Brain Tumor Classification (MRI)*, S. Bhuvaji et al., 2020: https://github.com/sartajbhuvaji/brain-tumor-classification-dataset

The dataset is **not** included in this repository. The notebook clones it automatically.
Exact duplicate files (237) are removed before a stratified 70 / 15 / 15 train / validation / test split (seed 42), leaving 3,027 unique images.

## Experiments

* **CNN-A**: 3 convolution blocks (16-32-64 filters), no batch normalization or dropout, 1.07 M parameters.
* **CNN-B**: 4 convolution blocks (16-32-64-128 filters) with batch normalization and dropout 0.4, 1.15 M parameters.
* **Basic** preprocessing: grayscale, resize to 128 × 128, tensor in [0, 1].
* **Enhanced** preprocessing: Basic + median filter, CLAHE contrast enhancement, z-score normalization and training-time augmentation.

## Results (test set, 455 images)

| Preprocessing | Model | Test accuracy (95 % CI) | Test macro-F1 |
|---|---|---|---|
| Basic | CNN-A | 0.851 (0.815–0.880) | 0.844 |
| Basic | CNN-B | 0.879 (0.846–0.906) | 0.867 |
| Enhanced | CNN-A | 0.888 (0.856–0.914) | 0.888 |
| Enhanced | CNN-B (main model) | 0.873 (0.839–0.900) | 0.867 |

The plain baseline (CNN-A + Basic) overfits (training accuracy 0.996). Enhanced preprocessing or batch normalization with dropout each improve it. The differences among the three regularized runs are within their confidence intervals (one run per setting).

## How to run

```bash
pip install torch torchvision opencv-python-headless scikit-learn pandas matplotlib jupyter
jupyter notebook CNN_Assignment_BrainTumor_MRI.ipynb   # then: Run All
```

Google Colab works too (Runtime → Run all; a GPU is optional). The saved outputs were produced on a CPU with seed 42; results on a GPU can differ slightly.
