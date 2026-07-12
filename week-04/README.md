🖼️ CIFAR-10 Image Classification — ANN vs CNN Architecture & Training Strategy Comparison

Week 4 Assignment | Data Science Internship

Author: Dhanraj Deshnukh

📌 Overview
This notebook builds a complete **image classification benchmark** on CIFAR-10, comparing how much of a model's performance comes from **architecture** (dense/ANN vs convolutional/CNN, shallow vs deep) versus **training strategy** (fixed epochs vs EarlyStopping, with vs without data augmentation). Rather than training one model and reporting one accuracy, six models are trained under controlled, identical data splits so each design choice's effect can be isolated and attributed.

The core question the notebook answers empirically, not just descriptively: **why does a CNN beat an ANN on images, and how much of the remaining gap can training strategy alone close without touching the architecture?**

📁 Project Structure
```
CIFAR10_ANN_CNN_Completed.ipynb   ← Main notebook (fully self-contained, dataset auto-downloads)
plots/                             ← Saved chart outputs (generated on run)
  ├── sample_images.png
  ├── ann_vs_cnn_val_accuracy.png
  ├── full_learning_curves.png
  ├── confusion_matrix_best_model.png
  └── final_accuracy_comparison.png
README.md                          ← This file
```

🗂️ Dataset
Source: `tf.keras.datasets.cifar10` (built into TensorFlow — no manual download or Kaggle account needed)

Unlike Week 3, this dataset **is embedded in the framework itself**, since CIFAR-10 is a standard benchmark shipped with Keras rather than an external CSV.

| Property | Value |
|---|---|
| Total images | 60,000 (50,000 train / 10,000 test) |
| Image size | 32×32×3 (RGB) |
| Classes | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Class balance | Perfectly balanced (6,000 images per class) |
| Total size on disk | ~170 MB (small relative to most CV benchmarks; downloads in under a minute) |
| Missing values | N/A (image tensor, not tabular) |
| Labels | Provided (this is supervised classification, unlike Week 3's derived labels) |

🔄 Pipeline Stages

| # | Stage | Description |
|---|---|---|
| 1 | Data Ingestion & Inspection | Load CIFAR-10 via Keras, check shapes, preview class names |
| 2 | Exploratory Data Analysis | Visualize sample images per class |
| 3 | Preprocessing | Normalize pixels 0–255 → 0–1; flatten to 3,072-length vectors for ANN input, keep 32×32×3 tensors for CNN input |
| 4 | Baseline ANN | 2 dense layers (512→256) with dropout — establishes the "ignores spatial structure" floor |
| 5 | Baseline CNN | 3 conv blocks (32→64→128) with BatchNorm + MaxPooling — establishes the "exploits spatial structure" comparison point |
| 6 | Architecture Scaling — Deep ANN | 4 dense layers (1024→512→256→128) with BatchNorm + Dropout at each stage |
| 7 | Architecture Scaling — Deep CNN | 4 conv blocks (32→64→128→256) with GlobalAveragePooling2D instead of Flatten |
| 8 | Training Strategy — EarlyStopping | Baseline CNN architecture, trained up to 20 epochs, `monitor='val_loss'`, `restore_best_weights=True` |
| 9 | Training Strategy — Data Augmentation | RandomFlip + RandomRotation + RandomZoom applied on-the-fly, trained with EarlyStopping |
| 10 | Model Evaluation | Test accuracy/loss for all 6 models, confusion matrix + classification report for the best CNN, full learning-curve overlay, comprehensive comparison table |

🏷️ Architecture & Training Strategy Design
No feature engineering applies here (raw pixels are the input), so the "engineering" happens at the **model design level**:

| Artifact | Formula / Logic | Rationale |
|---|---|---|
| ANN input | Image flattened to (3072,) vector | Establishes what's lost when spatial adjacency is discarded |
| CNN input | Image kept as (32,32,3) tensor | Preserves spatial adjacency for convolution to exploit |
| `GlobalAveragePooling2D` (Deep CNN) | Averages each feature map to one value instead of flattening | Cuts classifier-head parameters sharply, reduces overfitting vs a large `Flatten` → `Dense` |
| `EarlyStopping` callback | Stops when `val_loss` hasn't improved for 3 epochs, restores best weights | Removes the guesswork of picking a fixed epoch count by hand |
| Augmentation layers | `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)` applied only at train time | Synthetic variation forces the model to learn robust features instead of memorizing exact pixels |

🤖 Models Used

| Model | Type | Notes |
|---|---|---|
| ANN (baseline) | Dense, 2 hidden layers | 512→256 units, dropout 0.3, 10 epochs |
| Deep ANN | Dense, 4 hidden layers | 1024→512→256→128 units, BatchNorm + Dropout, 15 epochs |
| CNN (baseline) | Convolutional, 3 conv blocks | 32→64→128 filters, BatchNorm + MaxPooling, 10 epochs |
| Deep CNN | Convolutional, 4 conv blocks | 32→64→128→256 filters, GlobalAveragePooling2D, 15 epochs |
| CNN + EarlyStopping | Convolutional (baseline architecture) | Up to 20 epochs, stops automatically on `val_loss` plateau |
| Augmented CNN | Convolutional (baseline architecture) + augmentation layers | Up to 20 epochs with EarlyStopping, trained on flipped/rotated/zoomed images |

All classifiers evaluated on: **Test Accuracy, Test Loss, Trainable Parameter Count, Confusion Matrix (best model), Per-Class Precision/Recall/F1**

📈 Model Validation
- **Train/validation split (90/10)** used during training for every model, so validation-accuracy curves are directly comparable across all six.
- **Held-out test set** (the standard CIFAR-10 10,000-image test split) used only for final evaluation — never seen during training or validation, avoiding leakage.
- **EarlyStopping's actual stopping epoch** is logged and compared against the requested 20, showing directly how much of the fixed epoch budget was actually needed.
- **Train/validation accuracy gap** tracked qualitatively across architectures as the practical signal of overfitting, rather than relying on test accuracy alone.

📊 Key Findings
*(fill in the specific numbers after running the notebook — the cells are already built to produce each of these)*

| Stage | Finding |
|---|---|
| ANN vs CNN | Test accuracy gap between baseline ANN and baseline CNN — record value here |
| Architecture scaling | Whether Deep ANN/Deep CNN meaningfully beat their baselines, or mostly widen the train/val gap — record here |
| EarlyStopping | Epoch it actually stopped at (vs the 20 requested) and resulting test accuracy — record here |
| Augmentation | Test accuracy and train/val gap for the Augmented CNN vs CNN + EarlyStopping — record here |
| Best model | Which of the 6 models wins on test accuracy, and by how much — record here |
| Confusion matrix | Which class pairs are confused most (expect cat↔dog, automobile↔truck) — record here |
| Params vs performance | Whether the largest model (Deep CNN) is also the best-performing, or whether a smaller model + better training strategy wins — record here (this is the core actionable insight) |

🛠️ Tech Stack

| Library | Purpose |
|---|---|
| tensorflow / keras | Model building, training, layers, callbacks |
| numpy | Array/tensor manipulation |
| pandas | Results tables, comparison DataFrame |
| matplotlib, seaborn | Visualisation, confusion matrix heatmap |
| scikit-learn | `confusion_matrix`, `classification_report` |

▶️ How to Run
```
# 1. Dataset downloads automatically — no manual step needed.
#    (tf.keras.datasets.cifar10.load_data() handles it)

# 2. Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

# 3. Run
# Recommended: open in Google Colab and set
#   Runtime → Change runtime type → T4 GPU
# before running, since 6 models are trained sequentially.
jupyter notebook CIFAR10_ANN_CNN_Completed.ipynb
```
Run all cells top to bottom. Plots display inline. On a free Colab GPU the full notebook (all 6 models) runs in roughly 10–15 minutes; on CPU it will take substantially longer.

📋 Evaluation Metrics

| Metric | Formula / Definition | Interpretation |
|---|---|---|
| Test Accuracy | Correct predictions / total predictions on held-out test set | Primary headline metric, comparable across all 6 models |
| Test Loss | Sparse categorical cross-entropy on test set | Complements accuracy; useful when comparing models with similar accuracy |
| Trainable Parameters | Total learnable weights in the model | Lets "bigger model" be weighed against "better training strategy" directly |
| Precision / Recall / F1 (per class) | Standard classification metrics from `classification_report` | Reveals which specific classes the best model struggles with |
| Confusion Matrix | Predicted vs actual class counts | Visualizes exactly where misclassifications concentrate |
| Train/Val Accuracy Gap | Difference between train and validation accuracy across epochs | Qualitative overfitting signal, tracked via the learning-curve plots |

Week 4 Assignment — Data Science Internship
