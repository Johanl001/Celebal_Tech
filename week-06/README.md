Denoising Autoencoder on MNIST — Dense vs Convolutional Architecture Comparison

Week 6 Assignment | Data Science Internship

Author: Dhanraj Deshnukh

Overview
This notebook builds a deep learning model that removes noise from handwritten digit images using an autoencoder trained on MNIST. Rather than building a single model and reporting one reconstruction score, two encoder-decoder architectures — a Dense autoencoder and a Convolutional autoencoder — are trained under identical conditions (same noise level, same loss function, same training budget) so the effect of respecting spatial structure can be measured directly instead of assumed.

The core question the notebook answers empirically: how much does a convolutional encoder/decoder actually buy you over a fully connected one when the task is removing noise from an image, and where does each architecture start to fail?

Reference: general encoder/decoder structure follows the standard MNIST autoencoder pattern from public reference implementations (NvsYashwanth/MNIST-Autoencoder), adapted here specifically for denoising rather than plain reconstruction.

Project Structure
```
MNIST_Denoising_Autoencoder.ipynb   ← Main notebook (fully self-contained, dataset auto-downloads)
plots/                                ← Saved chart outputs (generated on run)
  ├── clean_vs_noisy_samples.png
  ├── training_loss_comparison.png
  └── denoising_output_comparison.png
README.md                             ← This file
```

Dataset
Source: `tf.keras.datasets.mnist` (built into TensorFlow — no manual download or Kaggle account needed)

| Property | Value |
|---|---|
| Total images | 70,000 (60,000 train / 10,000 test) |
| Image size | 28x28x1 (grayscale) |
| Classes | Digits 0-9 (not used as labels here — this is unsupervised reconstruction) |
| Noise type | Additive Gaussian noise, clipped to [0, 1] |
| Noise factor | 0.5 (tuned by visual inspection) |
| Missing values | N/A (image tensor, not tabular) |
| Labels | Not used — input is the noisy image, target is the corresponding clean image |

Pipeline Stages

| # | Stage | Description |
|---|---|---|
| 1 | Data Ingestion | Load MNIST via Keras, normalize pixel values to 0-1 |
| 2 | Noise Injection | Add Gaussian noise (factor 0.5) to train and test images, clip to valid range |
| 3 | Exploratory Visualization | Compare clean vs noisy digits side by side |
| 4 | Dense Autoencoder | Flatten to 784-length vector, compress to 32-unit bottleneck, reconstruct via mirrored dense layers |
| 5 | Convolutional Autoencoder | Conv2D + MaxPooling encoder, Conv2DTranspose decoder, preserves 2D spatial structure throughout |
| 6 | Training | Both models trained noisy-to-clean with EarlyStopping on validation loss |
| 7 | Loss Curve Comparison | Training and validation loss for both models plotted on shared axes |
| 8 | Denoised Output Comparison | Original, noisy, and reconstructed digits shown together for several test samples |
| 9 | Quantitative Evaluation | Test MSE and PSNR computed for both models, results tabulated |
| 10 | Analysis | Written observations on where and why the two architectures differ |

Architecture & Training Strategy Design
No manual feature engineering applies here (raw pixel values are the input), so the comparison happens at the model design level:

| Artifact | Formula / Logic | Rationale |
|---|---|---|
| Dense AE input | Image flattened to (784,) vector | Baseline that treats every pixel as an independent feature |
| Conv AE input | Image kept as (28,28,1) tensor | Preserves spatial adjacency for convolution to exploit |
| Bottleneck | 32-unit dense layer / two MaxPooling stages (conv) | Forces both models to learn a compressed representation, not just copy the input |
| Sigmoid output layer | Both models end in sigmoid activation | Matches the normalized [0,1] pixel range of the target images |
| Binary cross-entropy loss | Used for both models | Standard choice for autoencoders reconstructing values in [0,1] |
| `EarlyStopping` callback | Stops when `val_loss` hasn't improved for 5 epochs, restores best weights | Keeps the comparison fair by not hand-picking a fixed epoch count per model |

Models Used

| Model | Type | Notes |
|---|---|---|
| Dense Autoencoder | Fully connected, 784 to 32 to 784 | Baseline, no spatial awareness, fastest to train |
| Convolutional Autoencoder | Conv2D encoder, Conv2DTranspose decoder | Preserves 2D structure through the bottleneck |

Both trained with: `Adam` optimizer, `binary_crossentropy` loss, up to 30 epochs with EarlyStopping, noisy images as input and clean images as target.

Both models evaluated on: Test MSE, Test PSNR, Trainable Parameter Count, Epochs Run, Visual Reconstruction Quality

Model Validation
- Test set used as validation data during training for both models, so their loss curves are directly comparable.
- EarlyStopping's actual stopping epoch is logged per model and compared against the 30-epoch budget.
- Test MSE and PSNR computed on the full 10,000-image test set, not just the visual sample grid, so the quantitative comparison is not based on a handful of lucky examples.
- Reconstruction quality checked visually across multiple digit classes rather than a single example, since some digit pairs (4/9, 3/8) are more noise-sensitive than others.

Key Findings
(fill in the specific numbers after running the notebook — the cells are already built to produce each of these)

| Stage | Finding |
|---|---|
| Dense vs Conv | Test MSE and PSNR gap between the two models — record values here |
| Convergence speed | Epoch each model actually stopped at (vs the 30 requested) — record here |
| Visual quality | Which model produces sharper digit edges and fewer artifacts — record here |
| Failure cases | Which digit pairs remain hardest to denoise at noise_factor=0.5 — record here |
| Params vs performance | Whether the Conv AE's advantage comes from more parameters or from the architecture itself — record here (this is the core actionable insight) |
| Noise ceiling | Rough noise_factor value where reconstruction quality starts breaking down for both models — record here if the optional extension was run |

Tech Stack

| Library | Purpose |
|---|---|
| tensorflow / keras | Model building, training, layers, callbacks |
| numpy | Array manipulation, noise generation |
| pandas | Results table, comparison DataFrame |
| matplotlib | Sample visualization, loss curves, denoising comparison grid |

How to Run
```
# 1. Dataset downloads automatically — no manual step needed.
#    (tf.keras.datasets.mnist.load_data() handles it)

# 2. Install dependencies
pip install tensorflow numpy pandas matplotlib

# 3. Run
# Recommended: open in Google Colab (CPU runtime is sufficient given
# MNIST's small image size, GPU will still speed up training).
jupyter notebook MNIST_Denoising_Autoencoder.ipynb
```
Run all cells top to bottom. Plots display inline. On a free Colab CPU runtime the full notebook (both models, EarlyStopping-limited) runs in a few minutes; a GPU runtime will be faster still.

Evaluation Metrics

| Metric | Formula / Definition | Interpretation |
|---|---|---|
| Test MSE | Mean squared error between reconstructed and clean pixel values | Lower is better; primary headline metric |
| Test PSNR | `10 * log10(1 / MSE)`, measured in dB | Standard image-reconstruction metric; higher means output is closer to the clean original |
| Trainable Parameters | Total learnable weights in the model | Lets "architecture advantage" be checked against "just more parameters" |
| Validation Loss Curve | Binary cross-entropy per epoch, train vs validation | Shows convergence speed and whether either model overfits |
| Visual Reconstruction | Side-by-side grid of original, noisy, and denoised digits | Human-readable signal that MSE and PSNR alone do not fully capture |

Week 6 Assignment — Data Science Internship
