✍️ Text Generation — RNN vs LSTM vs GRU Architecture Comparison

Week 5 Assignment | Data Science Internship

Author: Dhanraj Deshnukh

📌 Overview
This notebook builds a controlled **sequence modelling benchmark** for next-word text generation, comparing three recurrent architectures — Vanilla RNN, LSTM, and GRU — trained under identical conditions (same embedding size, same hidden units, same training budget, same corpus). Rather than training one model and generating a few sentences, three models are trained side by side so that the effect of gating mechanisms on both quantitative metrics and generated text quality can be isolated and attributed.

The core question the notebook answers empirically, not just descriptively: **how much does gating (LSTM/GRU) actually buy you over a plain RNN on a small text corpus, and is the extra complexity of LSTM's three gates worth it over GRU's two?**

📁 Project Structure
```
Text_Generation_RNN_LSTM_GRU.ipynb   ← Main notebook (fully self-contained, corpus embedded)
plots/                                 ← Saved chart outputs (generated on run)
  ├── loss_comparison.png
  └── accuracy_comparison.png
README.md                              ← This file
```

🗂️ Dataset
Source: custom in-notebook text corpus (no external download required)

Unlike Week 4's CIFAR-10 dataset, this project uses a **small, hand-written corpus** of related sentences about sequence modelling itself, since the goal is architecture comparison rather than large-scale benchmark performance — a compact vocabulary keeps training fast while still producing enough repeated word patterns for each model to learn from.

| Property | Value |
|---|---|
| Raw corpus | 15 short sentences (single paragraph, no punctuation) |
| Vocabulary size | ~70–90 unique words (exact count printed on tokenization) |
| Sequence type | N-gram prefixes of each sentence, padded to a common length |
| Total training sequences | Derived from n-gram expansion of all 15 lines |
| Train/validation split | 85% / 15% |
| Labels | Next word in sequence (self-supervised, no external labels) |

🔄 Pipeline Stages

| # | Stage | Description |
|---|---|---|
| 1 | Setup & Reproducibility | Fix `numpy`/`tensorflow` random seeds for consistent runs |
| 2 | Corpus Definition | Define the 15-sentence text corpus embedded directly in the notebook |
| 3 | Tokenization | Word-level tokenization via Keras `Tokenizer`, vocabulary size computed |
| 4 | Sequence Generation | Build n-gram prefix sequences per line, pad to `max_sequence_len` |
| 5 | Train/Validation Split | 85/15 split via `train_test_split` for comparable curves across models |
| 6 | Model Building | Shared `build_model()` factory — identical embedding/Dense layers, only the recurrent cell changes |
| 7 | Training — SimpleRNN | Baseline recurrent model, EarlyStopping on `val_loss`, up to 150 epochs |
| 8 | Training — LSTM | Same setup, gated cell with input/forget/output gates |
| 9 | Training — GRU | Same setup, gated cell with reset/update gates only |
| 10 | Evaluation | Loss/accuracy curves, results table (loss, accuracy, perplexity, parameters, training time) for all 3 models |
| 11 | Text Generation | Temperature-based sampling, multiple seed phrases compared across all 3 models |

🏷️ Architecture & Training Strategy Design
No manual feature engineering applies here (raw token sequences are the input), so the comparison happens entirely at the **recurrent cell level**:

| Artifact | Formula / Logic | Rationale |
|---|---|---|
| Shared `build_model()` | `Embedding → {SimpleRNN / LSTM / GRU}(64) → Dense(softmax)` | Isolates the recurrent cell as the only variable between the 3 models |
| `EarlyStopping` callback | Stops when `val_loss` hasn't improved for 15 epochs, restores best weights | Removes hand-picked epoch counts, lets each architecture train to its own natural stopping point |
| Temperature sampling | `softmax(log(p)/T)` before sampling, `T = 0.8` | Avoids greedy-decoding loops on a small corpus; produces more natural, varied generated text |
| Validation perplexity | `exp(val_loss)` | More interpretable than raw cross-entropy for judging how "confident" each model is |
| Parameter count | `model.count_params()` per architecture | Lets "extra gating capacity" be weighed directly against training time and final loss |

🤖 Models Used

| Model | Type | Notes |
|---|---|---|
| SimpleRNN | Vanilla recurrent, 1 layer | 64 units, fewest parameters, fastest per-epoch training |
| LSTM | Gated recurrent, 1 layer | 64 units, 3 gates (input/forget/output), most parameters |
| GRU | Gated recurrent, 1 layer | 64 units, 2 gates (reset/update), fewer parameters than LSTM |

All three trained with: `Embedding(32-dim) → RNN cell(64) → Dense(softmax)`, `Adam(lr=0.01)`, `sparse_categorical_crossentropy`, up to 150 epochs with EarlyStopping.

All classifiers evaluated on: **Validation Loss, Validation Accuracy, Validation Perplexity, Trainable Parameter Count, Training Time, Generated Text Quality (qualitative)**

📈 Model Validation
- **Train/validation split (85/15)** used during training for every model, so loss/accuracy curves are directly comparable across all three.
- **EarlyStopping's actual stopping epoch** is logged per model and compared against the requested 150, showing how quickly each architecture converges.
- **Training time** recorded per model as the practical cost of extra gating complexity, not just parameter count on paper.
- **Generated text** compared across four seed phrases per model rather than a single example, so output quality isn't judged on one lucky (or unlucky) generation.

📊 Key Findings
*(fill in the specific numbers after running the notebook — the cells are already built to produce each of these)*

| Stage | Finding |
|---|---|
| RNN vs LSTM vs GRU | Final validation loss/accuracy for each — record values here |
| Convergence speed | Epoch each model actually stopped at (vs the 150 requested) — record here |
| Training time | Wall-clock time per model, and whether GRU's time savings over LSTM are meaningful at this scale — record here |
| Parameters vs performance | Whether LSTM's extra gate capacity translates into a meaningfully lower loss than GRU, or whether GRU matches it with fewer parameters — record here (this is the core actionable insight) |
| Generated text quality | Which model produces the most coherent continuations across the 4 seed phrases — record here |
| Perplexity | Validation perplexity per model, lower is better — record here |

🛠️ Tech Stack

| Library | Purpose |
|---|---|
| tensorflow / keras | Tokenization, model building, training, callbacks |
| numpy | Array manipulation, temperature-sampling logic |
| pandas | Results table, comparison DataFrame |
| matplotlib | Loss/accuracy curve visualisation |
| scikit-learn | `train_test_split` for train/validation split |

▶️ How to Run
```
# 1. No dataset download needed — corpus is embedded directly in the notebook.

# 2. Install dependencies
pip install tensorflow numpy pandas matplotlib scikit-learn

# 3. Run
# Recommended: open in Google Colab (CPU runtime is sufficient given the
# corpus size, but a T4 GPU will still speed up all 3 models).
jupyter notebook Text_Generation_RNN_LSTM_GRU.ipynb
```
Run all cells top to bottom. Plots display inline. On CPU the full notebook (all 3 models, EarlyStopping-limited) runs in a couple of minutes.

📋 Evaluation Metrics

| Metric | Formula / Definition | Interpretation |
|---|---|---|
| Validation Loss | Sparse categorical cross-entropy on held-out validation sequences | Primary headline metric, comparable across all 3 models |
| Validation Accuracy | Correct next-word predictions / total predictions on validation set | Complements loss with an interpretable percentage |
| Validation Perplexity | `exp(validation loss)` | Standard language-modelling metric; lower means less "surprised" by held-out text |
| Trainable Parameters | Total learnable weights in the model | Lets "more gating" be weighed against "training time" and "final loss" directly |
| Training Time | Wall-clock seconds to convergence (EarlyStopping-limited) | Practical cost of each architecture's extra complexity |
| Generated Text (qualitative) | Temperature-sampled continuations from 4 seed phrases per model | Human-readable signal that raw loss numbers alone don't capture |

Week 5 Assignment — Data Science Internship
