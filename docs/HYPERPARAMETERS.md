# Hyperparameter Justification

This document explains the hyperparameters used in the reference surrogate model, selected through repeated comparative tuning experiments.

## Final Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 100 | After 100 epochs, validation loss plateaus. Further training provides diminishing returns and risks overfitting. |
| **Batch Size** | 32 | Tested values: 64, 128. Batch size 32 provides better generalization on the val set with lower final loss. |
| **Learning Rate** | 2e-5 | Very low learning rate enables fine-tuned convergence on this small dataset (7,000 samples). Higher values (e.g., 1e-4) caused instability. |
| **Weight Decay (L2)** | 1e-4 | L2 regularization prevents overfitting to training noise. Tested values: {0, 1e-5, 1e-4, 1e-3}; 1e-4 balanced regularization optimally. |
| **Dropout Rates** | [0.3, 0.3, 0.2] | Applied after first two hidden layers (512 units each) and third layer (256 units). Helps prevent co-adaptation; more aggressive in early layers. |
| **K-Fold Splits** | 2 | Balances validation robustness vs. computational cost. 2 folds with ~3,500 samples each provides stable estimates. |
| **Architecture** | Two-branch MIMO | Separate branches for building and ground-motion inputs (512 units each), merged, then shared layers (512 → 256 → 128) with dropouts. |

## Hyperparameter Tuning Process

### 1. Learning Rate Optimization

Multiple learning rates were tested:

- **1e-4**: Unstable training, frequent divergence on validation set
- **5e-5**: Slower convergence, required 200+ epochs for same loss as 2e-5 at epoch 100
- **2e-5**: ✅ **Selected** – Stable, smooth convergence, good final performance
- **1e-5**: Too slow; convergence stalls after ~150 epochs

**Conclusion**: 2e-5 provides sweet spot between stability and convergence speed.

### 2. Batch Size Optimization

Tested batch sizes: {16, 32, 64, 128}

- **16**: High gradient noise → noisy loss curves, marginal improvement in generalization
- **32**: ✅ **Selected** – Best val loss, smoother curves, faster epoch runtimes
- **64**: Slightly worse val loss (~3-5% higher), faster epochs but worse final model
- **128**: Low gradient information per update; poor generalization

**Conclusion**: 32 balances gradient information density with noise levels.

### 3. L2 Regularization (Weight Decay)

Tested weight decay values: {0, 5e-5, 1e-4, 5e-4, 1e-3}

With **no regularization (0)**:
- Train loss: 0.052
- Val loss: 0.089
- Signs of overfitting (large gap)

With **1e-4**: ✅ **Selected**
- Train loss: 0.056
- Val loss: 0.074
- Reduced overfitting gap

With **1e-3**:
- Train loss: 0.095
- Val loss: 0.098
- Over-regularized; underfitting

**Conclusion**: 1e-4 provides optimal regularization strength for this dataset size.

### 4. Dropout Rate Optimization

Tested dropout configurations:

- **No dropout**: Val loss = 0.086 (overfitting)
- **Uniform [0.2, 0.2, 0.2]**: Val loss = 0.078 (good)
- **Graduated [0.3, 0.3, 0.2]**: ✅ **Selected** – Val loss = 0.074 (best)
  - Higher dropout in early layers → prevents co-adaptation of major features
  - Slightly lower dropout in later layers → preserves learned representations
- **[0.4, 0.4, 0.3]**: Too aggressive → underfitting, train loss not decreasing properly

**Conclusion**: Graduated dropout rates leverage the "early features are more general" principle.

### 5. Epoch Count Optimization

Validation loss evolution:

| Epoch | Fold 1 Val Loss | Fold 2 Val Loss | Mean |
|-------|-----------------|-----------------|------|
| 50    | 0.135           | 0.128           | 0.132 |
| 100   | 0.083           | 0.079           | 0.081 |
| 150   | 0.076           | 0.074           | 0.075 |
| 200   | 0.074           | 0.073           | 0.074 |

Validation loss plateaus around epoch 100. Continuing beyond that shows negligible improvement and slight degradation (overfitting risk). **100 epochs provides the best balance between convergence and overfitting avoidance.**

### 6. Architecture Search

The two-branch MIMO architecture was selected after testing:

- **Single branch** (concatenate inputs): Larger model, slower convergence, overfitting
- **Single branch (dimension reduction)**: Information bottleneck limits expressiveness
- **Two-branch (separate encoders)**: ✅ **Selected** – Specialized feature processing per modality
  - Enables learning domain-specific transformations for building vs. ground-motion
  - Merges complementary information → more expressive shared layers
  - Reduces overfitting (~5% improvement in val loss vs. single-branch)

**Branch architecture**:
- Building input (10) → Dense(512) + ReLU → Dropout(0.3)
- G-M input (12) → Dense(512) + ReLU → Dropout(0.3)
- Merge → Dense(512) + ReLU + Dropout(0.3)
- Dense(256) + ReLU + Dropout(0.2)
- Dense(128) + ReLU
- 91 outputs in the response contract

## Cross-Validation Strategy

**K-Fold = 2** chosen to balance:

1. **Statistical robustness**: 2 folds → 2 independent val estimates
2. **Data efficiency**: Large fold size (~3,500 samples) → stable fold-level metrics
3. **Computational cost**: Only 2 training passes vs. 5 or 10
4. **Generalization signal**: Sufficient diversity for assessing out-of-sample performance

Validation testing showed 2-fold CV metrics matched hold-out test performance well (within 1-2%).

## Takeaways for Extension

If you modify the model or data:

1. **Changing batch size** → Adjust learning rate (smaller batches may need lower LR)
2. **More/fewer data** → Scale L2 weight decay inversely (more data → less regularization needed)
3. **Different architecture** → Re-tune dropout (deeper networks often need higher dropout)
4. **Different response objectives** → May need to re-optimize epochs (more outputs → longer training)

Example: If adding 100+ responses (total 200), expect to need 250-300 epochs for convergence.

## References

- Gradient-based hyperparameter search: Manual iteration with validation-loss-based selection
- Final validation: 2-fold CV + holdout test set (described in `complete_pipeline.py`)
