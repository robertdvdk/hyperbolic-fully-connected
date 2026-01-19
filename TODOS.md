# TODOs: Lorentz BatchNorm Train/Eval Discrepancy

## Problem Summary

Validation metrics are extremely noisy (oscillating 10-70% accuracy) while training converges normally (~99% train accuracy, ~0.35 loss). Root cause is the Lorentz BatchNorm implementation using **linear EMA in hyperbolic space**, which is geometrically invalid.

The linear interpolation of space components:
```python
running_mean = (1-α) * old_mean + α * batch_mean
```
does not correspond to a meaningful hyperbolic average. This causes running statistics to drift from true population statistics, resulting in inconsistent normalization at eval time.

---

## Potential Solutions

### 1. Use Batch Statistics at Eval Time (Quick Fix)
**File**: `layers/LBatchNorm.py`

Modify `forward()` to always use batch statistics regardless of `self.training`:
- Remove the `if self.training` / `else` branch
- Always compute centroid and variance from the current batch
- Requires sufficiently large eval batch size for stability (e.g., 128+)

**Pros**: Simple, immediate fix
**Cons**: Eval results depend on batch size; single-sample inference won't work

---

### 2. Proper Hyperbolic EMA via Geodesic Interpolation
**File**: `layers/LBatchNorm.py`

Replace linear EMA with geodesic-based update using `lorentz_midpoint`:

```python
# Current (incorrect):
self.running_mean_space.mul_(1 - self.momentum).add_(
    mean[:, 1:, :, :] * self.momentum
)

# Proposed (geometrically correct):
# Store full Lorentz point, not just space components
running_mean_full = self.manifold.lorentz_midpoint(
    torch.stack([self.running_mean_full, mean], dim=-2),
    weights=torch.tensor([1 - self.momentum, self.momentum])
)
```

**Pros**: Geometrically principled, maintains valid hyperbolic statistics
**Cons**: More compute per update; need to store full Lorentz point instead of just space components

---

### 3. Remove BatchNorm Entirely
**File**: `layers/LResBlock.py`

Remove `self.bn1` and `self.bn2` from the ResBlock; rely on other regularization:
- Increase weight decay
- Add dropout (need hyperbolic-aware dropout or apply in tangent space)
- Use gradient clipping (already present)

**Pros**: Eliminates the problem entirely
**Cons**: May hurt convergence speed; hyperbolic networks might need some form of normalization

---

### 4. Lorentz InstanceNorm
**File**: New file `layers/LInstanceNorm.py`

Implement instance normalization that normalizes each sample independently:
- No running statistics needed
- Compute centroid and variance per-sample across spatial dimensions
- Train/eval behavior is identical

**Pros**: No train/eval discrepancy; works for any batch size including 1
**Cons**: You noted worse final accuracy in other codebase; less "global" normalization effect

---

### 5. Lorentz GroupNorm / LayerNorm
**File**: New file `layers/LGroupNorm.py` or `layers/LLayerNorm.py`

Implement GroupNorm or LayerNorm for Lorentz manifold:
- GroupNorm: normalize over groups of channels per sample
- LayerNorm: normalize over all channels per sample
- No running statistics, so no train/eval discrepancy

**Pros**: Commonly used in transformers; batch-size independent
**Cons**: Different normalization characteristics; may need hyperparameter tuning

---

### 6. Increase BatchNorm Momentum
**File**: `layers/LResBlock.py` (where BN is instantiated)

Change from default momentum=0.1 to higher value (0.3-0.5):
```python
self.bn1 = LorentzBatchNorm2d(num_features=output_dim, manifold=manifold, momentum=0.3)
```

**Pros**: Simple config change; makes running stats track batch stats more closely
**Cons**: Doesn't fix the fundamental geometric issue; running stats will be noisier

---

### 7. Hybrid: Batch Stats During Training, Accumulated Stats for Eval
**File**: `layers/LBatchNorm.py`

After training completes, do a full pass over training data to compute proper population statistics using `lorentz_midpoint` over all samples, then freeze these for eval.

**Pros**: Gets correct population statistics without modifying training
**Cons**: Requires post-training calibration step; extra complexity

---

## Recommended Investigation Order

1. **Quick validation**: Try solution #1 (batch stats at eval) to confirm the diagnosis
2. **Proper fix**: Implement solution #2 (geodesic EMA) for production use
3. **Alternative**: If #2 proves problematic, try #4 (InstanceNorm) or #5 (GroupNorm)
4. **Ablation**: Try #3 (no BatchNorm) to understand how much BN contributes to final accuracy

---

## Related Issues to Monitor

- **Coordinate scale growth**: Deep hyperbolic networks have exponentially growing coordinates. Monitor `x[:,0].max()` (time component) through layers. If exceeding ~10^6, numerical issues may compound the BN problems.

- **Variance computation**: Current `_compute_variance()` returns mean squared geodesic distance. Verify this is numerically stable at large coordinate scales.

- **16 BN layers total**: ResNet-18 has 2 BN per block × 8 blocks = 16 BN layers. Errors compound through all of them.
