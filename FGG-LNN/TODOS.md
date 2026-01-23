# TODOs: Lorentz BatchNorm Train/Eval Discrepancy

## Problem Summary

Validation metrics are extremely noisy (oscillating 10-70% accuracy) while training converges normally (~99% train accuracy, ~0.35 loss). Root cause is the Lorentz BatchNorm implementation having **inconsistent statistics between train and eval**.

---

## Experiments Completed

### #1. Use Batch Statistics at Eval Time ✓ TESTED
**Result**: Solves noise, but **decreases final performance**

**Conclusion**: Running statistics ARE valuable when correct. The noise comes from the train/eval discrepancy, not from the running stats concept itself.

---

### #2. Proper Hyperbolic EMA via Geodesic Interpolation ✓ TESTED
**Result**: Noise persists (no improvement)

**Implementation notes** (bugs fixed during implementation):
- Tensor shape bug: `torch.stack([running_mean_full, mean], dim=-2)` with shape `[1, C, 1, 1]` created `[1, C, 1, 2, 1]`, but `lorentz_midpoint` expects `[..., num_points, dim]` with manifold dim LAST
- Fixed to flatten to `[1, C]`, stack to `[1, 2, C]`, then reshape back
- Initialization bug: `running_mean_full` was initialized to zeros, should be manifold origin `[1/√κ, 0, 0, ...]`

**Conclusion**: Geodesic interpolation of the mean alone doesn't fix the problem. The issue is deeper.

---

### #3. Remove BatchNorm Entirely ✓ TESTED
**Result**: Training is **extremely slow** (barely learns for 7 epochs, then slowly improves)

```
Epoch 1-7: loss ≈ 4.60, acc ≈ 1% (random chance)
Epoch 8+: finally starts learning
```

**Conclusion**: BatchNorm is critical for optimization in hyperbolic networks. Removing it is not viable without significant changes to initialization/learning rate.

---

### #2b. Compute Variance Around Running Mean ✓ TESTED
**Change**: Instead of `var = variance(batch, batch_mean)`, use `var = variance(batch, running_mean)`

**Result**: More noise during training, BUT **better final performance**

**Why this makes sense**:
| Approach | running_var computed around | eval uses mean |
|----------|---------------------------|----------------|
| Original | batch_mean (varies) | running_mean |
| New | running_mean | running_mean |

The new approach has **internally consistent statistics** at eval time. The noise is a transient artifact of running_mean not being stable early in training, but final statistics are more accurate.

---

## Key Insight: The Real Problem

The core issue is that **logmap and parallel_transport are nonlinear operations**:

```python
v_at_mean = self.manifold.logmap(mean_flat, x_flat)  # base point matters!
v_at_origin = self.manifold.parallel_transport(mean_flat, v_at_mean, origin)
```

During training: centering uses `batch_mean`
During eval: centering uses `running_mean`

Even if `running_mean ≈ population_mean`, the tangent vectors computed from different base points are **fundamentally different**. This is why #1 (batch stats at eval) works - it maintains consistency in the logmap base point.

---

## Philosophical Issue: BatchNorm vs Hyperbolic Geometry

**Tension**: Hyperbolic space's power comes from exponential volume growth (trees of depth d need coordinates ~e^d). BatchNorm that normalizes to "unit variance" effectively caps this depth, collapsing the hierarchy.

**What we want to preserve**:
- **Radial depth** (distance from origin) → represents hierarchy level
- While controlling **angular spread** → prevents optimization issues

Standard BatchNorm normalizes both, which may be fundamentally at odds with hyperbolic geometry.

---

## Recommended Next Steps

### Option A: Stabilize Current Approach (variance around running_mean)
Since this gives better final performance despite noise:
- Lower momentum (0.05 instead of 0.1) for more stable EMA
- Warmup period: don't update running stats for first N epochs
- Just ignore noisy val metrics during training, trust final result

### Option B: Hybrid Centering
Use batch_mean for logmap (stable centering), but running_var for scaling:
```python
# During eval:
mean = self._compute_centroid(x)  # batch mean for consistent logmap
var = self.running_var            # running variance for scaling
```
This decouples the nonlinear centering (which needs consistency) from the linear scaling (which can use running stats).

### Option C: Learnable Depth Scaling
Allow the network to learn how much radial growth each layer needs:
```python
self.depth_scale = nn.Parameter(torch.ones(1) * 1.5)
# After normalization:
v_out = v_shifted_space * F.softplus(self.depth_scale)
```

### Option D: InstanceNorm
No running statistics, no train/eval discrepancy. Each sample normalized independently. Worth trying if other approaches fail.

### Option E: Calibration Pass (#7 from original list)
After training, do a full pass to compute true population statistics, then freeze for eval. Tests whether the issue is EMA accumulation vs true population stats.

---

## Current State of LBatchNorm.py

The current implementation has:
- ✓ Geodesic EMA for running_mean (via `lorentz_midpoint`)
- ✓ Proper manifold initialization (origin, not zeros)
- ✓ Variance computed around running_mean (consistent with eval)
- ✗ Still has train/eval noise (but better final performance)

---

## Untested Ideas

### Only Normalize Variance, Don't Center
Skip the logmap/transport-to-origin step entirely. Just scale geodesic distances without recentering. Points keep their depth, just get their spread controlled.

### Channel-wise Normalization Only
Normalize across channels at each spatial position, not across spatial positions. Different spatial locations can live at different depths.

### Multiplicative vs Additive Centering
Instead of subtracting mean (additive), divide by some reference (multiplicative). Might preserve hierarchical ratios better.

---

## Related Issues to Monitor

- **Coordinate scale growth**: Deep hyperbolic networks have exponentially growing coordinates. Monitor `x[:,0].max()` (time component) through layers.

- **16 BN layers total**: ResNet-18 has 2 BN per block × 8 blocks = 16 BN layers. Errors compound through all of them.

- **Fréchet mean of batch means ≠ Fréchet mean of all points**: The geodesic EMA of batch means may not converge to true population mean due to nonlinearity of Fréchet mean.
