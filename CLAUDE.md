# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements hyperbolic neural network layers using the **Lorentz model** (hyperboloid model) of hyperbolic geometry. The implementation follows a geometrically principled "distance-to-hyperplane" formulation for fully connected layers, analogous to how Euclidean linear layers can be interpreted as computing signed, scaled distances to hyperplanes.

The layers are designed to operate natively in hyperbolic space, which is useful for data with hierarchical structure (e.g., trees, graphs) since hyperbolic geometry naturally represents hierarchies through its exponential volume growth.

## Package Management

This project uses **uv** for dependency management:

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Linux/macOS

# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev
```

Dependencies include PyTorch with CUDA support (cu126), Lightning, wandb, geoopt, and hypll.

## Running Experiments

### CIFAR-100 Experiment
```bash
cd cifar_exp
python main.py
```

### WordNet Classification
```bash
cd wordnet_exp
python exp.py
```

### Experiment 2 (MLR with features)
```bash
# Preprocess features first
python exp2/preprocess_features.py

# Run hyperparameter search
python exp2/hyperparam_search.py

# Train from preprocessed features
python exp2/train_from_features.py
```

### Job Scripts
SLURM job scripts are available in the `jobs/` directory for running experiments on a cluster.

## Mathematical Foundation

### Distance-to-Hyperplane Formulation

The core innovation is a generalization of fully connected layers to the Lorentz manifold that converges to the Euclidean case as curvature approaches zero.

#### Spacelike Vectors and Hyperplanes

Each output neuron is parametrized by weights `w ∈ R^n` and bias `b ∈ R`. These define a **spacelike vector** `v` that is orthogonal to a geodesic hyperplane in the Lorentz model:

```
v = [sinh(-√κ b ||w||) ||w||, cosh(-√κ b ||w||) w]
```

Where:
- The first component is the "time" component in Minkowski space
- The remaining components are the "space" components
- `κ` is the curvature parameter (stored as `k` in code)

#### Pre-activations: Signed, Scaled Distances

For input `x` on the hyperboloid, the pre-activation is computed as:

```
z_i = (1/√κ) arcsinh(√κ (x ∘ v^(i)))
```

Where `∘` denotes the Minkowski inner product (Lorentz inner product). This computes the signed, scaled hyperbolic distance from `x` to the hyperplane defined by `v^(i)`.

**Key insight**: The scaling is *non-linear* (we scale the sinh of the angle, not the distance itself), but this converges to the Euclidean dot product as curvature approaches zero.

#### Lorentzian Activations

Activation functions are applied in a "Lorentzian" form:

```
f_Lorentzian,κ(x) = (1/√κ) arcsinh(√κ f((1/√κ) sinh(√κ x)))
```

This ensures that as `κ → 0`, the Lorentzian activation converges to the standard Euclidean activation `f`.

#### Output Construction

The output point on the manifold is constructed so its distances to coordinate hyperplanes equal the activations:

```
y = [√(1/κ + ||y_space||²), (1/√κ) sinh(√κ a)]
```

Where `a` is the vector of activations after applying the activation function.

#### Inference Optimization: Caching V

During training, the spacelike vectors `V` are computed from parameters `(W, b)` using sinh/cosh. During inference, since parameters are fixed, we can:
1. Pre-compute and cache the matrix `V`
2. Use only the cached `V` for forward passes
3. The mapping from `(w, b)` to `v` is invertible, so no extra memory is needed

This avoids repeated evaluation of expensive transcendental functions.

## Critical Numerical Stability Issues

### ⚠️ The Exponential Coordinate Problem

**This is the most important thing to understand when working with this codebase.**

#### Why Coordinates Grow Exponentially

In hyperbolic geometry, distances grow *logarithmically* with Euclidean coordinates in the Lorentz model. This means:
- To represent a hyperbolic distance that grows linearly (e.g., distance = 10, 20, 30...)
- Euclidean coordinates must grow exponentially (e.g., coordinates ~ e^10, e^20, e^30...)

**This exponential growth is mathematically correct and necessary.** It's a fundamental property of hyperbolic space.

#### The Float32 Catastrophe

Through multiple layers, points naturally drift far from the origin:
- By layer 4, time components can reach ~280,000
- Space components can reach ~45,000 per channel
- With 128 channels, space norms squared are ~10^11

The Lorentz constraint requires: `-time² + ||space||² = -1/κ`

With float32's ~7 significant digits, checking this constraint involves:
```
-7.9e10 + 2.6e11 ≈ -1.0
```

This is catastrophic cancellation - we're subtracting numbers at 10^11 scale to get -1. Float32 loses all meaningful precision.

#### Current Solution: Float64

The code currently uses **double precision everywhere** (`float64`/`torch.double()`). This works because float64 has ~15 significant digits, enough to maintain the manifold constraint even at large coordinate scales.

**However, this is not ideal:**
- 2x memory usage
- Slower computation (especially on GPUs)
- Still doesn't solve the fundamental problem at very deep networks

#### Areas Needing Improvement

The following approaches need to be implemented and tested:

1. **Relative Error Checking**: Instead of checking `|norm - target| < ε`, check `|norm - target| / scale < ε` where scale is the magnitude of coordinates.

2. **Numerically Stable Manifold Checks**: Check `time = √(||space||² + 1/κ)` directly instead of computing the difference `-time² + ||space||²`.

3. **Adaptive Tolerance**: Scale assertion tolerances based on coordinate magnitude: `tolerance = max(1e-3, 1e-6 * coordinate_scale)`.

4. **Point Normalization**: Periodically map points back toward the origin using logarithmic/exponential maps while preserving their relative positions.

5. **Mixed Precision**: Use float32 for most operations but cast to float64 only for critical manifold constraint computations.

6. **Gradient Clipping**: Prevent unbounded growth during training with careful gradient clipping and weight regularization.

**When working on stability, focus on `layers/lorentz.py` and `layers/lorentz_fc.py`, particularly:**
- `direct_concat()` in lorentz.py:224 - this is where the assertion failures occur
- `projection_space_orthogonal()` in lorentz.py:184 - this is where points can grow without bounds
- The manifold constraint checks throughout

## Architecture

### Core Components

#### `layers/` Package

The main package containing hyperbolic geometry implementations:

- **`lorentz.py`**: Core `Lorentz` class implementing the Lorentz model
  - **Curvature parameter** `k`: Stored as `c_softplus_inv` with softplus parameterization for positivity
  - **Exponential map** (`expmap0`, `expmap`): Maps tangent vectors from tangent space to the hyperboloid
    - `expmap0`: From origin (simplified formula)
    - `expmap`: From arbitrary base point
  - **Logarithmic map** (`logmap0`, `logmap`): Inverse of exponential map, projects hyperboloid points to tangent space
  - **Parallel transport**: Transports tangent vectors between different tangent spaces while preserving Riemannian metric
  - **Direct concatenation** (`direct_concat`): Combines multiple points on the manifold into a single point (used in convolutions)
    - ⚠️ This is a major source of numerical instability
  - **Lorentz midpoint** (`lorentz_midpoint`): Computes Fréchet mean of multiple points
  - **Poincaré-Lorentz conversions**: Convert between Poincaré ball and Lorentz hyperboloid models

- **`lorentz_fc.py`**: Hyperbolic fully connected and convolutional layers
  - **`Lorentz_fully_connected`**: Implements the distance-to-hyperplane formulation
    - `create_spacelike_vector()`: Constructs vector `v` from weights `U` and bias `a`
    - `compute_output_space()`: Computes `activation(x @ V)` where V is matrix of spacelike vectors
    - `projection_space_orthogonal()`: Projects output back to hyperboloid by computing time component
      - ⚠️ This can produce arbitrarily large coordinates
    - `mlr()`: Multinomial logistic regression mode - computes signed distances directly without manifold projection
    - **Supports `do_mlr=True`** for classification heads
  - **`Lorentz_Conv2d`**: Convolutional layer for Lorentz manifolds
    - Unfolds spatial patches using `unfold()`
    - Uses `direct_concat()` to combine each patch on the manifold
    - Applies `Lorentz_fully_connected` to combined patches
    - Reshapes output back to spatial format

- **`LBNorm.py`**: `LorentzBatchNorm` - Batch normalization for Lorentz manifolds
  - Computes **Fréchet mean** (Lorentz midpoint) of batch as center
  - Uses `logmap()` to move points to tangent space at mean
  - Uses `parallel_transport()` to transport tangent vectors to origin
  - Normalizes by Fréchet variance
  - Transports to learnable center `β` and maps back to manifold
  - Maintains running statistics (exponential moving average in tangent space)

- **`poincare.py`**: Poincaré ball model (alternative to Lorentz)
  - `Poincare`: Core manifold with curvature parameter `c`
  - `project()`: Projects points back into Poincaré ball (ensures norm < 1/√c)
  - `Poincare_linear`: Distance-to-hyperplane formulation in Poincaré ball
  - `PoincareActivation`: Applies activations by mapping to tangent space, applying function, mapping back

- **`chen.py`**: `ChenLinear` - Alternative formulation (Chen et al., 2020)
  - Uses standard Euclidean linear layer followed by projection
  - Optional normalization with learnable scale parameter
  - Simpler but less theoretically motivated than distance-to-hyperplane approach

- **`bdeir.py`**: Additional experimental layer implementation

### Experiment Structures

#### CIFAR Experiment (`cifar_exp/`)

- **`model.py`**: ResNet-style architecture using Lorentz layers
  - **`ResNetBlock`**: Two Lorentz convolutions (no skip connection in current version)
  - **`ourModel`**: Complete model with 7 residual blocks
    - Initial conv: 3 → 64 channels with stride 2
    - Blocks progressively increase channels: 64 → 64 → 128 → 128 → 256 → 256 → 512
    - Uses `LorentzBatchNorm` after initial conv (commented out in blocks)
    - Global Lorentz midpoint pooling to aggregate spatial features
    - Final classifier in MLR mode (`do_mlr=True`)

- **`main.py`**: Training script
  - CIFAR-100 classification (100 classes)
  - 200 epochs with batch size 256
  - **Maps images to hyperboloid** with `manifold.expmap0()` before processing
  - Optimizer: Adam with lr=1e-2, weight_decay=5e-4
  - Learning rate scheduling:
    - Warmup: 10 epochs from 0.01x to 1x
    - MultiStep decay at epochs 50, 110, 150 by factor 0.2
  - ⚠️ **Currently requires `.double()` everywhere** for numerical stability

#### Experiment 2 (`exp2/`)

Multinomial logistic regression on preprocessed features. Uses distance-to-hyperplane formulation for classification without deep networks.

### Key Usage Patterns

1. **Always initialize manifold first**:
   ```python
   manifold = Lorentz(k=1.0)  # Curvature k > 0
   layer = Lorentz_fully_connected(in_features=128, out_features=64, manifold=manifold)
   ```

2. **Map Euclidean inputs to hyperboloid**:
   ```python
   # For image data: map each pixel vector
   x_euclidean = images.permute(0, 2, 3, 1)  # (B, H, W, C)
   x_hyperbolic = manifold.expmap0(x_euclidean)  # (B, H, W, C+1)
   x_hyperbolic = x_hyperbolic.permute(0, 3, 1, 2)  # (B, C+1, H, W)
   ```

3. **Classification heads use MLR mode**:
   ```python
   classifier = Lorentz_fully_connected(
       in_features=512,
       out_features=num_classes,
       manifold=manifold,
       do_mlr=True  # Output logits directly, no manifold projection
   )
   ```

4. **Always use double precision** (until stability is improved):
   ```python
   model = MyModel(manifold).double().cuda()
   images = images.double().cuda()
   ```

5. **Manifold constraint checking**:
   - Many operations have assertions to verify points lie on the manifold
   - These check `-time² + ||space||² = -1/κ` within tolerance
   - ⚠️ These assertions fail in float32 due to the exponential coordinate problem
   - When debugging, look for assertions in `direct_concat()` and similar methods

## Development Notes

### Current State

- **Precision requirement**: All experiments currently use `float64` (double precision) for numerical stability
- **CUDA required**: Training assumes GPU availability (`.cuda()` calls throughout)
- **Reproducibility**: Random seeds are set via `seed_everything(seed)` in experiments
- **einops**: Used for complex tensor rearrangements in convolutional layers

### Known Issues & Areas for Improvement

1. **Float32 instability**: The exponential coordinate growth causes float32 to fail. This is the primary development focus.
   - See "Critical Numerical Stability Issues" section above
   - Potential solutions need implementation and testing

2. **Memory usage**: Float64 uses 2x memory compared to float32
   - Limits batch size and model size
   - Slower on GPU

3. **Batch normalization interaction**: LorentzBatchNorm may exacerbate coordinate growth
   - Currently commented out in some ResNet blocks
   - Needs investigation

4. **Residual connections**: The residual connections in the original design are commented out
   - May have contributed to instability
   - Worth revisiting after stability improvements

5. **Gradient flow**: Deep hyperbolic networks may have gradient flow issues
   - No explicit gradient clipping currently implemented
   - Weight initialization uses "eye" or "kaiming" strategies

### Testing Numerical Stability

When working on stability improvements, test with:

```python
# Create a simple test that goes deep enough to expose issues
manifold = Lorentz(k=1.0)
x = torch.randn(2, 3, 8, 8).float()  # Start with float32

# Map to hyperboloid
x = x.permute(0, 2, 3, 1)
x = manifold.expmap0(x.double())  # May need .double() initially
x = x.permute(0, 3, 1, 2)

# Apply multiple conv layers
for i in range(4):
    conv = Lorentz_Conv2d(in_channels=x.shape[1]-1, out_channels=64,
                          kernel_size=3, manifold=manifold, padding='same')
    x = conv(x)
    print(f"Layer {i}: time range [{x[:,0].min():.2f}, {x[:,0].max():.2f}], "
          f"space range [{x[:,1:].min():.2f}, {x[:,1:].max():.2f}]")

    # Check if still on manifold
    lorentz_norm = -x[:,0:1]**2 + (x[:,1:]**2).sum(dim=1, keepdim=True)
    print(f"  Manifold constraint deviation: {(lorentz_norm + 1/manifold.k()).abs().mean():.2e}")
```

Watch for:
- Exponential growth of coordinates (expected)
- Manifold constraint deviation growing (problem - should stay small relative to coordinate scale)
- NaN or Inf values (critical failure)

### Code Style

- Use descriptive variable names: `lorentz_norm_sq` not `ln2`
- Document tensor shapes in comments: `# (B, C, H, W)`
- Assertions should have informative error messages
- When adding epsilon for stability, comment why: `# Add eps to prevent sqrt(0)`
