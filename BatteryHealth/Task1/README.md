# Exponential Decay PINN - Zero-Shot Learning

This project implements a **Physics-Informed Neural Network (PINN)** to learn exponential decay behavior using the **zero-shot method** - training purely from physics constraints without any target data.

## Problem Statement

The PINN learns to solve the exponential decay ODE:
```
dy/dt + λy = 0
y(0) = y₀
```

where:
- `λ` is the decay rate
- `y₀` is the initial condition

## Zero-Shot Method

Unlike traditional supervised learning, this PINN:
- ❌ **No target data** is provided during training
- ✅ **Only physics loss** is used (the ODE residual)
- ✅ Learns by ensuring predictions satisfy the differential equation
- ✅ Uses automatic differentiation to compute dy/dt

## How It Works

1. **Neural Network**: Takes time `t` as input, outputs `y(t)`
2. **Physics Loss**: Computes `dy/dt + λy` using automatic differentiation - should be zero
3. **Initial Condition Loss**: Ensures `y(0) = y₀`
4. **Training**: Random collocation points are sampled; network adjusts weights to minimize physics residual

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete demo:
```bash
python exponential_decay_pinn.py
```

This will:
1. Initialize a PINN with random weights
2. Train using only physics loss (10,000 epochs)
3. Compare predictions with analytical solution
4. Generate visualization plots
5. Print accuracy metrics

## Results

The PINN successfully learns exponential decay behavior without ever seeing target data! The network discovers the solution purely by satisfying the physics equations.

**Expected output:**
- Training loss converges to near-zero
- Predictions closely match analytical solution: `y(t) = y₀ * exp(-λt)`
- Visualization showing prediction accuracy

## Key Features

- **Zero-shot learning**: No data labels required
- **Automatic differentiation**: PyTorch computes derivatives automatically
- **Physics-driven**: Network learns from fundamental equations
- **Collocation points**: Random sampling ensures good coverage
- **Visualization**: Comprehensive plots of predictions and losses

## Parameters

You can customize the PINN in `main()`:
- `decay_rate`: Controls how fast the decay occurs (λ)
- `y0`: Initial value at t=0
- `t_max`: Maximum time for training domain
- `n_collocation`: Number of physics evaluation points
- `epochs`: Training iterations

## Theory

The physics loss is:
```
L_physics = mean((dy/dt + λy)²)
```

The initial condition loss is:
```
L_ic = (y(0) - y₀)²
```

Total loss:
```
L_total = L_physics + 10 * L_ic
```

The network learns by gradient descent to minimize this combined loss.
