# Exponential Decay PINN - Zero-Shot Learning

This code trains a small physics-informed neural network (PINN) to solve the exponential decay equation

  dy/dt + λ y = 0,  y(0) = y0

Purpose
- Learn y(t) using only the physics. No labeled solution data is required.

Method 
- I build a neural network that maps time `t` to a prediction `y(t)`.
- I compute the derivative dy/dt using automatic differentiation and form a physics loss from the ODE residual dy/dt + λ y.
- I also add a loss for the initial condition so the network starts at y(0) = y0.
- The optimizer reduces the combined loss so the network learns a function that satisfies the ODE and the initial condition.

How to run
1. Install the requirements:

```bash
pip install -r requirements.txt
```

2. Run the script:

```bash
python exponential_decay_pinn.py
```

What the script does
- Samples collocation points in time and trains the PINN to minimize the physics residual and the initial condition error.
- Saves plots that compare the learned solution to the analytic solution y(t) = y0 * exp(-λ t).

Expected results
- The training loss should drop and the learned curve should match the analytic solution for the chosen `λ` and `y0`.

Main parameters you can change
- `decay_rate` (λ): how fast the solution decays.
- `y0`: initial value at t = 0.
- `t_max`: end time for training.
- `n_collocation`: number of physics points sampled.
- `epochs`: training iterations.

Theory (brief)
- Physics loss: L_physics = mean((dy/dt + λ y)^2). This pushes the network to satisfy the differential equation.
- Initial condition loss: L_ic = (y(0) - y0)^2. This forces the network to match the starting value.
- Total loss: L_total = L_physics + α * L_ic (α is a weight used in the code).

Notes
- This method is useful when you know the governing equation but do not have labeled solution data. The network learns by enforcing the physics.

