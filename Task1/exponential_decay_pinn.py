"""
Physics-Informed Neural Network for Exponential Decay (Zero-Shot Method)

This implementation learns the exponential decay ODE purely from physics constraints:
    dy/dt + decay_rate * y = 0
    Initial condition: y(0) = y0

No target data is used during training - the network learns by minimizing physics loss only.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad


class PINN(nn.Module):
    """Physics-Informed Neural Network for exponential decay."""
    
    def __init__(self, hidden_layers=[32, 32, 32]):
        super(PINN, self).__init__()
        
        # Build network layers
        layers = []
        layers.append(nn.Linear(1, hidden_layers[0]))
        layers.append(nn.Tanh())
        
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t):
        """Forward pass through the network."""
        return self.network(t)


class ExponentialDecayPINN:
    """PINN solver for exponential decay using zero-shot learning."""
    
    def __init__(self, decay_rate=0.5, y0=1.0, t_max=10.0, n_collocation=100):
        """
        Initialize the PINN solver.
        
        Args:
            decay_rate: Decay constant (lambda in dy/dt = -lambda*y)
            y0: Initial condition y(0)
            t_max: Maximum time value for training
            n_collocation: Number of collocation points for physics loss
        """
        self.decay_rate = decay_rate
        self.y0 = y0
        self.t_max = t_max
        self.n_collocation = n_collocation
        
        # Initialize network
        self.model = PINN(hidden_layers=[32, 32, 32])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Loss history
        self.loss_history = []
        self.physics_loss_history = []
        self.ic_loss_history = []
        
    def compute_physics_loss(self, t_collocation):
        """
        Compute physics loss based on the ODE: dy/dt + decay_rate * y = 0
        
        Args:
            t_collocation: Time points for evaluating physics
            
        Returns:
            Physics loss
        """
        # Enable gradient computation
        t_collocation.requires_grad_(True)
        
        # Forward pass
        y_pred = self.model(t_collocation)
        
        # Compute dy/dt using automatic differentiation
        dy_dt = grad(
            outputs=y_pred,
            inputs=t_collocation,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Physics residual: dy/dt + decay_rate * y = 0
        physics_residual = dy_dt + self.decay_rate * y_pred
        
        # Mean squared error of physics residual
        physics_loss = torch.mean(physics_residual ** 2)
        
        return physics_loss
    
    def compute_initial_condition_loss(self):
        """
        Compute loss for initial condition: y(0) = y0
        
        Returns:
            Initial condition loss
        """
        t_zero = torch.zeros(1, 1)
        y_pred_zero = self.model(t_zero)
        ic_loss = (y_pred_zero - self.y0) ** 2
        
        return ic_loss
    
    def train(self, epochs=3000, verbose=True, verbose_freq=500):
        """
        Train the PINN using zero-shot method (physics loss only).
        
        Args:
            epochs: Number of training epochs
            verbose: Whether to print training progress
            verbose_freq: Frequency of progress printing
        """
        print("=" * 60)
        print("ZERO-SHOT PINN TRAINING: Exponential Decay")
        print("=" * 60)
        print(f"Decay rate: {self.decay_rate}")
        print(f"Initial condition: y(0) = {self.y0}")
        print(f"Time domain: [0, {self.t_max}]")
        print(f"Collocation points: {self.n_collocation}")
        print(f"Training epochs: {epochs}")
        print("=" * 60)
        print("\nTraining with PHYSICS LOSS ONLY (no target data)...\n")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Generate random collocation points in [0, t_max]
            t_collocation = torch.rand(self.n_collocation, 1) * self.t_max
            
            # Compute losses
            physics_loss = self.compute_physics_loss(t_collocation)
            ic_loss = self.compute_initial_condition_loss()
            
            # Total loss (weighted combination)
            total_loss = physics_loss + 10.0 * ic_loss  # Weight IC loss higher
            
            # Backpropagation
            total_loss.backward()
            self.optimizer.step()
            
            # Record losses
            self.loss_history.append(total_loss.item())
            self.physics_loss_history.append(physics_loss.item())
            self.ic_loss_history.append(ic_loss.item())
            
            # Print progress
            if verbose and (epoch % verbose_freq == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:5d} | Total Loss: {total_loss.item():.6f} | "
                      f"Physics Loss: {physics_loss.item():.6f} | "
                      f"IC Loss: {ic_loss.item():.6f}")
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
    
    def predict(self, t):
        """
        Make predictions at given time points.
        
        Args:
            t: Time points (numpy array or torch tensor)
            
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        
        if isinstance(t, np.ndarray):
            t_tensor = torch.FloatTensor(t.reshape(-1, 1))
        else:
            t_tensor = t.reshape(-1, 1)
        
        with torch.no_grad():
            y_pred = self.model(t_tensor)
        
        return y_pred.numpy().flatten()
    
    def analytical_solution(self, t):
        """
        Compute analytical solution: y(t) = y0 * exp(-decay_rate * t)
        
        Args:
            t: Time points (numpy array)
            
        Returns:
            Analytical solution values
        """
        return self.y0 * np.exp(-self.decay_rate * t)
    
    def plot_results(self, n_points=200):
        """
        Plot PINN predictions vs analytical solution and loss history.
        
        Args:
            n_points: Number of points for plotting
        """
        # Generate time points
        t = np.linspace(0, self.t_max, n_points)
        
        # Get predictions and analytical solution
        y_pred = self.predict(t)
        y_true = self.analytical_solution(t)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Predictions vs Analytical Solution
        ax1 = axes[0, 0]
        ax1.plot(t, y_true, 'b-', linewidth=2, label='Analytical Solution')
        ax1.plot(t, y_pred, 'r--', linewidth=2, label='PINN Prediction')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('y(t)', fontsize=12)
        ax1.set_title('Zero-Shot PINN: Exponential Decay', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Error
        ax2 = axes[0, 1]
        error = np.abs(y_pred - y_true)
        ax2.plot(t, error, 'g-', linewidth=2)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Absolute Error', fontsize=12)
        ax2.set_title('Prediction Error', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Total Loss History
        ax3 = axes[1, 0]
        ax3.semilogy(self.loss_history, 'b-', linewidth=1.5)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Total Loss (log scale)', fontsize=12)
        ax3.set_title('Training Loss History', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Physics vs IC Loss
        ax4 = axes[1, 1]
        ax4.semilogy(self.physics_loss_history, 'r-', linewidth=1.5, label='Physics Loss')
        ax4.semilogy(self.ic_loss_history, 'g-', linewidth=1.5, label='IC Loss')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss (log scale)', fontsize=12)
        ax4.set_title('Loss Components', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('exponential_decay_pinn_results.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as 'exponential_decay_pinn_results.png'")
        plt.show()
        
        # Print error statistics
        print("\n" + "=" * 60)
        print("PREDICTION ACCURACY")
        print("=" * 60)
        print(f"Mean Absolute Error: {np.mean(error):.6f}")
        print(f"Max Absolute Error: {np.max(error):.6f}")
        print(f"Relative Error (L2): {np.linalg.norm(error) / np.linalg.norm(y_true):.6f}")
        print("=" * 60)


def main():
    """Main function to demonstrate zero-shot PINN for exponential decay."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize PINN solver
    print("\nInitializing PINN for exponential decay...\n")
    pinn = ExponentialDecayPINN(
        decay_rate=0.5,      # Decay constant
        y0=1.0,              # Initial condition
        t_max=10.0,          # Time domain
        n_collocation=100    # Number of physics evaluation points
    )
    
    # Train using zero-shot method (physics loss only)
    pinn.train(epochs=3000, verbose=True, verbose_freq=500)
    
    # Plot results
    pinn.plot_results(n_points=200)
    
    # Test predictions at specific points
    print("\nSample Predictions:")
    print("=" * 60)
    test_times = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    predictions = pinn.predict(test_times)
    analytical = pinn.analytical_solution(test_times)
    
    print(f"{'Time':<10} {'PINN':<15} {'Analytical':<15} {'Error':<15}")
    print("-" * 60)
    for t, pred, true in zip(test_times, predictions, analytical):
        error = abs(pred - true)
        print(f"{t:<10.2f} {pred:<15.6f} {true:<15.6f} {error:<15.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
