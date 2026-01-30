"""
Week 4: Enhanced SPM model with transformer integration.
Fits an SPM to measured data and can be used alongside the transformer for ensemble predictions.
"""
import numpy as np
from scipy.optimize import curve_fit


class EnhancedSPM:
    """
    Simplified Single-Particle Model (SPM) for battery simulation.
    
    Features:
    - OCV(SoC) mapping (polynomial fit)
    - Ohmic resistance (R0)
    - Charge transfer resistance (Rct)
    - Double-layer capacitance (Cdl)
    
    Can be trained on discharge data to estimate parameters.
    """

    def __init__(self, soc_breakpoints=None, ocv_values=None, R0=0.05, Rct=0.02, Cdl=5000):
        """
        Initialize SPM.
        
        Args:
            soc_breakpoints: SoC points for OCV (0-100%)
            ocv_values: OCV values at those SoC points
            R0: Ohmic resistance (ohms)
            Rct: Charge transfer resistance (ohms)
            Cdl: Double-layer capacitance (F)
        """
        self.R0 = R0
        self.Rct = Rct
        self.Cdl = Cdl
        
        # Default OCV curve if not provided
        if soc_breakpoints is None:
            self.soc_breakpoints = np.array([0, 20, 40, 60, 80, 100])
            self.ocv_values = np.array([3.0, 3.3, 3.5, 3.7, 3.85, 4.0])
        else:
            self.soc_breakpoints = soc_breakpoints
            self.ocv_values = ocv_values
        
        # Fit polynomial to OCV curve
        self.ocv_poly = np.polyfit(self.soc_breakpoints, self.ocv_values, 3)

    def ocv(self, soc):
        """Get OCV for given SoC (0-100%)."""
        return np.polyval(self.ocv_poly, soc)

    def simulate_voltage(self, soc_profile, current_profile, dt_vec):
        """
        Simulate voltage during discharge.
        
        Args:
            soc_profile: SoC array (0-100%)
            current_profile: Current array (A, positive = discharge)
            dt_vec: Time step array (hours)
        
        Returns:
            Simulated voltage array
        """
        n = len(soc_profile)
        v_ocv = np.array([self.ocv(s) for s in soc_profile])
        
        # Simple RC element for transient response
        v_rc = np.zeros(n)
        for i in range(1, n):
            tau = self.Rct * self.Cdl  # Time constant (seconds)
            dt_s = dt_vec[i] * 3600  # Convert hours to seconds
            alpha = np.exp(-dt_s / tau) if tau > 0 else 0
            v_rc[i] = alpha * v_rc[i-1] + (1 - alpha) * self.Rct * current_profile[i]
        
        # Total voltage = OCV - ohmic drop - RC transient
        v_terminal = v_ocv - self.R0 * current_profile - v_rc
        return v_terminal

    @staticmethod
    def fit_ohmic_resistance(voltage_array, current_array, soc_array=None):
        """
        Estimate R0 from V-I data using least squares.
        
        Args:
            voltage_array: Terminal voltage (V)
            current_array: Current (A)
            soc_array: Optional SoC profile to estimate OCV
        
        Returns:
            Estimated R0 (ohms)
        """
        if soc_array is not None:
            ocv_est = np.mean(voltage_array + 0.05 * current_array)
        else:
            ocv_est = np.mean(voltage_array)
        
        residual = voltage_array - ocv_est
        r0 = -np.sum(current_array * residual) / (np.sum(current_array ** 2) + 1e-9)
        return float(max(0.001, r0))

    def __repr__(self):
        return f"EnhancedSPM(R0={self.R0:.4f}Ω, Rct={self.Rct:.4f}Ω, Cdl={self.Cdl:.0f}F)"
