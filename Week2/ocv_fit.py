import numpy as np
import os


def load_coeffs(task_dir=None):
    if task_dir is None:
        task_dir = os.path.dirname(__file__)
    coeffs_path = os.path.join(task_dir, "ocv_coeffs.npy")
    coeffs = np.load(coeffs_path)
    return coeffs


def soc_to_voltage(soc, coeffs=None):
    """Return voltage for given SOC (scalar or array)."""
    coeffs = coeffs if coeffs is not None else load_coeffs()
    return np.polyval(coeffs, soc)


if __name__ == '__main__':
    coeffs = load_coeffs()
    print("Loaded OCV polynomial coefficients:", coeffs)
    # quick smoke test
    print("V at SOC=1.0:", soc_to_voltage(1.0, coeffs))