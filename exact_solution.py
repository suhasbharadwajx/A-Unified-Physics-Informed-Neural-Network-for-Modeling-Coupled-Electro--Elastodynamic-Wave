import numpy as np

pi = np.pi

def exact_solution(x):
    """
    Exact standing-wave solution for validation.
    
    Args:
        x: Numpy array of shape (N, 2) containing (x, t) coordinates
    
    Returns:
        out: Array of shape (N, 2) containing (u_exact, φ_exact)
    """
    X = x[:, 0:1]
    t = x[:, 1:2]
    
    u = np.sin(pi * X) * np.cos(pi * t)
    phi = 0.5 * np.sin(pi * X) * np.cos(pi * t)
    
    return np.hstack([u, phi])
