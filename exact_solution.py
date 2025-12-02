import numpy as np

pi = np.pi

def exact_solution(x):
    X = x[:, 0:1]
    t = x[:, 1:2]
    
    u = np.sin(pi * X) * np.cos(pi * t)
    phi = 0.5 * np.sin(pi * X) * np.cos(pi * t)
    
    return np.hstack([u, phi])
