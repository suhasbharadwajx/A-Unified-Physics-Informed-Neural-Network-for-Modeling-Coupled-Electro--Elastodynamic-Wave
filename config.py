"""Configuration file for 1D Piezoelectric PINNs"""

# Material parameters
MATERIAL_PARAMS = {
    'rho': 1.0,      # mass density
    'c_E': 1.0,      # elastic stiffness
    'e_33': 0.8,     # piezoelectric coupling (stress-charge form)
    'eps_S': 0.6,    # permittivity
}

# Network architecture
NETWORK_CONFIG = {
    'in_dim': 2,
    'out_dim': 2,
    'hidden_dim': 180,
    'n_layers': 8,
}

# Collocation points
COLLOCATION = {
    'n_pde': 20000,          # interior PDE collocation points
    'n_bc': 5000,            # boundary condition points
    'n_ic': 5000,            # initial condition points
    'batch_pde': 3000,       # mini-batch size for PDE
}

# Stage 1: Adam
STAGE_1_ADAM = {
    'lr': 2e-3,
    'beta1': 0.9,
    'beta2': 0.999,
    'epochs': 18000,
    'patience': 2000,
}

# Stage 2: AdamW
STAGE_2_ADAMW = {
    'lr': 8e-4,
    'weight_decay': 1.5e-5,
    'epochs': 12000,
    'patience': 1500,
}

# Stage 3: L-BFGS
STAGE_3_LBFGS = {
    'lr': 1.0,
    'max_iter': 600,
    'tolerance_grad': 1e-10,
    'tolerance_change': 1e-10,
    'history_size': 80,
}

# Loss weights
LOSS_WEIGHTS = {
    'w_bc': 500.0,
    'w_ic': 300.0,
}

# Evaluation
EVALUATION = {
    'grid_resolution': 450,
    'x_range': [0, 1],
    't_range': [0, 1],
    'gradient_clip': 5.0,
}

# Plotting
PLOTTING = {
    'times': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'figsize': (20, 12),
    'dpi': 200,
}
