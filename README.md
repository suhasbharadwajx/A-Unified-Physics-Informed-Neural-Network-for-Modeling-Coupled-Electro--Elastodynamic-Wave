# A-Unified-Physics-Informed-Neural-Network-for-Modeling-Coupled-Electro-Elastodynamic-Wave
Source code to the paper titled "A Unified Physics-Informed Neural Network for Modeling Coupled Electro- and Elastodynamic Wave Propagation Using 1D Piezoelectricity" by Suhas Suresh Bharadwaj and Reuben Thomas Thovelil.

# A Unified Physics-Informed Neural Network for Modeling Electro- and Elastodynamic Wave Propagation Using 1D Piezoelectricity

This repository contains the implementation of a physics-informed neural network (PINN) for solving coupled one-dimensional linear piezoelectric systems. The code trains a neural network to simultaneously predict mechanical displacement and electric potential fields while satisfying elastodynamic and electrostatic governing equations.

## Overview

Physics-informed neural networks embed partial differential equations directly into the neural network training process through soft constraints in the loss function. This work applies PINNs to a benchmark problem in piezoelectricity, where mechanical and electrical fields are coupled through constitutive relations. The study demonstrates how PINNs handle multiphysics problems while also documenting practical limitations on long-time horizons and coupled field interactions.

## Key Features

- **Hard-constraint enforcement** of boundary and initial conditions via basis-function output transformations
- **Three-stage optimization strategy** combining Adam, AdamW, and L-BFGS for robust convergence
- **Dual-field learning** of displacement and electric potential in a unified framework
- **Validation against analytical solution** for precise error characterization
- **GPU acceleration** via PyTorch DataParallel on dual NVIDIA T4 GPUs

## Problem Formulation

The 1D linear pieelectric system in stress-charge form consists of:

1. **Elastodynamics**: $\(\rho u_{tt} = \sigma_x\)$
2. **Stress relation**: $\(\sigma = c_E u_x - e_{33} \varphi_x\)$
3. **Electric displacement**: $\(D = e_{33} u_x + \varepsilon_S \varphi_x\)$
4. **Electrical equation**: $\(\varepsilon_0 \varphi_{tt} = -D_x\)$

With boundary conditions: $\(u(0,t) = u(1,t) = 0\), \(\varphi(0,t) = \varphi(1,t) = 0\)$

And initial conditions: $\(u(x,0) = \sin(\pi x)\), \(\varphi(x,0) = 0.5\sin(\pi x)\)$

The exact standing-wave solution is known analytically, enabling rigorous validation.

## Requirements
Python 3.8+
PyTorch 1.13+
NumPy
Matplotlib

## Install dependencies:
pip install torch numpy matplotlib

## For GPU support, ensure CUDA-compatible PyTorch is installed:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## Usage

### Basic Training

from pinn_piezo import PINNPiezoelectric

Initialize the PINN
pinn = PINNPiezoelectric(
layers=[2, 180, 180, 180, 180, 180, 180, 180, 180ailable() else 'cpu'
)

Train with three-stage optimization
history = pinn.train_three_stage(
n_collocation_pde=20000,
n_collocation_bc=5000,
n_collocation_ic=5000,
batch_size=3000
)

Evaluate on dense grid
x_test, t_test = np.linspace(0, 1, 450), np.linspace(0, 1, 450)
u_pred, phi_pred = pinn.predict(x_test, t_test)

Compute errors against exact solution
u_exact, phi_exact = exact_solution(x_test, t_test)
error_u = np.abs(u_pred - u_exact)
error_phi = np.abs(phi_pred - phi_exact)

### Configuration

Edit hyperparameters in the main script or pass them as arguments:
Training stages
stage_1_epochs = 18000 # Adam
stage_2_epochs = 12000 # AdamW
stage_3_iters = 600 # L-BFGS

Loss weights
w_BC = 500
w_IC = 300

Optimizer settings
lr_adam = 2e-3
lr_adamw = 8e-4
wd_adamw = 1.5e-5
lbfgs_history = 80

## Results

The PINN achieves:
- **Displacement error**: 2.34% relative L2 error
- **Electric potential error**: 4.87% relative L2 error
- **Boundary condition error**: < 10⁻⁶ absolute error
- **Initial condition error**: < 10⁻⁴ absolute error

Time-dependent error growth and amplitude loss at late times are observed, reflecting intrinsic PINN limitations on long-horizon coupled hyperbolic problems.

## Repository Structure

├── pinn_piezo.py # Core PINN implementation
├── train.py # Training script
├── evaluate.py # Evaluation and visualization
├── exact_solution.py # Analytical solution
├── config.py # Hyperparameter configuration
├── requirements.txt # Dependencies
├── results/ # Output plots and error data
└── README.md

## Key Files

- **`pinn_piezo.py`**: Main PINN class with forward pass, loss computation, and three-stage training
- **`train.py`**: End-to-end training pipeline
- **`evaluate.py`**: Generate error plots and compare with exact solution
- **`exact_solution.py`**: Analytical standing-wave solution for validation

## Reproducibility

All hyperparameters and training configurations are documented in `config.py`. To reproduce results:

python train.py --config config.py
python evaluate.py --model_checkpoint results/pinn_final.pt

The code uses fixed random seeds for reproducibility:
torch.manual_seed(42)
np.random.seed(42)

## Limitations and Future Work

Current implementation shows:
- Error growth over time (long-horizon limitation)
- Error amplification in coupled electric field
- Trade-offs between early-time and late-time accuracy

Potential improvements:
- **Temporal domain decomposition**: Train separate networks on time intervals
- **Autoregressive formulation**: Use network as time-stepping operator
- **Fourier features**: Replace tanh with periodic activations for oscillatory solutions
- **Adaptive collocation**: Focus sampling on high-error regions

## Citation

If you use this code in your research, please cite:

@article{Bharadwaj2025,
author = {Bharadwaj, Suhas Suresh and Thovelil, Reuben Thomas},
title = {A Unified Physics-Informed Neural Network for Modeling
Electro-Elastodynamic Wave Propagation Using 1D Piezoelectricity},
journal = {arXiv (preprint)},
year = {2025}
}

## License

MIT License — see LICENSE file for details.

## Contact

For questions or issues, open an issue on GitHub or contact the authors directly.

**Last updated**: December 2025

