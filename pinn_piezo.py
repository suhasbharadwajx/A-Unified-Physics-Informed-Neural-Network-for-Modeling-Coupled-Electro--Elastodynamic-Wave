import torch
import torch.nn as nn
import numpy as np
from config import MATERIAL_PARAMS, NETWORK_CONFIG, EVALUATION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pi = np.pi

# Unpack material parameters
rho = MATERIAL_PARAMS['rho']
c_E = MATERIAL_PARAMS['c_E']
e_33 = MATERIAL_PARAMS['e_33']
eps_S = MATERIAL_PARAMS['eps_S']

class PiezoelectricPINN(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, hidden=180, layers=8):
        super().__init__()
        modules = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*modules)
        
        # Xavier uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x_coord = x[:, 0:1]
        t_coord = x[:, 1:2]
        raw_out = self.net(x)
        u_raw, phi_raw = raw_out[:, 0:1], raw_out[:, 1:2]
        
        # Hard boundary conditions: u(0,t)=u(1,t)=0, φ(0,t)=φ(1,t)=0
        bc_basis = x_coord * (1.0 - x_coord)
        u = bc_basis * u_raw
        phi = bc_basis * phi_raw
        
        # Hard initial conditions: u(x,0)=sin(πx), φ(x,0)=0.5sin(πx)
        ic_weight = t_coord
        sin_x = torch.sin(pi * x_coord)
        u_ic = sin_x * (1.0 - ic_weight)
        phi_ic = 0.5 * sin_x * (1.0 - ic_weight)
        
        u = u + u_ic
        phi = phi + phi_ic
        
        return torch.cat([u, phi], dim=1)

def grad(x, f):
    return torch.autograd.grad(
        f, x, 
        grad_outputs=torch.ones_like(f),
        create_graph=True, 
        retain_graph=True, 
        allow_unused=True
    )[0]

def piezo_residual(x, model):
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    u, phi = out[:, 0:1], out[:, 1:2]
    
    # First derivatives
    g1 = grad(x, u)
    u_x = g1[:, 0:1]
    u_t = g1[:, 1:2]
    
    g2 = grad(x, phi)
    phi_x = g2[:, 0:1]
    phi_t = g2[:, 1:2]
    
    # Second derivatives
    u_xx = grad(x, u_x)[:, 0:1]
    u_tt = grad(x, u_t)[:, 1:2]
    phi_xx = grad(x, phi_x)[:, 0:1]
    phi_tt = grad(x, phi_t)[:, 1:2]
    
    # Stress-charge form constitutive relations
    sigma = c_E * u_x - e_33 * phi_x  # stress
    sigma_x = grad(x, sigma)[:, 0:1]
    
    D = e_33 * u_x + eps_S * phi_x  # electric displacement
    D_x = grad(x, D)[:, 0:1]
    
    # Residuals
    r1 = rho * u_tt - sigma_x           # elastodynamics
    r2 = eps_S * phi_tt + D_x           # electrical (Gauss law)
    
    return torch.cat([r1, r2], dim=1)
