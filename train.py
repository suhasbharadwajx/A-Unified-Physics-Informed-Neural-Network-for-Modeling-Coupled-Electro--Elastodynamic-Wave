import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import gc

from pinn_piezo import PiezoelectricPINN, piezo_residual
from exact_solution import exact_solution
from config import (
    MATERIAL_PARAMS, NETWORK_CONFIG, COLLOCATION, 
    LOSS_WEIGHTS, STAGE_1_ADAM, STAGE_2_ADAMW, STAGE_3_LBFGS,
    EVALUATION
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float32)

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f" GPU {i}: {torch.cuda.get_device_name(i)}")

# Initialize model
model = PiezoelectricPINN(
    in_dim=NETWORK_CONFIG['in_dim'],
    out_dim=NETWORK_CONFIG['out_dim'],
    hidden=NETWORK_CONFIG['hidden_dim'],
    layers=NETWORK_CONFIG['n_layers']
).to(device)

if torch.cuda.device_count() > 1:
    print(f"\nEnabling DataParallel for {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}\n")

# Collocation point preparation
Nd = COLLOCATION['n_pde']
Nb = COLLOCATION['n_bc']
Ni = COLLOCATION['n_ic']
batch_pde = COLLOCATION['batch_pde']

Xd_cpu = torch.rand(Nd, 2)
Xd_cpu[:, 0] *= 1.0
Xd_cpu[:, 1] *= 1.0

tb = torch.rand(Nb, 1)
xb0 = torch.zeros((Nb, 1))
xb1 = torch.ones((Nb, 1))
Xb = torch.cat([torch.hstack([xb0, tb]), torch.hstack([xb1, tb])], dim=0).to(device)

Xi = torch.rand(Ni, 2)
Xi[:, 1] = 0.0
Xi = Xi.to(device)

u_bc_exact = torch.zeros(Xb.shape[0], 1, device=device)
u_ic_exact = torch.tensor(exact_solution(Xi.cpu().numpy()), dtype=torch.float32, device=device)

def loss_fn():
    idx = torch.randperm(Nd)[:batch_pde]
    Xd = Xd_cpu[idx].to(device)
    res = piezo_residual(Xd, model)
    L_pde = (res**2).mean()
    
    u_b = model(Xb)
    L_bc = (u_b**2).mean()
    
    u_i = model(Xi)
    L_ic = ((u_i - u_ic_exact)**2).mean()
    
    L = L_pde + LOSS_WEIGHTS['w_bc'] * L_bc + LOSS_WEIGHTS['w_ic'] * L_ic
    return L, (L_pde, L_bc, L_ic)

# STAGE 1: Adam
print("STAGE 1: ADAM (18k epochs, dual T4)")
opt_adam = torch.optim.Adam(model.parameters(), lr=STAGE_1_ADAM['lr'], 
                           betas=(STAGE_1_ADAM['beta1'], STAGE_1_ADAM['beta2']))
best_loss = float('inf')
patience_ctr = 0

for ep in trange(STAGE_1_ADAM['epochs']):
    opt_adam.zero_grad()
    L, (Lp, Lb, Li) = loss_fn()
    L.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), EVALUATION['gradient_clip'])
    opt_adam.step()
    
    if L.item() < best_loss:
        best_loss = L.item()
        patience_ctr = 0
    else:
        patience_ctr += 1
    
    if (ep + 1) % 1000 == 0:
        print(f"Epoch {ep+1:5d}: L={L.item():.3e}, PDE={Lp.item():.3e}, BC={Lb.item():.3e}, IC={Li.item():.3e}")
    
    if patience_ctr >= STAGE_1_ADAM['patience']:
        print(f"\nEarly stopping at epoch {ep+1}")
        break

torch.cuda.empty_cache()
gc.collect()

# STAGE 2: AdamW
print("\nSTAGE 2: ADAMW (12k epochs, Dual T4)")
opt_adamw = torch.optim.AdamW(model.parameters(), lr=STAGE_2_ADAMW['lr'], 
                             weight_decay=STAGE_2_ADAMW['weight_decay'])
best_loss = float('inf')
patience_ctr = 0

for ep in trange(STAGE_2_ADAMW['epochs']):
    opt_adamw.zero_grad()
    L, (Lp, Lb, Li) = loss_fn()
    L.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), EVALUATION['gradient_clip'])
    opt_adamw.step()
    
    if L.item() < best_loss:
        best_loss = L.item()
        patience_ctr = 0
    else:
        patience_ctr += 1
    
    if (ep + 1) % 1000 == 0:
        print(f"Epoch {ep+1:5d}: L={L.item():.3e}, PDE={Lp.item():.3e}, BC={Lb.item():.3e}, IC={Li.item():.3e}")
    
    if patience_ctr >= STAGE_2_ADAMW['patience']:
        print(f"\nEarly stopping at epoch {ep+1}")
        break

torch.cuda.empty_cache()
gc.collect()

# STAGE 3: L-BFGS
print("\nSTAGE 3: L-BFGS (600 iterations)")
opt_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=STAGE_3_LBFGS['lr'],
    max_iter=STAGE_3_LBFGS['max_iter'],
    tolerance_grad=STAGE_3_LBFGS['tolerance_grad'],
    tolerance_change=STAGE_3_LBFGS['tolerance_change'],
    history_size=STAGE_3_LBFGS['history_size'],
    line_search_fn='strong_wolfe'
)

lbfgs_iter = 0
def closure():
    global lbfgs_iter
    opt_lbfgs.zero_grad()
    L, (Lp, Lb, Li) = loss_fn()
    L.backward()
    lbfgs_iter += 1
    if lbfgs_iter % 50 == 0:
        print(f" L-BFGS iter {lbfgs_iter}: Loss={L.item():.6e}")
    return L

opt_lbfgs.step(closure)
torch.cuda.empty_cache()
gc.collect()

# Save model
torch.save(model.state_dict(), 'results/pinn_model.pt')
print("Model saved to results/pinn_model.pt")
