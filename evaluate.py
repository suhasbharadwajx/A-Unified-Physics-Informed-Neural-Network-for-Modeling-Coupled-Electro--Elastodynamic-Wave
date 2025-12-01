import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn_piezo import PiezoelectricPINN
from exact_solution import exact_solution
from config import EVALUATION, PLOTTING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pi = np.pi

# Load model
model = PiezoelectricPINN().to(device)
model.load_state_dict(torch.load('results/pinn_model.pt', map_location=device))
model.eval()

# Evaluation grid
grid_res = EVALUATION['grid_resolution']
x_test = np.linspace(0, 1, grid_res)
t_test = np.linspace(0, 1, grid_res)
Xg, Tg = np.meshgrid(x_test, t_test)
XT = np.hstack([Xg.reshape(-1, 1), Tg.reshape(-1, 1)])
XT_t = torch.tensor(XT, dtype=torch.float32, device=device)

with torch.no_grad():
    pred = model(XT_t).detach().cpu().numpy()
    exact = exact_solution(XT)

# Global error metrics
print("\nGLOBAL ERROR METRICS:")
for idx, name in enumerate(['u (displacement)', 'φ (electric potential)']):
    err = np.linalg.norm(pred[:, idx] - exact[:, idx]) / (np.linalg.norm(exact[:, idx]) + 1e-14)
    print(f"{name:30s}: {err:.8e}")

# Time-slice error analysis
print("\nERROR ANALYSIS (per time slice):")
times_analysis = np.linspace(0, 1, 11)
for t_val in times_analysis:
    idx_t = np.argmin(np.abs(t_test - t_val))
    t_actual = t_test[idx_t]
    
    u_err = np.linalg.norm(pred[idx_t*grid_res:(idx_t+1)*grid_res, 0] - 
                           exact[idx_t*grid_res:(idx_t+1)*grid_res, 0]) / \
            (np.linalg.norm(exact[idx_t*grid_res:(idx_t+1)*grid_res, 0]) + 1e-14)
    phi_err = np.linalg.norm(pred[idx_t*grid_res:(idx_t+1)*grid_res, 1] - 
                             exact[idx_t*grid_res:(idx_t+1)*grid_res, 1]) / \
              (np.linalg.norm(exact[idx_t*grid_res:(idx_t+1)*grid_res, 1]) + 1e-14)
    
    print(f"t={t_actual:.2f}: u={u_err:.3e} φ={phi_err:.3e}")

# Plotting
times_plot = PLOTTING['times']
fields = ['u (displacement)', 'φ (electric potential)']
fig = plt.figure(figsize=PLOTTING['figsize'])

for j, field in enumerate(fields):
    # Solution plot
    ax1 = plt.subplot(2, 2, 2*j+1)
    for tv in times_plot:
        Xline = np.column_stack([x_test, np.full_like(x_test, tv)])
        Xt_line = torch.tensor(Xline, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_line = model(Xt_line).detach().cpu().numpy()[:, j]
        ex_line = exact_solution(Xline)[:, j]
        ax1.plot(x_test, ex_line, 'k--', alpha=0.2, linewidth=3, label='Exact' if tv == 0.0 else '')
        ax1.plot(x_test, pred_line, linewidth=2.2, label=f"PINN t={tv:.1f}")
    ax1.set_title(f"{field}: Solution", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Position x", fontsize=12)
    ax1.set_ylabel(field, fontsize=12)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Error plot
    ax2 = plt.subplot(2, 2, 2*j+2)
    for tv in times_plot:
        Xline = np.column_stack([x_test, np.full_like(x_test, tv)])
        Xt_line = torch.tensor(Xline, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_line = model(Xt_line).detach().cpu().numpy()[:, j]
        ex_line = exact_solution(Xline)[:, j]
        ax2.semilogy(x_test, np.abs(pred_line - ex_line) + 1e-16, linewidth=2.2, label=f"t={tv:.1f}")
    ax2.set_title(f"Absolute Error: {field.split()[0]}", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Position x", fontsize=12)
    ax2.set_ylabel("Absolute Error", fontsize=12)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim([1e-7, 1e-0])

plt.suptitle("1D Linear Piezoelectricity PINN", fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/piezoelectric_pinn.png', dpi=PLOTTING['dpi'], bbox_inches='tight')
print(f"\n✓ Plot saved to results/piezoelectric_pinn.png")
plt.show()
