import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

print("================================================================================")
print("PHASE 7: Burgers Equation Viscosity Sweep (Temporal Variance Validation)")
print("================================================================================")

# -------------------------------------------------------------
# 1. SETUP DOMAIN & EXACT SOLVER
# -------------------------------------------------------------
Nx = 100
Nt_pred = 50
x_np = np.linspace(-1, 1, Nx)
t_np = np.linspace(0, 1, Nt_pred)
dx = x_np[1] - x_np[0]

def get_exact_burgers(nu):
    Nx_fine = 400
    x_fine = np.linspace(-1, 1, Nx_fine)
    dx_fine = x_fine[1] - x_fine[0]
    
    def burgers_rhs(t, u):
        u_xx = np.zeros_like(u)
        u_x = np.zeros_like(u)
        
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx_fine**2
        # Upwind scheme for stability
        u_x[1:-1] = np.where(u[1:-1] > 0, 
                             (u[1:-1] - u[:-2]) / dx_fine, 
                             (u[2:] - u[1:-1]) / dx_fine)
        
        u_xx[0] = 0; u_xx[-1] = 0
        u_x[0] = 0; u_x[-1] = 0
        
        du_dt = -u * u_x + nu * u_xx
        du_dt[0] = 0
        du_dt[-1] = 0
        return du_dt
        
    u0 = -np.sin(np.pi * x_fine)
    sol = solve_ivp(burgers_rhs, [0, 1], u0, t_eval=t_np, method='BDF')
    
    # Interpolate back to original grid
    U_fine = sol.y # [Nx_fine, Nt_pred]
    U_exact = np.zeros((Nx, Nt_pred))
    for j in range(Nt_pred):
        U_exact[:, j] = np.interp(x_np, x_fine, U_fine[:, j])
    return U_exact

# -------------------------------------------------------------
# 2. PINN DEFINITION
# -------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

# Collocation points
N_f = 2000
x_f = (2.0 * torch.rand(N_f, 1) - 1.0).requires_grad_(True)
t_f = (torch.rand(N_f, 1)).requires_grad_(True)

# Initial condition points
x_ic = (2.0 * torch.rand(500, 1) - 1.0)
t_ic = torch.zeros_like(x_ic)
u_ic = -torch.sin(np.pi * x_ic)

# Boundary condition points
t_bc = torch.rand(500, 1)
x_bc_left = -torch.ones_like(t_bc)
x_bc_right = torch.ones_like(t_bc)

# Grid for evaluation
X_grid, T_grid = np.meshgrid(x_np, t_np, indexing='ij')
x_grid_flat = torch.tensor(X_grid.flatten(), dtype=torch.float32).unsqueeze(1).requires_grad_(True)
t_grid_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32).unsqueeze(1).requires_grad_(True)

# -------------------------------------------------------------
# 3. SWEEP & EVALUATION
# -------------------------------------------------------------
nu_list = [
    (0.01/np.pi, 'Reference 1'),
    (0.0040, 'Reference 2'),
    (0.0035, 'Interpolation'),
    (0.0020, 'Extrapolation'),
    (0.0050, 'Extrapolation')
]

results = {}

for nu, label in nu_list:
    print(f"\n--- Processing Viscosity (nu) = {nu:.5f} ({label}) ---")
    U_exact = get_exact_burgers(nu)
    
    pinn = PINN()
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    epochs = 3000
    
    snapshots = []
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Data loss
        u_pred_ic = pinn(x_ic, t_ic)
        loss_ic = nn.MSELoss()(u_pred_ic, u_ic)
        
        u_pred_left = pinn(x_bc_left, t_bc)
        u_pred_right = pinn(x_bc_right, t_bc)
        loss_bc = nn.MSELoss()(u_pred_left, torch.zeros_like(u_pred_left)) + \
                  nn.MSELoss()(u_pred_right, torch.zeros_like(u_pred_right))
                  
        # PDE loss
        u_pred_f = pinn(x_f, t_f)
        u_t = torch.autograd.grad(u_pred_f, t_f, grad_outputs=torch.ones_like(u_pred_f), create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred_f, x_f, grad_outputs=torch.ones_like(u_pred_f), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        f_pred = u_t + u_pred_f * u_x - nu * u_xx
        loss_f = nn.MSELoss()(f_pred, torch.zeros_like(f_pred))
        
        loss = loss_f + 100 * (loss_ic + loss_bc)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            with torch.no_grad():
                u_snap = pinn(x_grid_flat, t_grid_flat).view(Nx, Nt_pred).numpy()
                snapshots.append(u_snap.copy())
    
    # -------------------------------------------------------------
    # Evaluate at the end
    # -------------------------------------------------------------
    with torch.no_grad():
        final_pinn = pinn(x_grid_flat, t_grid_flat).view(Nx, Nt_pred).numpy()
    
    true_error = np.mean(np.abs(final_pinn - U_exact), axis=1) # [Nx]
    
    # Compute Temporal Variance (All snapshots for easy PDEs)
    S = np.array([s.flatten() for s in snapshots]).T # [Nx*Nt_pred, K]
    var_flat = np.var(S, axis=1)
    var_spatial = np.mean(var_flat.reshape(Nx, Nt_pred), axis=1) # [Nx]
    
    # Compute PDE Residual 
    u_grid = pinn(x_grid_flat, t_grid_flat)
    u_t = torch.autograd.grad(u_grid, t_grid_flat, grad_outputs=torch.ones_like(u_grid), create_graph=True)[0]
    u_x = torch.autograd.grad(u_grid, x_grid_flat, grad_outputs=torch.ones_like(u_grid), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_grid_flat, grad_outputs=torch.ones_like(u_x))[0]
    residual_field = (u_t + u_grid * u_x - nu * u_xx).view(Nx, Nt_pred).detach().numpy()
    res_spatial = np.mean(np.abs(residual_field), axis=1) # [Nx]
    
    # Correlations
    corr_var = np.corrcoef(true_error, var_spatial)[0, 1]
    corr_res = np.corrcoef(true_error, res_spatial)[0, 1]
    
    results[nu] = {
        'label': label,
        'corr_var': corr_var,
        'corr_res': corr_res,
        'true_error': true_error,
        'var_spatial': var_spatial,
        'res_spatial': res_spatial,
        'final_pinn': final_pinn,
        'exact': U_exact
    }

print("\n====================================================================================================")
print(f"{'Viscosity (nu)':<15} | {'Test Type':<15} | {'PDE Residual':<15} | {'Temporal Variance':<20} | {'Status'}")
print("====================================================================================================")
for nu, label in nu_list:
    r = results[nu]
    status = "SUCCESS" if (r['corr_var'] > r['corr_res'] and r['corr_var'] > 0.70) else "DEGRADED" if (r['corr_var'] > r['corr_res']) else "FAILED"
    print(f"{nu:<15.5f} | {label:<15} | {r['corr_res']:<15.4f} | {r['corr_var']:<20.4f} | {status}")
print("====================================================================================================")

# -------------------------------------------------------------
# 4. VISUALIZATION
# -------------------------------------------------------------
fig, axes = plt.subplots(len(nu_list), 3, figsize=(15, 3 * len(nu_list)))
fig.suptitle("Phase 7: Pointwise Temporal Variance across Burgers Viscosity Sweep", fontsize=16)

for i, (nu, label) in enumerate(nu_list):
    r = results[nu]
    
    # Plot 1: PINN Solution vs Exact
    axes[i, 0].plot(x_np, r['exact'][:, -1], 'k-', label='Exact (t=1.0)')
    axes[i, 0].plot(x_np, r['final_pinn'][:, -1], 'r--', label='PINN (t=1.0)')
    axes[i, 0].set_title(f'nu={nu:.5f} ({label}) Solution')
    if i == 0: axes[i, 0].legend()
    
    # Plot 2: True Error
    axes[i, 1].plot(x_np, r['true_error'], 'k-', label='True Error')
    axes[i, 1].set_title(f'True Spatial Error')
    if i == 0: axes[i, 1].legend()
    
    # Plot 3: Error Indicators
    axes[i, 2].plot(x_np, r['var_spatial'] / np.max(r['var_spatial']), 'b-', label=f'Temp Var (r={r["corr_var"]:.2f})')
    axes[i, 2].plot(x_np, r['res_spatial'] / np.max(r['res_spatial']), 'g--', label=f'Residual (r={r["corr_res"]:.2f})')
    axes[i, 2].set_title('Normalized Error Indicators')
    axes[i, 2].legend()

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/phase7_burgers_sweep.png", dpi=150)
print("\n✓ Phase 7 results figure saved -> outputs/phase7_burgers_sweep.png")
