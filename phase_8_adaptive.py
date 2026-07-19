"""
Spectral Error Indicators for Neural PDE Solvers
Phase 8: Active Adaptive Refinement (Closing the Loop)

This script demonstrates that Pointwise Temporal Variance can be used 
in real-time to actively guide collocation point sampling during PINN training,
outperforming both static sampling and the standard Residual-Based Adaptive Refinement (RAR).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json

print("=" * 70)
print("PHASE 8: Active Adaptive Refinement")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 1. GROUND TRUTH (Allen-Cahn)
# ─────────────────────────────────────────────────────────────
Nx = 100
Nt = 100
x = np.linspace(-1, 1, Nx)
t = np.linspace(0, 1, Nt)
dx = x[1] - x[0]

def allen_cahn_rhs(t, u):
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    u_xx[0] = (u[1] - 2*u[0] + u[-1]) / dx**2
    u_xx[-1] = (u[0] - 2*u[-1] + u[-2]) / dx**2
    return 0.0001 * u_xx - 5 * u**3 + 5 * u

print("Computing exact numerical solution for Allen-Cahn...")
u0 = (x**2) * np.cos(np.pi * x)
sol = solve_ivp(allen_cahn_rhs, [0, 1], u0, t_eval=t, method='RK45')
U_exact = sol.y # [Nx, Nt]

X_grid, T_grid = np.meshgrid(x, t, indexing='ij')
exact_flat = U_exact.flatten()
grid_pts = np.hstack((X_grid.flatten()[:, None], T_grid.flatten()[:, None]))
grid_tensor = torch.tensor(grid_pts, dtype=torch.float32, requires_grad=True)

# ─────────────────────────────────────────────────────────────
# 2. PINN DEFINITION
# ─────────────────────────────────────────────────────────────
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

def get_pde_residual(model, xt):
    x_t = xt[:, 0:1]
    t_t = xt[:, 1:2]
    u = model(x_t, t_t)
    u_t = torch.autograd.grad(u, t_t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_t, torch.ones_like(u_x), create_graph=True)[0]
    f = u_t - 0.0001 * u_xx + 5 * u**3 - 5 * u
    return f

N_ic = 200
N_bc = 200
x_ic = np.random.uniform(-1, 1, (N_ic, 1))
t_ic = np.zeros((N_ic, 1))
u_ic = (x_ic**2) * np.cos(np.pi * x_ic)
x_bc_left = np.full((N_bc, 1), -1.0)
x_bc_right = np.full((N_bc, 1), 1.0)
t_bc = np.random.uniform(0, 1, (N_bc, 1))

x_ic_t = torch.tensor(x_ic, dtype=torch.float32)
t_ic_t = torch.tensor(t_ic, dtype=torch.float32)
u_ic_t = torch.tensor(u_ic, dtype=torch.float32)
x_left_t = torch.tensor(x_bc_left, dtype=torch.float32, requires_grad=True)
x_right_t = torch.tensor(x_bc_right, dtype=torch.float32, requires_grad=True)
t_bc_t = torch.tensor(t_bc, dtype=torch.float32, requires_grad=True)

def calc_loss(model, xt_f):
    f = get_pde_residual(model, xt_f)
    loss_f = torch.mean(f**2)
    u_pred_ic = model(x_ic_t, t_ic_t)
    loss_ic = torch.mean((u_pred_ic - u_ic_t)**2)
    u_left = model(x_left_t, t_bc_t)
    u_right = model(x_right_t, t_bc_t)
    u_x_left = torch.autograd.grad(u_left, x_left_t, torch.ones_like(u_left), create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right, x_right_t, torch.ones_like(u_right), create_graph=True)[0]
    loss_bc = torch.mean((u_left - u_right)**2) + torch.mean((u_x_left - u_x_right)**2)
    return loss_f + 10 * loss_ic + 10 * loss_bc

# Hyperparameters for Adaptive Sampling
N_initial = 500
N_total = 2000
N_add = 150
epochs = 8000
adapt_every = 800

# ─────────────────────────────────────────────────────────────
# 3. TRAINING METHODS
# ─────────────────────────────────────────────────────────────
def train_static(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n--- Training Static Baseline (Seed {seed}) ---")
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Fair baseline: full 2000 points from the start
    pts = np.random.uniform([-1.0, 0.0], [1.0, 1.0], (N_total, 2))
    xt_f = torch.tensor(pts, dtype=torch.float32, requires_grad=True)
    
    for ep in range(epochs):
        optimizer.zero_grad()
        loss = calc_loss(model, xt_f)
        loss.backward()
        optimizer.step()
        if (ep + 1) % 2000 == 0:
            print(f"Static Epoch {ep+1}/{epochs}, Loss: {loss.item():.4f}")
            
    with torch.no_grad():
        preds = model(grid_tensor[:, 0:1], grid_tensor[:, 1:2]).numpy().flatten()
    mae = np.mean(np.abs(preds - exact_flat))
    return mae, pts

def train_rar(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n--- Training RAR (Residual Adaptive Refinement, Seed {seed}) ---")
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pts = np.random.uniform([-1.0, 0.0], [1.0, 1.0], (N_initial, 2))
    xt_f = torch.tensor(pts, dtype=torch.float32, requires_grad=True)
    
    for ep in range(epochs):
        optimizer.zero_grad()
        loss = calc_loss(model, xt_f)
        loss.backward()
        optimizer.step()
        
        if (ep + 1) % adapt_every == 0 and len(pts) < N_total:
            res = get_pde_residual(model, grid_tensor)
            res_val = torch.abs(res).detach().numpy().flatten()
            idx = np.argsort(res_val)[-N_add:]
            new_pts = grid_pts[idx]
            pts = np.vstack((pts, new_pts))
            xt_f = torch.tensor(pts, dtype=torch.float32, requires_grad=True)
            print(f"RAR Epoch {ep+1}: Added {N_add} points. Total: {len(pts)}")

    with torch.no_grad():
        preds = model(grid_tensor[:, 0:1], grid_tensor[:, 1:2]).numpy().flatten()
    mae = np.mean(np.abs(preds - exact_flat))
    return mae, pts

def train_tvar(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n--- Training TVAR (Temporal Variance Adaptive Refinement, Seed {seed}) ---")
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pts = np.random.uniform([-1.0, 0.0], [1.0, 1.0], (N_initial, 2))
    xt_f = torch.tensor(pts, dtype=torch.float32, requires_grad=True)
    
    snapshots = []
    
    for ep in range(epochs):
        optimizer.zero_grad()
        loss = calc_loss(model, xt_f)
        loss.backward()
        optimizer.step()
        
        if ep % (adapt_every // 10) == 0:
            with torch.no_grad():
                pred = model(grid_tensor[:, 0:1], grid_tensor[:, 1:2]).numpy().flatten()
                snapshots.append(pred)
                if len(snapshots) > 10:
                    snapshots.pop(0)
                    
        if (ep + 1) % adapt_every == 0 and len(pts) < N_total:
            stack = np.array(snapshots) # [10, Nx*Nt]
            var = np.var(stack, axis=0)
            idx = np.argsort(var)[-N_add:]
            new_pts = grid_pts[idx]
            pts = np.vstack((pts, new_pts))
            xt_f = torch.tensor(pts, dtype=torch.float32, requires_grad=True)
            print(f"TVAR Epoch {ep+1}: Added {N_add} points. Total: {len(pts)}")

    with torch.no_grad():
        preds = model(grid_tensor[:, 0:1], grid_tensor[:, 1:2]).numpy().flatten()
    mae = np.mean(np.abs(preds - exact_flat))
    return mae, pts

# ─────────────────────────────────────────────────────────────
# 4. EXECUTE AND VISUALIZE ACROSS SEEDS
# ─────────────────────────────────────────────────────────────
seeds = [42, 43, 44]
results = {
    "static": [],
    "rar": [],
    "tvar": []
}

final_pts_static = None
final_pts_rar = None
final_pts_tvar = None

for seed in seeds:
    mae_s, pts_s = train_static(seed)
    mae_r, pts_r = train_rar(seed)
    mae_t, pts_t = train_tvar(seed)
    
    results["static"].append(mae_s)
    results["rar"].append(mae_r)
    results["tvar"].append(mae_t)
    
    # Save the last seed's points for plotting
    final_pts_static = pts_s
    final_pts_rar = pts_r
    final_pts_tvar = pts_t

mean_static = np.mean(results["static"])
std_static = np.std(results["static"])

mean_rar = np.mean(results["rar"])
std_rar = np.std(results["rar"])

mean_tvar = np.mean(results["tvar"])
std_tvar = np.std(results["tvar"])

print("\n" + "="*50)
print("FINAL RESULTS ACROSS 3 SEEDS: MEAN ± STD MAE")
print(f"Static Uniform : {mean_static:.5f} ± {std_static:.5f}")
print(f"Standard RAR   : {mean_rar:.5f} ± {std_rar:.5f}")
print(f"TVAR (Ours)    : {mean_tvar:.5f} ± {std_tvar:.5f}")
print("="*50)

# Save results
with open("phase8_results.json", "w") as f:
    json.dump({
        "static": {"mean": mean_static, "std": std_static, "raw": results["static"]},
        "rar": {"mean": mean_rar, "std": std_rar, "raw": results["rar"]},
        "tvar": {"mean": mean_tvar, "std": std_tvar, "raw": results["tvar"]}
    }, f, indent=4)

# Visualization (using the last seed's final points)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (pts, title, mean_mae, std_mae) in enumerate([
    (final_pts_static, "Static Uniform", mean_static, std_static), 
    (final_pts_rar, "Residual Refinement (RAR)", mean_rar, std_rar), 
    (final_pts_tvar, "Temporal Variance (TVAR)", mean_tvar, std_tvar)]):
    
    ax = axes[i]
    cf = ax.contourf(T_grid, X_grid, U_exact, 50, cmap='RdBu', alpha=0.5)
    ax.scatter(pts[:, 1], pts[:, 0], s=2, c='black', alpha=0.6, label='Collocation Pts')
    ax.set_title(f"{title}\nMAE: {mean_mae:.4f} ± {std_mae:.4f}")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Space (x)")
    ax.legend(loc="upper right")

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/phase8_adaptive.pdf')
print("Saved comparison figure to outputs/phase8_adaptive.pdf")
