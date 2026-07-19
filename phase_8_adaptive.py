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

torch.manual_seed(42)
np.random.seed(42)

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

initial_pts = np.random.uniform([-1.0, 0.0], [1.0, 1.0], (N_initial, 2))

# ─────────────────────────────────────────────────────────────
# 3. TRAINING METHODS
# ─────────────────────────────────────────────────────────────
def train_static():
    print("\n--- Training Static Baseline ---")
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

def train_rar():
    print("\n--- Training RAR (Residual Adaptive Refinement) ---")
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pts = initial_pts.copy()
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

def train_tvar():
    print("\n--- Training TVAR (Temporal Variance Adaptive Refinement) ---")
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pts = initial_pts.copy()
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
# 4. EXECUTE AND VISUALIZE
# ─────────────────────────────────────────────────────────────
mae_static, pts_static = train_static()
mae_rar, pts_rar = train_rar()
mae_tvar, pts_tvar = train_tvar()

print("\n" + "="*50)
print("FINAL RESULTS: MEAN ABSOLUTE ERROR (MAE)")
print(f"Static Uniform : {mae_static:.6f}")
print(f"Standard RAR   : {mae_rar:.6f}")
print(f"TVAR (Ours)    : {mae_tvar:.6f}")
print("="*50)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (pts, title, mae) in enumerate([
    (pts_static, "Static Uniform", mae_static), 
    (pts_rar, "Residual Refinement (RAR)", mae_rar), 
    (pts_tvar, "Temporal Variance (TVAR)", mae_tvar)]):
    
    ax = axes[i]
    cf = ax.contourf(T_grid, X_grid, U_exact, 50, cmap='RdBu', alpha=0.5)
    ax.scatter(pts[:, 1], pts[:, 0], s=2, c='black', alpha=0.6, label='Collocation Pts')
    ax.set_title(f"{title}\nFinal MAE: {mae:.5f}")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Space (x)")
    ax.legend(loc="upper right")

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/phase8_adaptive.pdf')
print("Saved comparison figure to outputs/phase8_adaptive.pdf")
