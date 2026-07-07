"""
Spectral Error Indicators for Neural PDE Solvers
Phase 6: Deterministic Spectral Error Indicators

- Solves the Kovasznay flow benchmark (analytical 2D Navier-Stokes).
- Trains a vector-valued PINN (u, v, p).
- Extracts training snapshots.
- Computes three deterministic error indicators:
  1. DMD Reconstruction Error
  2. Unstable Mode Filtering
  3. Pointwise Temporal Variance
- Compares correlation of these indicators against the true error.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from pathlib import Path
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("PHASE 6: Deterministic Spectral Error Indicators")
print("=" * 80)

# Ground Truth
Re = 20.0
lam = 0.5 * Re - np.sqrt(0.25 * Re**2 + 4 * np.pi**2)
Nx, Ny = 50, 50
x_val = np.linspace(-0.5, 1.0, Nx)
y_val = np.linspace(-0.5, 0.5, Ny)
X_grid, Y_grid = np.meshgrid(x_val, y_val, indexing='ij')

U_exact = 1 - np.exp(lam * X_grid) * np.cos(2 * np.pi * Y_grid)
V_exact = (lam / (2 * np.pi)) * np.exp(lam * X_grid) * np.sin(2 * np.pi * Y_grid)
Mag_exact = np.sqrt(U_exact**2 + V_exact**2)

class NavierStokesPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

print("Training 2D Navier-Stokes PINN...")
pinn = NavierStokesPINN()
optimizer = optim.Adam(pinn.parameters(), lr=1e-3)

N_f = 5000
x_f = np.random.uniform(-0.5, 1.0, (N_f, 1))
y_f = np.random.uniform(-0.5, 0.5, (N_f, 1))
x_f_t = torch.tensor(x_f, dtype=torch.float32, requires_grad=True)
y_f_t = torch.tensor(y_f, dtype=torch.float32, requires_grad=True)

N_bc = 1500
x_bc_points = []; y_bc_points = []
for _ in range(N_bc // 4):
    x_bc_points.append([-0.5]); y_bc_points.append([np.random.uniform(-0.5, 0.5)])
for _ in range(N_bc // 4):
    x_bc_points.append([1.0]); y_bc_points.append([np.random.uniform(-0.5, 0.5)])
for _ in range(N_bc // 4):
    x_bc_points.append([np.random.uniform(-0.5, 1.0)]); y_bc_points.append([-0.5])
for _ in range(N_bc // 4):
    x_bc_points.append([np.random.uniform(-0.5, 1.0)]); y_bc_points.append([0.5])
x_bc = np.array(x_bc_points); y_bc = np.array(y_bc_points)
u_bc = 1 - np.exp(lam * x_bc) * np.cos(2 * np.pi * y_bc)
v_bc = (lam / (2 * np.pi)) * np.exp(lam * x_bc) * np.sin(2 * np.pi * y_bc)

x_bc_t = torch.tensor(x_bc, dtype=torch.float32)
y_bc_t = torch.tensor(y_bc, dtype=torch.float32)
u_bc_t = torch.tensor(u_bc, dtype=torch.float32)
v_bc_t = torch.tensor(v_bc, dtype=torch.float32)

x_grid_flat = torch.tensor(X_grid.flatten()[:, None], dtype=torch.float32)
y_grid_flat = torch.tensor(Y_grid.flatten()[:, None], dtype=torch.float32)

snapshots = []
epochs = 2500
snapshot_interval = 50

for ep in range(epochs):
    optimizer.zero_grad()
    out_f = pinn(x_f_t, y_f_t)
    u, v, p = out_f[:, 0:1], out_f[:, 1:2], out_f[:, 2:3]
    
    u_x = torch.autograd.grad(u, x_f_t, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_f_t, torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x_f_t, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y_f_t, torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x_f_t, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y_f_t, torch.ones_like(p), create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x_f_t, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_f_t, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x_f_t, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y_f_t, torch.ones_like(v_y), create_graph=True)[0]
    
    f_u = u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy)
    f_v = u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)
    f_c = u_x + v_y
    
    loss_f = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_c**2)
    
    out_bc = pinn(x_bc_t, y_bc_t)
    u_bc_pred, v_bc_pred = out_bc[:, 0:1], out_bc[:, 1:2]
    loss_bc = torch.mean((u_bc_pred - u_bc_t)**2) + torch.mean((v_bc_pred - v_bc_t)**2)
    
    out_p = pinn(x_f_t[0:1], y_f_t[0:1])
    loss_p = torch.mean(out_p[:, 2:3]**2)
    
    loss = loss_f + 10 * loss_bc + loss_p
    loss.backward()
    optimizer.step()
    
    if (ep + 1) % snapshot_interval == 0:
        with torch.no_grad():
            out_snap = pinn(x_grid_flat, y_grid_flat).detach().numpy()
            mag_snap = np.sqrt(out_snap[:, 0]**2 + out_snap[:, 1]**2)
            snapshots.append(mag_snap)

print(f"Collected {len(snapshots)} velocity magnitude snapshots.")

# Ground truth error
with torch.no_grad():
    out_grid = pinn(x_grid_flat, y_grid_flat).detach().numpy()
    u_pred_flat = out_grid[:, 0]
    v_pred_flat = out_grid[:, 1]
    mag_pred = np.sqrt(u_pred_flat**2 + v_pred_flat**2).reshape(Nx, Ny)
true_error_2d = np.abs(mag_pred - Mag_exact)
true_error_flat = true_error_2d.flatten()

# ─────────────────────────────────────────────────────────────
# 3. DETERMINISTIC ERROR INDICATORS
# ─────────────────────────────────────────────────────────────
print("Computing Deterministic Error Indicators...")

S = np.array(snapshots).T # [Nx*Ny, K]
X_mat = S[:, :-1]
Xp_mat = S[:, 1:]

# Standard exact DMD
U, sigma, Vt = svd(X_mat, full_matrices=False)
r = 25 # truncating small noise
U_r = U[:, :r]
sigma_r = sigma[:r]
Vt_r = Vt[:r, :]

A_tilde = U_r.T @ Xp_mat @ Vt_r.T @ np.diag(1.0 / sigma_r)
eigvals, W = np.linalg.eig(A_tilde)
Phi = Xp_mat @ Vt_r.T @ np.diag(1.0 / sigma_r) @ W

# Amplitudes b: Phi @ b = x_1
b, _, _, _ = np.linalg.lstsq(Phi, X_mat[:, 0], rcond=None)

# 1. DMD Reconstruction Error
# Use top 4 modes with largest amplitude
sort_idx_amp = np.argsort(np.abs(b))[::-1]
top_k_idx = sort_idx_amp[:4]

Phi_top = Phi[:, top_k_idx]
b_top = b[top_k_idx]
eigvals_top = eigvals[top_k_idx]

# Reconstruct final snapshot (K) using top modes
recon_final_flat = np.real(Phi_top @ (eigvals_top**(len(snapshots)-1) * b_top))
recon_error_flat = np.abs(S[:, -1] - recon_final_flat)
recon_error_2d = recon_error_flat.reshape(Nx, Ny)

# 2. Unstable Mode Filtering
# Find modes with eigenvalues NOT near 1
dist_to_1 = np.abs(eigvals - 1.0)
# Discard the 4 most stable modes
stable_idx = np.argsort(dist_to_1)[:4]
unstable_idx = [i for i in range(r) if i not in stable_idx]

Phi_unstable = Phi[:, unstable_idx]
b_unstable = b[unstable_idx]

# Weight modes by their amplitude to reflect actual energy contribution!
unstable_sum_flat = np.sum(np.abs(Phi_unstable * b_unstable), axis=1)
unstable_sum_2d = unstable_sum_flat.reshape(Nx, Ny)

# 3. Pointwise Temporal Variance
# Calculate variance only over the LAST 20 snapshots (last 1000 epochs)
# Early epochs have massive variance everywhere as the network is completely untrained.
variance_flat = np.var(S[:, -20:], axis=1)
variance_2d = variance_flat.reshape(Nx, Ny)

# ─────────────────────────────────────────────────────────────
# 4. METRICS & VISUALIZATION
# ─────────────────────────────────────────────────────────────
corr_recon = np.corrcoef(true_error_flat, recon_error_flat)[0, 1]
corr_unstable = np.corrcoef(true_error_flat, unstable_sum_flat)[0, 1]
corr_var = np.corrcoef(true_error_flat, variance_flat)[0, 1]

print("\n" + "="*70)
print("PHASE 6 METRICS (Deterministic Error Indicators)")
print("="*70)
print(f"1. DMD Recon Error Correlation      : {corr_recon:.4f}")
print(f"2. Unstable Mode Filter Correlation : {corr_unstable:.4f}")
print(f"3. Temporal Variance Correlation    : {corr_var:.4f}")
print("="*70)

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0a0a14')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
TXT = 'white'

def style_ax(ax, title):
    ax.set_facecolor('#12121e')
    ax.set_title(title, color=TXT, fontsize=10, fontweight='bold', pad=10)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a3a')

# Panel 0: Exact Flow
ax = fig.add_subplot(gs[0, 0])
im = ax.contourf(X_grid, Y_grid, Mag_exact, levels=50, cmap='magma')
ax.set_xlabel('x', color=TXT)
ax.set_ylabel('y', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, 'Exact Velocity Magnitude')

# Panel 1: PINN Prediction
ax = fig.add_subplot(gs[0, 1])
im = ax.contourf(X_grid, Y_grid, mag_pred, levels=50, cmap='magma')
ax.set_xlabel('x', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, 'PINN Prediction Magnitude')

# Panel 2: True Error
ax = fig.add_subplot(gs[0, 2])
im = ax.contourf(X_grid, Y_grid, true_error_2d, levels=50, cmap='Reds')
ax.set_xlabel('x', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, 'Absolute Error |PINN - Exact|')

# Panel 3: DMD Recon Error
ax = fig.add_subplot(gs[1, 0])
im = ax.contourf(X_grid, Y_grid, recon_error_2d, levels=50, cmap='Reds')
ax.set_xlabel('x', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, f'1. DMD Recon Error (r={corr_recon:.3f})')

# Panel 4: Unstable Filter
ax = fig.add_subplot(gs[1, 1])
im = ax.contourf(X_grid, Y_grid, unstable_sum_2d, levels=50, cmap='Reds')
ax.set_xlabel('x', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, f'2. Unstable Mode Sum (r={corr_unstable:.3f})')

# Panel 5: Temporal Variance
ax = fig.add_subplot(gs[1, 2])
im = ax.contourf(X_grid, Y_grid, variance_2d, levels=50, cmap='Reds')
ax.set_xlabel('x', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, f'3. Temporal Variance (r={corr_var:.3f})')

fig.suptitle(
    "Phase 6: Deterministic Spectral Error Indicators Comparison",
    color=TXT, fontsize=14, fontweight='bold', y=0.96
)

out_dir = Path('outputs')
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'phase6_comparison.png', dpi=160, bbox_inches='tight', facecolor='#0a0a14')
print("\n✓ Phase 6 results figure saved → outputs/phase6_comparison.png")
