"""
Spectral Error Indicators for Neural PDE Solvers
Phase 4B: Out-Of-Distribution (OOD) Generalization Test

- Solves the Allen-Cahn equation for TWO separate Initial Conditions (IC A and IC B).
- Trains PINN A on IC A and extracts DMD features.
- Trains PINN B on IC B and extracts DMD features.
- Trains a CNN strictly on PINN A's true errors and DMD features.
- Evaluates the frozen CNN on PINN B's DMD features (Zero-Shot).
- Compares Zero-Shot CNN performance vs standard PDE Residual on PINN B.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("PHASE 4B: OOD Generalization Test (Zero-Shot Error Prediction)")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# 1. GROUND TRUTH GENERATION (IC A and IC B)
# ─────────────────────────────────────────────────────────────
Nx = 256
Nt_pred = 100
x = np.linspace(-1, 1, Nx)
t_eval = np.linspace(0, 1, Nt_pred)
dx = x[1] - x[0]

def allen_cahn_rhs(t, u):
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    u_xx[0] = (u[1] - 2*u[0] + u[-1]) / dx**2
    u_xx[-1] = (u[0] - 2*u[-1] + u[-2]) / dx**2
    return 0.0001 * u_xx - 5 * u**3 + 5 * u

print("Computing exact numerical solutions for IC A and IC B...")
# IC A: x^2 cos(pi x)
u0_A = (x**2) * np.cos(np.pi * x)
sol_A = solve_ivp(allen_cahn_rhs, [0, 1], u0_A, t_eval=t_eval, method='RK45')
U_exact_A = sol_A.y  # [Nx, Nt]

# IC B: sin(pi x)
u0_B = np.sin(np.pi * x)
sol_B = solve_ivp(allen_cahn_rhs, [0, 1], u0_B, t_eval=t_eval, method='RK45')
U_exact_B = sol_B.y  # [Nx, Nt]

# ─────────────────────────────────────────────────────────────
# 2. PINN ARCHITECTURE & TRAINING LOOP
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

def train_pinn(u_ic_func, name):
    print(f"\n--- Training PINN {name} ---")
    pinn = PINN()
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)

    N_f = 8000
    x_f = np.random.uniform(-1, 1, (N_f, 1))
    t_f = np.random.uniform(0, 1, (N_f, 1))
    x_f_t = torch.tensor(x_f, dtype=torch.float32, requires_grad=True)
    t_f_t = torch.tensor(t_f, dtype=torch.float32, requires_grad=True)

    N_ic = 200
    x_ic = np.random.uniform(-1, 1, (N_ic, 1))
    t_ic = np.zeros((N_ic, 1))
    u_ic = u_ic_func(x_ic)
    x_ic_t = torch.tensor(x_ic, dtype=torch.float32)
    t_ic_t = torch.tensor(t_ic, dtype=torch.float32)
    u_ic_t = torch.tensor(u_ic, dtype=torch.float32)

    N_bc = 200
    t_bc = np.random.uniform(0, 1, (N_bc, 1))
    t_bc_t = torch.tensor(t_bc, dtype=torch.float32, requires_grad=True)
    x_left_t = torch.tensor(np.full((N_bc, 1), -1.0), dtype=torch.float32, requires_grad=True)
    x_right_t = torch.tensor(np.full((N_bc, 1), 1.0), dtype=torch.float32, requires_grad=True)

    X_grid, T_grid = np.meshgrid(x, t_eval, indexing='ij')
    x_grid_flat = torch.tensor(X_grid.flatten()[:, None], dtype=torch.float32)
    t_grid_flat = torch.tensor(T_grid.flatten()[:, None], dtype=torch.float32)

    snapshots = []
    epochs = 2000
    snapshot_interval = 40

    for ep in range(epochs):
        optimizer.zero_grad()
        u_f = pinn(x_f_t, t_f_t)
        u_t = torch.autograd.grad(u_f, t_f_t, torch.ones_like(u_f), create_graph=True)[0]
        u_x = torch.autograd.grad(u_f, x_f_t, torch.ones_like(u_f), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_f_t, torch.ones_like(u_x), create_graph=True)[0]
        f = u_t - 0.0001 * u_xx + 5 * u_f**3 - 5 * u_f
        loss_f = torch.mean(f**2)
        
        u_pred_ic = pinn(x_ic_t, t_ic_t)
        loss_ic = torch.mean((u_pred_ic - u_ic_t)**2)
        
        u_left = pinn(x_left_t, t_bc_t)
        u_right = pinn(x_right_t, t_bc_t)
        u_x_left = torch.autograd.grad(u_left, x_left_t, torch.ones_like(u_left), create_graph=True)[0]
        u_x_right = torch.autograd.grad(u_right, x_right_t, torch.ones_like(u_right), create_graph=True)[0]
        loss_bc = torch.mean((u_left - u_right)**2) + torch.mean((u_x_left - u_x_right)**2)
        
        loss = loss_f + 10 * loss_ic + 10 * loss_bc
        loss.backward()
        optimizer.step()
        
        if (ep + 1) % snapshot_interval == 0:
            with torch.no_grad():
                u_snap = pinn(x_grid_flat, t_grid_flat).view(Nx, Nt_pred).detach().numpy()
                snapshots.append(u_snap)

    x_grid_t = torch.tensor(X_grid.flatten()[:, None], dtype=torch.float32, requires_grad=True)
    t_grid_t = torch.tensor(T_grid.flatten()[:, None], dtype=torch.float32, requires_grad=True)
    u_pred_final = pinn(x_grid_t, t_grid_t)
    u_t_final = torch.autograd.grad(u_pred_final, t_grid_t, torch.ones_like(u_pred_final), create_graph=True)[0]
    u_x_final = torch.autograd.grad(u_pred_final, x_grid_t, torch.ones_like(u_pred_final), create_graph=True)[0]
    u_xx_final = torch.autograd.grad(u_x_final, x_grid_t, torch.ones_like(u_x_final), create_graph=True)[0]
    f_final = (u_t_final - 0.0001 * u_xx_final + 5 * u_pred_final**3 - 5 * u_pred_final).detach().numpy().reshape(Nx, Nt_pred)
    final_pinn = u_pred_final.detach().numpy().reshape(Nx, Nt_pred)
    
    return snapshots, final_pinn, f_final

# Run Training for both ICs
snapshots_A, final_pinn_A, f_final_A = train_pinn(lambda x: (x**2) * np.cos(np.pi * x), "A")
snapshots_B, final_pinn_B, f_final_B = train_pinn(lambda x: np.sin(np.pi * x), "B")

true_error_A = np.mean(np.abs(final_pinn_A - U_exact_A), axis=1)
residual_A = np.mean(np.abs(f_final_A), axis=1)

true_error_B = np.mean(np.abs(final_pinn_B - U_exact_B), axis=1)
residual_B = np.mean(np.abs(f_final_B), axis=1)

# ─────────────────────────────────────────────────────────────
# 3. DMD PIPELINE EXTRACTOR
# ─────────────────────────────────────────────────────────────
def extract_dmd_features(snapshots):
    S = np.array([s.flatten() for s in snapshots]).T
    X_mat = S[:, :-1]
    Xp_mat = S[:, 1:]
    U, sigma, Vt = svd(X_mat, full_matrices=False)
    r = 25
    U_r = U[:, :r]
    sigma_r = sigma[:r]
    Vt_r = Vt[:r, :]
    A_tilde = U_r.T @ Xp_mat @ Vt_r.T @ np.diag(1.0 / sigma_r)
    eigvals, W = np.linalg.eig(A_tilde)
    Phi = Xp_mat @ Vt_r.T @ np.diag(1.0 / sigma_r) @ W

    energies = np.abs(eigvals) * np.linalg.norm(Phi, axis=0)
    energies = energies / (energies.sum() + 1e-12)
    H = -np.sum(energies * np.log(energies + 1e-12))

    Phi_spatial = np.abs(Phi).reshape(Nx, Nt_pred, r).mean(axis=1)
    
    # Extract spatial windows
    X_cnn_pad = np.pad(Phi_spatial, ((10, 10), (0, 0)), mode='edge')
    windows = np.array([X_cnn_pad[i:i+21].T for i in range(Nx)]) # [Nx, r, 21]
    return windows

print("\nExtracting DMD features for A and B...")
windows_A = extract_dmd_features(snapshots_A)
windows_B = extract_dmd_features(snapshots_B)

# ─────────────────────────────────────────────────────────────
# 4. 1D CNN TRAINING (On PINN A only)
# ─────────────────────────────────────────────────────────────
class Conv1DNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 21, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.conv(x)

print("Training CNN solely on PINN A...")
train_mean = windows_A.mean(axis=0, keepdims=True)
train_std = windows_A.std(axis=0, keepdims=True) + 1e-8
win_A_norm = (windows_A - train_mean) / train_std
y_max_A = true_error_A.max() + 1e-8
y_A_scaled = true_error_A / y_max_A

win_A_t = torch.tensor(win_A_norm, dtype=torch.float32)
y_A_t = torch.tensor(y_A_scaled, dtype=torch.float32).unsqueeze(1)
loader_A = DataLoader(TensorDataset(win_A_t, y_A_t), batch_size=32, shuffle=True)

cnn = Conv1DNet(in_ch=25)
optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.005)
cnn.train()
for _ in range(120):  # Train on all of A
    for bx, by in loader_A:
        optimizer_cnn.zero_grad()
        loss_c = nn.MSELoss()(cnn(bx), by)
        loss_c.backward()
        optimizer_cnn.step()

# ─────────────────────────────────────────────────────────────
# 5. ZERO-SHOT EVALUATION (On PINN B)
# ─────────────────────────────────────────────────────────────
print("Evaluating Frozen CNN on PINN B (Zero-Shot Transfer)...")
cnn.eval()

# Must normalize B using A's statistics!
win_B_norm = (windows_B - train_mean) / train_std
win_B_t = torch.tensor(win_B_norm, dtype=torch.float32)

with torch.no_grad():
    # We predict the scaled error and unscale it using A's max, 
    # but practically we just care about correlation which is scale-invariant.
    preds_B = cnn(win_B_t).squeeze().numpy() * y_max_A
    pred_cnn_B = np.clip(preds_B, 0, None)

corr_cnn_B = np.corrcoef(true_error_B, pred_cnn_B)[0, 1]
corr_res_B = np.corrcoef(true_error_B, residual_B)[0, 1]

print("\n" + "="*80)
print("PHASE 4B ZERO-SHOT METRICS (Tested on PINN B)")
print("="*80)
print(f"Zero-Shot DMD+CNN Correlation : {corr_cnn_B:.4f}")
print(f"Standard PDE Residual Corr    : {corr_res_B:.4f}")
print("="*80)
if corr_cnn_B > corr_res_B:
    print("HOLY GRAIL ACHIEVED: The CNN learned a universal error mapping!")
else:
    print("Generalization failed: CNN overfit to IC A.")

# ─────────────────────────────────────────────────────────────
# 6. VISUALIZATION
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0a0a14')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
TXT = 'white'

def style_ax(ax, title):
    ax.set_facecolor('#12121e')
    ax.set_title(title, color=TXT, fontsize=10, fontweight='bold', pad=10)
    ax.tick_params(colors=TXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a3a')

# Panel 0: True Solution B
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(U_exact_B, aspect='auto', cmap='magma', origin='lower', vmin=-1, vmax=1)
ax.set_xlabel('Time', color=TXT)
ax.set_ylabel('Space', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, 'Exact Solution B: sin(pi x)')

# Panel 1: Absolute Error B
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(np.abs(final_pinn_B - U_exact_B), aspect='auto', cmap='Reds', origin='lower')
ax.set_xlabel('Time', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, 'Absolute Error |PINN B - True B|')

# Panel 2: Error Indicators on B
ax = fig.add_subplot(gs[1, :])
norm_true_B = true_error_B / true_error_B.max()
norm_cnn_B = pred_cnn_B / (pred_cnn_B.max() + 1e-8)
norm_res_B = residual_B / (residual_B.max() + 1e-8)

ax.plot(x, norm_true_B, color='#ffd700', lw=2.5, label='True Error B', alpha=0.9)
ax.plot(x, norm_cnn_B, color='#00d4ff', lw=2, linestyle='--', label=f'Zero-Shot DMD+CNN (r={corr_cnn_B:.3f})')
ax.plot(x, norm_res_B, color='#ff6b6b', lw=2, linestyle=':', label=f'PDE Residual (r={corr_res_B:.3f})')
ax.set_xlabel('Space (x)', color=TXT)
ax.legend(fontsize=10, facecolor='#1a1a2e', labelcolor=TXT)
style_ax(ax, 'Zero-Shot Generalization: DMD+CNN vs PDE Residual on Unseen PINN B')

fig.suptitle(
    "Phase 4B: Out-Of-Distribution Generalization on Unseen Initial Condition",
    color=TXT, fontsize=14, fontweight='bold', y=0.97
)

out_dir = Path('outputs')
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'phase4b_generalization.png', dpi=160, bbox_inches='tight', facecolor='#0a0a14')
print("\n✓ Phase 4B results figure saved → outputs/phase4b_generalization.png")
