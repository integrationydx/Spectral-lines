"""
Spectral Error Indicators for Neural PDE Solvers
Phase 5: 2D Navier-Stokes Generalization

- Solves the Kovasznay flow benchmark (analytical 2D Navier-Stokes).
- Trains a vector-valued PINN (u, v, p).
- Applies 2D DMD to velocity magnitude snapshots.
- Uses a 2D CNN on localized patches to predict true spatial error (OOF validation).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
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
print("PHASE 5: 2D Navier-Stokes Generalization (Kovasznay Flow)")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# 1. GROUND TRUTH (Kovasznay Flow Analytical Solution)
# ─────────────────────────────────────────────────────────────
Re = 20.0
lam = 0.5 * Re - np.sqrt(0.25 * Re**2 + 4 * np.pi**2)

Nx, Ny = 50, 50
x_val = np.linspace(-0.5, 1.0, Nx)
y_val = np.linspace(-0.5, 0.5, Ny)
X_grid, Y_grid = np.meshgrid(x_val, y_val, indexing='ij')

U_exact = 1 - np.exp(lam * X_grid) * np.cos(2 * np.pi * Y_grid)
V_exact = (lam / (2 * np.pi)) * np.exp(lam * X_grid) * np.sin(2 * np.pi * Y_grid)
P_exact = -0.5 * np.exp(2 * lam * X_grid)
Mag_exact = np.sqrt(U_exact**2 + V_exact**2)

# ─────────────────────────────────────────────────────────────
# 2. 2D VECTOR-VALUED PINN
# ─────────────────────────────────────────────────────────────
class NavierStokesPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3) # Outputs: u, v, p
        )
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

print("Training 2D Navier-Stokes PINN...")
pinn = NavierStokesPINN()
optimizer = optim.Adam(pinn.parameters(), lr=1e-3)

# Collocation points (Interior)
N_f = 5000
x_f = np.random.uniform(-0.5, 1.0, (N_f, 1))
y_f = np.random.uniform(-0.5, 0.5, (N_f, 1))
x_f_t = torch.tensor(x_f, dtype=torch.float32, requires_grad=True)
y_f_t = torch.tensor(y_f, dtype=torch.float32, requires_grad=True)

# Boundary points
N_bc = 1500
x_bc_points = []
y_bc_points = []
for _ in range(N_bc // 4): # Left
    x_bc_points.append([-0.5]); y_bc_points.append([np.random.uniform(-0.5, 0.5)])
for _ in range(N_bc // 4): # Right
    x_bc_points.append([1.0]); y_bc_points.append([np.random.uniform(-0.5, 0.5)])
for _ in range(N_bc // 4): # Bottom
    x_bc_points.append([np.random.uniform(-0.5, 1.0)]); y_bc_points.append([-0.5])
for _ in range(N_bc // 4): # Top
    x_bc_points.append([np.random.uniform(-0.5, 1.0)]); y_bc_points.append([0.5])

x_bc = np.array(x_bc_points)
y_bc = np.array(y_bc_points)

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
    
    # Physics Loss
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
    
    # Boundary Loss
    out_bc = pinn(x_bc_t, y_bc_t)
    u_bc_pred, v_bc_pred = out_bc[:, 0:1], out_bc[:, 1:2]
    loss_bc = torch.mean((u_bc_pred - u_bc_t)**2) + torch.mean((v_bc_pred - v_bc_t)**2)
    
    # Point loss for pressure gauge (fix p=0 at a random point to avoid drift)
    out_p = pinn(x_f_t[0:1], y_f_t[0:1])
    loss_p = torch.mean(out_p[:, 2:3]**2)
    
    loss = loss_f + 10 * loss_bc + loss_p
    loss.backward()
    optimizer.step()
    
    if (ep + 1) % snapshot_interval == 0:
        with torch.no_grad():
            out_snap = pinn(x_grid_flat, y_grid_flat).detach().numpy()
            u_snap, v_snap = out_snap[:, 0], out_snap[:, 1]
            mag_snap = np.sqrt(u_snap**2 + v_snap**2)
            snapshots.append(mag_snap)

print(f"Collected {len(snapshots)} velocity magnitude snapshots.")

# Extract Final State and Residual on Grid
x_grid_t = torch.tensor(X_grid.flatten()[:, None], dtype=torch.float32, requires_grad=True)
y_grid_t = torch.tensor(Y_grid.flatten()[:, None], dtype=torch.float32, requires_grad=True)

out_grid = pinn(x_grid_t, y_grid_t)
u, v, p = out_grid[:, 0:1], out_grid[:, 1:2], out_grid[:, 2:3]
u_x = torch.autograd.grad(u, x_grid_t, torch.ones_like(u), create_graph=True)[0]
u_y = torch.autograd.grad(u, y_grid_t, torch.ones_like(u), create_graph=True)[0]
v_x = torch.autograd.grad(v, x_grid_t, torch.ones_like(v), create_graph=True)[0]
v_y = torch.autograd.grad(v, y_grid_t, torch.ones_like(v), create_graph=True)[0]
p_x = torch.autograd.grad(p, x_grid_t, torch.ones_like(p), create_graph=True)[0]
p_y = torch.autograd.grad(p, y_grid_t, torch.ones_like(p), create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x_grid_t, torch.ones_like(u_x), create_graph=True)[0]
u_yy = torch.autograd.grad(u_y, y_grid_t, torch.ones_like(u_y), create_graph=True)[0]
v_xx = torch.autograd.grad(v_x, x_grid_t, torch.ones_like(v_x), create_graph=True)[0]
v_yy = torch.autograd.grad(v_y, y_grid_t, torch.ones_like(v_y), create_graph=True)[0]

f_u = u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy)
f_v = u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)
f_c = u_x + v_y

residual_magnitude = torch.sqrt(f_u**2 + f_v**2 + f_c**2).detach().numpy().reshape(Nx, Ny)

u_pred_flat = u.detach().numpy().flatten()
v_pred_flat = v.detach().numpy().flatten()
mag_pred = np.sqrt(u_pred_flat**2 + v_pred_flat**2).reshape(Nx, Ny)

true_error_2d = np.abs(mag_pred - Mag_exact)

# ─────────────────────────────────────────────────────────────
# 3. 2D DMD PIPELINE
# ─────────────────────────────────────────────────────────────
S = np.array(snapshots).T # [Nx*Ny, K]
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

Phi_spatial = np.abs(Phi).reshape(Nx, Ny, r)

# Extract 2D windows (9x9 patches)
pad = 4
Phi_pad = np.pad(Phi_spatial, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
windows = np.zeros((Nx * Ny, r, 9, 9))

idx = 0
for i in range(Nx):
    for j in range(Ny):
        windows[idx] = Phi_pad[i:i+9, j:j+9, :].transpose(2, 0, 1)
        idx += 1

true_error_flat = true_error_2d.flatten()
residual_flat = residual_magnitude.flatten()

# ─────────────────────────────────────────────────────────────
# 4. 2D CNN ERROR PREDICTOR (5-FOLD OOF)
# ─────────────────────────────────────────────────────────────
class Conv2DNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.conv(x)

print("Evaluating 2D DMD+CNN (5-Fold OOF)...")
pred_cnn_flat = np.zeros(Nx * Ny)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(windows):
    win_tr, win_va = windows[train_idx], windows[val_idx]
    y_tr, y_va = true_error_flat[train_idx], true_error_flat[val_idx]
    
    # Standardize
    train_mean = win_tr.mean(axis=0, keepdims=True)
    train_std = win_tr.std(axis=0, keepdims=True) + 1e-8
    win_tr = (win_tr - train_mean) / train_std
    win_va = (win_va - train_mean) / train_std
    
    y_max = y_tr.max() + 1e-8
    y_tr_scaled = y_tr / y_max
    
    win_tr_t = torch.tensor(win_tr, dtype=torch.float32)
    win_va_t = torch.tensor(win_va, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_scaled, dtype=torch.float32).unsqueeze(1)
    
    loader = DataLoader(TensorDataset(win_tr_t, y_tr_t), batch_size=64, shuffle=True)
    cnn = Conv2DNet(in_ch=r)
    optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.005)
    
    cnn.train()
    for _ in range(60):
        for bx, by in loader:
            optimizer_cnn.zero_grad()
            loss_c = nn.MSELoss()(cnn(bx), by)
            loss_c.backward()
            optimizer_cnn.step()
    
    cnn.eval()
    with torch.no_grad():
        preds = cnn(win_va_t).squeeze().numpy() * y_max
        pred_cnn_flat[val_idx] = np.clip(preds, 0, None)

pred_cnn_2d = pred_cnn_flat.reshape(Nx, Ny)

# ─────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────
corr_cnn = np.corrcoef(true_error_flat, pred_cnn_flat)[0, 1]
corr_res = np.corrcoef(true_error_flat, residual_flat)[0, 1]

print("\n" + "="*70)
print("PHASE 5 METRICS (2D Navier-Stokes Kovasznay Flow)")
print("="*70)
print(f"2D PDE Residual Correlation : {corr_res:.4f}")
print(f"2D DMD+CNN Correlation      : {corr_cnn:.4f}")
print("="*70)
if corr_cnn > corr_res:
    print("SUCCESS: 2D Spectral DMD features predict local error BETTER than the 2D PDE residual!")
else:
    print("BASELINE WINS: The residual is more informative.")

# ─────────────────────────────────────────────────────────────
# 6. VISUALIZATION
# ─────────────────────────────────────────────────────────────
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

# Panel 3: DMD Mode 1
ax = fig.add_subplot(gs[1, 0])
im = ax.contourf(X_grid, Y_grid, Phi_spatial[:,:,0], levels=50, cmap='viridis')
ax.set_xlabel('x', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, 'Dominant 2D DMD Spatial Mode')

# Panel 4: DMD+CNN Prediction
ax = fig.add_subplot(gs[1, 1])
im = ax.contourf(X_grid, Y_grid, pred_cnn_2d, levels=50, cmap='Reds')
ax.set_xlabel('x', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, f'2D DMD+CNN Error Prediction (r={corr_cnn:.3f})')

# Panel 5: Physics Residual
ax = fig.add_subplot(gs[1, 2])
im = ax.contourf(X_grid, Y_grid, residual_magnitude, levels=50, cmap='Reds')
ax.set_xlabel('x', color=TXT)
plt.colorbar(im, ax=ax).ax.tick_params(colors=TXT)
style_ax(ax, f'Raw 2D PDE Residual (r={corr_res:.3f})')

fig.suptitle(
    "Phase 5: 2D Navier-Stokes (Kovasznay Flow)",
    color=TXT, fontsize=14, fontweight='bold', y=0.96
)

out_dir = Path('outputs')
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'phase5_navierstokes.png', dpi=160, bbox_inches='tight', facecolor='#0a0a14')
print("\n✓ Phase 5 results figure saved → outputs/phase5_navierstokes.png")
