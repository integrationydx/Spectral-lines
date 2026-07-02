"""
Spectral Error Indicators for Neural PDE Solvers
Phase 5C: Mode Alignment & Parametric Generalization

- Simulates an aerodynamic parameter sweep across Re = [20, 22, 25, 30, 40].
- Applies Algorithmic Mode Alignment (Hungarian Algorithm) to align DMD modes
  across varying physical parameters before passing them to the CNN.
- Tests if Mode Alignment fixes the zero-shot parametric generalization failure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 5C: Algorithmic Mode Alignment & Generalization")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# FIXED GLOBAL SETTINGS & GRID
# ─────────────────────────────────────────────────────────────
def set_seeds():
    np.random.seed(42)
    torch.manual_seed(42)

Nx, Ny = 50, 50
x_val = np.linspace(-0.5, 1.0, Nx)
y_val = np.linspace(-0.5, 0.5, Ny)
X_grid, Y_grid = np.meshgrid(x_val, y_val, indexing='ij')

x_grid_flat = torch.tensor(X_grid.flatten()[:, None], dtype=torch.float32)
y_grid_flat = torch.tensor(Y_grid.flatten()[:, None], dtype=torch.float32)

set_seeds()
N_f = 5000
x_f_np = np.random.uniform(-0.5, 1.0, (N_f, 1))
y_f_np = np.random.uniform(-0.5, 0.5, (N_f, 1))
x_f_t = torch.tensor(x_f_np, dtype=torch.float32, requires_grad=True)
y_f_t = torch.tensor(y_f_np, dtype=torch.float32, requires_grad=True)

N_bc = 1500
x_bc_points, y_bc_points = [], []
for _ in range(N_bc // 4): x_bc_points.append([-0.5]); y_bc_points.append([np.random.uniform(-0.5, 0.5)])
for _ in range(N_bc // 4): x_bc_points.append([1.0]); y_bc_points.append([np.random.uniform(-0.5, 0.5)])
for _ in range(N_bc // 4): x_bc_points.append([np.random.uniform(-0.5, 1.0)]); y_bc_points.append([-0.5])
for _ in range(N_bc // 4): x_bc_points.append([np.random.uniform(-0.5, 1.0)]); y_bc_points.append([0.5])

x_bc_np = np.array(x_bc_points)
y_bc_np = np.array(y_bc_points)
x_bc_t = torch.tensor(x_bc_np, dtype=torch.float32)
y_bc_t = torch.tensor(y_bc_np, dtype=torch.float32)

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

# ─────────────────────────────────────────────────────────────
# REYNOLDS SWEEP & DMD EXTRACTION
# ─────────────────────────────────────────────────────────────
reynolds_numbers = [20, 22, 25, 30, 40]
data_store = {}

for Re in reynolds_numbers:
    print(f"\n--- Processing Reynolds Number = {Re} ---")
    set_seeds()
    
    lam = 0.5 * Re - np.sqrt(0.25 * Re**2 + 4 * np.pi**2)
    U_exact = 1 - np.exp(lam * X_grid) * np.cos(2 * np.pi * Y_grid)
    V_exact = (lam / (2 * np.pi)) * np.exp(lam * X_grid) * np.sin(2 * np.pi * Y_grid)
    Mag_exact = np.sqrt(U_exact**2 + V_exact**2)
    
    u_bc_t = torch.tensor(1 - np.exp(lam * x_bc_np) * np.cos(2 * np.pi * y_bc_np), dtype=torch.float32)
    v_bc_t = torch.tensor((lam / (2 * np.pi)) * np.exp(lam * x_bc_np) * np.sin(2 * np.pi * y_bc_np), dtype=torch.float32)
    
    pinn = NavierStokesPINN()
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    
    snapshots = []
    epochs = 1500
    snapshot_interval = 30
    
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
        loss_bc = torch.mean((out_bc[:, 0:1] - u_bc_t)**2) + torch.mean((out_bc[:, 1:2] - v_bc_t)**2)
        
        out_p = pinn(x_f_t[0:1], y_f_t[0:1])
        loss_p = torch.mean(out_p[:, 2:3]**2)
        
        loss = loss_f + 10 * loss_bc + loss_p
        loss.backward()
        optimizer.step()
        
        if (ep + 1) % snapshot_interval == 0:
            with torch.no_grad():
                out_snap = pinn(x_grid_flat, y_grid_flat).detach().numpy()
                snapshots.append(np.sqrt(out_snap[:, 0]**2 + out_snap[:, 1]**2))
    
    # Final Error & Residual Evaluation
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
    
    mag_pred = np.sqrt(u.detach().numpy()**2 + v.detach().numpy()**2).reshape(Nx, Ny)
    true_error_2d = np.abs(mag_pred - Mag_exact)
    
    # DMD
    S = np.array(snapshots).T
    U, sigma, Vt = svd(S[:, :-1], full_matrices=False)
    r = 25
    U_r = U[:, :r]; sigma_r = sigma[:r]; Vt_r = Vt[:r, :]
    A_tilde = U_r.T @ S[:, 1:] @ Vt_r.T @ np.diag(1.0 / sigma_r)
    eigvals, W = np.linalg.eig(A_tilde)
    Phi = S[:, 1:] @ Vt_r.T @ np.diag(1.0 / sigma_r) @ W
    Phi_spatial = np.real(Phi).reshape(Nx, Ny, r)
    
    data_store[Re] = {
        'true_error': true_error_2d.flatten(),
        'residual': residual_magnitude.flatten(),
        'Phi_raw': Phi_spatial
    }

# ─────────────────────────────────────────────────────────────
# ALGORITHMIC MODE ALIGNMENT (Hungarian Matching)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("Performing Mode Alignment via Hungarian Algorithm...")

Phi_ref = data_store[20]['Phi_raw'] # Shape: (Nx, Ny, r)
r = Phi_ref.shape[2]

Phi_ref_flat = Phi_ref.reshape(-1, r)
norms_ref = np.linalg.norm(Phi_ref_flat, axis=0) + 1e-10
Phi_ref_norm = Phi_ref_flat / norms_ref

data_store[20]['Phi_aligned'] = Phi_ref.copy()

for Re in [22, 25, 30, 40]:
    Phi_test = data_store[Re]['Phi_raw']
    Phi_test_flat = Phi_test.reshape(-1, r)
    norms_test = np.linalg.norm(Phi_test_flat, axis=0) + 1e-10
    Phi_test_norm = Phi_test_flat / norms_test
    
    cos_sim = Phi_ref_norm.T @ Phi_test_norm
    cost_matrix = 1.0 - np.abs(cos_sim)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    Phi_test_aligned = np.zeros_like(Phi_test)
    for ref_i, test_j in zip(row_ind, col_ind):
        sign = np.sign(cos_sim[ref_i, test_j])
        if sign == 0: sign = 1.0
        Phi_test_aligned[:, :, ref_i] = sign * Phi_test[:, :, test_j]
    
    data_store[Re]['Phi_aligned'] = Phi_test_aligned
    avg_similarity = np.mean(np.abs(cos_sim[row_ind, col_ind]))
    print(f"Re={Re:<2} aligned to Re=20. Average Channel Similarity: {avg_similarity:.4f}")

def extract_windows(Phi_spatial):
    pad = 4
    Phi_pad = np.pad(Phi_spatial, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    windows = np.zeros((Nx * Ny, r, 9, 9))
    idx = 0
    for i in range(Nx):
        for j in range(Ny):
            windows[idx] = Phi_pad[i:i+9, j:j+9, :].transpose(2, 0, 1)
            idx += 1
    return windows

for Re in reynolds_numbers:
    data_store[Re]['windows'] = extract_windows(data_store[Re]['Phi_aligned'])

# ─────────────────────────────────────────────────────────────
# TRAIN CNN ON ALIGNED [Re=20, Re=25] ENVELOPE
# ─────────────────────────────────────────────────────────────
set_seeds()
print("\nTraining CNN on ALIGNED [Re=20, Re=25] Envelope...")
train_windows = np.vstack([data_store[20]['windows'], data_store[25]['windows']])
train_errors = np.concatenate([data_store[20]['true_error'], data_store[25]['true_error']])

train_mean = train_windows.mean(axis=0, keepdims=True)
train_std = train_windows.std(axis=0, keepdims=True) + 1e-8
train_windows_scaled = (train_windows - train_mean) / train_std
y_max = train_errors.max() + 1e-8
train_errors_scaled = train_errors / y_max

win_tr_t = torch.tensor(train_windows_scaled, dtype=torch.float32)
y_tr_t = torch.tensor(train_errors_scaled, dtype=torch.float32).unsqueeze(1)
loader = DataLoader(TensorDataset(win_tr_t, y_tr_t), batch_size=128, shuffle=True)

cnn = Conv2DNet(in_ch=25)
optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.005)

cnn.train()
for _ in range(80):
    for bx, by in loader:
        optimizer_cnn.zero_grad()
        loss_c = nn.MSELoss()(cnn(bx), by)
        loss_c.backward()
        optimizer_cnn.step()

cnn.eval()

# ─────────────────────────────────────────────────────────────
# ZERO-SHOT INFERENCE ON ALIGNED FEATURES
# ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"{'Re':<5} | {'Test Type':<15} | {'PDE Residual':<15} | {'Aligned DMD+CNN':<20} | {'Status'}")
print("="*80)

results = {}
for Re, t_type in zip([22, 30, 40], ['Interpolation', 'Extrapolation', 'Extrapolation']):
    win_test = data_store[Re]['windows']
    y_test = data_store[Re]['true_error']
    residual = data_store[Re]['residual']
    
    win_test_scaled = (win_test - train_mean) / train_std
    win_test_t = torch.tensor(win_test_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        preds = cnn(win_test_t).squeeze().numpy() * y_max
        preds = np.clip(preds, 0, None)
    
    corr_cnn = np.corrcoef(y_test, preds)[0, 1]
    corr_res = np.corrcoef(y_test, residual)[0, 1]
    results[Re] = {'preds': preds, 'y': y_test, 'corr_cnn': corr_cnn, 'corr_res': corr_res}
    
    status = "SUCCESS" if (corr_cnn > corr_res and corr_cnn > 0.70) else "DEGRADED" if (corr_cnn > corr_res) else "FAILED"
    print(f"{Re:<5} | {t_type:<15} | {corr_res:<15.4f} | {corr_cnn:<20.4f} | {status}")
print("="*80)

# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────
out_dir = Path('outputs')
out_dir.mkdir(exist_ok=True)
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0a0a14')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
TXT = 'white'

def style_ax(ax, title):
    ax.set_facecolor('#12121e')
    ax.set_title(title, color=TXT, fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a3a')

row = 0
for Re in [22, 30, 40]:
    ax = fig.add_subplot(gs[row, 0])
    im = ax.contourf(X_grid, Y_grid, results[Re]['y'].reshape(Nx,Ny), levels=50, cmap='magma')
    style_ax(ax, f'Re={Re} True Error')
    
    ax = fig.add_subplot(gs[row, 1])
    im = ax.contourf(X_grid, Y_grid, results[Re]['preds'].reshape(Nx,Ny), levels=50, cmap='magma')
    style_ax(ax, f'Re={Re} Aligned CNN (r={results[Re]["corr_cnn"]:.3f})')
    
    ax = fig.add_subplot(gs[row, 2])
    im = ax.contourf(X_grid, Y_grid, data_store[Re]['residual'].reshape(Nx,Ny), levels=50, cmap='Reds')
    style_ax(ax, f'Re={Re} Residual (r={results[Re]["corr_res"]:.3f})')
    
    ax = fig.add_subplot(gs[row, 3])
    im = ax.contourf(X_grid, Y_grid, data_store[Re]['Phi_aligned'][:,:,0], levels=50, cmap='viridis')
    style_ax(ax, f'Re={Re} ALIGNED DMD Mode 1')
    
    row += 1

fig.suptitle("Phase 5C: Parametric Generalization (With Mode Alignment)", color=TXT, fontsize=16, fontweight='bold', y=0.96)
plt.savefig(out_dir / 'phase5c_aligned_generalization.png', dpi=160, bbox_inches='tight', facecolor='#0a0a14')
print("\n✓ Phase 5C metrics figure saved → outputs/phase5c_aligned_generalization.png")
