import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# Enforce guide's aesthetic requirements
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'

# Define the Kramers 1D double-well potential
x = np.linspace(-2, 2, 500)
U = x**4 - 2*x**2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, U, 'b-', linewidth=2.5, label='Potential U(x)')

# Annotate minima and saddle point
ax.scatter([-1, 1], [-1, -1], color='red', s=100, zorder=5, label='Minima ($x_m$)')
ax.scatter([0], [0], color='green', s=100, zorder=5, label='Saddle Point ($x_s$)')

# Add annotations for delta U
ax.annotate('', xy=(0, 0), xytext=(0, -1),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text(0.1, -0.5, '$\Delta U$', fontsize=14, verticalalignment='center')

# Highlight the metastable trap
ax.text(-1, -1.3, 'Metastable\nLocal Minimum', fontsize=12, ha='center')
ax.text(1, -1.3, 'Global\nMinimum', fontsize=12, ha='center')

# Styling
ax.set_title("Eyring-Kramers 1D Escape Potential", fontsize=16)
ax.set_xlabel("State space (x)", fontsize=14)
ax.set_ylabel("Loss Landscape / Energy (U)", fontsize=14)
ax.legend(fontsize=12)

# Save
os.makedirs("outputs", exist_ok=True)
plt.tight_layout()
plt.savefig("outputs/double_well.pdf", format="pdf")
print("Saved double well diagram to outputs/double_well.pdf")
