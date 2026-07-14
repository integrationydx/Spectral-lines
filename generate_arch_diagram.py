import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

# Ensure we use the correct font and no gridlines as requested
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']

boxes = [
    ("PINN Training\n(Epochs)", 0.05, 0.3),
    ("Snapshot\nExtraction", 0.25, 0.3),
    ("Dynamic Mode\nDecomposition", 0.45, 0.3),
    ("CNN Error\nPredictor", 0.65, 0.3),
    ("True Spatial\nError Map", 0.85, 0.3)
]

for i, (text, x, y) in enumerate(boxes):
    rect = patches.Rectangle((x, y), 0.12, 0.4, linewidth=1.5, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    ax.text(x + 0.06, y + 0.2, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    if i < len(boxes) - 1:
        # Draw arrow
        ax.annotate('', xy=(x + 0.16, y + 0.2), xytext=(x + 0.12, y + 0.2),
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

plt.tight_layout()
plt.savefig('outputs/Architecture_Diagram.pdf', dpi=300)
plt.close()
