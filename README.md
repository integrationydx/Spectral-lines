# Spectral Error Indicators for Neural PDE Solvers

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6.svg?logo=scipy)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

Physics-Informed Neural Networks (PINNs) often produce solutions that appear globally accurate
while concealing large local errors. This repository implements a novel, post-hoc spectral error
indicator framework that analyzes intermediate PINN training snapshots using Dynamic Mode
Decomposition (DMD).

By extracting spectral features — modal energy distribution, eigenvalue drift, and spectral
entropy — from training epoch sequences, we train a lightweight regression model to map these
features to local PDE solution errors. This provides a solver-agnostic mechanism to flag
high-error spatial regions and drive adaptive collocation refinement, without requiring access
to ground-truth data at inference time.

---

## Key Features

- **Unsupervised Feature Extraction** — DMD tracks spatial modes and eigenvalue trajectories across training epochs with no labels required.
- **Spatial Error Regression** — DMD-derived spectral signatures are mapped to local absolute errors using Ridge Regression (Phase 1), with XGBoost / CNN planned for Phase 2.
- **Adaptive Refinement** — High-error hotspots are flagged dynamically to guide targeted collocation point placement.
- **Solver-Agnostic** — Operates entirely on snapshot matrices; independent of the underlying PINN architecture.

---

## Results — Phase 1 (Burgers' Equation Benchmark)

Phase 1 implements the full pipeline on a synthetic Burgers' equation benchmark with a known
analytical solution (Cole-Hopf transformation), allowing exact ground-truth error maps to be
computed for validation.

| Metric | Value |
|---|---|
| DMD Rank (r) | 15 modes |
| Spectral Entropy H | 2.6943 |
| Top Eigenvalue \|λ\| | 1.0000 (neutrally stable) |
| Regression MAE | 0.00095 |
| Regression R² | 0.082 |
| Pearson Correlation | 0.287 |
| Adaptive Refinement Gain | 16.1% error reduction in flagged regions |
| High-error points flagged | 25 / 100 spatial points |

**Key observations:**
- Spectral entropy decreases monotonically across training epochs, confirming it carries convergence information.
- The adaptive refinement module correctly localizes the shock front (|x| < 0.15) without being told where it is.
- R² of 0.082 indicates the linear regression head is too weak — upgrading to XGBoost / MLP is the immediate next step (Phase 2).

![Phase 1 Results](outputs/spectral_error_results.png)

---

## Repository Structure

```
Qanad-Spectral-lines-/
├── outputs/                          # Generated plots and figures
│   └── spectral_error_results.png   # 9-panel Phase 1 results figure
├── spectral_error_pipeline.py        # Phase 1 end-to-end pipeline (run this)
├── Spectral_Error_Indicators_Research_Proposal.pdf
├── review1.txt
├── requirements.txt
└── README.md
```

> Note: `src/`, `scripts/`, and `data/` folders will be added in Phase 2 when the modular
> PyTorch PINN implementation is introduced.

---

## Installation

```bash
git clone https://github.com/integrationydx/Qanad-Spectral-lines-.git
cd Qanad-Spectral-lines-
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

```bash
python spectral_error_pipeline.py
```

This single script runs the complete Phase 1 pipeline:
1. Generates the Burgers' equation ground truth via Cole-Hopf transformation
2. Simulates 40 PINN training snapshots with realistic local error near the shock front
3. Runs Exact DMD (rank-15) on the snapshot sequence
4. Extracts spectral features: eigenvalues, modal energies, spectral entropy
5. Trains a Ridge Regression spatial error estimator
6. Runs adaptive collocation refinement simulation
7. Saves a 9-panel results figure to `outputs/spectral_error_results.png`

**Dependencies:** NumPy, SciPy, Matplotlib, Scikit-learn — no PyTorch required for Phase 1.

---

## Validation Datasets

| # | Dataset | Purpose | Status |
|---|---|---|---|
| 1 | **Burgers' Equation (1D)** | Primary synthetic benchmark — known analytical solution via Cole-Hopf | ✅ Phase 1 complete |
| 2 | **Allen-Cahn Equation** | Stiff nonlinear PINN failure benchmark — sharp phase interface | 🔜 Phase 3 |
| 3 | **Navier-Stokes (Cylinder Wake)** | Generalization to vector-valued multi-physics PDE | 🔜 Phase 4 |

---

## Roadmap

- [x] Phase 1 — Synthetic Burgers' benchmark, DMD pipeline, Ridge regression baseline
- [ ] Phase 2 — Replace Ridge with XGBoost / MLP; real PyTorch PINN with genuine snapshots
- [ ] Phase 3 — Allen-Cahn benchmark
- [ ] Phase 4 — Navier-Stokes generalization
- [ ] Phase 5 — Solver-agnostic extension to FNO / DeepONet

---

## Prior Work

| Reference | Contribution | Gap This Project Fills |
|---|---|---|
| Raissi et al. (2019) | Original PINN formulation | No error estimation or adaptive refinement |
| Dwivedi & Srinivasan (2020) | Residual-based adaptive collocation | Residual ≠ true error in stiff/nonlinear problems |
| Lu et al. (2021) — DeepXDE | Residual-driven point selection | Same limitation — not spectrally informed |
| Schmid (2010) | DMD for fluid dynamics | Applied to physical snapshots, not NN training dynamics |
| Yang & Perdikaris (2021) — B-PINNs | Bayesian UQ for PINNs | Expensive (MCMC); not solver-agnostic |
| **This work** | DMD on training snapshots → spectral features → spatial error regression | Lightweight, solver-agnostic, no ground truth at inference |

---

## Author

**Aditya Alur**
PES University, EC Campus

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.