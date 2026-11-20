# nsc-gaba-glutamate-dynamics

Simulation, closed-form equilibrium, stability, and figure-generation code for:

> **Glutamate-Driven Neurogenesis Under Michaelis-Menten GABA Inhibitory Feedback: A Dynamical Systems Analysis of Neural Stem Cell Lineage Progression**  

## Overview

`S1_complete_code.py` is a self-contained Python script that:
1. Defines the five-compartment ordinary differential equation (ODE) model of the neural stem cell lineage—neural stem cells ($X$), transit-amplifying cells ($Y$), progenitor neurons ($Z$), mature neurons ($U$), and extracellular GABA ($A$)—alongside its analytical $5 \times 5$ Jacobian.
2. Evaluates all four closed-form equilibrium classes ($E_0$, $E_Z$, $E_{YZ}$, and $E_{XYZ}$) and determines local asymptotic stability directly from the analytical Jacobian eigenvalues.
3. Executes an automated verification suite comparing closed-form results against implicit Radau ODE integrations, finite-difference Jacobians, random-start Newton equilibrium searches, linear-noise approximations, and structural model extensions.
4. Generates all manuscript tables (`./output/`) and publication figures (`./figures/`, 600 dpi PNG/TIFF and vector PDF).

## Requirements

* Python $\ge 3.9$
* `numpy >= 1.23`
* `scipy >= 1.10`
* `matplotlib >= 3.7`
* `Pillow >= 9.0`

Install dependencies via:
```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run verification suite, generate tables, and export all figures (Figs. 1–10, Fig. S1)
python S1_complete_code.py

# Run numerical verification suite only
python S1_complete_code.py --verify

# Generate specific figures only
python S1_complete_code.py --figs 1 3 9 S1

# Quick check using coarser parameter grids
python S1_complete_code.py --fast
```
