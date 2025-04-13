# From Noise to Information: Reconstructing Quantum States from Wigner Functions

<div align="center">
  <img src="images/anb_logo.png" alt="Cat" width="150" />
</div>
<div align="center">
<strong>YQuantum 2025 - Alice &amp; Bob YQuantum Challenge</strong>
</div>

## Authors  
- **Devesh Vedantha** - DePaul University  
- **Shilpi Shah** - Rutgers University 
- **Sana Muneer** - Southern Connecticut State University
- **Russell Yang** - Harvard University  
- **Sang Hyun Kim** - Purdue University  

--- 

## Overview

For quantum computing, the [Wigner function](https://en.wikipedia.org/wiki/Wigner_quasiprobability_distribution) is a natural way to measure quantum states. It provides a clear view of the state, making it ideal for state tomography ([quantum tomography](https://en.wikipedia.org/wiki/Quantum_tomography))—the method we use to see what’s happening inside a quantum computer.

<div align="center">
  <img src="images/cat.gif" alt="Cat" width="300" />
</div>

# YQuantum Alice & Bob Challenge Submission  
Quantum State Tomography using Wigner Functions

This repository presents our complete implementation for the YQuantum Alice & Bob Challenge. We follow the structure defined in the official challenge document and provide high-quality, reproducible results for each task. All components—from Wigner simulation to supervised denoising and accelerated fitting—are modular and independently executable.

## Files to Review

Each task corresponds to a Jupyter notebook, as listed below. Please refer directly to these notebooks for implementation details, results, and visualizations.

| Section | Task Description                                                 | File Name         |
|---------|------------------------------------------------------------------|-------------------|
| 1.A     | Generate Wigner Functions                                        | `1a.ipynb`        |
| 1.B     | Density Matrix Reconstruction from Wigner Data                   | `1b.ipynb`        |
| 1.C     | Robustness: Noise and Real Experimental Data                     | `1c_part_1.ipynb`  |
|         |                                                                  | `1c_part_2.ipynb`  |
| 2.A     | Affine Distortion Correction and Gaussian Denoising              | `2a.ipynb`        |
| 2.C     | Accelerated Quantum State Reconstruction (fit function speed-up) | `2c.ipynb`        |

*Note: Task 2.B (Supervised Learning for Wigner Denoising) is not included in this version.*

## Summary of Tasks

### 1.A — Wigner Function Generation
- Simulations of Fock states, coherent states, 2- and 3-cat states.
- Dissipative cat state modeled using a two-photon exchange Hamiltonian.
- Wigner functions visualized and animated using `dynamiqs`.

### 1.B — Density Matrix Reconstruction
- Reconstruction of $\rho$ from Wigner function $W(x,p)$ using displaced parity measurements.
- Solved a constrained convex least-squares optimization.
- Evaluation via fidelity, purity, and eigenvalue spectra.

### 1.C — Robustness to Noise and Real Data
- Reconstruction performance analyzed under Gaussian noise $\mathcal{N}(0, \sigma^2)$.
- Applied reconstruction pipeline to real experimental Wigner datasets.
- Fidelity plotted against noise level $\sigma$.

### 2.A — Affine Correction and Denoising
- Estimated unknown affine parameters $(a, b)$ and applied correction.
- Applied 2D Gaussian filtering with tunable $\sigma$.
- Compared fidelity before and after denoising.

### 2.C — Accelerated Fitting Pipeline
- Optimized the reconstruction pipeline for large grids and high Fock-space dimensions.
- Benchmarked against original implementation (runtime vs fidelity).

## Authors

- Devesh Vedantha — DePaul University  
- Shilpi Shah — Rutgers University  
- Sana Muneer — Southern Connecticut State University  
- Russell Yang — Harvard University  
- Sang Hyun Kim — Purdue University

---

We invite reviewers to explore the notebooks listed above. Each file is self-contained and carefully documented to demonstrate both theoretical understanding and technical execution of the quantum tomography tasks.

