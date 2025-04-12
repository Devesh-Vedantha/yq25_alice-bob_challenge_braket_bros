# Wigner Function Processing and Quantum State Reconstruction

This repository contains the implementation for the Alice & Bob YQuantum Challenge 2025. The code handles Wigner function generation, quantum state reconstruction from Wigner data, and denoising techniques.

## Setup and Requirements

### Dependencies

The main dependencies for this project are:

```bash
numpy
scipy
matplotlib
cvxpy
tqdm
pickle
jax
dynamiqs
```

You can install all requirements using:

```bash
pip install numpy scipy matplotlib cvxpy tqdm
pip install 'dynamiqs[full]'  # This will also install JAX
```

## Code Structure

The repository is organized into several Python modules:

1. **decode_wigner.py**: Loads and visualizes Wigner function data from pickle files
2. **wigner_to_density.py**: Implements quantum state reconstruction from Wigner data
3. **wigner_denoising.py**: Provides tools for denoising and correcting affine distortions
4. **generate_wigner.py**: Generates Wigner functions for various quantum states using dynamiqs
5. **run_wigner_analysis.py**: Combines all functionality for testing and demonstration

## How to Use

### 1. Decoding Pickle Files

To examine the contents of a Wigner function pickle file:

```bash
python decode_wigner.py
```

This will load example pickle files and display their content.

### 2. Generating Wigner Functions

To generate Wigner functions for various quantum states:

```bash
python generate_wigner.py
```

This will:
- Create Fock states, coherent states, and cat states
- Simulate the dissipative cat state evolution
- Save the Wigner functions as pickle files in the `generated_states` directory
- Generate visualization plots and GIFs

### 3. Reconstructing Density Matrices

To reconstruct a density matrix from a Wigner function:

```python
from wigner_to_density import reconstruct_from_wigner_data
from decode_wigner import load_wigner_data

# Load Wigner data
X, Y, Z = load_wigner_data("data/experimental/wigner_fock_zero.pickle")

# Reconstruct density matrix
rho = reconstruct_from_wigner_data(X, Y, Z, fock_dim=8, n_samples=10)
```

### 4. Denoising Wigner Functions

To denoise a Wigner function and correct affine distortions:

```python
from wigner_denoising import correct_affine_distortion, apply_gaussian_filter
from decode_wigner import load_wigner_data

# Load Wigner data
X, Y, Z = load_wigner_data("data/synthetic/noisy_wigner_0.pickle")

# Correct affine distortion
Z_corrected, a, b = correct_affine_distortion(X, Y, Z)

# Apply Gaussian filter for denoising
Z_filtered = apply_gaussian_filter(Z_corrected, sigma=1.0)
```

### 5. Running the Complete Analysis Pipeline

To run the complete analysis pipeline:

```bash
python run_wigner_analysis.py
```

This will:
- Test reconstruction accuracy for different Fock space dimensions
- Evaluate robustness to noise
- Test affine correction for different distortion parameters
- Process experimental Wigner function data
- Process synthetic noisy Wigner function data

## Data Structure

The Wigner function data is stored as pickle files containing a tuple `(X, Y, Z)`:
- `X`: x-coordinate grid (position)
- `Y`: y-coordinate grid (momentum)
- `Z`: Wigner function values on the grid

There are two main data directories:
- `data/experimental/`: Contains real experimental Wigner function data
- `data/synthetic/`: Contains synthetic Wigner function data, including both clean and noisy versions

## Implementation Details

### Quantum State Reconstruction

The reconstruction process follows these steps:
1. Convert Wigner values to measurement probabilities
2. Construct measurement operators for each displacement point
3. Set up a convex optimization problem to find the density matrix that best explains the data
4. Solve the optimization problem with constraints (positive semidefinite, trace=1)

### Denoising and Affine Correction

The denoising process includes:
1. Estimating and correcting affine distortions (scale and offset)
2. Applying Gaussian filtering to reduce noise
3. Optimizing filter width to maximize fidelity with the original state

## Evaluation Metrics

The code includes several metrics to evaluate reconstruction quality:
- **Fidelity**: Measures how similar two quantum states are
- **Purity**: Measures how "pure" a quantum state is
- **Eigenvalue distribution**: Shows the distribution of eigenvalues of the density matrix
- **Expected photon number**: Calculates the average photon number of the state 