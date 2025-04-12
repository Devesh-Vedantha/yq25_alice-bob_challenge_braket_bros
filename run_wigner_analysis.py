import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
from tqdm import tqdm

# Import our modules
from decode_wigner import load_wigner_data, plot_wigner
from wigner_to_density import (
    reconstruct_from_wigner_data, 
    calculate_fidelity
)
from wigner_denoising import (
    add_gaussian_noise,
    apply_affine_distortion,
    correct_affine_distortion,
    apply_gaussian_filter,
    evaluate_denoising,
    optimize_filter_width,
    plot_denoising_comparison,
    plot_fidelity_vs_noise,
    process_noisy_wigner,
    compare_with_reference
)

def test_reconstruction_accuracy(file_path, dim_range=range(5, 31, 5)):
    """Test reconstruction accuracy for different Fock space dimensions.
    
    Args:
        file_path: Path to the Wigner function data file
        dim_range: Range of dimensions to test
    """
    print(f"Testing reconstruction accuracy for {os.path.basename(file_path)}")
    
    # Load Wigner data
    data = load_wigner_data(file_path)
    
    if data is None:
        return
    
    X, Y, Z = data
    
    # Create a "ground truth" reference with a high dimension
    high_dim = max(dim_range) + 10
    rho_ref = reconstruct_from_wigner_data(X, Y, Z, fock_dim=high_dim, n_samples=15)
    
    # Test different dimensions
    dims = []
    fidelities = []
    times = []
    
    for dim in dim_range:
        print(f"Testing dimension {dim}...")
        start_time = time.time()
        rho = reconstruct_from_wigner_data(X, Y, Z, fock_dim=dim, n_samples=10)
        end_time = time.time()
        
        # Calculate fidelity with reference
        fidelity = calculate_fidelity(rho_ref[:dim, :dim], rho)
        
        dims.append(dim)
        fidelities.append(fidelity)
        times.append(end_time - start_time)
        
        print(f"  Fidelity: {fidelity:.4f}, Time: {end_time - start_time:.2f}s")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Fidelity vs. dimension
    ax1.plot(dims, fidelities, 'o-')
    ax1.set_xlabel('Fock Space Dimension')
    ax1.set_ylabel('Fidelity')
    ax1.set_title('Reconstruction Fidelity vs. Dimension')
    ax1.grid(True)
    
    # Time vs. dimension
    ax2.plot(dims, times, 's-')
    ax2.set_xlabel('Fock Space Dimension')
    ax2.set_ylabel('Reconstruction Time (s)')
    ax2.set_title('Reconstruction Time vs. Dimension')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"reconstruction_accuracy_{os.path.basename(file_path).replace('.pickle', '')}.png", dpi=300)
    plt.show()
    
    return dims, fidelities, times

def test_noise_robustness(file_path, noise_levels=np.linspace(0.01, 0.3, 10), fock_dim=8):
    """Test robustness to noise for a given Wigner function.
    
    Args:
        file_path: Path to the Wigner function data file
        noise_levels: Levels of Gaussian noise to test
        fock_dim: Dimension for density matrix reconstruction
    """
    print(f"Testing noise robustness for {os.path.basename(file_path)}")
    
    # Load Wigner data
    data = load_wigner_data(file_path)
    
    if data is None:
        return
    
    X, Y, Z_true = data
    
    # Reconstruct original state for reference
    rho_true = reconstruct_from_wigner_data(X, Y, Z_true, fock_dim=fock_dim)
    
    # Test different noise levels
    fidelities_raw = []
    fidelities_denoised = []
    
    for noise_level in tqdm(noise_levels):
        # Add noise
        Z_noisy = add_gaussian_noise(Z_true, sigma=noise_level)
        
        # Reconstruct from noisy data
        rho_noisy = reconstruct_from_wigner_data(X, Y, Z_noisy, fock_dim=fock_dim)
        
        # Denoise with optimal filter width
        optimal_sigma, _ = optimize_filter_width(X, Y, Z_true, Z_noisy, fock_dim=fock_dim, 
                                              sigma_range=np.linspace(0.1, 2.0, 10))
        Z_denoised = apply_gaussian_filter(Z_noisy, sigma=optimal_sigma)
        
        # Reconstruct from denoised data
        rho_denoised = reconstruct_from_wigner_data(X, Y, Z_denoised, fock_dim=fock_dim)
        
        # Calculate fidelities
        fidelity_raw = calculate_fidelity(rho_true, rho_noisy)
        fidelity_denoised = calculate_fidelity(rho_true, rho_denoised)
        
        fidelities_raw.append(fidelity_raw)
        fidelities_denoised.append(fidelity_denoised)
        
        print(f"Noise level: {noise_level:.3f}, Raw fidelity: {fidelity_raw:.4f}, Denoised fidelity: {fidelity_denoised:.4f}")
    
    # Plot results
    plot_fidelity_vs_noise(noise_levels, fidelities_raw, fidelities_denoised,
                          title=f"Fidelity vs. Noise Level: {os.path.basename(file_path)}")
    
    # Save plot
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, fidelities_raw, 'o-', label='Raw')
    plt.plot(noise_levels, fidelities_denoised, 's-', label='Denoised')
    plt.xlabel('Noise Level (σ)')
    plt.ylabel('Fidelity')
    plt.title(f"Fidelity vs. Noise Level: {os.path.basename(file_path)}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"noise_robustness_{os.path.basename(file_path).replace('.pickle', '')}.png", dpi=300)
    
    return noise_levels, fidelities_raw, fidelities_denoised

def test_affine_correction(file_path, a_range=np.linspace(0.8, 1.5, 5), b_range=np.linspace(-0.1, 0.1, 5), fock_dim=8):
    """Test affine correction for different distortion parameters.
    
    Args:
        file_path: Path to the Wigner function data file
        a_range: Range of scaling factors to test
        b_range: Range of offsets to test
        fock_dim: Dimension for density matrix reconstruction
    """
    print(f"Testing affine correction for {os.path.basename(file_path)}")
    
    # Load Wigner data
    data = load_wigner_data(file_path)
    
    if data is None:
        return
    
    X, Y, Z_true = data
    
    # Reconstruct original state for reference
    rho_true = reconstruct_from_wigner_data(X, Y, Z_true, fock_dim=fock_dim)
    
    # Initialize results
    fidelities_raw = np.zeros((len(a_range), len(b_range)))
    fidelities_corrected = np.zeros((len(a_range), len(b_range)))
    
    # Test different combinations of a and b
    for i, a in enumerate(a_range):
        for j, b in enumerate(b_range):
            # Apply affine distortion
            Z_distorted = apply_affine_distortion(Z_true, a=a, b=b)
            
            # Reconstruct from distorted data
            rho_distorted = reconstruct_from_wigner_data(X, Y, Z_distorted, fock_dim=fock_dim)
            
            # Correct affine distortion
            Z_corrected, a_est, b_est = correct_affine_distortion(X, Y, Z_distorted)
            
            # Reconstruct from corrected data
            rho_corrected = reconstruct_from_wigner_data(X, Y, Z_corrected, fock_dim=fock_dim)
            
            # Calculate fidelities
            fidelity_raw = calculate_fidelity(rho_true, rho_distorted)
            fidelity_corrected = calculate_fidelity(rho_true, rho_corrected)
            
            fidelities_raw[i, j] = fidelity_raw
            fidelities_corrected[i, j] = fidelity_corrected
            
            print(f"a={a:.2f}, b={b:.2f}, Raw fidelity: {fidelity_raw:.4f}, Corrected fidelity: {fidelity_corrected:.4f}")
            print(f"Estimated parameters: a={a_est:.2f}, b={b_est:.2f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw fidelity
    im1 = ax1.imshow(fidelities_raw, extent=[b_range[0], b_range[-1], a_range[0], a_range[-1]], 
                    origin='lower', aspect='auto', cmap='viridis')
    ax1.set_xlabel('Offset (b)')
    ax1.set_ylabel('Scale (a)')
    ax1.set_title('Raw Fidelity')
    plt.colorbar(im1, ax=ax1)
    
    # Corrected fidelity
    im2 = ax2.imshow(fidelities_corrected, extent=[b_range[0], b_range[-1], a_range[0], a_range[-1]], 
                    origin='lower', aspect='auto', cmap='viridis')
    ax2.set_xlabel('Offset (b)')
    ax2.set_ylabel('Scale (a)')
    ax2.set_title('Corrected Fidelity')
    plt.colorbar(im2, ax=ax2)
    
    plt.suptitle(f"Affine Correction: {os.path.basename(file_path)}")
    plt.tight_layout()
    plt.savefig(f"affine_correction_{os.path.basename(file_path).replace('.pickle', '')}.png", dpi=300)
    plt.show()
    
    return a_range, b_range, fidelities_raw, fidelities_corrected

def process_experimental_data(file_path, fock_dim=8, optimal_sigma=1.0):
    """Process experimental Wigner function data.
    
    Args:
        file_path: Path to the experimental Wigner function data file
        fock_dim: Dimension for density matrix reconstruction
        optimal_sigma: Standard deviation for Gaussian filter
    """
    print(f"Processing experimental data: {os.path.basename(file_path)}")
    
    # Load Wigner data
    data = load_wigner_data(file_path)
    
    if data is None:
        return
    
    X, Y, Z = data
    
    # Plot original Wigner function
    plot_wigner(X, Y, Z, title=f"Original Wigner Function: {os.path.basename(file_path)}")
    
    # Correct affine distortion
    Z_corrected, a, b = correct_affine_distortion(X, Y, Z)
    
    # Apply Gaussian filter for denoising
    Z_filtered = apply_gaussian_filter(Z_corrected, sigma=optimal_sigma)
    
    # Plot comparison
    plot_denoising_comparison(X, Y, Z, Z, Z_filtered, 
                             title=f"Wigner Function Processing: {os.path.basename(file_path)}")
    
    # Reconstruct density matrix
    rho = reconstruct_from_wigner_data(X, Y, Z_filtered, fock_dim=fock_dim)
    
    # Plot eigenvalues
    eigenvalues = np.linalg.eigvalsh(rho)
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(eigenvalues)), np.flip(eigenvalues))
    plt.title(f"Eigenvalues: {os.path.basename(file_path)}")
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig(f"eigenvalues_{os.path.basename(file_path).replace('.pickle', '')}.png", dpi=300)
    plt.show()
    
    # Calculate purity
    purity = np.trace(rho @ rho)
    print(f"Purity: {purity:.4f}")
    
    # Calculate expected photon number
    n_op = np.zeros((fock_dim, fock_dim))
    for n in range(fock_dim):
        n_op[n, n] = n
    n_avg = np.real(np.trace(rho @ n_op))
    print(f"Expected photon number: {n_avg:.4f}")
    
    return rho

def process_synthetic_noisy_data(noisy_file, reference_file=None, fock_dim=8):
    """Process synthetic noisy Wigner function data.
    
    Args:
        noisy_file: Path to the noisy Wigner function data file
        reference_file: Path to the reference (clean) Wigner function data file, if available
        fock_dim: Dimension for density matrix reconstruction
    """
    print(f"Processing synthetic noisy data: {os.path.basename(noisy_file)}")
    
    # Load noisy data
    noisy_data = load_wigner_data(noisy_file)
    
    if noisy_data is None:
        return
    
    X, Y, Z_noisy = noisy_data
    
    # Optimize filter width if reference is available
    if reference_file is not None:
        reference_data = load_wigner_data(reference_file)
        if reference_data is not None:
            _, _, Z_ref = reference_data
            optimal_sigma, _ = optimize_filter_width(X, Y, Z_ref, Z_noisy, fock_dim=fock_dim)
        else:
            optimal_sigma = 1.0
    else:
        optimal_sigma = 1.0
    
    # Process noisy Wigner function
    Z_corrected, a, b = correct_affine_distortion(X, Y, Z_noisy)
    Z_filtered = apply_gaussian_filter(Z_corrected, sigma=optimal_sigma)
    
    # Plot comparison
    if reference_file is not None and reference_data is not None:
        _, _, Z_ref = reference_data
        plot_denoising_comparison(X, Y, Z_ref, Z_noisy, Z_filtered, 
                                 title=f"Synthetic Data Processing: {os.path.basename(noisy_file)}")
    else:
        plot_denoising_comparison(X, Y, Z_filtered, Z_noisy, Z_filtered, 
                                 title=f"Synthetic Data Processing: {os.path.basename(noisy_file)}")
    
    # Reconstruct density matrices
    rho_noisy = reconstruct_from_wigner_data(X, Y, Z_noisy, fock_dim=fock_dim)
    rho_processed = reconstruct_from_wigner_data(X, Y, Z_filtered, fock_dim=fock_dim)
    
    # Compare if reference is available
    if reference_file is not None and reference_data is not None:
        _, _, Z_ref = reference_data
        rho_ref = reconstruct_from_wigner_data(X, Y, Z_ref, fock_dim=fock_dim)
        
        fidelity_raw = calculate_fidelity(rho_ref, rho_noisy)
        fidelity_processed = calculate_fidelity(rho_ref, rho_processed)
        
        print(f"Raw fidelity: {fidelity_raw:.4f}")
        print(f"Processed fidelity: {fidelity_processed:.4f}")
        print(f"Improvement: {(fidelity_processed - fidelity_raw) * 100:.2f}%")
    
    return rho_processed

def main():
    """Main function to run the complete analysis pipeline."""
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # 1. Test reconstruction accuracy
    print("=== Testing Reconstruction Accuracy ===")
    test_reconstruction_accuracy("data/experimental/wigner_fock_zero.pickle")
    
    # 2. Test noise robustness
    print("\n=== Testing Noise Robustness ===")
    test_noise_robustness("data/synthetic/quantum_state_0.pickle")
    
    # 3. Test affine correction
    print("\n=== Testing Affine Correction ===")
    test_affine_correction("data/synthetic/quantum_state_0.pickle")
    
    # 4. Process experimental data
    print("\n=== Processing Experimental Data ===")
    for file_name in ["wigner_fock_zero.pickle", "wigner_fock_one.pickle", "wigner_cat_plus.pickle"]:
        file_path = f"data/experimental/{file_name}"
        process_experimental_data(file_path)
    
    # 5. Process synthetic noisy data
    print("\n=== Processing Synthetic Noisy Data ===")
    # Assume first 8 noisy Wigner functions have reference data available
    for i in range(8):
        noisy_file = f"data/synthetic/noisy_wigner_{i}.pickle"
        reference_file = f"data/synthetic/quantum_state_{i}.pickle"
        process_synthetic_noisy_data(noisy_file, reference_file)
    
    # Process remaining noisy Wigner functions without reference
    for i in range(8, 16):
        noisy_file = f"data/synthetic/noisy_wigner_{i}.pickle"
        process_synthetic_noisy_data(noisy_file)

if __name__ == "__main__":
    main() 