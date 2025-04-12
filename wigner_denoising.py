import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Check for required packages
try:
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import minimize
    from scipy.interpolate import griddata
    SCIPY_INSTALLED = True
except ImportError:
    SCIPY_INSTALLED = False
    print("Warning: SciPy is not installed. Some functionality will be limited.")
    print("To install SciPy, run: pip install scipy")

# Check if our local modules are available
try:
    from decode_wigner import load_wigner_data
    from wigner_to_density import reconstruct_from_wigner_data, calculate_fidelity
    LOCAL_MODULES_AVAILABLE = True
except ImportError:
    LOCAL_MODULES_AVAILABLE = False
    print("Warning: Local modules not found. Some functionality will be limited.")

def estimate_background_offset(Z, percentile=5):
    """Estimate background offset 'b' in the affine transformation W_measured = a*W + b.
    
    Args:
        Z: Wigner function values
        percentile: Percentile of values to consider as background
        
    Returns:
        Estimated background offset
    """
    # Assume background is represented by the lowest values in the distribution
    background = np.percentile(Z, percentile)
    return background

def estimate_scaling_factor(Z, b, target_integral=1.0, dx=1.0, dy=1.0):
    """Estimate scaling factor 'a' in the affine transformation W_measured = a*W + b.
    
    Args:
        Z: Wigner function values
        b: Background offset
        target_integral: Target value for the integral (1.0 for proper normalization)
        dx, dy: Grid spacing in x and y dimensions
        
    Returns:
        Estimated scaling factor
    """
    # Current integral value after subtracting background
    current_integral = np.sum(Z - b) * dx * dy
    
    # Estimate scaling factor
    a = current_integral / target_integral
    
    return a

def correct_affine_distortion(X, Y, Z, dx=None, dy=None, percentile=5):
    """Correct affine distortion in Wigner function: W_measured = a*W + b.
    
    Args:
        X, Y: Coordinate grids
        Z: Wigner function values
        dx, dy: Grid spacing (calculated from X, Y if not provided)
        percentile: Percentile to use for background estimation
        
    Returns:
        Corrected Wigner function values
    """
    # Calculate grid spacing if not provided
    if dx is None:
        if len(X.shape) == 1:
            dx = (X[-1] - X[0]) / (len(X) - 1)
        else:
            dx = (X[0, -1] - X[0, 0]) / (X.shape[1] - 1)
    
    if dy is None:
        if len(Y.shape) == 1:
            dy = (Y[-1] - Y[0]) / (len(Y) - 1)
        else:
            dy = (Y[-1, 0] - Y[0, 0]) / (Y.shape[0] - 1)
    
    # Estimate background offset
    b = estimate_background_offset(Z, percentile)
    
    # Estimate scaling factor
    a = estimate_scaling_factor(Z, b, target_integral=1.0, dx=dx, dy=dy)
    
    print(f"Estimated affine parameters: a = {a}, b = {b}")
    
    # Apply correction
    Z_corrected = (Z - b) / a
    
    return Z_corrected, a, b

def apply_gaussian_filter(Z, sigma=1.0):
    """Apply Gaussian filter to Wigner function to reduce noise.
    
    Args:
        Z: Wigner function values
        sigma: Standard deviation of the Gaussian filter
        
    Returns:
        Filtered Wigner function values
    """
    if SCIPY_INSTALLED:
        return gaussian_filter(Z, sigma=sigma)
    else:
        print("Warning: SciPy not installed, using simple mean filter instead")
        # Simple mean filter as fallback
        Z_filtered = Z.copy()
        for i in range(1, Z.shape[0] - 1):
            for j in range(1, Z.shape[1] - 1):
                Z_filtered[i, j] = np.mean(Z[i-1:i+2, j-1:j+2])
        return Z_filtered

def add_gaussian_noise(Z, sigma=0.1):
    """Add Gaussian noise to Wigner function for testing.
    
    Args:
        Z: Wigner function values
        sigma: Standard deviation of the noise
        
    Returns:
        Noisy Wigner function values
    """
    noise = np.random.normal(0, sigma, Z.shape)
    return Z + noise

def apply_affine_distortion(Z, a=1.2, b=0.05):
    """Apply affine distortion to Wigner function for testing.
    
    Args:
        Z: Wigner function values
        a: Scaling factor
        b: Background offset
        
    Returns:
        Distorted Wigner function values
    """
    return a * Z + b

def evaluate_denoising(X, Y, Z_true, Z_noisy, Z_denoised, fock_dim=8):
    """Evaluate denoising performance by comparing reconstructed density matrices.
    
    Args:
        X, Y: Coordinate grids
        Z_true: True Wigner function values
        Z_noisy: Noisy Wigner function values
        Z_denoised: Denoised Wigner function values
        fock_dim: Dimension for density matrix reconstruction
        
    Returns:
        Tuple of (fidelity_raw, fidelity_denoised)
    """
    if not LOCAL_MODULES_AVAILABLE:
        print("Error: Local modules not available for density matrix reconstruction")
        return None
    
    # Reconstruct density matrices
    rho_true = reconstruct_from_wigner_data(X, Y, Z_true, fock_dim=fock_dim)
    rho_noisy = reconstruct_from_wigner_data(X, Y, Z_noisy, fock_dim=fock_dim)
    rho_denoised = reconstruct_from_wigner_data(X, Y, Z_denoised, fock_dim=fock_dim)
    
    # Calculate fidelities
    fidelity_raw = calculate_fidelity(rho_true, rho_noisy)
    fidelity_denoised = calculate_fidelity(rho_true, rho_denoised)
    
    print(f"Fidelity before denoising: {fidelity_raw:.4f}")
    print(f"Fidelity after denoising: {fidelity_denoised:.4f}")
    print(f"Improvement: {(fidelity_denoised - fidelity_raw) * 100:.2f}%")
    
    return fidelity_raw, fidelity_denoised

def optimize_filter_width(X, Y, Z_true, Z_noisy, fock_dim=8, sigma_range=np.linspace(0.1, 3.0, 20)):
    """Optimize Gaussian filter width to maximize fidelity.
    
    Args:
        X, Y: Coordinate grids
        Z_true: True Wigner function values
        Z_noisy: Noisy Wigner function values
        fock_dim: Dimension for density matrix reconstruction
        sigma_range: Range of sigma values to try
        
    Returns:
        Tuple of (optimal_sigma, fidelities)
    """
    if not LOCAL_MODULES_AVAILABLE or not SCIPY_INSTALLED:
        print("Error: Required modules not available for filter optimization")
        return 1.0, None  # Return default sigma
    
    fidelities = []
    
    # Reconstruct true density matrix once
    rho_true = reconstruct_from_wigner_data(X, Y, Z_true, fock_dim=fock_dim)
    
    for sigma in sigma_range:
        # Apply Gaussian filter
        Z_filtered = apply_gaussian_filter(Z_noisy, sigma=sigma)
        
        # Reconstruct density matrix
        rho_filtered = reconstruct_from_wigner_data(X, Y, Z_filtered, fock_dim=fock_dim)
        
        # Calculate fidelity
        fidelity = calculate_fidelity(rho_true, rho_filtered)
        fidelities.append(fidelity)
        
        print(f"Sigma = {sigma:.2f}, Fidelity = {fidelity:.4f}")
    
    # Find optimal sigma
    optimal_idx = np.argmax(fidelities)
    optimal_sigma = sigma_range[optimal_idx]
    
    print(f"Optimal sigma: {optimal_sigma:.2f}, Max Fidelity: {fidelities[optimal_idx]:.4f}")
    
    return optimal_sigma, fidelities

def plot_denoising_comparison(X, Y, Z_true, Z_noisy, Z_denoised, title="Wigner Function Denoising"):
    """Plot comparison of true, noisy, and denoised Wigner functions.
    
    Args:
        X, Y: Coordinate grids
        Z_true: True Wigner function values
        Z_noisy: Noisy Wigner function values
        Z_denoised: Denoised Wigner function values
        title: Plot title
    """
    # Create a meshgrid if X and Y are 1D arrays
    if len(X.shape) == 1 and len(Y.shape) == 1:
        X_mesh, Y_mesh = np.meshgrid(X, Y)
    else:
        X_mesh, Y_mesh = X, Y
    
    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Find global min/max for consistent colormap
    vmin = min(Z_true.min(), Z_noisy.min(), Z_denoised.min())
    vmax = max(Z_true.max(), Z_noisy.max(), Z_denoised.max())
    
    # Plot true Wigner function
    im1 = axes[0].pcolormesh(X_mesh, Y_mesh, Z_true, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title("True Wigner Function")
    axes[0].set_xlabel("Position (x)")
    axes[0].set_ylabel("Momentum (p)")
    axes[0].axis('square')
    
    # Plot noisy Wigner function
    im2 = axes[1].pcolormesh(X_mesh, Y_mesh, Z_noisy, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title("Noisy Wigner Function")
    axes[1].set_xlabel("Position (x)")
    axes[1].set_ylabel("Momentum (p)")
    axes[1].axis('square')
    
    # Plot denoised Wigner function
    im3 = axes[2].pcolormesh(X_mesh, Y_mesh, Z_denoised, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
    axes[2].set_title("Denoised Wigner Function")
    axes[2].set_xlabel("Position (x)")
    axes[2].set_ylabel("Momentum (p)")
    axes[2].axis('square')
    
    # Add colorbar
    plt.colorbar(im3, ax=axes.ravel().tolist())
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_fidelity_vs_noise(noise_levels, fidelities_raw, fidelities_denoised, title="Fidelity vs. Noise Level"):
    """Plot fidelity vs. noise level for raw and denoised reconstructions.
    
    Args:
        noise_levels: Array of noise levels
        fidelities_raw: Fidelities for raw reconstructions
        fidelities_denoised: Fidelities for denoised reconstructions
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, fidelities_raw, 'o-', label='Raw')
    plt.plot(noise_levels, fidelities_denoised, 's-', label='Denoised')
    plt.xlabel('Noise Level (σ)')
    plt.ylabel('Fidelity')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fidelity_vs_filter_width(sigma_values, fidelities, optimal_sigma, title="Fidelity vs. Filter Width"):
    """Plot fidelity vs. filter width.
    
    Args:
        sigma_values: Array of filter widths
        fidelities: Corresponding fidelities
        optimal_sigma: Optimal filter width
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_values, fidelities, 'o-')
    plt.axvline(x=optimal_sigma, color='r', linestyle='--', label=f'Optimal σ = {optimal_sigma:.2f}')
    plt.xlabel('Filter Width (σ)')
    plt.ylabel('Fidelity')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def process_noisy_wigner(file_path, optimal_sigma=1.0, fock_dim=8):
    """Process a noisy Wigner function by applying affine correction and Gaussian filtering.
    
    Args:
        file_path: Path to the pickle file
        optimal_sigma: Standard deviation for Gaussian filter
        fock_dim: Dimension for density matrix reconstruction
        
    Returns:
        Tuple of (X, Y, Z_corrected, Z_filtered)
    """
    if not LOCAL_MODULES_AVAILABLE:
        print("Error: Local modules not available for processing")
        return None
    
    # Load Wigner data
    data = load_wigner_data(file_path)
    
    if data is None:
        return None
    
    X, Y, Z_noisy = data
    
    # Correct affine distortion
    Z_corrected, a, b = correct_affine_distortion(X, Y, Z_noisy)
    
    # Apply Gaussian filter
    Z_filtered = apply_gaussian_filter(Z_corrected, sigma=optimal_sigma)
    
    return X, Y, Z_corrected, Z_filtered

def compare_with_reference(noisy_file, reference_file, optimal_sigma=1.0, fock_dim=8):
    """Compare denoised Wigner function with a reference.
    
    Args:
        noisy_file: Path to the noisy Wigner function
        reference_file: Path to the reference Wigner function
        optimal_sigma: Standard deviation for Gaussian filter
        fock_dim: Dimension for density matrix reconstruction
        
    Returns:
        Tuple of (fidelity_raw, fidelity_denoised)
    """
    if not LOCAL_MODULES_AVAILABLE or not SCIPY_INSTALLED:
        print("Error: Required modules not available for comparison")
        return None
    
    # Load data
    noisy_data = load_wigner_data(noisy_file)
    reference_data = load_wigner_data(reference_file)
    
    if noisy_data is None or reference_data is None:
        return None
    
    X_noisy, Y_noisy, Z_noisy = noisy_data
    X_ref, Y_ref, Z_ref = reference_data
    
    # Ensure grids match or interpolate if necessary
    if X_noisy.shape != X_ref.shape or Y_noisy.shape != Y_ref.shape:
        print("Grids don't match, interpolating reference data...")
        
        if len(X_ref.shape) == 1 and len(Y_ref.shape) == 1:
            # Regular grid interpolation
            from scipy.interpolate import RegularGridInterpolator
            interpolator = RegularGridInterpolator((X_ref, Y_ref), Z_ref.T)
            
            X_mesh, Y_mesh = np.meshgrid(X_noisy, Y_noisy)
            points = np.vstack([X_mesh.flatten(), Y_mesh.flatten()]).T
            Z_ref_interp = interpolator(points).reshape(X_mesh.shape)
            
        else:
            # General interpolation
            points_ref = np.vstack([X_ref.flatten(), Y_ref.flatten()]).T
            points_noisy = np.vstack([X_noisy.flatten(), Y_noisy.flatten()]).T
            Z_ref_interp = griddata(points_ref, Z_ref.flatten(), points_noisy, method='cubic').reshape(X_noisy.shape)
        
        Z_ref = Z_ref_interp
    
    # Process noisy Wigner function
    Z_corrected, a, b = correct_affine_distortion(X_noisy, Y_noisy, Z_noisy)
    Z_filtered = apply_gaussian_filter(Z_corrected, sigma=optimal_sigma)
    
    # Evaluate denoising
    fidelity_raw, fidelity_denoised = evaluate_denoising(X_noisy, Y_noisy, Z_ref, Z_noisy, Z_filtered, fock_dim=fock_dim)
    
    # Plot comparison
    plot_denoising_comparison(X_noisy, Y_noisy, Z_ref, Z_noisy, Z_filtered, title="Wigner Function Denoising")
    
    return fidelity_raw, fidelity_denoised

def test_denoising_basic():
    """Basic test of denoising without dependencies."""
    # Create a simple test pattern
    grid_size = 50
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Create a simple Gaussian function
    sigma_x = 1.0
    sigma_y = 1.0
    Z_true = np.exp(-(X**2/(2*sigma_x**2) + Y**2/(2*sigma_y**2)))
    
    # Add noise
    Z_noisy = add_gaussian_noise(Z_true, sigma=0.1)
    
    # Apply affine distortion
    Z_distorted = apply_affine_distortion(Z_noisy, a=1.2, b=0.05)
    
    # Correct affine distortion
    Z_corrected, a, b = correct_affine_distortion(X, Y, Z_distorted)
    
    # Apply Gaussian filter
    Z_filtered = apply_gaussian_filter(Z_corrected, sigma=1.0)
    
    # Plot comparison
    plot_denoising_comparison(X, Y, Z_true, Z_distorted, Z_filtered, title="Basic Denoising Test")

if __name__ == "__main__":
    # Basic test that works without dependencies
    test_denoising_basic()
    
    # Only run more advanced tests if dependencies are available
    if LOCAL_MODULES_AVAILABLE and SCIPY_INSTALLED:
        # Load a synthetic Wigner function if available
        ref_file = "data/synthetic/quantum_state_0.pickle"
        if os.path.exists(ref_file):
            ref_data = load_wigner_data(ref_file)
            
            if ref_data:
                X, Y, Z_true = ref_data
                
                # Create noisy version for testing
                Z_noisy = apply_affine_distortion(Z_true, a=1.2, b=0.05)
                Z_noisy = add_gaussian_noise(Z_noisy, sigma=0.1)
                
                # Correct affine distortion
                Z_corrected, a, b = correct_affine_distortion(X, Y, Z_noisy)
                
                # Apply Gaussian filter
                Z_filtered = apply_gaussian_filter(Z_corrected, sigma=1.0)
                
                # Evaluate denoising
                fidelity_raw, fidelity_denoised = evaluate_denoising(X, Y, Z_true, Z_noisy, Z_filtered)
                
                # Plot comparison
                plot_denoising_comparison(X, Y, Z_true, Z_noisy, Z_filtered) 