import numpy as np
import scipy.linalg as la
import pickle
import matplotlib.pyplot as plt
import sys
import os

# Check for required packages
CVXPY_INSTALLED = True
try:
    import cvxpy as cp
except ImportError:
    CVXPY_INSTALLED = False
    print("Warning: CVXPY package is not installed.")
    print("Density matrix reconstruction will use SciPy optimization methods, which may be less accurate.")
    print("To install CVXPY, run: pip install cvxpy")

def create_fock_state(n, dim):
    """Create a Fock state |n⟩ in the number basis.
    
    Args:
        n: Photon number
        dim: Hilbert space dimension
        
    Returns:
        Fock state as a column vector
    """
    state = np.zeros((dim, 1))
    if n < dim:
        state[n, 0] = 1.0
    return state

def create_coherent_state(alpha, dim):
    """Create a coherent state |α⟩ in the number basis.
    
    Args:
        alpha: Complex displacement parameter
        dim: Hilbert space dimension
        
    Returns:
        Coherent state as a column vector
    """
    n = np.arange(dim)
    state = np.exp(-0.5 * np.abs(alpha)**2) * (alpha**n) / np.sqrt(np.math.factorial(n))
    return state.reshape(-1, 1)

def create_annihilation_operator(dim):
    """Create annihilation operator in the number basis.
    
    Args:
        dim: Hilbert space dimension
        
    Returns:
        Annihilation operator as a matrix
    """
    a = np.zeros((dim, dim))
    for i in range(1, dim):
        a[i-1, i] = np.sqrt(i)
    return a

def create_displacement_operator(alpha, dim):
    """Create displacement operator D(α) in the number basis.
    
    Args:
        alpha: Complex displacement parameter
        dim: Hilbert space dimension
        
    Returns:
        Displacement operator as a matrix
    """
    # For better numerical accuracy, use a larger dimension and then truncate
    work_dim = dim + 10
    a = create_annihilation_operator(work_dim)
    a_dag = a.T.conj()
    
    # D(α) = exp(α a† - α* a)
    D = la.expm(alpha * a_dag - np.conj(alpha) * a)
    
    # Truncate back to the original dimension
    return D[:dim, :dim]

def create_parity_operator(dim):
    """Create parity operator P in the number basis.
    
    Args:
        dim: Hilbert space dimension
        
    Returns:
        Parity operator as a matrix
    """
    P = np.zeros((dim, dim))
    for n in range(dim):
        P[n, n] = (-1)**n
    return P

def create_displaced_parity_operator(alpha, dim):
    """Create displaced parity operator E_α = 1/2(I + D(α)PD(α)†).
    
    Args:
        alpha: Complex displacement parameter
        dim: Hilbert space dimension
        
    Returns:
        Displaced parity operator as a matrix
    """
    D = create_displacement_operator(alpha, dim)
    P = create_parity_operator(dim)
    I = np.eye(dim)
    
    # E_α = 1/2(I + D(α)PD(α)†)
    E = 0.5 * (I + D @ P @ D.conj().T)
    
    return E

def wigner_to_measurement_prob(wigner_value):
    """Convert Wigner function value to measurement probability.
    
    Args:
        wigner_value: Value of the Wigner function at a point
        
    Returns:
        Probability of getting +1 in a displaced parity measurement
    """
    return 0.5 * (1 + np.pi/2 * wigner_value)

def reconstruct_density_matrix(wigner_data, alphas, dim, use_cvxpy=True):
    """Reconstruct density matrix from Wigner function data.
    
    Args:
        wigner_data: Dictionary mapping complex displacements (alphas) to Wigner function values
        alphas: List of complex displacements where Wigner function was sampled
        dim: Hilbert space dimension for reconstruction
        use_cvxpy: Whether to use CVXPY (True) or SciPy (False) for optimization
        
    Returns:
        Reconstructed density matrix
    """
    # Convert Wigner values to measurement probabilities
    probs = {alpha: wigner_to_measurement_prob(wigner_data[alpha]) for alpha in alphas}
    
    # Create measurement operators for each displacement
    operators = {alpha: create_displaced_parity_operator(alpha, dim) for alpha in alphas}
    
    # If CVXPY is installed and requested, use it
    if CVXPY_INSTALLED and use_cvxpy:
        # Set up CVXPY optimization problem
        rho_var = cp.Variable((dim, dim), hermitian=True)
        
        # Define the objective: minimize sum of squared errors
        objective = 0
        for alpha in alphas:
            E_alpha = operators[alpha]
            p_alpha = probs[alpha]
            predicted_prob = cp.real(cp.trace(E_alpha @ rho_var))
            objective += cp.square(predicted_prob - p_alpha)
        
        # Constraints: rho is PSD and has trace 1
        constraints = [rho_var >> 0, cp.trace(rho_var) == 1]
        
        # Solve the problem
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve()
        
        # Get the reconstructed density matrix
        rho = rho_var.value
        
    else:
        # Alternative approach using SciPy
        # We'll use a parameterization that ensures rho is PSD and has trace 1
        from scipy.optimize import minimize
        
        def objective_function(params):
            # Convert parameters to a valid density matrix
            rho = params_to_density_matrix(params, dim)
            
            # Compute squared error
            error = 0
            for alpha in alphas:
                E_alpha = operators[alpha]
                p_alpha = probs[alpha]
                predicted_prob = np.real(np.trace(E_alpha @ rho))
                error += (predicted_prob - p_alpha)**2
            
            return error
        
        # Initial guess: maximally mixed state
        initial_guess = np.random.randn(dim**2 * 2)  # Real and imaginary parts
        
        # Minimize the objective function
        result = minimize(objective_function, initial_guess, method='L-BFGS-B')
        
        # Convert optimized parameters back to density matrix
        rho = params_to_density_matrix(result.x, dim)
    
    return rho

def params_to_density_matrix(params, dim):
    """Convert optimization parameters to a valid density matrix.
    
    Args:
        params: Optimization parameters
        dim: Hilbert space dimension
        
    Returns:
        Valid density matrix (PSD with trace 1)
    """
    # Reshape parameters into a complex matrix
    n_params = len(params) // 2
    T = params[:n_params] + 1j * params[n_params:]
    T = T.reshape(dim, dim)
    
    # Ensure Hermiticity
    T = 0.5 * (T + T.conj().T)
    
    # Construct rho = T†T and normalize
    rho = T.conj().T @ T
    rho = rho / np.trace(rho)
    
    return rho

def calculate_fidelity(rho1, rho2):
    """Calculate fidelity between two density matrices.
    
    Args:
        rho1, rho2: Density matrices
        
    Returns:
        Fidelity between the states
    """
    # F(ρ1, ρ2) = [Tr(√(√ρ1 ρ2 √ρ1))]²
    sqrt_rho1 = la.sqrtm(rho1)
    fidelity = np.trace(la.sqrtm(sqrt_rho1 @ rho2 @ sqrt_rho1))
    return np.real(fidelity)**2

def interpolate_wigner_data(X, Y, Z, alphas):
    """Interpolate Wigner function data to get values at specific displacements.
    
    Args:
        X, Y: Coordinate grids
        Z: Wigner function values on the grid
        alphas: List of complex displacements to interpolate at
        
    Returns:
        Dictionary mapping displacements to interpolated Wigner values
    """
    from scipy.interpolate import RegularGridInterpolator, griddata
    
    # Create interpolator for the Wigner function
    if len(X.shape) == 1 and len(Y.shape) == 1:
        interpolator = RegularGridInterpolator((X, Y), Z.T)
        
        # Interpolate at each alpha point
        wigner_data = {}
        for alpha in alphas:
            x, y = alpha.real, alpha.imag
            if x >= X.min() and x <= X.max() and y >= Y.min() and y <= Y.max():
                wigner_data[alpha] = interpolator([x, y])[0]
        
    else:
        # For non-rectangular grids, use general interpolation
        
        # Reshape the grid points into a list of (x,y) coordinates
        points = np.vstack([X.flatten(), Y.flatten()]).T
        values = Z.flatten()
        
        # Interpolate at each alpha point
        wigner_data = {}
        for alpha in alphas:
            x, y = alpha.real, alpha.imag
            if (x >= X.min() and x <= X.max() and y >= Y.min() and y <= Y.max()):
                wigner_data[alpha] = griddata(points, values, np.array([[x, y]]), method='cubic')[0]
    
    return wigner_data

def sample_phase_space_points(x_max, y_max, n_points):
    """Generate a set of phase-space points for sampling the Wigner function.
    
    Args:
        x_max, y_max: Maximum values of x and y coordinates
        n_points: Number of points to sample in each dimension
        
    Returns:
        List of complex displacements
    """
    # Create a grid of points
    x = np.linspace(-x_max, x_max, n_points)
    y = np.linspace(-y_max, y_max, n_points)
    xx, yy = np.meshgrid(x, y)
    
    # Convert to complex values
    alphas = xx.flatten() + 1j * yy.flatten()
    
    return alphas

def reconstruct_from_wigner_data(X, Y, Z, fock_dim=10, n_samples=10):
    """Reconstruct density matrix from Wigner function data.
    
    Args:
        X, Y: Coordinate grids
        Z: Wigner function values on the grid
        fock_dim: Dimension of the Fock space to use
        n_samples: Number of sample points to use in each dimension
        
    Returns:
        Reconstructed density matrix
    """
    # Determine sampling range based on the data
    x_max = max(abs(X.min()), abs(X.max()))
    y_max = max(abs(Y.min()), abs(Y.max()))
    
    # Sample phase space points
    alphas = sample_phase_space_points(x_max, y_max, n_samples)
    
    # Interpolate Wigner function to get values at the sample points
    wigner_data = interpolate_wigner_data(X, Y, Z, alphas)
    
    # Reconstruct density matrix using CVXPY if available, otherwise use SciPy
    rho = reconstruct_density_matrix(wigner_data, list(wigner_data.keys()), fock_dim, use_cvxpy=CVXPY_INSTALLED)
    
    return rho

def plot_density_matrix(rho, title="Density Matrix"):
    """Plot the density matrix.
    
    Args:
        rho: Density matrix
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot real part
    im1 = ax1.matshow(np.real(rho), cmap='RdBu_r')
    ax1.set_title("Real Part")
    plt.colorbar(im1, ax=ax1)
    
    # Plot imaginary part
    im2 = ax2.matshow(np.imag(rho), cmap='RdBu_r')
    ax2.set_title("Imaginary Part")
    plt.colorbar(im2, ax=ax2)
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def test_reconstruction(state_type="fock", param=0, dim=10, n_samples=10):
    """Test reconstruction accuracy for a known state.
    
    Args:
        state_type: Type of state ("fock", "coherent", "cat")
        param: Parameter for the state (n for Fock, alpha for coherent/cat)
        dim: Hilbert space dimension
        n_samples: Number of sample points in each dimension
    """
    # Create the state
    if state_type == "fock":
        state = create_fock_state(param, dim)
        rho_true = state @ state.conj().T
        state_name = f"Fock |{param}⟩"
    elif state_type == "coherent":
        state = create_coherent_state(param, dim)
        rho_true = state @ state.conj().T
        state_name = f"Coherent |α={param}⟩"
    else:
        # Default to vacuum state
        state = create_fock_state(0, dim)
        rho_true = state @ state.conj().T
        state_name = "Vacuum"
    
    # Generate Wigner function
    x_max = y_max = 5.0
    grid_size = 51
    x = np.linspace(-x_max, x_max, grid_size)
    y = np.linspace(-y_max, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Sample Wigner function
    alphas = sample_phase_space_points(x_max, y_max, grid_size)
    alphas_reshaped = alphas.reshape(grid_size, grid_size)
    
    # Compute Wigner function
    Z = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            alpha = alphas_reshaped[i, j]
            E_alpha = create_displaced_parity_operator(alpha, dim)
            Z[i, j] = (2/np.pi) * np.real(np.trace(E_alpha @ rho_true)) - (2/np.pi)
    
    # Reconstruct density matrix
    rho = reconstruct_from_wigner_data(x, y, Z, fock_dim=dim, n_samples=n_samples)
    
    # Calculate fidelity
    fidelity = calculate_fidelity(rho_true, rho)
    
    print(f"Reconstruction for {state_name}:")
    print(f"Fidelity: {fidelity:.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, Z, cmap='RdBu_r', shading='auto')
    plt.colorbar(label='W(x,p)')
    plt.contour(X, Y, Z, colors='black', alpha=0.5)
    plt.title(f'Wigner Function for {state_name}')
    plt.xlabel('Position (x)')
    plt.ylabel('Momentum (p)')
    plt.axis('equal')
    plt.show()
    
    # Plot density matrix
    plot_density_matrix(rho, title=f"Reconstructed Density Matrix for {state_name}")
    
    return rho_true, rho, fidelity

if __name__ == "__main__":
    # Test reconstruction for a Fock state
    test_reconstruction(state_type="fock", param=1, dim=10, n_samples=10)
    
    # Test reconstruction for a coherent state
    test_reconstruction(state_type="coherent", param=2.0, dim=10, n_samples=10) 