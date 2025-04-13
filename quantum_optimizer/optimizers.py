from scipy import optimize
import qutip as qt
from cmaes import CMA, CMAwM
import numpy as np

def to_list_complex(input_list, start, stop):
    """
    Converts a real-valued parameter list into a list of complex numbers.
    """
    def to_complex(a, b):
        return a + 1j * b
    return [to_complex(input_list[i], input_list[i + 1]) for i in range(start, stop, 2)]

def fitness_func(quantum_state, params):
    """
    Calculates the inverse fidelity (1 - fidelity) for a given parameter set.
    Args:
        quantum_state (QuantumState): The quantum state object.
        params (list): Real-valued parameters to optimize.
    Returns:
        float: Inverse fidelity (1 - fidelity).
    """
    N = quantum_state.N_blocks
    betas = [complex(params[i], params[i + 1]) for i in range(0, 2 * N, 2)]
    phis = params[2 * N:3 * N]
    thetas = params[3 * N:4 * N]
    state = quantum_state.gate_sequence(betas, phis, thetas, N)
    return quantum_state.inverse_fidelity(state)  # Returns 1 - fidelity


def scipy_optimize(quantum_state, method="BFGS", tol=1e-12, max_iterations=5000):
    """
    Optimizes the quantum state using scipy.optimize and tracks inverse fidelity progress.
    Args:
        quantum_state (QuantumState): The quantum state object.
        method (str): Optimization method (e.g., 'nelder-mead').
        tol (float): Tolerance for optimization.
        max_iterations (int): Maximum number of iterations.
    Returns:
        tuple: (result, inverse_fidelity_progress), where inverse_fidelity_progress is a list of 1 - fidelity values.
    """
    N = quantum_state.N_blocks
    initial_guess = np.ones(4 * N)
    bounds = [(-10, 10)] * (2 * N) + [(-np.pi, np.pi)] * (2 * N)
    inverse_fidelity_progress = []

    def callback(params):
        """
        Callback function to track inverse fidelity at each iteration.
        """
        fitness = fitness_func(quantum_state, params)
        inverse_fidelity_progress.append(fitness)
        print(f"Current 1 - Fidelity = {fitness:.6f}", end='\r')

    result = optimize.minimize(
        lambda params: fitness_func(quantum_state, params),
        initial_guess,
        method=method,
        tol=tol,
        bounds=bounds,
        callback=callback,
        options={"disp": True, "maxiter": max_iterations}
    )
    return result, inverse_fidelity_progress

def cmaes_optimize(quantum_state, sigma=1.0, max_generations=500):
    """
    Optimizes the quantum state using the CMA-ES algorithm and tracks inverse fidelity.
    Args:
        quantum_state (QuantumState): The quantum state object.
        sigma (float): Initial step size for CMA-ES.
        max_generations (int): Maximum number of generations.
    Returns:
        tuple: (best_solution, inverse_fidelity_progress), where inverse_fidelity_progress is a list of 1 - fidelity values.
    """
    N = quantum_state.N_blocks
    initial_guess = np.ones(4 * N)
    bounds = np.array([(-40, 40)] * (2 * N) + [(-np.pi, np.pi)] * (2 * N))

    optimizer = CMA(mean=initial_guess, sigma=sigma, bounds=bounds, lr_adapt=True)
    best_solution = None
    best_fitness = float("inf")
    inverse_fidelity_progress = []  # Track 1 - fidelity values
    
    for generation in range(max_generations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            fitness = fitness_func(quantum_state, x)
            solutions.append((x, fitness))
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = x
        optimizer.tell(solutions)

        # Track inverse fidelity progress
        inverse_fidelity_progress.append(best_fitness)
        print(f"Generation {generation}: 1 - Fidelity = {best_fitness:.6f}", end='\r')

        if optimizer.should_stop():
            break

    return best_solution, inverse_fidelity_progress

def cmawm_optimize(quantum_state, sigma=1.0, max_generations=500, callback=None):
    """
    Optimizes the quantum state using the CMAwM algorithm and tracks inverse fidelity.
    Args:
        quantum_state (QuantumState): The quantum state object.
        sigma (float): Initial step size for CMAwM.
        max_generations (int): Maximum number of generations.
        callback (callable, optional): Function called after each generation with the current best fitness.
    Returns:
        tuple: (best_solution, inverse_fidelity_progress), where inverse_fidelity_progress is a list of 1 - fidelity values.
    """
    N = quantum_state.N_blocks
    dim = 4 * N
    bounds = np.array([[-10, 10]] * (2 * N) + [[-np.pi, np.pi]] * (2 * N))
    steps = np.zeros(dim)  # Continuous optimization

    # Initialize CMAwM optimizer
    optimizer = CMAwM(mean=np.zeros(dim), sigma=sigma, bounds=bounds, steps=steps)
    best_solution = None
    best_fitness = float("inf")
    inverse_fidelity_progress = []  # Track 1 - fidelity values
    evals = 0

    print(" evals    1 - Fidelity")
    print("======================")

    for generation in range(max_generations):
        solutions = []
        for _ in range(optimizer.population_size):
            x_for_eval, x_for_tell = optimizer.ask()
            fitness = fitness_func(quantum_state, x_for_eval)  # Evaluate fitness
            evals += 1
            solutions.append((x_for_tell, fitness))
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = x_for_eval
        optimizer.tell(solutions)

        # Track progress
        inverse_fidelity_progress.append(best_fitness)
        if callback:
            callback(generation, best_fitness)  # Invoke callback with current generation and fitness
        print(f"{evals:5d}    {best_fitness:10.6f}", end='\r')

        if optimizer.should_stop():
            break

    return best_solution, inverse_fidelity_progress