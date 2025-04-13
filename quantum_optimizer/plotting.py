import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

def Qfunction(state, range=6):
    """
    Plots the Q-function of the given quantum state.
    Args:
        state (qutip.Qobj): The quantum state to plot.
        range (int): The range of the Q-function plot.
    """
    rho = qt.ptrace(state, 1)  # Partial trace over the cavity mode
    xvec = np.linspace(-range, range, 200)
    W = qt.wigner(rho, xvec, xvec, g=2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(xvec, xvec, W, 100, cmap=plt.get_cmap('RdBu'))
    ax.set_xlabel(r'Re $\alpha$', fontsize=18)
    ax.set_ylabel(r'Im $\alpha$', fontsize=18)
    plt.show()

def plot_fidelity_progress(fidelities):
    """
    Plots the fidelity progress during optimization.
    Args:
        fidelities (list or np.ndarray): The fidelity values over optimization iterations.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(fidelities, label="Fidelity", marker='o')
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Fidelity", fontsize=14)
    plt.title("Fidelity Progress During Optimization", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

def plot_parameters(betas, phis, thetas):
    """
    Plots the optimized parameters for visualization.
    Args:
        betas (list of complex): Beta values (displacements).
        phis (list of float): Phi values (angles for rotations).
        thetas (list of float): Theta values (angles for rotations).
    """
def plot_fidelity_progress(fidelities, label=None):
    """
    Plots the fidelity progress during optimization on a logarithmic scale.
    Args:
        fidelities (list or np.ndarray): The fidelity values over optimization iterations.
        label (str): Label for the plot (useful for comparisons).
    """
    plt.plot(1-fidelities, label=label, marker='o')
    plt.yscale("log")
    plt.xlabel("Iteration/Generation", fontsize=14)
    plt.ylabel("Fidelity", fontsize=14)
    plt.title("Logarithmic Fidelity Progress During Optimization", fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    if label:
        plt.legend(fontsize=12)
    plt.show()

def plot_comparison(cmaes_inverse_fidelities, scipy_inverse_fidelities):
    """
    Plots the inverse fidelity (1 - fidelity) progress for both CMA-ES and Scipy optimizations.
    Args:
        cmaes_inverse_fidelities (list or np.ndarray): CMA-ES inverse fidelity values over generations.
        scipy_inverse_fidelities (list or np.ndarray): Scipy inverse fidelity values over iterations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cmaes_inverse_fidelities, label="CMA-ES 1 - Fidelity", marker='o')
    plt.plot(scipy_inverse_fidelities, label="Scipy Optimize 1 - Fidelity", marker='x')
    plt.yscale("log")  # Use logarithmic scale for better visualization
    plt.xlabel("Iteration/Generation", fontsize=14)
    plt.ylabel("1 - Fidelity", fontsize=14)
    plt.title("Inverse Fidelity Progress Comparison", fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)
    plt.show()
