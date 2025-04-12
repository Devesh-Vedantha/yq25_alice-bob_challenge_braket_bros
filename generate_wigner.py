import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
import os
import sys

# Check for required packages
DEPENDENCIES_INSTALLED = True
MISSING_PACKAGES = []

# Try to import JAX and dynamiqs
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    DEPENDENCIES_INSTALLED = False
    MISSING_PACKAGES.append("jax")

try:
    import dynamiqs as dq
except ImportError:
    DEPENDENCIES_INSTALLED = False
    MISSING_PACKAGES.append("dynamiqs")

# If dependencies are missing, provide installation instructions
if not DEPENDENCIES_INSTALLED:
    print("Error: Missing required packages for Wigner function generation:")
    for pkg in MISSING_PACKAGES:
        print(f"  - {pkg}")
    print("\nPlease install the required packages using:")
    print("  pip install 'dynamiqs[full]'")
    print("\nThis will install both dynamiqs and JAX.")
    print("For more information, visit: https://www.dynamiqs.org/stable/")
    
    # If this script is run directly (not imported), exit
    if __name__ == "__main__":
        sys.exit(1)

# Only define dynamiqs-dependent functions if dependencies are installed
if DEPENDENCIES_INSTALLED:
    def create_fock_state(n, dim=30):
        """Create a Fock state |n⟩ in the dynamiqs framework.
        
        Args:
            n: Photon number
            dim: Hilbert space dimension
            
        Returns:
            dynamiqs State object
        """
        # Create the state
        state = dq.State(dq.fock(dim, n))
        return state

    def create_coherent_state(alpha, dim=30):
        """Create a coherent state |α⟩ in the dynamiqs framework.
        
        Args:
            alpha: Complex displacement parameter
            dim: Hilbert space dimension
            
        Returns:
            dynamiqs State object
        """
        # Create the state
        state = dq.State(dq.coherent(dim, alpha))
        return state

    def create_cat_state(alpha, cat_type='even', dim=30):
        """Create a cat state in the dynamiqs framework.
        
        Args:
            alpha: Complex displacement parameter
            cat_type: Type of cat state ('even', 'odd', or 'plus', 'minus')
            dim: Hilbert space dimension
            
        Returns:
            dynamiqs State object
        """
        # Create coherent states |α⟩ and |-α⟩
        if cat_type in ['even', 'plus']:
            # |ψ⟩ ∝ |α⟩ + |-α⟩
            state_vec = dq.normalize(dq.coherent(dim, alpha) + dq.coherent(dim, -alpha))
        elif cat_type in ['odd', 'minus']:
            # |ψ⟩ ∝ |α⟩ - |-α⟩
            state_vec = dq.normalize(dq.coherent(dim, alpha) - dq.coherent(dim, -alpha))
        else:
            raise ValueError(f"Unknown cat_type: {cat_type}")
        
        # Create the state
        state = dq.State(state_vec)
        return state

    def create_three_component_cat(alpha, phases=[0, 2*np.pi/3, 4*np.pi/3], dim=30):
        """Create a three-component cat state in the dynamiqs framework.
        
        Args:
            alpha: Magnitude of displacement
            phases: List of phases for the components
            dim: Hilbert space dimension
            
        Returns:
            dynamiqs State object
        """
        # Create a superposition of three coherent states with different phases
        state_vec = dq.coherent(dim, alpha * np.exp(1j * phases[0]))
        for phase in phases[1:]:
            state_vec = state_vec + dq.coherent(dim, alpha * np.exp(1j * phase))
        
        # Normalize and create the state
        state_vec = dq.normalize(state_vec)
        state = dq.State(state_vec)
        return state

    def compute_wigner_function(state, grid_size=51, xmax=5.0, ymax=5.0):
        """Compute the Wigner function of a state.
        
        Args:
            state: dynamiqs State object
            grid_size: Number of points in each dimension
            xmax, ymax: Grid limits
            
        Returns:
            tuple: (X, Y, W) where X, Y are meshgrid coordinates and W is the Wigner function
        """
        # Create a grid of points
        x = np.linspace(-xmax, xmax, grid_size)
        y = np.linspace(-ymax, ymax, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Convert to complex grid
        alpha_grid = X + 1j * Y
        
        # Compute Wigner function
        W = dq.wigner(state.vec, alpha_grid)
        
        return X, Y, W

    def save_wigner_data(X, Y, W, filename):
        """Save Wigner function data to a pickle file.
        
        Args:
            X, Y: Coordinate grids
            W: Wigner function values
            filename: Output filename
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save data as a tuple
        with open(filename, 'wb') as f:
            pickle.dump((X, Y, W), f)
        
        print(f"Wigner data saved to {filename}")

    def simulate_two_photon_exchange(g2=1.0, epsilon_b=-4.0, kappa_b=10.0, sim_time=4.0, dim_a=30, dim_b=10, dt=0.01):
        """Simulate the dissipative cat state evolution with a two-photon exchange Hamiltonian.
        
        Args:
            g2: Coupling strength
            epsilon_b: Drive strength
            kappa_b: Dissipation rate
            sim_time: Total simulation time
            dim_a: Hilbert space dimension for the memory mode
            dim_b: Hilbert space dimension for the buffer mode
            dt: Time step for simulation
            
        Returns:
            tuple: (times, states_a) where times is an array of time points and states_a are the reduced states of mode a
        """
        # Create a bipartite system with two modes
        system = dq.CompositeSystem([dim_a, dim_b])
        
        # Get operators for both subsystems
        a = system.operator('a', 0)  # Annihilation operator for the memory mode
        b = system.operator('a', 1)  # Annihilation operator for the buffer mode
        
        # Define the Hamiltonian
        H = g2 * a**2 * b.dag() + g2.conjugate() * a.dag()**2 * b + epsilon_b * b.dag() + epsilon_b.conjugate() * b
        
        # Define the collapse operators (dissipation)
        c_ops = [np.sqrt(kappa_b) * b]
        
        # Initial state: vacuum in both modes
        initial_state = dq.State(system.ground_state())
        
        # Create the solver
        solver = dq.Solver(
            system=system,
            hamiltonian=H,
            dissipators=c_ops,
            stepper='adaptive',
            dt=dt,
        )
        
        # Run the simulation
        result = solver.run(
            initial_state=initial_state,
            times=np.arange(0, sim_time, dt),
            store_states=True,
        )
        
        # Extract the reduced states of the memory mode
        states_a = []
        for state in result.states:
            # Trace out the buffer mode to get the memory mode state
            rho_a = state.partial_trace([1])
            states_a.append(rho_a)
        
        return result.times, states_a

    def create_wigner_animation(times, states, grid_size=51, xmax=5.0, ymax=5.0, filename='cat_evolution.gif'):
        """Create an animation of Wigner function evolution.
        
        Args:
            times: Array of time points
            states: List of quantum states
            grid_size: Number of points in each dimension
            xmax, ymax: Grid limits
            filename: Output filename
        """
        # Create a figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Compute the first Wigner function to set up the plot
        X, Y, W = compute_wigner_function(states[0], grid_size, xmax, ymax)
        
        # Find global min/max for consistent colormap
        all_W_values = []
        for state in states[::10]:  # Sample every 10th state to save computation
            _, _, W_sample = compute_wigner_function(state, grid_size, xmax, ymax)
            all_W_values.append(W_sample)
        
        vmin = min(np.min(W) for W in all_W_values)
        vmax = max(np.max(W) for W in all_W_values)
        
        # Initial plot
        im = ax.pcolormesh(X, Y, W, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        contour = ax.contour(X, Y, W, colors='black', alpha=0.5)
        
        plt.colorbar(im, ax=ax, label='W(x,p)')
        ax.set_title(f'Wigner Function, t = {times[0]:.2f}')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Momentum (p)')
        ax.axis('square')
        
        # Update function for animation
        def update(frame):
            ax.clear()
            
            # Compute Wigner function for this frame
            X, Y, W = compute_wigner_function(states[frame], grid_size, xmax, ymax)
            
            # Update plot
            im = ax.pcolormesh(X, Y, W, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
            contour = ax.contour(X, Y, W, colors='black', alpha=0.5)
            
            ax.set_title(f'Wigner Function, t = {times[frame]:.2f}')
            ax.set_xlabel('Position (x)')
            ax.set_ylabel('Momentum (p)')
            ax.axis('square')
            
            return im, contour
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=range(0, len(times), max(1, len(times)//100)), 
                                     blit=False, repeat=True)
        
        # Save as GIF
        ani.save(filename, writer='pillow', fps=10)
        
        plt.close(fig)
        print(f"Animation saved to {filename}")

    def generate_all_states():
        """Generate and save Wigner functions for all required states."""
        output_dir = "generated_states"
        os.makedirs(output_dir, exist_ok=True)
        
        # Parameters
        dim = 30
        grid_size = 101
        xmax = ymax = 5.0
        
        # 1. Fock states
        for n in range(4):  # |0⟩, |1⟩, |2⟩, |3⟩
            state = create_fock_state(n, dim)
            X, Y, W = compute_wigner_function(state, grid_size, xmax, ymax)
            
            # Save Wigner data
            filename = f"{output_dir}/fock_state_{n}.pickle"
            save_wigner_data(X, Y, W, filename)
            
            # Plot and save
            plt.figure(figsize=(8, 6))
            plt.pcolormesh(X, Y, W, cmap='RdBu_r', shading='auto')
            plt.colorbar(label='W(x,p)')
            plt.contour(X, Y, W, colors='black', alpha=0.5)
            plt.title(f'Wigner Function for Fock State |{n}⟩')
            plt.xlabel('Position (x)')
            plt.ylabel('Momentum (p)')
            plt.axis('equal')
            plt.savefig(f"{output_dir}/fock_state_{n}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Coherent states
        alphas = [1.0, 2.0, 3.0]
        for alpha in alphas:
            state = create_coherent_state(alpha, dim)
            X, Y, W = compute_wigner_function(state, grid_size, xmax, ymax)
            
            # Save Wigner data
            filename = f"{output_dir}/coherent_state_alpha_{alpha}.pickle"
            save_wigner_data(X, Y, W, filename)
            
            # Plot and save
            plt.figure(figsize=(8, 6))
            plt.pcolormesh(X, Y, W, cmap='RdBu_r', shading='auto')
            plt.colorbar(label='W(x,p)')
            plt.contour(X, Y, W, colors='black', alpha=0.5)
            plt.title(f'Wigner Function for Coherent State |α={alpha}⟩')
            plt.xlabel('Position (x)')
            plt.ylabel('Momentum (p)')
            plt.axis('equal')
            plt.savefig(f"{output_dir}/coherent_state_alpha_{alpha}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Cat states (2-component)
        alphas = [2.0, 3.0]
        cat_types = ['even', 'odd']
        for alpha in alphas:
            for cat_type in cat_types:
                state = create_cat_state(alpha, cat_type, dim)
                X, Y, W = compute_wigner_function(state, grid_size, xmax, ymax)
                
                # Save Wigner data
                filename = f"{output_dir}/cat_state_{cat_type}_alpha_{alpha}.pickle"
                save_wigner_data(X, Y, W, filename)
                
                # Plot and save
                plt.figure(figsize=(8, 6))
                plt.pcolormesh(X, Y, W, cmap='RdBu_r', shading='auto')
                plt.colorbar(label='W(x,p)')
                plt.contour(X, Y, W, colors='black', alpha=0.5)
                plt.title(f'Wigner Function for {cat_type.capitalize()} Cat State, α={alpha}')
                plt.xlabel('Position (x)')
                plt.ylabel('Momentum (p)')
                plt.axis('equal')
                plt.savefig(f"{output_dir}/cat_state_{cat_type}_alpha_{alpha}.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # 4. Cat states (3-component)
        alpha = 2.0
        state = create_three_component_cat(alpha, dim=dim)
        X, Y, W = compute_wigner_function(state, grid_size, xmax, ymax)
        
        # Save Wigner data
        filename = f"{output_dir}/three_component_cat_alpha_{alpha}.pickle"
        save_wigner_data(X, Y, W, filename)
        
        # Plot and save
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(X, Y, W, cmap='RdBu_r', shading='auto')
        plt.colorbar(label='W(x,p)')
        plt.contour(X, Y, W, colors='black', alpha=0.5)
        plt.title(f'Wigner Function for 3-Component Cat State, α={alpha}')
        plt.xlabel('Position (x)')
        plt.ylabel('Momentum (p)')
        plt.axis('equal')
        plt.savefig(f"{output_dir}/three_component_cat_alpha_{alpha}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Dissipative cat state simulation
        print("Simulating dissipative cat state evolution...")
        times, states = simulate_two_photon_exchange()
        
        # Create animation
        create_wigner_animation(times, states, filename=f"{output_dir}/dissipative_cat_evolution.gif")
        
        # Save states at specific time points
        save_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
        for i, idx in enumerate(save_indices):
            t = times[idx]
            state = states[idx]
            X, Y, W = compute_wigner_function(state, grid_size, xmax, ymax)
            
            # Save Wigner data
            filename = f"{output_dir}/dissipative_cat_t_{t:.2f}.pickle"
            save_wigner_data(X, Y, W, filename)
            
            # Plot and save
            plt.figure(figsize=(8, 6))
            plt.pcolormesh(X, Y, W, cmap='RdBu_r', shading='auto')
            plt.colorbar(label='W(x,p)')
            plt.contour(X, Y, W, colors='black', alpha=0.5)
            plt.title(f'Wigner Function for Dissipative Cat State, t={t:.2f}')
            plt.xlabel('Position (x)')
            plt.ylabel('Momentum (p)')
            plt.axis('equal')
            plt.savefig(f"{output_dir}/dissipative_cat_t_{t:.2f}.png", dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    # Only run if dynamiqs and JAX are installed
    if DEPENDENCIES_INSTALLED:
        # Generate all required states
        generate_all_states()
    else:
        # Already printed error message above
        pass 