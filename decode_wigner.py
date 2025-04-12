import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys

def load_wigner_data(file_path):
    """Load Wigner function data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        tuple: (X, Y, Z) where X, Y are coordinate ranges and Z is the Wigner function values
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different data formats
        if isinstance(data, tuple) and len(data) == 3:
            X, Y, Z = data
            return X, Y, Z
        elif isinstance(data, list) and len(data) == 3:
            # Convert list to tuple
            X, Y, Z = data
            print(f"Data was in list format, converted to tuple.")
            return X, Y, Z
        elif isinstance(data, dict) and all(k in data for k in ['X', 'Y', 'Z']):
            # If it's a dictionary with the right keys
            return data['X'], data['Y'], data['Z']
        else:
            print(f"Unexpected data format: {type(data)}")
            print(f"Data structure: {data if len(str(data)) < 100 else str(data)[:100] + '...'}")
            
            # Try to extract X, Y, Z by position if it's an iterable
            if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                try:
                    # Try to unpack as three items
                    items = list(data)
                    if len(items) >= 3:
                        print("Attempting to extract X, Y, Z by position...")
                        X, Y, Z = items[0], items[1], items[2]
                        return X, Y, Z
                except:
                    pass
            
            return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        
        # Get detailed exception information
        import traceback
        traceback.print_exc()
        
        return None

def plot_wigner(X, Y, Z, title="Wigner Function", save_path=None):
    """Plot the Wigner function.
    
    Args:
        X: X-coordinate range
        Y: Y-coordinate range
        Z: Wigner function values (2D matrix)
        title: Plot title
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=(10, 8))
    
    # Create a meshgrid if X and Y are 1D arrays
    if len(X.shape) == 1 and len(Y.shape) == 1:
        X_mesh, Y_mesh = np.meshgrid(X, Y)
    else:
        X_mesh, Y_mesh = X, Y
    
    # Plot the Wigner function as a 2D color map
    plt.pcolormesh(X_mesh, Y_mesh, Z, cmap='RdBu_r', shading='auto')
    plt.colorbar(label='W(x,p)')
    
    # Add contour lines for clarity
    contour = plt.contour(X_mesh, Y_mesh, Z, colors='black', alpha=0.5)
    plt.clabel(contour, inline=True, fontsize=8)
    
    plt.title(title)
    plt.xlabel('Position (x)')
    plt.ylabel('Momentum (p)')
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_wigner_data(file_path):
    """Load and analyze a Wigner function from a pickle file."""
    data = load_wigner_data(file_path)
    
    if data is None:
        return
    
    X, Y, Z = data
    
    # Print basic info about the data
    print(f"File: {os.path.basename(file_path)}")
    print(f"X shape: {X.shape}, type: {X.dtype}")
    print(f"Y shape: {Y.shape}, type: {Y.dtype}")
    print(f"Z shape: {Z.shape}, type: {Z.dtype}")
    
    # Get min/max values
    try:
        print(f"X range: {X.min()} to {X.max()}")
        print(f"Y range: {Y.min()} to {Y.max()}")
        print(f"Z min: {Z.min()}, max: {Z.max()}")
    except Exception as e:
        print(f"Error calculating min/max: {e}")
    
    # Calculate some properties of the Wigner function
    try:
        dx = (X.max() - X.min()) / (len(X) - 1) if len(X.shape) == 1 else (X[0, -1] - X[0, 0]) / (X.shape[1] - 1)
        dy = (Y.max() - Y.min()) / (len(Y) - 1) if len(Y.shape) == 1 else (Y[-1, 0] - Y[0, 0]) / (Y.shape[0] - 1)
        
        # Integration of W(x,p) over phase space should be 1 for proper normalization
        integral = np.sum(Z) * dx * dy
        print(f"Integral of W(x,p): {integral:.6f} (should be close to 1 for proper normalization)")
    except Exception as e:
        print(f"Error calculating properties: {e}")
    
    # Plot the Wigner function
    try:
        plot_wigner(X, Y, Z, title=f"Wigner Function: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error plotting: {e}")
    
    return X, Y, Z

def list_available_files():
    """List all available Wigner function files in the data directory."""
    print("Available Wigner function files:")
    
    # Check experimental data
    print("\nExperimental data:")
    exp_dir = "data/experimental"
    if os.path.exists(exp_dir):
        for file in sorted(os.listdir(exp_dir)):
            if file.endswith(".pickle"):
                print(f"  - {exp_dir}/{file}")
    else:
        print("  Experimental data directory not found.")
    
    # Check synthetic data
    print("\nSynthetic data:")
    syn_dir = "data/synthetic"
    if os.path.exists(syn_dir):
        for file in sorted(os.listdir(syn_dir)):
            if file.endswith(".pickle"):
                print(f"  - {syn_dir}/{file}")
    else:
        print("  Synthetic data directory not found.")

def inspect_pickle_file(file_path):
    """Inspect the structure of a pickle file without fully loading it."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nInspecting file: {file_path}")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, tuple):
            print(f"Tuple length: {len(data)}")
            for i, item in enumerate(data):
                print(f"  Item {i}: type={type(item)}, ", end="")
                if hasattr(item, 'shape'):
                    print(f"shape={item.shape}, dtype={item.dtype}")
                else:
                    print(f"value={str(item)[:50] + '...' if len(str(item)) > 50 else item}")
        elif isinstance(data, list):
            print(f"List length: {len(data)}")
            for i, item in enumerate(data):
                print(f"  Item {i}: type={type(item)}, ", end="")
                if hasattr(item, 'shape'):
                    print(f"shape={item.shape}, dtype={item.dtype}")
                else:
                    print(f"value={str(item)[:50] + '...' if len(str(item)) > 50 else item}")
        elif isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            for k, v in data.items():
                print(f"  Key '{k}': type={type(v)}, ", end="")
                if hasattr(v, 'shape'):
                    print(f"shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"value={str(v)[:50] + '...' if len(str(v)) > 50 else v}")
        else:
            print(f"Data content: {str(data)[:100] + '...' if len(str(data)) > 100 else data}")
        
        return data
    except Exception as e:
        print(f"Error inspecting file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # List available files
    list_available_files()
    
    # Check if a specific file was provided as argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            print(f"\nInspecting specific file: {file_path}")
            data = inspect_pickle_file(file_path)
            if data is not None:
                analyze_wigner_data(file_path)
        else:
            print(f"File not found: {file_path}")
    else:
        # Try experimental files
        experimental_files = [
            "data/experimental/wigner_fock_zero.pickle", 
            "data/experimental/wigner_fock_one.pickle",
            "data/experimental/wigner_cat_plus.pickle"
        ]
        
        for exp_file in experimental_files:
            if os.path.exists(exp_file):
                print(f"\nInspecting experimental file: {exp_file}")
                data = inspect_pickle_file(exp_file)
                if data is not None:
                    analyze_wigner_data(exp_file)
                break  # Just analyze one file for demonstration
        
        # Try synthetic files
        synthetic_files = [
            "data/synthetic/quantum_state_0.pickle",
            "data/synthetic/noisy_wigner_0.pickle"
        ]
        
        for syn_file in synthetic_files:
            if os.path.exists(syn_file):
                print(f"\nInspecting synthetic file: {syn_file}")
                data = inspect_pickle_file(syn_file)
                if data is not None:
                    analyze_wigner_data(syn_file)
                break  # Just analyze one file for demonstration 