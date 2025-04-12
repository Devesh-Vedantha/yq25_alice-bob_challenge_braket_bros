import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import cm

def load_pickle_file(file_path):
    """Load data from a pickle file, handling different formats."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Try to extract X, Y, Z based on the data structure
        if isinstance(data, (tuple, list)) and len(data) >= 3:
            # Assume first three items are X, Y, Z
            X, Y, Z = data[0], data[1], data[2]
            return X, Y, Z
        elif isinstance(data, dict) and all(k in data for k in ['X', 'Y', 'Z']):
            # Dictionary with X, Y, Z keys
            return data['X'], data['Y'], data['Z']
        else:
            print(f"Unexpected data format: {type(data)}")
            return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def plot_wigner(X, Y, Z, title="Wigner Function", save_path=None):
    """Plot a Wigner function."""
    plt.figure(figsize=(10, 8))
    
    # Create a meshgrid if X and Y are 1D arrays
    if len(X.shape) == 1 and len(Y.shape) == 1:
        X_mesh, Y_mesh = np.meshgrid(X, Y)
    else:
        X_mesh, Y_mesh = X, Y
    
    # Plot as a 2D color map
    plt.pcolormesh(X_mesh, Y_mesh, Z, cmap='RdBu_r', shading='auto')
    plt.colorbar(label='W(x,p)')
    
    # Add contour lines for clarity
    try:
        contour = plt.contour(X_mesh, Y_mesh, Z, colors='black', alpha=0.5)
        plt.clabel(contour, inline=True, fontsize=8)
    except:
        print("Warning: Could not create contour lines")
    
    plt.title(title)
    plt.xlabel('Position (x)')
    plt.ylabel('Momentum (p)')
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def list_pickle_files(directory):
    """List all pickle files in a directory."""
    files = []
    if os.path.exists(directory):
        for file in sorted(os.listdir(directory)):
            if file.endswith('.pickle'):
                files.append(os.path.join(directory, file))
    return files

def print_file_list(files):
    """Print a numbered list of files."""
    for i, file in enumerate(files):
        print(f"[{i+1}] {os.path.basename(file)}")

def main():
    print("Wigner Function Visualizer\n")
    
    # Find all pickle files
    exp_dir = "data/experimental"
    syn_dir = "data/synthetic"
    
    exp_files = list_pickle_files(exp_dir)
    syn_files = list_pickle_files(syn_dir)
    
    all_files = exp_files + syn_files
    
    # If no files found
    if not all_files:
        print("No pickle files found in data directories.")
        return
    
    # Show all available files
    print("Available files:")
    print("\nExperimental files:")
    print_file_list(exp_files)
    print("\nSynthetic files:")
    print_file_list(syn_files)
    
    # Check if a specific file was provided as argument
    if len(sys.argv) > 1:
        try:
            # Check if it's a number (index)
            idx = int(sys.argv[1]) - 1
            if 0 <= idx < len(all_files):
                file_path = all_files[idx]
            else:
                file_path = sys.argv[1]  # Assume it's a path
        except ValueError:
            file_path = sys.argv[1]  # Not a number, assume it's a path
        
        if os.path.exists(file_path):
            print(f"\nVisualizing: {file_path}")
            data = load_pickle_file(file_path)
            if data:
                X, Y, Z = data
                plot_wigner(X, Y, Z, title=f"Wigner Function: {os.path.basename(file_path)}")
        else:
            print(f"File not found: {file_path}")
    else:
        # If no argument provided, try the first experimental file
        if exp_files:
            file_path = exp_files[0]
            print(f"\nVisualizing first experimental file: {file_path}")
            data = load_pickle_file(file_path)
            if data:
                X, Y, Z = data
                plot_wigner(X, Y, Z, title=f"Wigner Function: {os.path.basename(file_path)}")

if __name__ == "__main__":
    main() 