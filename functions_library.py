import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# function 1- read pickle file (do not use on its own)
def read_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# function 2 - load data from pickle file (THIS is the one to use- returns Z, the 2D array of Wigner function)
def load_data_from_pickle(file_path):
    if os.path.exists(file_path):
        data = read_pickle(file_path)
        if data is not None and isinstance(data, (list, tuple)) and len(data) > 2:
            Z = data[2]
            print(f"Z: {Z}")
            return Z
        else:
            print("Data does not have a valid item 2.")
            return None
    else:
        print(f"File does not exist: {file_path}")
        return None
    
# function 3 - describe data (do not use on its own)
def describe_data(name, data):
    print(f"\n{name} — type: {type(data)}")

    if isinstance(data, (list, tuple)):
        print(f"  ➤ container with {len(data)} elements")
        for i, item in enumerate(data):
            print(f"    - Item {i}: type={type(item)}, shape={np.shape(item)}")
    elif isinstance(data, dict):
        print(f"  ➤ dictionary with {len(data)} keys")
        for key, value in data.items():
            print(f"    - Key: {key}, shape: {np.shape(value)}, type: {type(value)}")
        if "x" in data and "y" in data:
            plt.plot(data["x"], data["y"])
            plt.show()
    elif hasattr(data, "shape"):
        print(f"  ➤ array-like: shape={data.shape}, dtype={data.dtype}")
    else:
        print(f"  ➤ unknown structure")

# function 4 - visualize data (THIS is the one to use for plotting)
def visualize_Z_if_possible(data): # changed from item2 to Z to standardize
    if isinstance(data, (list, tuple)) and len(data) > 2:
        Z = data[2]
        if isinstance(Z, np.ndarray) and Z.ndim == 2:
            plt.figure(figsize=(6, 5))
            plt.contourf(Z, 100, cmap='RdBu_r')
            plt.colorbar()
            plt.title("Visualization of item 2")
            plt.xlabel("Axis 1")
            plt.ylabel("Axis 0")
            plt.tight_layout()
            plt.show()
        else:
            print("Z is not a 2D array, cannot visualize as contour.")
    else:
        print("Data does not have Z.")

# function 5 - add gaussian elimination
def add_gaussian_elimination(Z, sigma=0.1):
    noise = np.random.normal(0, sigma, Z.shape)
    w_noisy = Z + noise
    return w_noisy
