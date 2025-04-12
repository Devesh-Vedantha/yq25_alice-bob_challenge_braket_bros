# necessary imports - can be removed from this file
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# read given input .pickle file 
# function 1
def read_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# function 1.5
def load_data_from_pickle(file_path): # MUST BE USED IMMEDIATELY AFTER READ_PICKLE, the only thing we care about is Z
    if os.path.exists(file_path):
        data = read_pickle(file_path)
        Z = data[2]
        print(f"Z: {Z}")
        return Z

file_path = "ADD ABSOLUTE_FILE_PATH HERE" # change based on desired file
Z = load_data_from_pickle(file_path)
    
# add gaussian noise
def add_gaussian_noise(Z, sigma=0.1):
    # note Z = data[2]
    noise = np.random.normal(0, sigma, Z.shape)
    wigner_noisy = Z + noise
    # print(wigner_noisy) - used for testing
    return wigner_noisy
# example usage
wigner_noisy = add_gaussian_noise(Z, sigma=0.1)

# function 2
def visualize_item2_if_possible(data):
    if isinstance(data, (list, tuple)) and len(data) > 2:
        item2 = data[2]
        if isinstance(item2, np.ndarray) and item2.ndim == 2:
            plt.figure(figsize=(6, 5))
            plt.contourf(item2, 100, cmap='RdBu_r')
            plt.colorbar()
            plt.title("Visualization of item 2")
            plt.xlabel("Axis 1")
            plt.ylabel("Axis 0")
            plt.tight_layout()
            plt.show()
        else:
            print("item 2 is not a 2D array, cannot visualize as contour.")
    else:
        print("Data does not have an item 2.")


base_path = "" # change to own path

for subfolder in ["experimental", "synthetic"]:
    dir_path = os.path.join(base_path, subfolder)
    if not os.path.isdir(dir_path):
        continue

    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".pickle"):
            file_path = os.path.join(dir_path, filename)
            data = read_pickle(file_path)
            if data is not None:
                describe_data(filename, data)
                visualize_item2_if_possible(data)

# function 2.5 - necessary function for visualize
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
        plt.plot(data["x"], data["y"])
        plt.show()
    elif hasattr(data, "shape"):
        print(f"  ➤ array-like: shape={data.shape}, dtype={data.dtype}")
    else:
        print(f"  ➤ unknown structure")
