# necessary imports - can be removed from this file
import pickle
import os
import numpy as np

# read given input .pickle file 
# method 1
def read_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# method 1.5
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
