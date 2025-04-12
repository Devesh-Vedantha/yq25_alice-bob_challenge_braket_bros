# necessary imports - can be removed from this file
import pickle
import os
import numpy as np

# read given input .pickle file 
def read_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
# example usage
file_path = "ADD ABSOLUTE_FILE_PATH HERE" # change based on desired file

if os.path.exists(file_path):
    data = read_pickle(file_path)
    print(data)
    
# add gaussian noise
def add_gaussian_noise(Z, sigma=0.1):
    noise = np.random.normal(0, sigma, Z.shape)
    wigner_noisy = Z + noise
    # print(wigner_noisy) - used for testing
    return wigner_noisy
# example usage

