import pickle
import os

def read_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

file_path = "/common/home/sps247/code/yq25_alice-bob_challenge_braket_bros/data/experimental/wigner_cat_minus.pickle" # change based on desired file

if os.path.exists(file_path):
    data = read_pickle(file_path)
    print(data)
