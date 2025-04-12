import pickle
import os
import sys

def inspect_pickle_file(file_path):
    """Try to load a pickle file and print basic information about its contents."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nInspecting file: {file_path}")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, tuple) or isinstance(data, list):
            print(f"Length: {len(data)}")
            for i, item in enumerate(data):
                print(f"  Item {i}: type={type(item)}")
                if hasattr(item, 'shape'):
                    print(f"    shape: {item.shape}, dtype: {item.dtype}")
                elif hasattr(item, '__len__'):
                    print(f"    length: {len(item)}")
        elif isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
        
        return True
    except Exception as e:
        print(f"Error loading file: {e}")
        return False

def list_files(directory):
    """List pickle files in the given directory."""
    files = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith('.pickle'):
                files.append(os.path.join(directory, file))
    return files

def main():
    print("Pickle Inspector\n")
    
    # List all available pickle files
    exp_dir = "data/experimental"
    syn_dir = "data/synthetic"
    
    exp_files = list_files(exp_dir)
    syn_files = list_files(syn_dir)
    
    print(f"Found {len(exp_files)} experimental files and {len(syn_files)} synthetic files.")
    
    # Try to inspect the first file from each directory
    if exp_files:
        print("\nInspecting first experimental file...")
        inspect_pickle_file(exp_files[0])
    
    if syn_files:
        print("\nInspecting first synthetic file...")
        inspect_pickle_file(syn_files[0])
    
    # Allow inspecting a specific file if provided as an argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            print(f"\nInspecting specified file: {file_path}")
            inspect_pickle_file(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main() 