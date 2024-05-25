import numpy as np

def read_npy(file_path):
    data = np.load(file_path)
    return data