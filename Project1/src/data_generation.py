
import numpy as np

"""
Generate random data and save in data.npy
"""

def generateData(length):
    """
    param length: length of output array
    return: numpy array of random numbers from a uniform distribution over [0, 1)
    """
    x = np.random.rand(length, 1)
    y = np.random.rand(length, 1)
    return np.c_[x, y]

if __name__ == '__main__':
    # Generate data
    data = generateData(1000)
    np.save('data.npy', data)
