import os
import h5py
import argparse
from scipy import misc

def open(path):
    with h5py.File(path, 'r') as f:
        output = f['data'][:]
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str)
    args = parser.parse_args()

    data = open(args.input_path)
    for n,a in enumerate(data):
        print a.shape
        img = a[:,:,:3]
        s = a[:,:,3]
        print img.shape
        misc.imsave("{}_img.jpg".format(n), img)
        misc.imsave("{}_s.jpg".format(n), s)