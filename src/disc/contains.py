import os
import argparse
import numpy as np
import h5py
import pandas as pd

import utils

def open_h5file(file_path):
    with h5py.File(file_path, 'r') as f:
            output = f['allprob'][:]
            if output.dtype == 'uint8':
                return output.astype('float32')/255
            else:
                return output.astype('float32')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--project', type=str, required=True, help="Project name")
    parser.add_argument("-o", '--output', type=str, required=True, help="Output file name")
    args = parser.parse_args()

    config = utils.get_config(args.project)
    im_list = utils.open_im_list(config["im_list"])
    categories = utils.categories
    #im_list = im_list[:100]

    root_allprob = os.path.join(config["pspnet_prediction"], 'all_prob')
    print root_allprob
    
    n = len(im_list)
    contains_matrix = np.zeros((n, 150), dtype=int)

    for i,im in enumerate(im_list):
        print im

        fn_allprob = os.path.join(root_allprob, im.replace('.jpg', '.h5'))
        all_prob = open_h5file(fn_allprob)
        contains = np.amax(all_prob > 0.5, axis=(1,2))
        contains = contains.astype(int)
        print contains
        contains_matrix[i,:] = contains

    # Write to output
    df = pd.DataFrame(contains_matrix, index=im_list, columns=categories)
    df.to_csv(args.output + ".csv")


