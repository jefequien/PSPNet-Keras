import os
import argparse
import numpy as np

from evaluator import Evaluator
from datasource import DataSource

import utils

def plot(im_list, evaluator, category):
    for line in im_list:
        im = line.split()[0]
        pr_iou = line.split()[1]

        results = evaluator.get_results(im)
        iou = results["iou"][category-1]

        print im, pr_iou, iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of run")
    parser.add_argument("-p", "--project", required=True, help="Project name")
    parser.add_argument("-c", "--category", required=True, help="Category")
    parser.add_argument('-i', '--im_list', type=str, help="Specific image list")
    args = parser.parse_args()

    config = utils.get_config(args.project)
    config["pspnet_prediction"] = args.prediction
    datasource = DataSource(config)

    evaluator = Evaluator(args.name, args.project, datasource)

    im_list = utils.open_im_list(args.im_list)

    plot(im_list, evaluator, args.category)