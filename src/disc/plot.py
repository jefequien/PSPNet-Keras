import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ggplot import *
import pandas as pd

from utils.evaluator import Evaluator
from utils.data import DataSource
from utils.recorder import Recorder

import utils

def scatterplot(recorder_fn, evaluated, category, title=None):
    recorder = Recorder(recorder_fn)
    im_list = recorder.record.keys()
    scores, gt_scores = get_scores(im_list, recorder)
    ious = get_ious(im_list, evaluated, category)

    if title is None:
        title = "Discriminator for category {}".format(category)

    nonzero = ious > 0
    num_zero = np.count_nonzero(~nonzero)
    print "Total:", len(scores)
    print "Zero IOUs: ", num_zero

    scatter_fn = recorder_fn.replace(".txt", "_scatter.png")
    df = pd.DataFrame({'IOUs':ious, 'Predicted Scores':scores})
    scatter = ggplot(df,aes(x='IOUs',y='Predicted Scores')) + geom_point(alpha = 0.3) + ggtitle(title)
    scatter.save(scatter_fn)
    print "Plot saved: {}".format(scatter_fn)

def get_scores(im_list, recorder):
    scores = []
    gt_scores = []
    for im in im_list:
        scores.append(recorder.record[im][0])
        gt_scores.append(recorder.record[im][1])
    return np.array(scores), np.array(gt_scores)

def get_ious(im_list, evaluated, category):
    ious = []
    for im in im_list:
        results = evaluated.get_results(im)
        iou = results["iou"][category-1]
        ious.append(iou)
    return np.array(ious)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of run")
    parser.add_argument("-p", "--project", required=True, help="Project name")
    parser.add_argument("-c", "--category", type=int, required=True, help="Category")
    parser.add_argument('-i', '--im_list', type=str, help="Specific image list")
    args = parser.parse_args()

    config = utils.get_config(args.project)
    datasource = DataSource(config)


    plot(recorder, evaluator, args.category, args.project)
