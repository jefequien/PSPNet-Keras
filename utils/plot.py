import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ggplot import *
import pandas as pd

from evaluator import Evaluator
from datasource import DataSource
from recorder import Recorder

import utils

def get_scores(im_list, recorder):
    scores = []
    gt_scores = []
    for im in im_list:
        scores.append(recorder.record[im][0])
        gt_scores.append(recorder.record[im][1])
    return np.array(scores), np.array(gt_scores)

def get_ious(im_list, evaluator, category):
    ious = []
    for im in im_list:
        results = evaluator.get_results(im)
        iou = results["iou"][category-1]
        ious.append(iou)
    return np.array(ious)

def plot(recorder, evaluator, category, project):
    im_list = recorder.record.keys()
    scores, gt_scores = get_scores(im_list, recorder)
    ious = get_ious(im_list, evaluator, category)

    nonzero = ious > 0
    num_zero = np.count_nonzero(~nonzero)

    print "Total:", len(scores)
    print "Zero IOUs: ", num_zero

    title = "Discriminator ({}, {})".format(project, category)
    save_file = "{}_{}_scatter.png".format(project, category)
    df = pd.DataFrame({'IOUs':ious, 'Predicted Scores':scores})
    scatter = ggplot(df,aes(x='IOUs',y='Predicted Scores')) + geom_point(alpha = 0.3) + ggtitle(title)
    scatter.save(save_file)

    title = "Discriminator GT ({}, {})".format(project, category)
    save_file = "{}_{}_gt_scatter.png".format(project, category)
    df = pd.DataFrame({'IOUs':ious, 'Predicted Scores':gt_scores})
    scatter = ggplot(df,aes(x='IOUs',y='Predicted Scores')) + geom_point(alpha = 0.3) + ggtitle(title)
    scatter.save(save_file)
    '''
    plt.scatter(ious,scores,s=1)
    plt.xlabel("IOU")
    plt.ylabel("Predicted Score")
    plt.title("Discriminator ({}, {})".format(project, category))
    plt.savefig("{}_{}_scatter.png".format(project, category))
    plt.close()

    plt.scatter(ious,gt_scores,s=1)
    plt.xlabel("IOU")
    plt.ylabel("Predicted Score")
    plt.title("Discriminator GT ({}, {})".format(project, category))
    plt.savefig("{}_{}_gt_scatter.png".format(project, category))
    plt.close()
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of run")
    parser.add_argument("-p", "--project", required=True, help="Project name")
    parser.add_argument("-c", "--category", type=int, required=True, help="Category")
    parser.add_argument('-i', '--im_list', type=str, help="Specific image list")
    args = parser.parse_args()

    config = utils.get_config(args.project)
    datasource = DataSource(config)

    evaluator = Evaluator(args.name, args.project, datasource)
    recorder = Recorder(args.im_list)

    plot(recorder, evaluator, args.category, args.project)
