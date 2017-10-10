import os
import argparse
import numpy as np
import h5py
import time

import utils
import file_utils

DIR = "predictions/results/"
if not os.path.exists(DIR):
    os.makedirs(DIR)

NUMCLASS = 150

class Evaluator:

    def __init__(self, name, project, config):
        fname = "{}-{}.h5".format(name, project)
        self.fname = os.path.join(DIR, fname)

        self.config = config
        self.im_list = utils.open_im_list(self.config["im_list"])
        self.n = len(self.im_list)

        self.save_freq = 10
        
        # Metrics
        # im x NUMCLASS
        self.load()

        assert self.precision.shape[0] == self.n
        assert self.recall.shape[0] == self.n
        assert self.iou.shape[0] == self.n

    def get_result(self, im):
        idx = self.im_list.index(im)
        result = {}
        result["precision"] = self.precision[idx]
        result["recall"] = self.recall[idx]
        result["iou"] = self.iou[idx]
        return result

    def evaluate(self):
        for i in xrange(self.n):
            im = self.im_list[i]
            if self.is_evaluated(i):
                print im, "Done."
            else:
                print im
                im = self.im_list[i]
                gt, _ = file_utils.get_ground_truth(im, self.config, one_hot=True)
                ap, _ = file_utils.get_all_prob(im, self.config)
                results = evaluate_prediction(gt, ap)

                self.precision[i] = results[0]
                self.recall[i] = results[1]
                self.iou[i] = results[2]

                print "Precision: {}".format(np.nanmean(self.precision[i]))
                print "Recall: {}".format(np.nanmean(self.recall[i]))
                print "IOU: {}".format(np.nanmean(self.iou[i]))

                if i % self.save_freq == 0:
                    self.save()

    def is_evaluated(self, i):
        precision = self.precision[i]
        recall = self.recall[i]
        iou = self.iou[i]
        return np.any(~np.isnan(precision)) or np.any(~np.isnan(recall)) or np.any(~np.isnan(iou))

    def get_indicies(self, im_list):
        return [self.im_list.index(im) for im in im_list]

    def get_image_metrics(self, im, category):
        c = category - 1
        i = self.im_list.index(im)
        precision = self.precision[i,c]
        recall = self.recall[i,c]
        iou = self.iou[i,c]
        return [precision, recall, iou]

    def save(self):
        with h5py.File(self.fname, 'w') as f:
            f.create_dataset('precision', data=self.precision)
            f.create_dataset('recall', data=self.recall)
            f.create_dataset('iou', data=self.iou)

    def load(self):
        if os.path.exists(self.fname):
            print "Loading..."
            with h5py.File(self.fname, 'r') as f:
                self.precision = f['precision'][:]
                self.recall = f['recall'][:]
                self.iou = f['iou'][:]
        else:
            self.precision = np.zeros((self.n, NUMCLASS))
            self.recall = np.zeros((self.n, NUMCLASS))
            self.iou = np.zeros((self.n, NUMCLASS))
            self.precision[:] = np.nan
            self.recall[:] = np.nan
            self.iou[:] = np.nan


def evaluate_prediction(gt,pr):
    pr = pr > 0.5

    # precision, recall, iou
    results = np.zeros((3,150))
    for i in xrange(150):
        gt_s = gt[:,:,i]
        pr_s = pr[:,:,i]

        intersection = np.logical_and(gt_s, pr_s)
        union = np.logical_or(gt_s, pr_s)

        precision, recall, iou = np.nan, np.nan, np.nan
        if np.sum(pr_s) != 0:
            precision = 1.*np.sum(intersection)/np.sum(pr_s)
        if np.sum(gt_s) != 0:
            recall = 1.*np.sum(intersection)/np.sum(gt_s)
        if np.sum(union) != 0:
            iou = 1.0*np.sum(intersection)/np.sum(union)

        results[:,i] = [precision, recall, iou]
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", required=True, help="Project name")
    parser.add_argument("--prediction", required=True, help="Pspnet prediction")
    parser.add_argument("-n", "--name", required=True, help="Name of run")
    args = parser.parse_args()

    config = utils.get_config(args.project)
    config["pspnet_prediction"] = args.prediction

    evaluator = Evaluator(args.name, args.project, config)
    evaluator.evaluate()

