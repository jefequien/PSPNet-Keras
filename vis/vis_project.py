import argparse
import os
import random
import uuid
import time
import numpy as np
import pandas as pd

import utils
from utils.datasource import DataSource
from utils.evaluator import Evaluator
from vis_image import ImageVisualizer

TMP_DIR = "tmp/"
IMAGES_DIR = "tmp/images/"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

class ProjectVisualizer:

    def __init__(self, project, datasource, MAX=100, evaluator=None):
        self.project = project
        self.image_visualizer = ImageVisualizer(project, datasource)
        self.evaluator = evaluator
        self.MAX = MAX

        fname = "{}_{}.html".format(project, int(time.time()))
        self.output_path = os.path.join(TMP_DIR, fname)

        self.init_output_file(datasource)

    def init_output_file(self, datasource):
        head = str(datasource.config)
        body = ""
        html = "<html><head>" + head + "</head><body>" + body + "</body></html>"
        with open(self.output_path, 'w') as f:
            f.write(html)

        # Print link to output file
        root = "/data/vision/oliva/scenedataset/"
        abs_path = os.path.abspath(self.output_path)
        rel_path = os.path.relpath(abs_path, root)
        print "http://places.csail.mit.edu/{}".format(rel_path)

    def visualize_images(self, im_list, category=None):
        for n, line in enumerate(im_list[:self.MAX]):
            print n, line
            self.add_image_section(line, category=category)

    def add_image_section(self, line, category=None):
        im = line.split()[0]
        image_tags = []

        paths = self.image_visualizer.visualize(im)
        order = np.arange(0,150)
        if category is None:
            paths1 = self.image_visualizer.visualize_all_categories(im)
            paths.update(paths1)
            order = paths["order"]
            del paths["order"]
        else:
            paths2 = self.image_visualizer.visualize_category(im, category)
            paths.update(paths2)

        for key in paths:
            tag = self.get_image_tag(paths[key])
            image_tags.append(tag)

        # Results
        results = self.evaluator.get_results(im)

        # Build section
        title = "{} {}".format(self.project, line)
        img_section = ' '.join(image_tags)
        results_section = self.build_results_section(results, order)
        section = "<br><br>{}<br><br>{}<br>{}".format(title, img_section, results_section)

        # Append to body
        with open(self.output_path, 'r') as f:
            html = f.read()
        new_html = html.replace("</body>", "{}</body>".format(section))
        with open(self.output_path, 'w') as f:
            f.write(new_html)

    def build_results_section(self, results, order):
        keys = []
        values = []
        for key in results.keys():
            keys.append(key)
            values.append(results[key])
        values = np.stack(values)

        sorted_values = values[:,order]
        df = pd.DataFrame(sorted_values, index=keys, columns=order+1)
        html = df.to_html()
        return html

    def get_image_tag(self, path):
        if os.path.isabs(path):
            # Symlink into tmp image directory
            path = self.symlink(path)

        path = os.path.relpath(path, os.path.dirname(self.output_path))
        return "<img src=\"{}\" height=\"256px\">".format(path)

    def symlink(self, path):
        fn = "{}.jpg".format(uuid.uuid4().hex)
        dst = os.path.join(IMAGES_DIR, fn)
        os.symlink(path, dst)
        return dst

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True, help="Name of run")
    parser.add_argument('-p', '--project', type=str, required=True, help="Project name")
    parser.add_argument("--prediction", type=str, required=True, help="")
    parser.add_argument('-r', '--randomize', action='store_true', default=False, help="Randomize image list")
    parser.add_argument('-i', '--im_list', type=str, help="Specific image list")
    parser.add_argument('-N', '--number', type=int, default=10, help="Number of images")
    parser.add_argument('-s', '--start', type=int, default=0, help="Number of images")
    parser.add_argument('-c', '--category', type=int, help="Category")
    args = parser.parse_args()

    # Configuration
    config = utils.get_config(args.project)
    if args.prediction is not None:
        config["pspnet_prediction"] = args.prediction

    datasource = DataSource(config)
    evaluator = Evaluator(args.name, args.project, datasource) # Evaluation results
    vis = ProjectVisualizer(args.project, datasource, MAX=args.number, evaluator=evaluator)

    # Image List
    im_list = None
    if args.im_list:
        # Open specific image list
        im_list = utils.open_im_list(args.im_list)
    elif args.category:
        im_list = evaluator.get_im_list_by_category(args.category)
    else:
        # Open default image list
        im_list = utils.open_im_list(config["im_list"])

    if args.randomize:
        # Shuffle image list
        random.seed(3)
        random.shuffle(im_list)

    im_list = im_list[args.start:]
    vis.visualize_images(im_list, category=args.category)

