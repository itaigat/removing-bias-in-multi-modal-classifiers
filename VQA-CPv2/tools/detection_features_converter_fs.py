"""
Reads in a tsv file with pre-trained bottom up attention features and
stores them in per-image files.
"""
from __future__ import print_function

import os
import sys
from os.path import join, exists

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import numpy as np


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
infile = 'data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
h_trainval_file = 'data/trainval36.hdf5'
feature_length = 2048
num_fixed_boxes = 36


if __name__ == '__main__':
    output_dir = 'data/trainval_features/'
    os.mkdir(output_dir)

    n_images = 123287

    pbar = tqdm(total=n_images, ncols=100, desc="converting-features")
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            pbar.update(1)
            num_boxes = int(item['num_boxes'])
            image_id = int(item['image_id'])
            out_file = join(output_dir, str(image_id) + ".bin")
            if exists(out_file):
              raise RuntimeError()
            arr = np.frombuffer(
              base64.decodestring(item['features']),
              dtype=np.float32).reshape((num_boxes, -1))
            arr.tofile(out_file)

    print("done!")
