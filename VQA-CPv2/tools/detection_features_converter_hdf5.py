"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.
"""
from __future__ import print_function

import os
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import cPickle
import numpy as np
import utils


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
infile = 'data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
h_trainval_file = 'data/trainval36.hdf5'
trainval_indices_file = 'data/trainval36_imgid2idx.pkl'
feature_length = 2048
num_fixed_boxes = 36


if __name__ == '__main__':
    h_trainval_file = h5py.File(h_trainval_file, "w")

    trainval_indices = {}

    n_images = 123287
    train_img_features = h_trainval_file.create_dataset(
        'image_features', (n_images, num_fixed_boxes, feature_length), 'f')

    counter = 0

    pbar = tqdm(total=n_images, ncols=100, desc="converting-features")
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            pbar.update(1)
            num_boxes = int(item['num_boxes'])
            image_id = int(item['image_id'])
            trainval_indices[image_id] = counter
            train_img_features[counter, :, :] = np.frombuffer(
                base64.decodestring(item['features']),
                dtype=np.float32).reshape((num_boxes, -1))
            counter += 1

    cPickle.dump(trainval_indices, open(trainval_indices_file, 'wb'))
    h_trainval_file.close()
    print("done!")
