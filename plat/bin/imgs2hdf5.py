#!/usr/bin/env python

"""Convert a directory of images to an hdf5 dataset."""
import argparse
import glob
import random
import os

import numpy as np
import sys
from scipy.misc import imread

def loadImageOrNone(f):
    try:
        im = imread(f, mode='RGB')
        return im
    except:
        print("Could not read {}, continuing".format(f))
        return None

def split_to_numpy_features(filelist):
    images_all = map(loadImageOrNone, filelist)
    keepers = [x for x in images_all if x is not None]
    images_array = np.asarray(keepers)
    features_array = np.asarray([[f[:,:,0], f[:,:,1], f[:,:,2]] for f in images_array])
    return features_array

from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path
import fuel
import h5py
from fuel.converters.base import fill_hdf5_file, check_exists

def save_fuel_dataset(output_filename, train_features, valid_features, test_features):
    datasource_dir = fuel.config.data_path[0]

    """Converts the dataset to HDF5.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    output_path = os.path.join(datasource_dir, output_filename)
    h5file = h5py.File(output_path, mode='w')

    data = (('train', 'features', train_features),
            ('valid', 'features', valid_features),
            ('test', 'features', test_features))
    fill_hdf5_file(h5file, data)
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'channel'
    h5file['features'].dims[2].label = 'height'
    h5file['features'].dims[3].label = 'width'

    h5file.flush()
    h5file.close()

    return (output_path,)

def create_dataset(dataset, train_files, valid_files, test_files):
    train_features = split_to_numpy_features(train_files)
    valid_features = split_to_numpy_features(valid_files)
    test_features = split_to_numpy_features(test_files)
    print("Shapes: {}, {}, {}".format(train_features.shape, valid_features.shape, test_features.shape))
    filename = "{}.hdf5".format(dataset)
    save_fuel_dataset(filename, train_features, valid_features, test_features)

def main(cliargs):
    parser = argparse.ArgumentParser(description="Convert imgs to hdf5 dataset")
    parser.add_argument(dest='path', type=str,
                        default=None, help="Glob for input images")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Output dataset name.")
    parser.add_argument('--color-convert', dest='color_convert',
                        default=False, action='store_true',
                        help="Convert source dataset to color from grayscale.")
    parser.add_argument('--unshuffle', dest='unshuffle',
                        default=False, action='store_true',
                        help="Disable shuffling.")
    parser.add_argument("--cap", type=int,
                        default=None, help="Cap dataset size to integer")
    parser.add_argument("--seed", type=int,
                        default=None, help="Optional random seed")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument('--percent-valid', dest='percent_valid', default=5, type=float,
                        help="Percent of datset to be validation split")
    parser.add_argument('--percent-test', dest='percent_test', default=5, type=float,
                        help="Percent of datset to be test split")
    args = parser.parse_args(cliargs)

    if args.path is None:
        print("Input path needed")
        sys.exit(0)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    filelist = glob.glob(args.path)
    num_files = len(filelist)
    print("Found {} input files".format(num_files))
    if not args.unshuffle:
        random.shuffle(filelist)

    if args.cap is not None:
        dataset_size = args.cap
    else:
        dataset_size = num_files

    # allocate all files to a split
    num_valid_max = int(args.percent_valid * num_files / 100.0)
    num_test_max = int(args.percent_test * num_files / 100.0)
    num_train_max = num_files - num_valid_max - num_test_max

    # indexes to split points
    train_start = 0
    valid_start = num_train_max
    test_start = num_train_max + num_valid_max

    # determine how many files will in in stored dataset
    num_valid = int(args.percent_valid * dataset_size / 100.0)
    num_test = int(args.percent_test * dataset_size / 100.0)
    num_train = dataset_size - num_valid - num_test

    # print("Creating splits: {}, {}, {}".format(num_train, num_valid, num_test))

    train_files = filelist[train_start:(train_start+num_train)]
    valid_files = filelist[valid_start:(valid_start+num_valid)]
    test_files = filelist[test_start:(test_start+num_test)]

    print("Splits: {}, {}, {}".format(len(train_files), len(valid_files), len(test_files)))
    create_dataset(args.dataset, train_files, valid_files, test_files)    

if __name__ == '__main__':
    main(sys.argv[1:])
