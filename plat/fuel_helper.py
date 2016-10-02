#!/usr/bin/env python

"""Routines for accessing fuel datasets."""
import numpy as np
from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.schemes import SequentialExampleScheme, ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import AgnosticSourcewiseTransformer

def get_dataset_iterator(dataset, split, include_features=True, include_targets=False, unit_scale=True):
    """Get iterator for dataset, split, targets (labels) and scaling (from 255 to 1.0)"""
    sources = []
    sources = sources + ['features'] if include_features else sources
    sources = sources + ['targets'] if include_targets else sources
    if split == "all":
        splits = ('train', 'valid', 'test')
    elif split == "nontrain":
        splits = ('valid', 'test')
    else:
        splits = (split,)

    dataset_fname = find_in_data_path("{}.hdf5".format(dataset))
    datastream = H5PYDataset(dataset_fname, which_sets=splits,
                             sources=sources)
    if unit_scale:
        datastream.default_transformers = uint8_pixels_to_floatX(('features',))

    train_stream = DataStream.default_stream(
        dataset=datastream,
        iteration_scheme=SequentialExampleScheme(datastream.num_examples))

    it = train_stream.get_epoch_iterator()
    return it

# get images from dataset. numanchors=None to get all. image_size only needed for color conversion
def get_anchor_images(dataset, split, offset=0, stepsize=1, numanchors=150, allowed=None, prohibited=None, image_size=64, color_convert=False, include_targets=True, unit_scale=True):
    """Get images in np array with filters"""
    it = get_dataset_iterator(dataset, split, include_targets=include_targets, unit_scale=unit_scale)

    anchors = []
    for i in range(offset):
        cur = it.next()
    try:
        while numanchors == None or len(anchors) < numanchors:
            cur = it.next()
            for s in range(stepsize-1):
                it.next()
            candidate_passes = True
            if allowed:
                for p in allowed:
                    if(cur[1][p] != 1):
                        candidate_passes = False
            if prohibited:
                for p in prohibited:
                    if(cur[1][p] != 0):
                        candidate_passes = False

            if candidate_passes:
                if color_convert:
                    anchors.append(np.tile(cur[0].reshape(1, image_size, image_size), (3, 1, 1)))
                else:
                    anchors.append(cur[0])
    except StopIteration:
        if numanchors is not None:
            print("Warning: only read {} of {} requested anchor images".format(len(anchors), numanchors))

    return np.array(anchors)

class Colorize(AgnosticSourcewiseTransformer):
    """Triples a grayscale image to convert it to color.

    This transformer can be used to adapt a one one-channel
    grayscale image to a network that expects three color
    channels by simply replictaing the single channel
    three times.
    """
    def __init__(self, data_stream, **kwargs):
        super(Colorize, self).__init__(
             data_stream=data_stream,
             produces_examples=data_stream.produces_examples,
             **kwargs)

    def transform_any_source(self, source, _):
        iters, dim, height, width = source.shape
        return np.tile(source.reshape(iters*dim, 1, 64, 64), (1, 3, 1, 1))

class Scrubber(AgnosticSourcewiseTransformer):
    """Used to whitelist selected labels for training.

    This transformer zeros out all but selected attribute
    labels for training. This allows custom classifiers to
    be built on a subset of available features

    Parameters
    ----------
    allowed : list of int
        Whitelist of indexes to allow in training. For example,
        passing [4, 7] means only labels at positions 4 and 7
        will be used for training, and all others will be
        zeroed out.
    """
    def __init__(self, data_stream, allowed, **kwargs):
        super(Scrubber, self).__init__(
             data_stream=data_stream,
             produces_examples=data_stream.produces_examples,
             **kwargs)
        self.allowed = [0] * 40
        for a in allowed:
            self.allowed[a] = 1

    def transform_any_source(self, source, _):
        return [a*b for a, b in zip(self.allowed, source)]


class StretchLabels(AgnosticSourcewiseTransformer):
    """Used to stretch a set of vectors by zero-padding.

    This transformer appends zeros to a vector to get it
    to a predefined length.

    Parameters
    ----------
    length : int
        Target lenth of vector.
    """
    def __init__(self, data_stream, length=64, **kwargs):
        super(StretchLabels, self).__init__(
             data_stream=data_stream,
             produces_examples=data_stream.produces_examples,
             **kwargs)
        self.length = length

    def transform_any_source(self, source, _):
        if len(source > 0):
            npad = self.length - len(source[0])
            return [np.pad(a, pad_width=(0, npad), mode='constant',
                           constant_values=0) for a in source]
        else:
            return source

#copied from discgen.utils
def create_streams(train_set, valid_set, test_set, training_batch_size,
                   monitoring_batch_size):
    """Creates data streams for training and monitoring.

    Parameters
    ----------
    train_set : :class:`fuel.datasets.Dataset`
        Training set.
    valid_set : :class:`fuel.datasets.Dataset`
        Validation set.
    test_set : :class:`fuel.datasets.Dataset`
        Test set.
    monitoring_batch_size : int
        Batch size for monitoring.
    include_targets : bool
        If ``True``, use both features and targets. If ``False``, use
        features only.

    Returns
    -------
    rval : tuple of data streams
        Data streams for the main loop, the training set monitor,
        the validation set monitor and the test set monitor.

    """
    main_loop_stream = DataStream.default_stream(
        dataset=train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, training_batch_size))
    train_monitor_stream = DataStream.default_stream(
        dataset=train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, monitoring_batch_size))
    valid_monitor_stream = DataStream.default_stream(
        dataset=valid_set,
        iteration_scheme=ShuffledScheme(
            valid_set.num_examples, monitoring_batch_size))
    test_monitor_stream = DataStream.default_stream(
        dataset=test_set,
        iteration_scheme=ShuffledScheme(
            test_set.num_examples, monitoring_batch_size))

    return (main_loop_stream, train_monitor_stream, valid_monitor_stream,
            test_monitor_stream)

def create_custom_streams(filename, training_batch_size, monitoring_batch_size,
                          include_targets=False, color_convert=False,
                          allowed=None, stretch=False,
                          split_names=['train', 'valid', 'test']):
    """Creates data streams from fuel hdf5 file.

    Currently features must be 64x64.

    Parameters
    ----------
    filename : string
        basename to hdf5 file for input
    training_batch_size : int
        Batch size for training.
    monitoring_batch_size : int
        Batch size for monitoring.
    include_targets : bool
        If ``True``, use both features and targets. If ``False``, use
        features only.
    color_convert : bool
        If ``True``, input is assumed to be one-channel, and so will
        be transformed to three-channel by duplication.

    Returns
    -------
    rval : tuple of data streams
        Data streams for the main loop, the training set monitor,
        the validation set monitor and the test set monitor.

    """
    sources = ('features', 'targets') if include_targets else ('features',)

    dataset_fname = find_in_data_path(filename+'.hdf5')
    data_train = H5PYDataset(dataset_fname, which_sets=[split_names[0]],
                             sources=sources)
    data_valid = H5PYDataset(dataset_fname, which_sets=[split_names[1]],
                             sources=sources)
    data_test = H5PYDataset(dataset_fname, which_sets=[split_names[2]],
                            sources=sources)
    data_train.default_transformers = uint8_pixels_to_floatX(('features',))
    data_valid.default_transformers = uint8_pixels_to_floatX(('features',))
    data_test.default_transformers = uint8_pixels_to_floatX(('features',))

    results = create_streams(data_train, data_valid, data_test,
                             training_batch_size, monitoring_batch_size)

    if color_convert:
        results = tuple(map(
                    lambda s: Colorize(s, which_sources=('features',)),
                    results))

    # wrap labels in stretcher if requested
    if stretch:
        results = tuple(map(
                    lambda s: StretchLabels(s, which_sources=('targets',)),
                    results))

    # wrap labels in scrubber if not all labels are allowed
    if allowed:
        results = tuple(map(
                    lambda s: Scrubber(s, allowed=allowed,
                                       which_sources=('targets',)),
                    results))

    return results
