#!/usr/bin/env python

"""Plots model samples."""
import argparse

import numpy as np
import random
import sys
import json
import datetime
import os
import glob
from braceexpand import braceexpand

from plat.fuel_helper import get_dataset_iterator
from plat.grid_layout import grid2img, create_gradient_grid, create_mine_grid, create_chain_grid, create_fan_grid
from plat.utils import offset_from_string

g_image_size = 128

def real_glob(rglob):
    glob_list = braceexpand(rglob)
    files = []
    for g in glob_list:
        files = files + glob.glob(g)
    return files

# returns new version of images, rows, cols
def add_shoulders(images, anchor_images, rows, cols):
    n_anchors = len(anchor_images)
    if n_anchors == 1:
        ncols = cols + 1
        col_offset = 0
    else:
        ncols = cols + 2
        col_offset = 1
    nimages = []
    cur_im = 0
    for j in range(rows):
        for i in range(ncols):
            if i == 0 and j == 0 and n_anchors > 0:
                nimages.append(anchor_images[0])
            elif i == 0 and j == rows-1 and n_anchors > 1:
                nimages.append(anchor_images[1])
            elif i == ncols-1 and j == 0 and n_anchors > 2:
                nimages.append(anchor_images[2])
            elif i > 0 and i < ncols-col_offset:
                nimages.append(images[cur_im])
                cur_im = cur_im + 1
            else:
                nimages.append(None)
    return nimages, rows, ncols

# returns list of latent variables to support rows x cols 
def generate_latent_grid(z_dim, rows, cols, fan, gradient, spherical, gaussian, anchors, anchor_images, mine, chain, spacing, analogy, rand_uniform=False):
    if fan:
        z = create_fan_grid(z_dim, cols, rows)
    elif gradient:
        z = create_gradient_grid(rows, cols, z_dim, analogy, anchors, spherical, gaussian)
    elif mine:
        z = create_mine_grid(rows, cols, z_dim, spacing, anchors, spherical, gaussian)
    elif chain:
        z = create_chain_grid(rows, cols, z_dim, spacing, anchors, spherical, gaussian)
    else:
        if rand_uniform:
            z = np.random.uniform(-1, 1, size=(rows * cols, z_dim))
        else:
            z = np.random.normal(loc=0, scale=1, size=(rows * cols, z_dim))

    return z

# this function can fill in placeholders for %DATE%, %SIZE% and %SEQ%
def emit_filename(filename, args, image_size):
    datestr = datetime.datetime.now().strftime("%Y%m%d")
    filename = filename.replace('%DATE%', datestr)
    filename = filename.replace('%SIZE%', "{:d}".format(image_size))
    if args is not None:
        if args.model:
            model = args.model.replace(".", "_")
        else:
            model = "NoModel"
        if args.seed:
            seed = "{:d}".format(args.seed)
        else:
            seed = "NoSeed"
        filename = filename.replace('%MODEL%', model)
        filename = filename.replace('%OFFSET%', "{:d}".format(args.offset))
        filename = filename.replace('%SEED%', seed)
        filename = filename.replace('%ROWS%', "{:d}".format(args.rows))
        filename = filename.replace('%COLS%', "{:d}".format(args.cols))
    if '%SEQ%' in filename:
        # determine what the next available number is
        cur_seq = 1
        candidate = filename.replace('%SEQ%', "{:02d}".format(cur_seq))
        while os.path.exists(candidate):
            cur_seq = cur_seq + 1
            candidate = filename.replace('%SEQ%', "{:02d}".format(cur_seq))
        filename = candidate
    return filename

def grid_from_latents(z, dmodel, rows, cols, anchor_images, tight, shoulders, save_path, args=None, batch_size=24):
    z_queue = z[:]
    samples = None
    # print("========> DECODING {} at a time".format(batch_size))
    while(len(z_queue) > 0):
        cur_z = z_queue[:batch_size]
        z_queue = z_queue[batch_size:]
        decoded = dmodel.sample_at(cur_z)
        if samples is None:
            samples = decoded
        else:
            samples = np.concatenate((samples, decoded), axis=0)

    # samples = dmodel.sample_at(z)

    if shoulders:
        samples, rows, cols = add_shoulders(samples, anchor_images, rows, cols)

    try:
        one_sample = next(item for item in samples if item is not None)
    except StopIteration:
        print("No samples found to save")
        return

    # each sample is 3xsizexsize
    image_size = one_sample.shape[1]
    final_save_path = emit_filename(save_path, args, image_size);
    print("Saving image file {}".format(final_save_path))
    img = grid2img(samples, rows, cols, not tight)
    img.save(final_save_path)


def surround_anchors(rows, cols, anchors, rand_anchors):
    newanchors = []
    cur_anc = 0
    cur_rand = 0
    for r in range(rows):
        for c in range(cols):
            if r == 0 or c == 0 or r == rows-1 or c == cols-1:
                newanchors.append(rand_anchors[cur_rand])
                cur_rand = cur_rand + 1
            else:
                newanchors.append(anchors[cur_anc])
                cur_anc = cur_anc + 1
    return newanchors

def vector_to_json_array(v):
    return json.dumps(v.tolist())

def output_vectors(vectors):
    print("VECTOR OUTPUT BEGIN")
    print("JSON#[")
    for v in vectors[:-1]:
        print("JSON#{},".format(vector_to_json_array(v)))
    for v in vectors[-1:]:
        print("JSON#{}".format(vector_to_json_array(v)))
    print("JSON#]")
    print("VECTOR OUTPUT END")

def anchors_from_offsets(anchor, offsets, x_indices_str, y_indices_str, x_minscale, y_minscale, x_maxscale, y_maxscale):
    dim = len(anchor)
    x_offset = offset_from_string(x_indices_str, offsets, dim)
    y_offset = offset_from_string(y_indices_str, offsets, dim)

    newanchors = []
    newanchors.append(anchor + x_minscale * x_offset + y_minscale * y_offset)
    newanchors.append(anchor + x_minscale * x_offset + y_maxscale * y_offset)
    newanchors.append(anchor + x_maxscale * x_offset + y_minscale * y_offset)
    newanchors.append(anchor + x_maxscale * x_offset + y_maxscale * y_offset)
    return np.array(newanchors)

def anchors_noise_offsets(anchors, offsets, rows, cols, spacing, z_step, x_indices_str, y_indices_str, x_minscale, y_minscale, x_maxscale, y_maxscale):
    from noise import pnoise3
    from plat.interpolate import lerp

    only_anchor = None
    if len(anchors) == 1:
        only_anchor = anchors[0]

    dim = len(anchors[0])
    x_offset = offset_from_string(x_indices_str, offsets, dim)
    y_offset = offset_from_string(y_indices_str, offsets, dim)

    num_row_anchors = (rows + spacing - 1) / spacing
    num_col_anchors = (cols + spacing - 1) / spacing

    newanchors = []
    cur_anchor_index = 0
    for j in range(num_row_anchors):
        y_frac = float(j) / num_row_anchors
        for i in range(num_col_anchors):
            if only_anchor is None:
                cur_anchor = anchors[cur_anchor_index]
                cur_anchor_index += 1
            else:
                cur_anchor = only_anchor
            x_frac = float(i) / num_col_anchors
            n1 = 0.5 * (1.0 + pnoise3(x_frac, y_frac, z_step, octaves=4, repeatz=2))
            n2 = 0.5 * (1.0 + pnoise3(100+x_frac, 100+y_frac, z_step, octaves=4, repeatz=2))
            x_scale = lerp(n1, x_minscale, x_maxscale)
            y_scale = lerp(n2, y_minscale, y_maxscale)
            # print("{}, {} produced {} -> {}, {} = {}".format(i,j,n1,x_minscale, x_maxscale,x_scale))
            newanchors.append(cur_anchor + x_scale * x_offset + y_scale * y_offset)
    return np.array(newanchors)

def get_global_offset(offsets, indices_str, scale):
    dim = len(offsets[0])
    global_offset = offset_from_string(indices_str, offsets, dim)
    return scale * global_offset

def stream_output_vectors(dmodel, dataset, split, batch_size=20, color_convert=False):
    it = get_dataset_iterator(dataset, split)
    done = False

    sys.stderr.write("Streaming output vectors to stdout (batch size={})\n".format(batch_size))

    print("VECTOR OUTPUT BEGIN")
    print("JSON#[")

    num_output = 0
    while not done:
        anchors = []
        try:
            for i in range(batch_size):
                cur = it.next()
                if color_convert:
                    anchors.append(np.tile(cur[0].reshape(1, g_image_size, g_image_size), (3, 1, 1)))
                else:
                    anchors.append(cur[0])
            anchors_input = np.array(anchors)
            latents = dmodel.encode_images(anchors_input)
            num_output += len(latents)
            for v in latents:
                print("JSON#{},".format(vector_to_json_array(v)))
        except StopIteration:
            # process any leftovers
            if len(anchors) > 0:
                anchors_input = np.array(anchors)
                latents = dmodel.encode_images(anchors_input)
                num_output += len(latents)
                # end cut-n-paste
                for v in latents[:-1]:
                    print("JSON#{},".format(vector_to_json_array(v)))
                for v in latents[-1:]:
                    print("JSON#{}".format(vector_to_json_array(v)))
            done = True

    # for v in vectors[-1:]:
    #     print("{}".format(vector_to_json_array(v)))

    print("JSON#]")
    print("VECTOR OUTPUT END")
    sys.stderr.write("Done streaming {} vectors\n".format(num_output))