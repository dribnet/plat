#!/usr/bin/env python

"""Plots model samples."""
import argparse

import numpy as np
import random
import sys
import json
import math
import datetime
import os
import glob
from braceexpand import braceexpand

from plat.fuel_helper import get_dataset_iterator
from plat.grid_layout import grid2img, create_gradient_grid, create_mine_grid, create_chain_grid, create_fan_grid
from plat.utils import offset_from_string
from plat.interpolate import lerp

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
def emit_filename(filename, template_dict, args):
    datestr = datetime.datetime.now().strftime("%Y%m%d")
    filename = filename.replace('%DATE%', datestr)

    for key in template_dict:
        pattern = "%{}%".format(key)
        value = "{}".format(template_dict[key])
        filename = filename.replace(pattern, value)

    if args is not None:
        # legacy replacements
        if args.model:
            model = args.model.replace(".", "_")
        else:
            model = "NoModel"
        if args.seed:
            seed = "{:d}".format(args.seed)
        else:
            seed = "NoSeed"
        if '%MODEL%' in filename:
            filename = filename.replace('%MODEL%', model)
        if '%OFFSET%' in filename:
            filename = filename.replace('%OFFSET%', "{:d}".format(args.offset))
        if '%SEED%' in filename:
            filename = filename.replace('%SEED%', seed)
        if '%ROWS%' in filename:
            filename = filename.replace('%ROWS%', "{:d}".format(args.rows))
        if '%COLS%' in filename:
            filename = filename.replace('%COLS%', "{:d}".format(args.cols))
        if '%INDEX%' in filename:
            filename = filename.replace('%INDEX%', "{}".format(args.anchor_offset_x))
    if '%SEQ%' in filename:
        # determine what the next available number is
        cur_seq = 1
        candidate = filename.replace('%SEQ%', "{:02d}".format(cur_seq))
        while os.path.exists(candidate):
            cur_seq = cur_seq + 1
            candidate = filename.replace('%SEQ%', "{:02d}".format(cur_seq))
        filename = candidate
    return filename

def grid_from_latents(z, dmodel, rows, cols, anchor_images, tight, shoulders, save_path, args=None, batch_size=24, template_dict={}):
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
    template_dict["SIZE"] = one_sample.shape[1]
    final_save_path = emit_filename(save_path, template_dict, args)
    after_last_slash = final_save_path.rfind("/") + 1
    outfile_temp = final_save_path[:after_last_slash] + '_' + final_save_path[after_last_slash:]
    print("Saving image file {}".format(final_save_path))
    dirname = os.path.dirname(final_save_path)
    if dirname != '' and not os.path.exists(dirname):
        os.makedirs(dirname)
    img = grid2img(samples, rows, cols, not tight)
    img.save(outfile_temp)
    os.rename(outfile_temp, final_save_path)
    os.system("touch {}".format(final_save_path))

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

def output_vectors(vectors, outfile):
    print("VECTOR OUTPUT BEGIN")
    f_out = open(outfile, 'w+')
    f_out.write("[\n")
    for v in vectors[:-1]:
        f_out.write("{}\n,".format(vector_to_json_array(v)))
    for v in vectors[-1:]:
        f_out.write("{}\n".format(vector_to_json_array(v)))
    f_out.write("]\n")
    f_out.close();
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

def compute_wave(offset, clip_wave):
    # sine wave from 0-1 to 0-1. 0 outside of [0,1]
    if clip_wave and (offset < 0.0 or offset > 1.0):
        return 0
    else:
        return 0.5 * (1.0 - math.cos(offset * 2 * math.pi))

def distance_2d(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def anchors_wave_offsets(anchors, offsets, rows, cols, spacing, radial_wave, clip_wave, z_step, x_indices_str, x_minscale, x_maxscale):
    only_anchor = None
    if len(anchors) == 1:
        only_anchor = anchors[0]

    dim = len(anchors[0])
    x_offset = offset_from_string(x_indices_str, offsets, dim)

    num_row_anchors = (rows + spacing - 1) / spacing
    num_col_anchors = (cols + spacing - 1) / spacing

    newanchors = []
    cur_anchor_index = 0
    center_pt = [(num_col_anchors-1) / 2.0, (num_row_anchors-1) / 2.0]
    max_dist = distance_2d([0, 0], center_pt)
    for j in range(num_row_anchors):
        for i in range(num_col_anchors):
            if only_anchor is None:
                cur_anchor = anchors[cur_anchor_index]
                cur_anchor_index += 1
            else:
                cur_anchor = only_anchor
            cur_dist = distance_2d([i, j], center_pt)
            if radial_wave:
                x_frac = (max_dist-cur_dist) / max_dist
            else:
                x_frac = float(i) / num_col_anchors
            wave_val = z_step + x_frac
            n1 = compute_wave(wave_val, clip_wave)
            x_scale = lerp(n1, x_minscale, x_maxscale)
            # if wave_val < 0.0 or wave_val > 1.0:
            #     x_scale = x_minscale
            # else:
            #     if wave_val < 0.5:
            #         n1 = wave_val * 2
            #     else:
            #         n1 = (1.0 - wave_val) * 2
            #     x_scale = lerp(n1, x_minscale, x_maxscale)
            # print("{}, {} produced {} -> {}, {} = {}".format(i,j,n1,x_minscale, x_maxscale,x_scale))
            newanchors.append(cur_anchor + x_scale * x_offset)
    return np.array(newanchors)

def anchors_noise_offsets(anchors, offsets, rows, cols, spacing, z_step, x_indices_str, y_indices_str, x_minscale, y_minscale, x_maxscale, y_maxscale):
    from noise import pnoise3

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

# TODO: this should probably be refactored from above
def anchors_json_offsets(anchors, offsets, rows, cols, spacing, z_step, x_indices_str, y_indices_str, x_minscale, y_minscale, x_maxscale, y_maxscale, range_data):
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
            n1 = range_data[z_step][0]
            n2 = range_data[z_step][1]
            x_scale = lerp(n1, x_minscale, x_maxscale)
            y_scale = lerp(n2, y_minscale, y_maxscale)
            # print("{}, {} produced {} -> {}, {} = {}".format(i,j,n1,x_minscale, x_maxscale,x_scale))
            newanchors.append(cur_anchor + x_scale * x_offset + y_scale * y_offset)
    return np.array(newanchors)

def get_global_offset(offsets, indices_str, scale):
    dim = len(offsets[0])
    global_offset = offset_from_string(indices_str, offsets, dim)
    return scale * global_offset

def stream_output_vectors(dmodel, dataset, split, outfile=None, batch_size=20, color_convert=False):
    it = get_dataset_iterator(dataset, split)
    done = False

    # sys.stderr.write("Streaming output vectors to stdout (batch size={})\n".format(batch_size))

    print("VECTOR OUTPUT BEGIN")
    f_out = open(outfile, 'w+')
    f_out.write("[\n")

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
                f_out.write("{},\n".format(vector_to_json_array(v)))
        except StopIteration:
            # process any leftovers
            if len(anchors) > 0:
                anchors_input = np.array(anchors)
                latents = dmodel.encode_images(anchors_input)
                num_output += len(latents)
                # end cut-n-paste
                for v in latents[:-1]:
                    f_out.write("{},\n".format(vector_to_json_array(v)))
                for v in latents[-1:]:
                    f_out.write("{}\n".format(vector_to_json_array(v)))
            done = True

    # for v in vectors[-1:]:
    #     print("{}".format(vector_to_json_array(v)))

    f_out.write("]\n")
    f_out.close();
    print("VECTOR OUTPUT END")
    sys.stderr.write("Done streaming {} vectors\n".format(num_output))