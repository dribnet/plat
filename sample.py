# THIS IS HERE AS A TEMPORARY KLUDGE TO SUPPORT CHECKPOINTS

#!/usr/bin/env python

"""Plots model samples."""
import argparse

import numpy as np
import random
import sys
import json

from plat.grid_layout import grid2img, create_gradient_grid, create_mine_grid, create_chain_grid, create_fan_grid
from plat.utils import anchors_from_image, get_json_vectors, offset_from_string

import importlib
g_image_size = 128

def lazy_init_fuel_dependencies():
    try:
        from chips.fuel_helper import get_dataset_iterator, get_anchor_images
        return get_dataset_iterator, get_anchor_images
    except ImportError as e:
        # raise ImportError('<any message you want here>')
        print("Error: Failed fuel dependency")
        print e
        exit(1);


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

def grid_from_latents(z, dmodel, rows, cols, anchor_images, tight, shoulders, save_path, batch_size=24):
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

    print('Preparing image grid...')
    img = grid2img(samples, rows, cols, not tight)
    img.save(save_path)


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
    get_dataset_iterator, _ = lazy_init_fuel_dependencies()
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

def run_with_args(args, dmodel, cur_anchor_image, cur_save_path, cur_z_step):
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    anchor_images = None
    if args.anchors:
        _, get_anchor_images = lazy_init_fuel_dependencies()
        allowed = None
        prohibited = None
        include_targets = False
        if(args.allowed):
            include_targets = True
            allowed = map(int, args.allowed.split(","))
        if(args.prohibited):
            include_targets = True
            prohibited = map(int, args.prohibited.split(","))
        anchor_images = get_anchor_images(args.dataset, args.split, args.offset, args.stepsize, args.numanchors, allowed, prohibited, args.image_size, args.color_convert, include_targets=include_targets)

    if cur_anchor_image is not None:
        _, _, anchor_images = anchors_from_image(cur_anchor_image, image_size=(args.image_size, args.image_size))
        if args.offset > 0:
            anchor_images = anchor_images[args.offset:]
        # untested
        if args.numanchors is not None:
            anchor_images = anchor_images[:args.numanchors]

    if args.passthrough:
        print('Preparing image grid...')
        img = grid2img(anchor_images, args.rows, args.cols, not args.tight)
        img.save(cur_save_path)
        sys.exit(0)

    if dmodel is None:
        model_class_parts = args.model_class.split(".")
        model_class_name = model_class_parts[-1]
        model_module_name = ".".join(model_class_parts[:-1])
        print("Loading {} interface from {}".format(model_class_name, model_module_name))        
        ModelClass = getattr(importlib.import_module(model_module_name), model_class_name)
        print("Loading model from {}".format(args.model))
        dmodel = ModelClass(filename=args.model)

    if anchor_images is not None:
        x_queue = anchor_images[:]
        anchors = None
        # print("========> ENCODING {} at a time".format(args.batch_size))
        while(len(x_queue) > 0):
            cur_x = x_queue[:args.batch_size]
            x_queue = x_queue[args.batch_size:]
            encoded = dmodel.encode_images(cur_x)
            if anchors is None:
                anchors = encoded
            else:
                anchors = np.concatenate((anchors, encoded), axis=0)

        # anchors = dmodel.encode_images(anchor_images)
    elif args.anchor_vectors is not None:
        anchors = get_json_vectors(args.anchor_vectors)
    else:
        anchors = None

    if args.invert_anchors:
        anchors = -1 * anchors

    if args.encoder:
        if anchors is not None:
            output_vectors(anchors)
        else:
            stream_output_vectors(dmodel, args.dataset, args.split, batch_size=args.batch_size)
        sys.exit(0)

    global_offset = None
    if args.anchor_offset is not None:
        # compute anchors as offsets from existing anchor
        offsets = get_json_vectors(args.anchor_offset)
        if args.anchor_noise:
            anchors = anchors_noise_offsets(anchors, offsets, args.rows, args.cols, args.spacing,
                cur_z_step, args.anchor_offset_x, args.anchor_offset_y,
                args.anchor_offset_x_minscale, args.anchor_offset_y_minscale, args.anchor_offset_x_maxscale, args.anchor_offset_y_maxscale)
        else:
            anchors = anchors_from_offsets(anchors[0], offsets, args.anchor_offset_x, args.anchor_offset_y,
                args.anchor_offset_x_minscale, args.anchor_offset_y_minscale, args.anchor_offset_x_maxscale, args.anchor_offset_y_maxscale)

    if args.global_offset is not None:
        offsets = get_json_vectors(args.global_offset)
        if args.global_ramp:
            offsets = cur_z_step * offsets
        global_offset =  get_global_offset(offsets, args.global_indices, args.global_scale)

    z_dim = dmodel.get_zdim()
    # I don't remember what partway/encircle do so they are not handling the chain layout
    # this handles the case (at least) of mines with random anchors
    if (args.partway is not None) or args.encircle or (args.mine and anchors is None):
        srows=((args.rows // args.spacing) + 1)
        scols=((args.cols // args.spacing) + 1)
        rand_anchors = generate_latent_grid(z_dim, rows=srows, cols=scols, fan=False, gradient=False,
            spherical=False, gaussian=False, anchors=None, anchor_images=None, mine=False, chain=False,
            spacing=args.spacing, analogy=False, rand_uniform=args.uniform)
        if args.partway is not None:
            l = len(rand_anchors)
            clipped_anchors = anchors[:l]
            anchors = (1.0 - args.partway) * rand_anchors + args.partway * clipped_anchors
        elif args.encircle:
            anchors = surround_anchors(srows, scols, anchors, rand_anchors)
        else:
            anchors = rand_anchors
    z = generate_latent_grid(z_dim, args.rows, args.cols, args.fan, args.gradient, not args.linear, args.gaussian,
            anchors, anchor_images, args.mine, args.chain, args.spacing, args.analogy)
    if global_offset is not None:
        z = z + global_offset

    grid_from_latents(z, dmodel, args.rows, args.cols, anchor_images, args.tight, args.shoulders, cur_save_path, args.batch_size)
    return dmodel

def main(cliargs):
    parser = argparse.ArgumentParser(description="Plot model samples")
    parser.add_argument("--interface", dest='model_class', type=str,
                        default="plat.interface.discgen.DiscGenModel", help="class encapsulating model")
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument("--rows", type=int, default=5,
                        help="number of rows of samples to display")
    parser.add_argument("--cols", type=int, default=5,
                        help="number of columns of samples to display")
    parser.add_argument("--save-path", type=str, default=None,
                        help="where to save the generated samples")
    parser.add_argument('--fan', dest='fan', default=False, action='store_true')
    parser.add_argument('--analogy', dest='analogy', default=False, action='store_true')
    parser.add_argument('--global-offset', dest='global_offset', default=None,
                        help="use json file as source of global offsets")
    parser.add_argument('--global-indices', dest='global_indices', default=None, type=str,
                        help="offset indices to apply globally")
    parser.add_argument('--global-scale', dest='global_scale', default=1.0, type=float,
                        help="scaling factor for global offset")
    parser.add_argument('--global-ramp', dest='global_ramp', default=False, action='store_true',
                        help="ramp global effect with z-step")
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")
    parser.add_argument('--anchor-offset-x', dest='anchor_offset_x', default="5", type=str,
                        help="which indices to combine for x offset")
    parser.add_argument('--anchor-offset-y', dest='anchor_offset_y', default="39", type=str,
                        help="which indices to combine for y offset")
    parser.add_argument('--anchor-offset-x-minscale', dest='anchor_offset_x_minscale', default=0, type=float,
                        help="scaling factor for min x offset")
    parser.add_argument('--anchor-offset-y-minscale', dest='anchor_offset_y_minscale', default=0, type=float,
                        help="scaling factor for min y offset")
    parser.add_argument('--anchor-offset-x-maxscale', dest='anchor_offset_x_maxscale', default=2.0, type=float,
                        help="scaling factor for min x offset")
    parser.add_argument('--anchor-offset-y-maxscale', dest='anchor_offset_y_maxscale', default=2.0, type=float,
                        help="scaling factor for min y offset")
    parser.add_argument('--anchor-noise', dest='anchor_noise', default=False, action='store_true',
                        help="interpret anchor offsets as noise paramaters")
    parser.add_argument('--gradient', dest='gradient', default=False, action='store_true')
    parser.add_argument('--linear', dest='linear', default=False, action='store_true')
    parser.add_argument('--gaussian', dest='gaussian', default=False, action='store_true')
    parser.add_argument('--uniform', dest='uniform', default=False, action='store_true',
                        help="Random prior is uniform [-1,1] (not gaussian)")
    parser.add_argument('--tight', dest='tight', default=False, action='store_true')
    parser.add_argument("--seed", type=int,
                default=None, help="Optional random seed")
    parser.add_argument('--mine', dest='mine', default=False, action='store_true')
    parser.add_argument('--chain', dest='chain', default=False, action='store_true')
    parser.add_argument('--encircle', dest='encircle', default=False, action='store_true')
    parser.add_argument('--partway', dest='partway', type=float, default=None)
    parser.add_argument("--spacing", type=int, default=3,
                        help="spacing of mine grid, w & h must be multiples +1")
    parser.add_argument('--anchors', dest='anchors', default=False, action='store_true',
                        help="use reconstructed images instead of random ones")
    parser.add_argument('--anchor-image', dest='anchor_image', default=None,
                        help="use image as source of anchors")
    parser.add_argument('--anchor-vectors', dest='anchor_vectors', default=None,
                        help="use json file as source of anchors")
    parser.add_argument('--invert-anchors', dest='invert_anchors',
                        default=False, action='store_true',
                        help="Use antipode of given anchors.")
    parser.add_argument("--numanchors", type=int, default=150,
                        help="number of anchors to generate")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Dataset for anchors.")
    parser.add_argument('--color-convert', dest='color_convert',
                        default=False, action='store_true',
                        help="Convert source dataset to color from grayscale.")
    parser.add_argument('--split', dest='split', default="all",
                        help="Which split to use from the dataset (train/nontrain/valid/test/any).")
    parser.add_argument("--offset", type=int, default=0,
                        help="data offset to skip")
    parser.add_argument("--stepsize", type=int, default=1,
                        help="data step size from offset")
    parser.add_argument("--allowed", dest='allowed', type=str, default=None,
                        help="Only allow whitelisted labels L1,L2,...")
    parser.add_argument("--prohibited", dest='prohibited', type=str, default=None,
                        help="Only allow blacklisted labels L1,L2,...")
    parser.add_argument('--passthrough', dest='passthrough', default=False, action='store_true',
                        help="Use originals instead of reconstructions")
    parser.add_argument('--shoulders', dest='shoulders', default=False, action='store_true',
                        help="Append anchors to left/right columns")
    parser.add_argument('--encoder', dest='encoder', default=False, action='store_true',
                        help="Ouput dataset as encoded vectors")
    parser.add_argument("--batch-size", dest='batch_size',
                        type=int, default=64,
                        help="batch size when encoding vectors")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument('--anchor-image-template', dest='anchor_image_template', default=None,
                        help="template for anchor image filename")
    parser.add_argument('--save-path-template', dest='save_path_template', default=None,
                        help="template for save path filename")
    parser.add_argument('--range', dest='range', default=None,
                        help="low,high integer range for tempalte run")
    parser.add_argument('--z-step', dest='z_step', default=0.01, type=float,
                        help="variable that gets stepped each template step")
    parser.add_argument('--z-initial', dest='z_initial', default=0.0, type=float,
                        help="initial value of variable stepped each template step")
    args = parser.parse_args(cliargs)

    dmodel = None
    cur_z_step = args.z_initial
    if args.range is None:
        run_with_args(args, dmodel, args.anchor_image, args.save_path, cur_z_step)
    else:
        template_low, template_high = map(int, args.range.split(","))
        for i in range(template_low, template_high + 1):
            if args.anchor_image_template is not None:
                cur_anchor_image = args.anchor_image_template.format(i)
            else:
                cur_anchor_image = args.anchor_image
            cur_save_path = args.save_path_template.format(i)
            print("Saving: {}".format(cur_save_path))
            dmodel = run_with_args(args, dmodel, cur_anchor_image, cur_save_path, cur_z_step)
            cur_z_step += args.z_step

if __name__ == '__main__':
    main(sys.argv[1:])
