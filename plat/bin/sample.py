#!/usr/bin/env python

"""Plots model samples."""
import argparse

import numpy as np
import random
import sys
import json
import datetime
import os

from plat.fuel_helper import get_anchor_images
from plat.grid_layout import grid2img
from plat.utils import anchors_from_image, anchors_from_filelist, get_json_vectors
import plat.sampling
from plat import zoo

def run_with_args(args, dmodel, cur_anchor_image, cur_save_path, cur_z_step):
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    anchor_images = None
    if args.anchors:
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

    if args.anchor_glob is not None:
        files = plat.sampling.real_glob(args.anchor_glob)
        if args.offset > 0:
            files = files[args.offset:]
        if args.stepsize > 1:
            files = files[::args.stepsize]
        if args.numanchors is not None:
            files = files[:args.numanchors]
        anchor_images = anchors_from_filelist(files)
        print("Read {} images from {} files".format(len(anchor_images), len(files)))

    if cur_anchor_image is not None:
        _, _, anchor_images = anchors_from_image(cur_anchor_image, image_size=(args.image_size, args.image_size))
        if args.offset > 0:
            anchor_images = anchor_images[args.offset:]
        if args.stepsize > 0:
            anchor_images = anchor_images[::args.stepsize]
        if args.numanchors is not None:
            anchor_images = anchor_images[:args.numanchors]

    if args.passthrough:
        # determine final filename string
        image_size = anchor_images[0].shape[1]
        save_path = plat.sampling.emit_filename(cur_save_path, args, image_size);
        print("Preparing image file {}".format(save_path))
        img = grid2img(anchor_images, args.rows, args.cols, not args.tight)
        img.save(save_path)
        sys.exit(0)

    if dmodel is None:
        dmodel = zoo.load_model(args.model, args.model_file, args.model_type, args.model_interface)

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
            plat.sampling.output_vectors(anchors)
        else:
            plat.sampling.stream_output_vectors(dmodel, args.dataset, args.split, batch_size=args.batch_size)
        sys.exit(0)

    global_offset = None
    if args.anchor_offset is not None:
        # compute anchors as offsets from existing anchor
        offsets = get_json_vectors(args.anchor_offset)
        if args.anchor_noise:
            anchors = plat.sampling.anchors_noise_offsets(anchors, offsets, args.rows, args.cols, args.spacing,
                cur_z_step, args.anchor_offset_x, args.anchor_offset_y,
                args.anchor_offset_x_minscale, args.anchor_offset_y_minscale, args.anchor_offset_x_maxscale, args.anchor_offset_y_maxscale)
        else:
            anchors = plat.sampling.anchors_from_offsets(anchors[0], offsets, args.anchor_offset_x, args.anchor_offset_y,
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
        rand_anchors = plat.sampling.generate_latent_grid(z_dim, rows=srows, cols=scols, fan=False, gradient=False,
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
    z = plat.sampling.generate_latent_grid(z_dim, args.rows, args.cols, args.fan, args.gradient, not args.linear, args.gaussian,
            anchors, anchor_images, args.mine, args.chain, args.spacing, args.analogy)
    if global_offset is not None:
        z = z + global_offset

    plat.sampling.grid_from_latents(z, dmodel, args.rows, args.cols, anchor_images, args.tight, args.shoulders, cur_save_path, args, args.batch_size)
    return dmodel

def sample(parser, context, args):
    parser.add_argument("--interface", dest='model_class', type=str,
                        default=None, help="class encapsulating model")
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="name of model in plat zoo")
    parser.add_argument("--model-file", dest='model_file', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument("--model-type", dest='model_type', type=str, default=None,
                        help="the type of model (usually inferred from filename)")
    parser.add_argument("--model-interface", dest='model_interface', type=str,
                        default=None,
                        help="class interface for model (usually inferred from model-type)")
    parser.add_argument("--rows", type=int, default=3,
                        help="number of rows of samples to display")
    parser.add_argument("--cols", type=int, default=6,
                        help="number of columns of samples to display")
    parser.add_argument("--save-path", type=str, default="plat_%DATE%_%MODEL%_%SEQ%.png",
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
    parser.add_argument("--spacing", type=int, default=1,
                        help="spacing of mine grid, w & h must be multiples +1")
    parser.add_argument('--anchors', dest='anchors', default=False, action='store_true',
                        help="use reconstructed images instead of random ones")
    parser.add_argument('--anchor-glob', dest='anchor_glob', default=None,
                        help="use file glob source of anchors")
    parser.add_argument('--anchor-image', dest='anchor_image', default=None,
                        help="use image as source of anchors")
    parser.add_argument('--anchor-vectors', dest='anchor_vectors', default=None,
                        help="use json file as source of anchors")
    parser.add_argument('--invert-anchors', dest='invert_anchors',
                        default=False, action='store_true',
                        help="Use antipode of given anchors.")
    parser.add_argument("--numanchors", type=int, default=None,
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
    args = parser.parse_args(args)

    # check for model download first
    if args.model is not None:
        zoo.check_model_download(args.model)

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
    parser = argparse.ArgumentParser(description="Plot model samples")
    sample(parser, None, sys.argv[1:])