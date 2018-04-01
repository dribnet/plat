#!/usr/bin/env python

"""Plots model samples."""
import argparse

import numpy as np
import random
import sys
import json
import datetime
import os

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# from plat.fuel_helper import get_anchor_images, get_anchor_labels
from plat.grid_layout import grid2img
from plat.utils import anchors_from_image, anchors_from_filelist, get_json_vectors,get_json_vectors_list
import plat.sampling
from plat import zoo

def run_with_args(args, dmodel, cur_anchor_image, cur_save_path, cur_z_step, cur_basename="basename", range_data=None, template_dict={}):
    anchor_images = None
    anchor_labels = None
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
        if args.with_labels:
            anchor_labels = get_anchor_labels(args.dataset, args.split, args.offset, args.stepsize, args.numanchors)

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
        if len(anchor_images) == 0:
            print("No images, cannot contine")
            sys.exit(0)

    if cur_anchor_image is not None:
        _, _, anchor_images = anchors_from_image(cur_anchor_image, image_size=(args.image_size, args.image_size))
        if args.offset > 0:
            anchor_images = anchor_images[args.offset:]
        if args.stepsize > 0:
            anchor_images = anchor_images[::args.stepsize]
        if args.numanchors is not None:
            anchor_images = anchor_images[:args.numanchors]

    # at this point we can make a dummy anchor_labels if we need
    if anchor_images is not None and anchor_labels is None:
        anchor_labels = [None] * len(anchor_images)

    if args.passthrough:
        # determine final filename string
        image_size = anchor_images[0].shape[1]
        save_path = plat.sampling.emit_filename(cur_save_path, {}, args);
        print("Preparing image file {}".format(save_path))
        img = grid2img(anchor_images, args.rows, args.cols, not args.tight)
        img.save(save_path)
        sys.exit(0)

    if dmodel is None:
        dmodel = zoo.load_model(args.model, args.model_file, args.model_type, args.model_interface)

    embedded = None
    if anchor_images is not None:
        x_queue = anchor_images[:]
        c_queue = anchor_labels[:]
        anchors = None
        # print("========> ENCODING {} at a time".format(args.batch_size))
        while(len(x_queue) > 0):
            cur_x = x_queue[:args.batch_size]
            cur_c = c_queue[:args.batch_size]
            x_queue = x_queue[args.batch_size:]
            c_queue = c_queue[args.batch_size:]
            # TODO: remove vestiges of conditional encode/decode
            # encoded = dmodel.encode_images(cur_x, cur_c)
            encoded = dmodel.encode_images(cur_x)
            try:
                emb_l = dmodel.embed_labels(cur_c)
            except AttributeError:
                emb_l = [None] * args.batch_size
            if anchors is None:
                anchors = encoded
                embedded = emb_l
            else:
                anchors = np.concatenate((anchors, encoded), axis=0)
                embedded = np.concatenate((embedded, emb_l), axis=0)

        # anchors = dmodel.encode_images(anchor_images)
    elif args.anchor_vectors is not None:
        anchors = get_json_vectors(args.anchor_vectors)
    else:
        anchors = None

    if args.invert_anchors:
        anchors = -1 * anchors

    if args.encoder:
        if anchors is not None:
            plat.sampling.output_vectors(anchors, args.save_path)
        else:
            plat.sampling.stream_output_vectors(dmodel, args.dataset, args.split, args.save_path, batch_size=args.batch_size)
        sys.exit(0)

    global_offset = None
    if args.anchor_offset is not None:
        # compute anchors as offsets from existing anchor
        offsets = get_json_vectors_list(args.anchor_offset)
        if args.anchor_wave:
            anchors = plat.sampling.anchors_wave_offsets(anchors, offsets, args.rows, args.cols, args.spacing,
                args.radial_wave, args.clip_wave, cur_z_step, args.anchor_offset_x,
                args.anchor_offset_x_minscale, args.anchor_offset_x_maxscale)
        elif args.anchor_noise:
            anchors = plat.sampling.anchors_noise_offsets(anchors, offsets, args.rows, args.cols, args.spacing,
                cur_z_step, args.anchor_offset_x, args.anchor_offset_y,
                args.anchor_offset_x_minscale, args.anchor_offset_y_minscale, args.anchor_offset_x_maxscale, args.anchor_offset_y_maxscale)
        elif range_data is not None:
            anchors = plat.sampling.anchors_json_offsets(anchors, offsets, args.rows, args.cols, args.spacing,
                cur_z_step, args.anchor_offset_x, args.anchor_offset_y,
                args.anchor_offset_x_minscale, args.anchor_offset_y_minscale, args.anchor_offset_x_maxscale, args.anchor_offset_y_maxscale,
                range_data)
        else:
            anchors = plat.sampling.anchors_from_offsets(anchors[0], offsets, args.anchor_offset_x, args.anchor_offset_y,
                args.anchor_offset_x_minscale, args.anchor_offset_y_minscale, args.anchor_offset_x_maxscale, args.anchor_offset_y_maxscale)

    if args.global_offset is not None:
        offsets = get_json_vectors(args.global_offset)
        if args.global_ramp:
            offsets = cur_z_step * offsets
        global_offset =  plat.sampling.get_global_offset(offsets, args.global_indices, args.global_scale)

    z_dim = dmodel.get_zdim()
    # I don't remember what partway/encircle do so they are not handling the chain layout
    # this handles the case (at least) of mines with random anchors
    if (args.partway is not None) or args.encircle or (anchors is None):
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
            anchors, anchor_images, True, args.chain, args.spacing, args.analogy)
    if global_offset is not None:
        z = z + global_offset

    template_dict["BASENAME"] = cur_basename
    # emb_l = None
    # emb_l = [None] * len(z)
    embedded_labels = None
    # TODO: this could be more elegant
    if embedded is not None and embedded[0] is not None:
        if args.clone_label is not None:
            embedded_labels = np.tile(embedded[args.clone_label], [len(z), 1])
        else:
            embedded_labels = plat.sampling.generate_latent_grid(z_dim, args.rows, args.cols, args.fan, args.gradient, not args.linear, args.gaussian,
                    embedded, anchor_images, True, args.chain, args.spacing, args.analogy)

    #TODO - maybe not best way to check if labels are valid
    # if anchor_labels is None or anchor_labels[0] is None:
    #     emb_l = [None] * len(z)
    plat.sampling.grid_from_latents(z, dmodel, args.rows, args.cols, anchor_images, args.tight, args.shoulders, cur_save_path, args, args.batch_size, template_dict=template_dict, emb_l=embedded_labels)
    return dmodel

class AnchorFileHandler(FileSystemEventHandler):
    last_processed = None
    # if we compute path lists, they are cached
    anchor_path_list = None
    anchor_path_names = None

    def setup(self, args, dmodel, save_path, cur_z_step):
        self.args = args
        self.dmodel = dmodel
        self.save_path = save_path
        self.cur_z_step = cur_z_step

    def process(self, anchor):
        template_dict = {}
        basename = os.path.basename(anchor)
        if basename[0] == '.' or basename[0] == '_':
            print("Skipping anchor: {}".format(anchor))
            return;

        if anchor == self.last_processed:
            print("Skipping duplicate anchor: {}".format(anchor))
            return
        else:
            print("Processing anchor: {}".format(anchor))
            self.last_processed = anchor

        barename = os.path.splitext(basename)[0]
        z_range = None
        range_data = None
        if self.args.multistrip is not None:
            for n in range(self.args.multistrip):
                self.args.anchor_offset_x = "{:d}".format(n)
                self.dmodel = run_with_args(self.args, self.dmodel, anchor, self.save_path, self.cur_z_step, barename, template_dict=template_dict)
        elif self.args.anchor_jsons:
            if self.anchor_path_list is None:
                print("Reading anchor-jsons")
                self.anchor_path_list = []
                self.anchor_path_names = []
                json_list = plat.sampling.real_glob(self.args.anchor_jsons)
                for j in json_list:
                    base = os.path.basename(j)
                    self.anchor_path_names.append(os.path.splitext(base)[0])
                    print("Opening {}".format(j))
                    with open(j) as json_file:
                        range_data = np.array(json.load(json_file)["points"])
                        print(range_data.shape)
                        self.anchor_path_list.append(range_data)
            cur_z_step = 0
            z_step = 1
        elif self.args.range is not None:
            z_range = map(int, self.args.range.split(","))
            z_step = args.z_step
            cur_z_step = args.z_initial
        if z_range is not None or self.anchor_path_list is not None:
            if self.anchor_path_list is not None:
                random_index = random.randint(0, len(self.anchor_path_list)-1)
                print("Generating sequence with anchor path {}".format(self.anchor_path_names[random_index]))
                range_data = self.anchor_path_list[random_index]
                template_dict["PATHNAME"] = self.anchor_path_names[random_index]
                template_dict["PATHLEN"] = "{:03d}".format(len(range_data))
                z_range = 0, len(range_data) - 1
            template_low, template_high = z_range
            for i in range(template_low, template_high + 1):
                template_dict["CURZ"] = "{:03d}".format(i)
                # this is the tricky part to merge?
                if anchor:
                    cur_anchor_image = anchor
                elif self.args.anchor_image_template is not None:
                    cur_anchor_image = self.args.anchor_image_template.format(i)
                else:
                    cur_anchor_image = self.args.anchor_image
                cur_save_path = self.args.save_path_template.format(i)
                self.dmodel = run_with_args(self.args, self.dmodel, cur_anchor_image, cur_save_path, cur_z_step, cur_basename=barename, range_data=range_data, template_dict=template_dict)
                cur_z_step += z_step
        else:
            self.dmodel = run_with_args(self.args, self.dmodel, anchor, self.save_path, self.cur_z_step, barename, template_dict=template_dict)

    def on_modified(self, event):
        if not event.is_directory:
            self.process(event.src_path)

def sample(parser, context, args):
    parser.add_argument('--preload-model', default=False, action='store_true',
                        help="Load the model first before starting processing")
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="name of model")
    parser.add_argument("--model-file", dest='model_file', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument("--model-type", dest='model_type', type=str, default=None,
                        help="the type of model (usually inferred from filename)")
    parser.add_argument("--model-interface", dest='model_interface', type=str,
                        default=None,
                        help="class interface for model (usually inferred from model-type)")
    parser.add_argument("--rows", type=int, default=3,
                        help="number of rows of samples to display")
    parser.add_argument("--cols", type=int, default=7,
                        help="number of columns of samples to display")
    parser.add_argument("--outfile", dest='save_path', type=str, default="plat_%DATE%_%MODEL%_%SEQ%.png",
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
    parser.add_argument('--anchor-wave', dest='anchor_wave', default=False, action='store_true',
                        help="interpret anchor offsets as wave paramaters")
    parser.add_argument('--radial-wave', dest='radial_wave', default=False, action='store_true',
                        help="anchor-wave mode is radial")
    parser.add_argument('--clip-wave', dest='clip_wave', default=False, action='store_true',
                        help="anchor-wave mode is clipped (don't wrap)")
    parser.add_argument('--anchor-noise', dest='anchor_noise', default=False, action='store_true',
                        help="interpret anchor offsets as noise paramaters")
    parser.add_argument('--anchor-jsons', dest='anchor_jsons', default=False,
                        help="a json paths in n dimensions")
    parser.add_argument('--gradient', dest='gradient', default=False, action='store_true')
    parser.add_argument('--linear', dest='linear', default=False, action='store_true')
    parser.add_argument('--gaussian', dest='gaussian', default=False, action='store_true')
    parser.add_argument('--uniform', dest='uniform', default=False, action='store_true',
                        help="Random prior is uniform [-1,1] (not gaussian)")
    parser.add_argument('--tight', dest='tight', default=False, action='store_true')
    parser.add_argument("--seed", type=int,
                default=None, help="Optional random seed")
    parser.add_argument('--chain', dest='chain', default=False, action='store_true')
    parser.add_argument('--encircle', dest='encircle', default=False, action='store_true')
    parser.add_argument('--partway', dest='partway', type=float, default=None)
    parser.add_argument("--spacing", type=int, default=1,
                        help="spacing of mine grid, rows,cols must be multiples of spacing +1")
    parser.add_argument('--anchors', dest='anchors', default=False, action='store_true',
                        help="use reconstructed images instead of random ones")
    parser.add_argument('--anchor-glob', dest='anchor_glob', default=None,
                        help="use file glob source of anchors")
    parser.add_argument('--anchor-directory', dest='anchor_dir', default=None,
                        help="monitor directory for anchors")
    parser.add_argument('--watch', dest='watch', default=False, action='store_true',
                        help="monitor anchor-directory indefinitely")
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
    parser.add_argument("--with-labels", dest='with_labels', default=False, action='store_true',
                        help="use labels for conditioning information")
    parser.add_argument("--clone-label", dest='clone_label', type=int, default=None,
                        help="clone given label (used with --with-labels)")
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
    parser.add_argument('--outfile-template', dest='save_path_template', default=None,
                        help="template for save path filename")
    parser.add_argument('--multistrip', dest='multistrip', default=None, type=int,
                        help="update anchor-offset-x for each entry in anchor-offset")
    parser.add_argument('--range', dest='range', default=None,
                        help="low,high integer range for tempalte run")
    parser.add_argument('--z-step', dest='z_step', default=0.01, type=float,
                        help="variable that gets stepped each template step")
    parser.add_argument('--z-initial', dest='z_initial', default=0.0, type=float,
                        help="initial value of variable stepped each template step")
    args = parser.parse_args(args)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    dmodel = None
    if args.preload_model:
        dmodel = zoo.load_model(args.model, args.model_file, args.model_type, args.model_interface)
    z_range = None
    range_data = None
    event_handler = AnchorFileHandler()
    cur_z_step = args.z_initial

    barename = None
    if args.anchor_image:
        basename = os.path.basename(args.anchor_image)
        barename = os.path.splitext(basename)[0]

    if args.anchor_dir:
        event_handler.setup(args, dmodel, args.save_path, cur_z_step)

        for f in sorted(os.listdir(args.anchor_dir)):
            full_path = os.path.join(args.anchor_dir, f)
            if os.path.isfile(full_path):
                event_handler.process(full_path)

        if args.watch:
            print("Watching anchor directory {}".format(args.anchor_dir))
            observer = Observer()
            observer.schedule(event_handler, path=args.anchor_dir, recursive=False)
            observer.start()

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()
    elif args.anchor_jsons:
        event_handler.setup(args, dmodel, args.save_path, cur_z_step)
        event_handler.process(args.anchor_image)
    elif args.range is not None:
        # TODO: migrate this case to event handler like anchor_jsons above
        z_range = map(int, args.range.split(","))
        z_step = args.z_step
        cur_z_step = args.z_initial
        if z_range is not None:
            template_low, template_high = z_range
            for i in range(template_low, template_high + 1):
                if args.anchor_image_template is not None:
                    cur_anchor_image = args.anchor_image_template.format(i)
                else:
                    cur_anchor_image = args.anchor_image
                cur_save_path = args.save_path_template.format(i)
                dmodel = run_with_args(args, dmodel, cur_anchor_image, cur_save_path, cur_z_step, range_data=range_data)
                cur_z_step += z_step
    else:
        run_with_args(args, dmodel, args.anchor_image, args.save_path, cur_z_step, barename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot model samples")
    sample(parser, None, sys.argv[1:])