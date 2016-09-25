#!/usr/bin/env python

"""Plots model samples."""
import argparse

import numpy as np
import random
import sys
import json
from scipy.misc import imread, imsave

from plat.utils import anchors_from_image, get_json_vectors, offset_from_string
from plat.canvas_layout import create_mine_canvas

from PIL import Image
import importlib

channels = 4

# modified from http://stackoverflow.com/a/3375291/1010653
def alpha_composite(src, src_mask, dst):
    '''
    Return the alpha composite of src and dst.

    Parameters:
    src -- RGBA in range 0.0 - 1.0
    dst -- RGBA in range 0.0 - 1.0

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    '''
    out = np.empty(dst.shape, dtype = 'float')
    alpha = np.index_exp[3:, :, :]
    rgb = np.index_exp[:3, :, :]
    epsilon = 0.001
    src_a = np.maximum(src_mask, epsilon)
    dst_a = np.maximum(dst[alpha], epsilon)
    out[alpha] = src_a+dst_a*(1-src_a)
    old_setting = np.seterr(invalid = 'ignore')
    out[rgb] = (src[rgb]*src_a + dst[rgb]*dst_a*(1-src_a))/out[alpha]
    np.seterr(**old_setting)
    np.clip(out,0,1.0)
    return out

def additive_composite(src, src_mask, dst):
    '''
    Return the additive composite of src and dst.
    '''
    out = np.empty(dst.shape, dtype = 'float')
    alpha = np.index_exp[3:, :, :]
    rgb = np.index_exp[:3, :, :]
    out[alpha] = np.maximum(src_mask,dst[alpha])
    out[rgb] = np.maximum(src[rgb],dst[rgb])
    np.clip(out,0,1.0)
    return out

# gsize = 64
# gsize2 = gsize/2

class Canvas:
    """Simple Canvas Thingy"""

    def __init__(self, width, height, xmin, xmax, ymin, ymax, mask_name, image_size, init_black=False):
        self.pixels = np.zeros((channels, height, width))
        if init_black:
            alpha_channel = np.index_exp[3:, :, :]
            self.pixels[alpha_channel] = 1.0
        self.canvas_xmin = 0
        self.canvas_xmax = width
        self.canvas_ymin = 0
        self.canvas_ymax = height
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.canvas_xspread = self.canvas_xmax - self.canvas_xmin
        self.canvas_yspread = self.canvas_ymax - self.canvas_ymin
        self.xspread = self.xmax - self.xmin
        self.yspread = self.ymax - self.ymin
        self.xspread_ratio = float(self.canvas_xspread) / self.xspread
        self.yspread_ratio = float(self.canvas_yspread) / self.yspread

        self.gsize = image_size
        self.gsize2 = image_size/2

        _, _, mask_images = anchors_from_image("mask/{}_mask{}.png".format(mask_name, image_size), image_size=(image_size, image_size))
        # _, _, mask_images = anchors_from_image("mask/rounded_mask{}.png".format(gsize), image_size=(gsize, gsize))
        # _, _, mask_images = anchors_from_image("mask/hexagons/hex1_{}_blur.png".format(gsize), image_size=(gsize, gsize))
        self.mask = mask_images[0][0]

    # To map
    # [A, B] --> [a, b]
    # use this formula
    # (val - A)*(b-a)/(B-A) + a
    # A,B is virtual
    # a,b is canvas
    def map_to_canvas(self, x, y):
        new_x = int((x - self.xmin) * self.xspread_ratio + self.canvas_xmin)
        new_y = int((y - self.ymin) * self.yspread_ratio + self.canvas_ymin)
        return new_x, new_y

    def place_square(self, x, y):
        square = np.zeros((channels, self.gsize, self.gsize))
        square.fill(1)
        cx, cy = self.map_to_canvas(x, y)
        self.pixels[:, (cy-self.gsize2):(cy+self.gsize2), (cx-self.gsize2):(cx+self.gsize2)] = square

    def check_bounds(self, cx, cy):
        border = self.gsize2
        if (cx < self.canvas_xmin + border) or (cy < self.canvas_ymin + border) or (cx >= self.canvas_xmax - border) or (cy >= self.canvas_ymax - border):
            return False
        return True

    def place_image(self, im, x, y, additive=False):
        square = im
        cx, cy = self.map_to_canvas(x, y)
        if self.check_bounds(cx, cy):
            if additive:
                self.pixels[:, (cy-self.gsize2):(cy+self.gsize2), (cx-self.gsize2):(cx+self.gsize2)] = \
                    additive_composite(im, self.mask, self.pixels[:, (cy-self.gsize2):(cy+self.gsize2), (cx-self.gsize2):(cx+self.gsize2)])
            else:
                self.pixels[:, (cy-self.gsize2):(cy+self.gsize2), (cx-self.gsize2):(cx+self.gsize2)] = \
                    alpha_composite(im, self.mask, self.pixels[:, (cy-self.gsize2):(cy+self.gsize2), (cx-self.gsize2):(cx+self.gsize2)])

    def save(self, save_path):
        out = np.dstack(self.pixels)
        out = (255 * out).astype(np.uint8)
        img = Image.fromarray(out)
        img.save(save_path)

def apply_anchor_offsets(anchor, offsets, a, b, a_indices_str, b_indices_str):
    sa = 2.0 * (a - 0.5)
    sb = 2.0 * (b - 0.5)
    dim = len(anchor)
    a_offset = offset_from_string(a_indices_str, offsets, dim)
    b_offset = offset_from_string(b_indices_str, offsets, dim)
    new_anchor = anchor + sa * a_offset + sb * b_offset
    # print(a, a*a_offset)
    return new_anchor

def make_mask_layout(height, width, radius):
    I = np.zeros((height, width)).astype(np.uint8)
    center = np.array([width/2.0, height/2.0])
    for y in range(height):
        for x in range(width):
            x_off = 0.5
            if y%2 == 0:
                x_off = 1.0
            pos = np.array([x+x_off, y+0.5])
            length = np.linalg.norm(pos - center)
            if length < radius:
                I[y][x] = 255
    return I

def main(cliargs):
    parser = argparse.ArgumentParser(description="Plot model samples")
    parser.add_argument("--interface", dest='model_class', type=str,
                        default="plat.interface.discgen.DiscGenModel", help="class encapsulating model")
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument("--width", type=int, default=512,
                        help="width of canvas to render in pixels")
    parser.add_argument("--height", type=int, default=512,
                        help="height of canvas to render in pixels")
    parser.add_argument("--rows", type=int, default=3,
                        help="number of rows of anchors")
    parser.add_argument("--cols", type=int, default=3,
                        help="number of columns of anchors")
    parser.add_argument("--xmin", type=int, default=0,
                        help="min x in virtual space")
    parser.add_argument("--xmax", type=int, default=100,
                        help="max x in virtual space")
    parser.add_argument("--ymin", type=int, default=0,
                        help="min y in virtual space")
    parser.add_argument("--ymax", type=int, default=100,
                        help="max y in virtual space")
    parser.add_argument("--save-path", type=str, default="out.png",
                        help="where to save the generated samples")
    parser.add_argument("--seed", type=int,
                        default=None, help="Optional random seed")
    parser.add_argument('--anchor-image', dest='anchor_image', default=None,
                        help="use image as source of anchors")
    parser.add_argument('--anchor-mine', dest='anchor_mine', default=None,
                        help="use image as single source of mine coordinates")    
    parser.add_argument('--random-mine', dest='random_mine', default=False, action='store_true',
                        help="use random sampling as source of mine coordinates")
    parser.add_argument('--additive', dest='additive', default=False, action='store_true',
                        help="use additive compositing")
    parser.add_argument('--mask-name', dest='mask_name', default="rounded",
                        help="prefix name for alpha mask to use (full/rounded/hex")
    parser.add_argument('--mask-layout', dest='mask_layout', default=None,
                        help="use image as source of mine grid points")
    parser.add_argument('--mask-scale', dest='mask_scale', default=1.0, type=float,
                        help="Scale mask layout (squeeze)")
    parser.add_argument('--mask-width', dest='mask_width', type=int, default=15,
                        help="width for computed mask")
    parser.add_argument('--mask-height', dest='mask_height', type=int, default=15,
                        help="height for computed mask")
    parser.add_argument('--mask-radius', dest='mask_radius', default=None, type=float,
                        help="radius for computed mask")
    parser.add_argument('--layout', dest='layout', default=None,
                        help="layout json file")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=100,
                        help="number of images to decode at once")
    parser.add_argument('--passthrough', dest='passthrough', default=False, action='store_true',
                        help="Use originals instead of reconstructions")
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")
    parser.add_argument('--anchor-offset-a', dest='anchor_offset_a', default="42", type=str,
                        help="which indices to combine for offset a")
    parser.add_argument('--anchor-offset-b', dest='anchor_offset_b', default="31", type=str,
                        help="which indices to combine for offset b")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    args = parser.parse_args(cliargs)

    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)

    anchor_images = None
    if args.anchor_image is not None:
        _, _, anchor_images = anchors_from_image(args.anchor_image, image_size=(args.image_size, args.image_size))
    elif args.anchor_mine is not None:
        _, _, anchor_images = anchors_from_image(args.anchor_mine, image_size=(args.image_size, args.image_size))

    anchors = None
    if not args.passthrough:
        model_class_parts = args.model_class.split(".")
        model_class_name = model_class_parts[-1]
        model_module_name = ".".join(model_class_parts[:-1])
        print("Loading {} interface from {}".format(model_class_name, model_module_name))        
        ModelClass = getattr(importlib.import_module(model_module_name), model_class_name)
        print("Loading model from {}".format(args.model))
        dmodel = ModelClass(filename=args.model)

        if anchor_images is not None:
            anchors = dmodel.encode_images(anchor_images)

    if anchors is None:
        anchors = np.random.normal(loc=0, scale=1, size=(args.cols * args.rows, 100))

    anchor_offsets = None
    if args.anchor_offset is not None:
        # compute anchors as offsets from existing anchor
        anchor_offsets = get_json_vectors(args.anchor_offset)

    canvas = Canvas(args.width, args.height, args.xmin, args.xmax, args.ymin, args.ymax, args.mask_name, args.image_size)
    workq = []

    do_hex = True

    if args.layout:
        with open(args.layout) as json_file:
            layout_data = json.load(json_file)
        xy = np.array(layout_data["xy"])
        roots = layout_data["r"]
        for i, pair in enumerate(xy):
            x = pair[0] * canvas.xmax
            y = pair[1] * canvas.ymax
            a = pair[0]
            b = pair[1]
            r = roots[i]
            if args.passthrough:
                output_image = anchor_images[r]
                canvas.place_image(output_image, x, y, args.additive)
            else:
                if args.anchor_mine is not None or args.random_mine:
                    z = create_mine_canvas(args.rows, args.cols, b, a, anchors)
                elif anchor_offsets is not None:
                    z = apply_anchor_offsets(anchors[r], anchor_offsets, a, b, args.anchor_offset_a, args.anchor_offset_b)
                else:
                    z = anchors[r]
                workq.append({
                        "z": z,
                        "x": x,
                        "y": y
                    })

    elif args.mask_layout or args.mask_radius:
        if args.mask_layout:
            rawim = imread(args.mask_layout);
            if len(rawim.shape) == 2:
                im_height, im_width = rawim.shape
                mask_layout = rawim
            else:
                im_height, im_width, _ = rawim.shape
                mask_layout = rawim[:,:,0]
        else:
            im_height, im_width = args.mask_height, args.mask_width
            mask_layout = make_mask_layout(im_height, im_width, args.mask_radius)
        for xpos in range(im_width):
            for ypos in range(im_height):
                a = float(xpos) / (im_width - 1)
                if do_hex and ypos % 2 == 0:
                    a = a + 0.5 / (im_width - 1)
                x = args.mask_scale * canvas.xmax * a
                b = float(ypos) / (im_height - 1)
                y = args.mask_scale * canvas.ymax * b
                if not mask_layout[ypos][xpos] > 128:
                    pass
                elif args.passthrough:
                    output_image = anchor_images[0]
                    canvas.place_image(output_image, x, y, args.additive)
                else:
                    if len(anchors) == 1 or anchor_offsets is not None:
                        z = apply_anchor_offsets(anchors[0], anchor_offsets, a, b, args.anchor_offset_a, args.anchor_offset_b)
                    else:
                        z = create_mine_canvas(args.rows, args.cols, b, a, anchors)
                    workq.append({
                            "z": z,
                            "x": x,
                            "y": y
                        })

    while(len(workq) > 0):
        curq = workq[:args.batch_size]
        workq = workq[args.batch_size:]
        latents = [e["z"] for e in curq]
        images = dmodel.sample_at(np.array(latents))
        for i in range(len(curq)):
            canvas.place_image(images[i], curq[i]["x"], curq[i]["y"], args.additive)

    canvas.save(args.save_path)

if __name__ == '__main__':
    main(sys.argv[1:])
