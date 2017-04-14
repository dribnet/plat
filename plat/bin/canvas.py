#!/usr/bin/env python

"""Plots model samples."""
import argparse

import numpy as np
import random
import sys
import json
from scipy.misc import imread, imsave, imresize
import os

from plat.utils import anchors_from_image, get_json_vectors, offset_from_string
from plat.canvas_layout import create_mine_canvas
import plat.sampling

from PIL import Image
import importlib
from plat import zoo

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

    src_shape = src.shape
    if src_shape[1] == 1 and src_shape[2] == 1:
        return out

    alpha = np.index_exp[3:, :, :]
    rgb = np.index_exp[:3, :, :]
    epsilon = 0.001
    if src_mask is not None:
        src_a = np.maximum(src_mask, epsilon)
    else:
        src_a = 1.0
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
    if src_mask is not None:
        out[alpha] = np.maximum(src_mask,dst[alpha])
    else:
        out[alpha] = 1.0
    out[rgb] = np.maximum(src[rgb],dst[rgb])
    np.clip(out,0,1.0)
    return out

# gsize = 64
# gsize2 = gsize/2

class Canvas:
    """Simple Canvas Thingy"""

    def __init__(self, width, height, xmin, xmax, ymin, ymax, mask_name, image_size, do_check_bounds, init_black=False):
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

        self.do_check_bounds = do_check_bounds

        self.canvas_xspread = self.canvas_xmax - self.canvas_xmin
        self.canvas_yspread = self.canvas_ymax - self.canvas_ymin
        self.xspread = self.xmax - self.xmin
        self.yspread = self.ymax - self.ymin
        self.xspread_ratio = float(self.canvas_xspread) / self.xspread
        self.yspread_ratio = float(self.canvas_yspread) / self.yspread

        self.gsize = image_size
        self.gsize2 = image_size/2
        self.gsize4 = image_size/4

        if mask_name is not None:
            _, _, mask_images = anchors_from_image("mask/{}_mask{}.png".format(mask_name, image_size), image_size=(image_size, image_size))
            # _, _, mask_images = anchors_from_image("mask/rounded_mask{}.png".format(gsize), image_size=(gsize, gsize))
            # _, _, mask_images = anchors_from_image("mask/hexagons/hex1_{}_blur.png".format(gsize), image_size=(gsize, gsize))
            self.mask = mask_images[0][0]
        else:
            self.mask = None

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

    def set_background(self, fname):
        rawim = imread(fname);
        h, w, c = rawim.shape
        if h > self.canvas_ymax:
            h = self.canvas_ymax
        if w > self.canvas_xmax:
            w = self.canvas_xmax
        s_im = np.asarray([rawim[:,:,0]/255.0, rawim[:,:,1]/255.0, rawim[:,:,2]/255.0])            
        self.pixels[0:3, 0:h, 0:w] = s_im[:,0:h,0:w]
        self.pixels[3, 0:h, 0:w] = 1

    def place_square(self, x, y, s):
        square = np.zeros((channels, self.gsize, self.gsize))
        square.fill(1)
        cx, cy = self.map_to_canvas(x, y)
        self.pixels[:, (cy-self.gsize2):(cy+self.gsize2), (cx-self.gsize2):(cx+self.gsize2)] = square

    def check_bounds(self, cx, cy, border):
        if not self.do_check_bounds:
            return True
        if (cx < self.canvas_xmin) or (cy < self.canvas_ymin) or (cx > self.canvas_xmax - border) or (cy > self.canvas_ymax - border):
            return False
        return True

    def get_anchor(self, x, y, im_size):
        cx, cy = self.map_to_canvas(x, y)
        im_size = int(im_size)
        border = self.gsize2
        anchor_im = self.pixels[0:3, cy-border:cy+border, cx-border:cx+border]
        anchor = np.zeros([3, im_size, im_size])
        tc, th, tw = anchor_im.shape
        anchor[:, 0:th, 0:tw] = anchor_im
        return anchor.astype('float32')

    def place_image(self, im, x, y, additive=False, scale=None):
        # print("place_image {} at {}, {} with scale {}".format(im.shape, x, y, scale))
        if scale is not None:
            border = int(scale)
            slices = [
                slice(0, 4),
                slice(y, y+border),
                slice(x, x+border)
            ]
            out_stack = np.dstack(im)
            out_stack = (255 * out_stack).astype(np.uint8)
            rawim = imresize(out_stack, (border, border))
            s_im = np.asarray([rawim[:,:,0]/255.0, rawim[:,:,1]/255.0, rawim[:,:,2]/255.0])
        else:
            cx, cy = self.map_to_canvas(x, y)
            border = self.gsize2
            slices = [
                slice(0, 4),
                slice(cy-border, cy+border),
                slice(cx-border, cx+border)
            ]
            s_im = im

        if not self.check_bounds(x, y, border):
            return
        if additive:
            self.pixels[slices] = additive_composite(s_im, self.mask, self.pixels[slices])
        else:
            self.pixels[slices] = alpha_composite(s_im, self.mask, self.pixels[slices])

    def save(self, save_path):
        print("Preparing image file {}".format(save_path))
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

def canvas(parser, context, args):
    parser = argparse.ArgumentParser(description="Plot model samples")
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="name of model in plat zoo")
    parser.add_argument("--model-file", dest='model_file', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument("--model-type", dest='model_type', type=str, default=None,
                        help="the type of model (usually inferred from filename)")
    parser.add_argument("--model-interface", dest='model_interface', type=str,
                        default=None,
                        help="class interface for model (usually inferred from model-type)")
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
    parser.add_argument("--outfile", dest='save_path', type=str, default="canvas_%DATE%_%MODEL%_%SEQ%.png",
                        help="where to save the generated samples")
    parser.add_argument("--seed", type=int,
                        default=None, help="Optional random seed")
    parser.add_argument('--do-check-bounds', dest='do_check_bounds', default=False, action='store_true',
                        help="clip to drawing bounds")
    parser.add_argument('--background-image', dest='background_image', default=None,
                        help="use image initial background")
    parser.add_argument('--anchor-image', dest='anchor_image', default=None,
                        help="use image as source of anchors")
    parser.add_argument('--anchor-mine', dest='anchor_mine', default=None,
                        help="use image as single source of mine coordinates")    
    parser.add_argument('--anchor-canvas', dest='anchor_canvas', default=False, action='store_true',
                        help="anchor image from canvas")
    parser.add_argument('--random-mine', dest='random_mine', default=False, action='store_true',
                        help="use random sampling as source of mine coordinates")
    parser.add_argument('--additive', dest='additive', default=False, action='store_true',
                        help="use additive compositing")
    parser.add_argument('--mask-name', dest='mask_name', default=None,
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
    parser.add_argument('--layout-scale', dest='layout_scale', default=1, type=int,
                        help="Scale layout")
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
    parser.add_argument('--global-offset', dest='global_offset', default=None,
                        help="use json file as source of global offsets")
    parser.add_argument('--global-indices', dest='global_indices', default=None, type=str,
                        help="offset indices to apply globally")
    parser.add_argument('--global-scale', dest='global_scale', default=1.0, type=float,
                        help="scaling factor for global offset")
    args = parser.parse_args(args)

    template_dict = {}
    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)

    global_offset = None
    if args.global_offset is not None:
        offsets = get_json_vectors(args.global_offset)
        global_offset = plat.sampling.get_global_offset(offsets, args.global_indices, args.global_scale)

    anchor_images = None
    if args.anchor_image is not None:
        _, _, anchor_images = anchors_from_image(args.anchor_image, image_size=(args.image_size, args.image_size))
    elif args.anchor_mine is not None:
        _, _, anchor_images = anchors_from_image(args.anchor_mine, image_size=(args.image_size, args.image_size))
        basename = os.path.basename(args.anchor_mine)
        template_dict["BASENAME"] = os.path.splitext(basename)[0]

    anchors = None
    if not args.passthrough:
        dmodel = zoo.load_model(args.model, args.model_file, args.model_type, args.model_interface)

        workq = anchor_images[:]
        anchors_list = []
        while(len(workq) > 0):
            print("Processing {} anchors".format(args.batch_size))
            curq = workq[:args.batch_size]
            workq = workq[args.batch_size:]
            cur_anchors = dmodel.encode_images(curq)
            for c in cur_anchors:
                anchors_list.append(c)
        anchors = np.asarray(anchors_list)

    if anchors is None:
        anchors = np.random.normal(loc=0, scale=1, size=(args.cols * args.rows, 100))

    anchor_offsets = None
    if args.anchor_offset is not None:
        # compute anchors as offsets from existing anchor
        anchor_offsets = get_json_vectors(args.anchor_offset)

    canvas = Canvas(args.width, args.height, args.xmin, args.xmax, args.ymin, args.ymax, args.mask_name, args.image_size, args.do_check_bounds)
    if args.background_image is not None:
        canvas.set_background(args.background_image)
    workq = []

    do_hex = True

    if args.layout:
        with open(args.layout) as json_file:
            layout_data = json.load(json_file)
        xy = np.array(layout_data["xy"])
        grid_size = layout_data["size"]
        roots = layout_data["r"]
        if "s" in layout_data:
            s = layout_data["s"]
        else:
            s = None
        for i, pair in enumerate(xy):
            x = pair[0] * canvas.canvas_xmax / grid_size[0]
            y = pair[1] * canvas.canvas_ymax / grid_size[1]
            a = (pair[0] + 0.5 * s[i]) / float(grid_size[0])
            b = (pair[1] + 0.5 * s[i]) / float(grid_size[1])
            r = roots[i]
            if s is None:
                scale = args.layout_scale
            else:
                scale = s[i] * args.layout_scale
            # print("Placing {} at {}, {} because {},{} and {}, {}".format(scale, x, y, canvas.canvas_xmax, canvas.canvas_ymax, grid_size[0], grid_size[1]))
            if args.passthrough:
                output_image = anchor_images[r]
                canvas.place_image(output_image, x, y, args.additive, scale=scale)
            else:
                if args.anchor_mine is not None or args.random_mine:
                    z = create_mine_canvas(args.rows, args.cols, b, a, anchors)
                elif anchor_offsets is not None:
                    z = apply_anchor_offsets(anchors[r], anchor_offsets, a, b, args.anchor_offset_a, args.anchor_offset_b)
                else:
                    z = anchors[r]

                if global_offset is not None:
                    z = z + global_offset
                # print("Storing {},{} with {}".format(x, y, len(z)))
                workq.append({
                        "z": z,
                        "x": x,
                        "y": y,
                        "s": scale
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
                    if args.anchor_canvas:
                        cur_anchor_image = canvas.get_anchor(x, y, args.image_size)
                    else:
                        cur_anchor_image = anchor_images[0]
                    canvas.place_image(cur_anchor_image, x, y, args.additive, None)
                else:
                    if args.anchor_canvas:
                        cur_anchor_image = canvas.get_anchor(x, y, args.image_size)
                        zs = dmodel.encode_images([cur_anchor_image])
                        z = zs[0]
                    elif len(anchors) == 1 or anchor_offsets is not None:
                        z = apply_anchor_offsets(anchors[0], anchor_offsets, a, b, args.anchor_offset_a, args.anchor_offset_b)
                    else:
                        z = create_mine_canvas(args.rows, args.cols, b, a, anchors)

                    if global_offset is not None:
                        z = z + global_offset

                    workq.append({
                            "z": z,
                            "x": x,
                            "y": y,
                            "s": None
                        })

    while(len(workq) > 0):
        curq = workq[:args.batch_size]
        workq = workq[args.batch_size:]
        latents = [e["z"] for e in curq]
        images = dmodel.sample_at(np.array(latents))
        for i in range(len(curq)):
            # print("Placing {},{} with {}".format(curq[i]["x"], curq[i]["y"], len(latents)))
            canvas.place_image(images[i], curq[i]["x"], curq[i]["y"], args.additive, scale=curq[i]["s"])
            # print("Placed")

    template_dict["SIZE"] = args.image_size
    outfile = plat.sampling.emit_filename(args.save_path, template_dict, args);
    canvas.save(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot model samples on canvas")
    canvas(parser, None, sys.argv[1:])
