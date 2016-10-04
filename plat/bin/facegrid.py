"""Does facegrid stuff."""
import argparse

import numpy as np
import os
import json
import sys

from PIL import Image
from annoy import AnnoyIndex
from sklearn.manifold import TSNE
from fuel_helper import get_anchor_images
from plat.utils import anchors_from_image, json_list_to_array

def build_annoy_index(encoded, outfile):
    input_shape = encoded.shape
    f = input_shape[1]
    t = AnnoyIndex(f, metric='angular')  # Length of item vector that will be indexed
    for i,v in enumerate(encoded):
        t.add_item(i, v)

    t.build(100) # 10 trees
    if outfile is not None:
        t.save(outfile)

    return t

def load_annoy_index(infile, z_dim):
    t = AnnoyIndex(z_dim)
    t.load(infile) # super fast, will just mmap the file
    return t

def neighbors_to_grid(neighbors, imdata, gsize, with_center=False):
    canvas = np.zeros((gsize*3, gsize*5, 3)).astype(np.uint8)
    if with_center:
        offsets = [ [0, 0] ]
    else:
        offsets = []
    offsets += [
        [-1, 0],
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
        [0, -2],
        [0, 2],
        [-1, -2],
        [1, -2],
        [-1, 2],
        [1, 2]
    ]
    cx = gsize*2
    cy = gsize*1
    for i, offset in enumerate(offsets):
        n = neighbors[i]
        im = np.dstack(imdata[n]).astype(np.uint8)
        offy = cy + gsize*offset[0]
        offx = cx + gsize*offset[1]
        canvas[offy:offy+gsize, offx:offx+gsize, :] = im
    return Image.fromarray(canvas)

def debug_save_plot(xy, filename):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_bgcolor('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.autoscale_view(True,True,True)
    ax.invert_yaxis()
    ax.scatter(xy[:,0],xy[:,1],  edgecolors='none',marker='s',s=7.5)  # , c = vectors[:,:3]
    plt.savefig(filename)

def neighbors_to_rfgrid(neighbors, encoded, imdata, gsize, gridw, gridh):
    canvas = np.zeros((gsize*gridh, gsize*gridw, 3)).astype(np.uint8)

    vectors_list = []
    for n in neighbors:
        vectors_list.append(encoded[n])
    vectors = np.array(vectors_list)

    RS = 20150101
    # xy = bh_sne(vectors, perplexity=4., theta=0)
    xy = TSNE(init='pca', learning_rate=100, random_state=RS, method='exact', perplexity=4).fit_transform(vectors)

    # debug_save_plot(xy, "plot.png")

    from rasterfairy import rasterfairy
    grid_xy, quadrants = rasterfairy.transformPointCloud2D(xy,target=(gridw,gridh))
    indices = []
    for i in range(gridw * gridh):
        indices.append(quadrants[i]["indices"][0])

    i = 0
    for cur_y in range(gridh):
        for cur_x in range(gridw):
            cur_index = indices[i]
            n = neighbors[cur_index]
            im = np.dstack(imdata[n]).astype(np.uint8)
            offy = gsize * cur_y
            offx = gsize * cur_x
            canvas[offy:offy+gsize, offx:offx+gsize, :] = im
            i = i + 1

    return Image.fromarray(canvas)

def main(cliargs):
    parser = argparse.ArgumentParser(description="Plot model samples")
    parser.add_argument("--model-module", dest='model_module', type=str,
                        default="utils.interface", help="module encapsulating model")
    parser.add_argument("--model-class", dest='model_class', type=str,
                        default="DiscGenModel", help="class encapsulating model")
    parser.add_argument('--build-annoy', dest='build_annoy',
                        default=False, action='store_true')
    parser.add_argument("--jsons", type=str, default=None,
                        help="Comma separated list of json arrays")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Source dataset.")
    parser.add_argument('--dataset-image', dest='dataset_image', default=None,
                        help="use image as source dataset")
    parser.add_argument("--dataset-offset", dest='dataset_offset', type=int, default=0,
                        help="dataset offset to skip")
    parser.add_argument("--dataset-max", type=int, default=None,
                        help="Source dataset.")
    parser.add_argument('--seeds-image', dest='seeds_image', default=None,
                        help="image source of seeds")
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="model for encoding when seeds-images is enabled")
    parser.add_argument('--annoy-index', dest='annoy_index', default=None,
                        help="Annoy index.")
    parser.add_argument('--split', dest='split', default="all",
                        help="Which split to use from the dataset (train/nontrain/valid/test/any).")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument("--z-dim", dest='z_dim', type=int, default=100,
                        help="z dimension")
    parser.add_argument('--outdir', dest='outdir', default="neighborgrids",
                        help="Output dir for neighborgrids.")
    parser.add_argument('--outfile', dest='outfile', default="index_{:03d}.png",
                        help="Output file (template) for neighborgrids.")
    parser.add_argument("--outgrid-width", dest='outgrid_width', type=int, default=5,
                        help="width of output grid")
    parser.add_argument("--outgrid-height", dest='outgrid_height', type=int, default=3,
                        help="height of output grid")
    parser.add_argument('--range', dest='range', default="0",
                        help="Range of indexes to run.")
    args = parser.parse_args(cliargs)

    encoded = json_list_to_array(args.jsons)
    # print(encoded.shape)
    if args.build_annoy:
        aindex = build_annoy_index(encoded, args.annoy_index)
        sys.exit(0)

    # open annoy index and spit out some neighborgrids
    aindex = load_annoy_index(args.annoy_index, args.z_dim)
    if args.dataset is not None:
        anchor_images = get_anchor_images(args.dataset, args.split, offset=args.dataset_offset, numanchors=args.dataset_max, unit_scale=False)
        image_size = anchor_images.shape[2]
    # dataset_image requires image_size
    if args.dataset_image is not None:
        image_size = args.image_size
        _, _, anchor_images = anchors_from_image(args.dataset_image, image_size=(image_size, image_size), unit_scale=False)
        if args.dataset_offset > 0:
            anchor_images = anchor_images[args.dataset_offset:]
        if args.dataset_max is not None:
            anchor_images = anchor_images[:args.dataset_max]


    r = map(int, args.range.split(","))

    core_dataset_size = len(anchor_images)
    if(len(encoded) != core_dataset_size):
        print("Warning: {} vectors and {} images".format(len(encoded), core_dataset_size))
    if args.seeds_image is not None:
        image_size = args.image_size
        _, _, extra_images = anchors_from_image(args.seeds_image, image_size=(image_size, image_size), unit_scale=False)
        net_inputs = (extra_images / 255.0).astype('float32')

        print('Loading saved model')
        ModelClass = getattr(importlib.import_module(args.model_module), args.model_class)
        dmodel = ModelClass(filename=args.model)

        image_vectors = dmodel.encode_images(net_inputs)
        num_extras = len(extra_images)
        encoded = np.concatenate((encoded, image_vectors), axis=0)
        anchor_images = np.concatenate((anchor_images, extra_images), axis=0)
        # for now, override given range
        r = [core_dataset_size, core_dataset_size + num_extras]

    print anchor_images.shape

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if len(r) == 1:
        r = [r[0], r[0]+1]
    num_out_cells = args.outgrid_width * args.outgrid_height
    for i in range(r[0], r[1]):
        if i < core_dataset_size:
            neighbors = aindex.get_nns_by_item(i, num_out_cells, include_distances=True) # will find the 20 nearest neighbors
            file_num = i
        else:
            neighbors = aindex.get_nns_by_vector(encoded[i], num_out_cells-1, include_distances=True) # will find the 20 nearest neighbors
            neighbors[0].append(i)
            neighbors[1].append(0)
            file_num = i - core_dataset_size

        g = neighbors_to_rfgrid(neighbors[0], encoded, anchor_images, image_size, args.outgrid_width, args.outgrid_height)
        out_template = "{}/{}".format(args.outdir, args.outfile)
        g.save(out_template.format(file_num))

if __name__ == '__main__':
    main(sys.argv[1:])
