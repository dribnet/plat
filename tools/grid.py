import argparse
import json
import sys
import numpy as np

def generate_layout_dict():
    p = []
    r = []
    xy = []
    s = []
    w = 10
    h = 6
    for x in range(w):
        for y in range(h):
            p.append(-1)
            r.append(0)
            sx = (x + 0.5) / w
            sy = (y + 0.5) / h
            xy.append([sx, sy])
            s.append(1)
    return {"p": p,
        "r": r,
        "xy": xy,
        "s": s}

def random_layout(w, h, limits):
    p = []
    r = []
    xy = []
    s = []
    lookup = np.zeros((h, w))
    pairs = [[x, y] for x in range(w) for y in range(h)]
    num_pairs = len(pairs)
    for i in range(len(limits)):
        num_spaces = len(limits) - i
        shuffle = np.random.permutation(pairs)
        num_found = 0
        cur_pair_index = 0
        while cur_pair_index < num_pairs and num_found < limits[i]:
            x, y = shuffle[cur_pair_index]
            cur_pair_index = cur_pair_index + 1
            if x <= w - num_spaces and y <= h - num_spaces:
                lookup_slice = [slice(y, y+num_spaces), slice(x, x+num_spaces)]
                ls = lookup[lookup_slice]
                if np.max(ls) == 0:
                    num_found = num_found + 1
                    lookup[lookup_slice] = 1
                    p.append(-1)
                    r.append(0)
                    xy.append([x, y])
                    s.append(num_spaces)
    return {
        "size": [w, h],
        "p": p,
        "r": r,
        "xy": xy,
        "s": s}

def save_dict(d, filename):
    with open(filename, 'w') as outfile:
        json.dump(d, outfile)

def gengrid(parser, context, args):
    parser = argparse.ArgumentParser(description="Plot model samples")
    parser.add_argument("--rows", dest='rows', type=int, default=None,
                        help="number of rows a finest level")
    parser.add_argument("--cols", dest='cols', type=int, default=None,
                        help="number of cols a finest level")
    parser.add_argument("--limits", dest='limits', default=None,
                        help="comma separated list of integer limits for octaves")
    parser.add_argument("--outfile", dest='outfile', type=str, default="outfile.json",
                        help="where to save the generated samples")
    args = parser.parse_args(args)
    if args.limits is None:
        limits = []
    else:
        limits = map(int, args.limits.split(","))
    if args.rows is None or args.cols is None:
        d = generate_layout_dict()
    else:
        d = random_layout(args.cols, args.rows, limits)
    save_dict(d, args.outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate layout")
    gengrid(parser, None, sys.argv[1:])
