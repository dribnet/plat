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

def random_layout(w, h):
    p = []
    r = []
    xy = []
    s = []
    lookup = np.zeros((h, w))
    for n in range(0):
        x = np.random.random_integers(0, w - 4)
        y = np.random.random_integers(0, h - 4)
        lookup_slice = [slice(y, y+4), slice(x, x+4)]
        ls = lookup[lookup_slice]
        if np.max(ls) == 0:
            lookup[lookup_slice] = 1
            p.append(-1)
            r.append(0)
            sx = (x + 2.0) / w
            sy = (y + 2.0) / h
            xy.append([sx, sy])
            s.append(1)
    for n in range(10000):
        x = np.random.random_integers(0, w - 2)
        y = np.random.random_integers(0, h - 2)
        lookup_slice = [slice(y, y+2), slice(x, x+2)]
        ls = lookup[lookup_slice]
        if np.max(ls) == 0:
            lookup[lookup_slice] = 1
            p.append(-1)
            r.append(0)
            sx = (x + 1.0) / w
            sy = (y + 1.0) / h
            xy.append([sx, sy])
            s.append(1)
    for x in range(w):
        for y in range(h):
            lookup_slice = [slice(y, y+1), slice(x, x+1)]
            ls = lookup[lookup_slice]
            if np.max(ls) == 0:
                lookup[lookup_slice] = 1
                p.append(-1)
                r.append(0)
                sx = (x + 0.5) / w
                sy = (y + 0.5) / h
                xy.append([sx, sy])
                s.append(0)
    return {"p": p,
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
    parser.add_argument("--outfile", dest='outfile', type=str, default="outfile.json",
                        help="where to save the generated samples")
    args = parser.parse_args(args)
    if args.rows is None or args.cols is None:
        d = generate_layout_dict()
    else:
        d = random_layout(args.cols, args.rows)
    save_dict(d, args.outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate layout")
    gengrid(parser, None, sys.argv[1:])
