import argparse
import json
import sys

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

def save_dict(d, filename):
    with open(filename, 'w') as outfile:
        json.dump(d, outfile)

def gengrid(parser, context, args):
    parser = argparse.ArgumentParser(description="Plot model samples")
    parser.add_argument("--outfile", dest='outfile', type=str, default="outfile.json",
                        help="where to save the generated samples")
    args = parser.parse_args(args)
    d = generate_layout_dict()
    save_dict(d, args.outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate layout")
    gengrid(parser, None, sys.argv[1:])
