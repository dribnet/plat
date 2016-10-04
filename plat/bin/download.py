import argparse
import sys
from plat import zoo

def download(parser, context, args):
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="Name of model in plat zoo")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Name of dataset")
    args = parser.parse_args(args)

    if args.model is not None:
        zoo.download_model(args.model)
        sys.exit(0)

    print("Currently only model supported. Try --model <modelname>")
    sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot model samples")
    download(parser, None, sys.argv[1:])