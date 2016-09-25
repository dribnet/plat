import argparse
from pydoc import locate
from lib import utils
import os

class IganModel:
    def __init__(self, filename=None, model=None):
        if model is not None:
            # error
            return
        # figure out name,type from filename
        model_name, model_type = os.path.basename(filename).split(".")
        # initialize model and constrained optimization problem
        model_class = locate('model_def.%s' % model_type)
        self.model_G = model_class.Model(model_name=model_name, model_file=filename)

    def encode_images(self, images):
        # error?
        return

    def get_zdim(self):
        # ?
        return 100

    def sample_at(self, z):
        samples = self.model_G.gen_samples(z0=z, n=1, batch_size=1, nz=100,  use_transform=False)
        samples = (samples + 1.0) / 2.0
        return samples
