### This is a copy of ali/interface.py from the ALI project

import theano
from theano import tensor
from blocks.serialization import load
from blocks.utils import shared_floatx

class AliModel:
    def __init__(self, filename=None, model=None):
        if model is not None:
            self.model, = model.top_bricks
        else:
            with open(filename, 'rb') as src:
                main_loop = load(src)
                self.model, = main_loop.model.top_bricks

    def encode_images(self, images):
        x = tensor.tensor4('features')
        latents = theano.function([x], self.model.encode(x))(images)
        num_samples, z_dim, _x, _y = latents.shape
        # print("SO", num_samples, z_dim, _x, _y)
        # print("AND", images.shape)
        return latents.reshape(num_samples, z_dim)

    def get_zdim(self):
        input_shape = self.model.encoder.get_dim('output')
        return input_shape[0]

    def sample_at(self, z):
        num_samples, z_dim = z.shape
        sz = shared_floatx(z.reshape(num_samples, z_dim, 1, 1))
        x = self.model.sample(sz)
        samples = theano.function([], x)()
        return samples
