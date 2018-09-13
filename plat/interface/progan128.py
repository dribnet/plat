import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class Model:
    def __init__(self, filename=None, model=None):
        """
        """
        self.decoder = hub.Module("https://tfhub.dev/google/progan-128/1")
        self.session = None

    def get_session(self):
        if self.session is None:
          self.session = tf.Session()
          self.session.run(tf.global_variables_initializer())
        return self.session

    def encode_images(self, images):
        """
        Encode images x => z

        images is an n x 3 x s x s numpy array where:
          n = number of images
          3 = R G B channels
          s = size of image (eg: 64, 128, etc)
          pixels values for each channel are encoded [0,1]

        returns an n x z numpy array where:
          n = len(images)
          z = dimension of latent space
        """
        # todo
        pass

    def get_zdim(self):
        """
        Returns the integer dimension of the latent z space
        """
        return self.decoder.get_input_info_dict()['latent_vector'].get_shape()[1]

    def sample_at(self, z):
        """
        Decode images z => x

        z is an n x z numpy array where:
          n = len(images)
          z = dimension of latent space

        return images as an n x 3 x s x s numpy array where:
          n = number of images
          3 = R G B channels
          s = size of image (eg: 64, 128, etc)
          pixels values for each channel are encoded [0,1]
        """
        decoded_fn = self.decoder(z)
        decoded = self.get_session().run(decoded_fn)
        # print(decoded.shape)
        channel_first = np.rollaxis(decoded, 3, 1)  
        # print(channel_first.shape)
        return channel_first
