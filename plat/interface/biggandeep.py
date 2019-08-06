import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import random
import os

class Model:
    def __init__(self, filename=None, model=None):
        """
        """
        if 'IMAGENET_INDEX' in os.environ:
          self.y_index = int(os.getenv('IMAGENET_INDEX'))
          print("imagenet index set to {}".format(self.y_index))
        else:
          self.y_index = random.randint(0, 999)
          print("imagenet index randomly set to {}".format(self.y_index))
        self.decoder = hub.Module("https://tfhub.dev/deepmind/biggan-deep-256/1")
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
        # return self.decoder.get_input_info_dict()['latent_vector'].get_shape()[1]
        return 128

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
        z_len = len(z)
        truncation = 0.5  # scalar truncation value in [0.0, 1.0]
        # z = truncation * tf.random.truncated_normal([1, 128])  # noise sample
        y_index = [self.y_index] * z_len # class index
        # y_index = tf.random.uniform([1], maxval=1000, dtype=tf.int32)
        y = tf.one_hot(y_index, 1000) 

        decoded_fn = self.decoder(dict(y=y, z=z, truncation=0.5))
        # decoded_fn = self.decoder(z)
        decoded = self.get_session().run(decoded_fn)
        decoded = (0.5 * decoded) + 0.5;
        # print(decoded.shape)
        channel_first = np.rollaxis(decoded, 3, 1)  
        # print(channel_first.shape)
        return channel_first
