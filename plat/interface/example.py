### This is a documented example of a model interface compatible with plat.

class ExampleModel:
    def __init__(self, filename=None, model=None):
        """
        Initializate class give either a filename or a model

        Usually this method will load a model from disk and store internally,
        but model can also be provided directly instead (useful when training)
        """
        pass

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
        pass

    def get_zdim(self):
        """
        Returns the integer dimension of the latent z space
        """
        pass

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
        pass