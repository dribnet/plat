from __future__ import division, print_function

import os
import shutil
import theano
import theano.tensor as T

from blocks.extensions.saveload import Checkpoint

from plat.sampling import grid_from_latents
from plat.grid_layout import create_chain_grid
from plat.fuel_helper import get_anchor_images

class SampleCheckpoint(Checkpoint):
    def __init__(self, interface, z_dim, image_size, channels, dataset, split, save_subdir, **kwargs):
        super(SampleCheckpoint, self).__init__(path=None, **kwargs)
        self.interface = interface
        self.image_size = image_size
        self.channels = channels
        self.save_subdir = save_subdir
        self.iteration = 0
        self.epoch_src = "{0}/sample.png".format(save_subdir)
        self.rows=7
        self.cols=10
        self.spacing = 3
        self.z_dim = z_dim
        numanchors = 10 + 10

        self.anchor_images = get_anchor_images(dataset, split, numanchors=numanchors, image_size=image_size, include_targets=False)
        if not os.path.exists(self.save_subdir):
            os.makedirs(self.save_subdir)

    def do(self, callback_name, *args):
        """Sample the model and save images to disk
        """
        dmodel = self.interface(model=self.main_loop.model)
        anchors = dmodel.encode_images(self.anchor_images)
        z = create_chain_grid(self.rows, self.cols, self.z_dim, self.spacing, anchors=anchors, spherical=True, gaussian=False)
        grid_from_latents(z, dmodel, rows=self.rows, cols=self.cols, anchor_images=None, tight=False, shoulders=False, save_path=self.epoch_src, batch_size=12)
        if os.path.exists(self.epoch_src):
            epoch_dst = "{0}/epoch-{1:03d}.png".format(self.save_subdir, self.iteration)
            self.iteration = self.iteration + 1
            shutil.copy2(self.epoch_src, epoch_dst)
            os.system("convert -delay 5 -loop 1 {0}/epoch-*.png {0}/training.gif".format(self.save_subdir))
