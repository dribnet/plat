from scipy.misc import imread
import numpy as np
import json

def anchors_from_image(fname, channels=3, image_size=(64,64), unit_scale=True):
    """Get a series of small images from a single large image"""
    rawim = imread(fname);
    if(channels == 1):
        im_height, im_width, im_channels = rawim.shape
        mixedim = rawim
    else:
        im_height, im_width, im_channels = rawim.shape
        mixedim = np.asarray([rawim[:,:,0], rawim[:,:,1], rawim[:,:,2]])

    pairs = []
    target_shape = (channels, image_size[0], image_size[1])
    height, width = image_size

    # first build a list of num images in datastream
    datastream_images = []
    steps_y = int(im_height / height)
    steps_x = int(im_width / width)
    # while cur_x + width <= im_width and len(datastream_images) < num:
    for j in range(steps_y):
        cur_y = j * height
        for i in range(steps_x):
            cur_x = i * width
            if(channels == 1):
                # entry = (mixedim[cur_y:cur_y+height, cur_x:cur_x+width] / 255.0).astype('float32')
                entry = mixedim[cur_y:cur_y+height, cur_x:cur_x+width]
            else:
                # entry = (mixedim[0:im_channels, cur_y:cur_y+height, cur_x:cur_x+width] / 255.0).astype('float32')
                entry = mixedim[0:im_channels, cur_y:cur_y+height, cur_x:cur_x+width]
            if unit_scale:
                entry = (entry / 255.0).astype('float32')
            datastream_images.append(entry)

    return steps_y, steps_x, np.array(datastream_images)

def get_json_vectors(filename):
    """Return np array of vectors from json source"""
    with open(filename) as json_file:
        json_data = json.load(json_file)
    return np.array(json_data)

def offset_from_string(x_indices_str, offsets, dim):
    """Return json vectors from index shorthand list (allows inverting)"""
    x_offset = np.zeros((dim,))
    if x_indices_str[0] == ",":
        x_indices_str = x_indices_str[1:]
    x_indices = map(int, x_indices_str.split(","))
    for x_index in x_indices:
        if x_index < 0:
            scaling = -1.0
            x_index = -x_index
        else:
            scaling = 1.0
        x_offset += scaling * offsets[x_index]
    return x_offset
