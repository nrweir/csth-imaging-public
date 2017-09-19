#!/usr/bin/env
# -*- coding: utf-8 -*-
"""Functions for loading .czi files for analysis.

IMPORTANT NOTE: launching this script will yield importerrors for tifffile.c
and for czifile.pyx. You can ignore these unless you're using JPG-formatted
images.
"""

import czifile  # czi file import classes and methods
import numpy as np
import xml.etree.ElementTree as ET  # for parsing czi file metadata


def load_single_czi(path):
    """Load .czi data into a numpy array and output along with wavelength IDs.

    Arguments:
        path: the path to the .czi file. can be either absolute or relative to
            the current directory.

    Returns a tuple with 2 components:
    - a 4D numpy of shape [C,Z,Y,X]
    - a tuple with integers representing the excitation wavelengths for each
      channel
    """
    czi_file = czifile.CziFile(path)
    im_array = czi_file.asarray()  # extract image array from test_czi
    im_array = np.squeeze(im_array)  # eliminate single-dimensional axes
    # get channels
    channels = [int(np.rint(float(child.text))) for child in
                czi_file.metadata.iter('ExcitationWavelength')]
    # if there was only one channel, this will get rid of the C dimension.
    # add back if so.
    if len(channels) == 1:
        if len(im_array.shape) == 2:  # if it's not a Z stack
            im_array = np.expand_dims(im_array, axis=0)  # once for channel
            im_array = np.expand_dims(im_array, axis=0)  # once for Z
        elif len(im_array.shape) == 3:  # if it is a Z-stack
            im_array = np.expand_dims(im_array, axis=0)  # once for channel
    else:
        if len(im_array.shape) == 3:  # if it has 3 channels but only 1 Z slice
            im_array = np.expand_dims(im_array, axis=1)  # add Z axis

    return (im_array, channels)


def load_multi_czi(path):
    """Load multi-image czi file into a numpy array and output with wl IDs.

    Arguments:
        path (str): Path to the multi-image czi.

    Returns a tuple consisting of two components:
    - a 5D NumPy array with shape [IMG,C,Z,Y,X]
    - a tuple with integers representing the excitation wavelengths for each
      channel
    """
    czi_file = czifile.CziFile(path)
    im_array = czi_file.asarray()  # extract image array from test_czi
    if len(im_array.shape) == 8:  # if this is a single-image czi
        return load_single_czi(path)
    elif len(im_array.shape) == 9:
        im_array = np.squeeze(im_array)  # eliminate single-dimensional axes
        # get channels
        channels = [int(np.rint(float(child.text))) for child in
                    czi_file.metadata.iter('ExcitationWavelength')]
        # if there was only one channel, this will get rid of the C dimension.
        # add back if so.
        if len(channels) == 1:
            if len(im_array.shape) == 3:  # if it's not a Z stack but has >1 im
                im_array = np.expand_dims(im_array, axis=1)  # once for channel
                im_array = np.expand_dims(im_array, axis=1)  # once for Z
            elif len(im_array.shape) == 4:  # if it is a Z-stack
                im_array = np.expand_dims(im_array, axis=1)  # once for channel
        else:
            if len(im_array.shape) == 4:  # it has mult C but only 1 Z
                im_array = np.expand_dims(im_array, axis=2)  # add Z axis
        return (im_array, channels)
    else:
        raise ValueError('Received an unexpected czi file shape.')
