#!/usr/bin/env
# -*- coding: utf-8 -*-
"""Classes and methods for watershed segmentation of cells using nuclei."""

import numpy as np
from csth_analysis import czi_io
from csth_analysis import find_cells
from pyto_segmenter.PexSegment import PexSegmenter
import scipy.ndimage.morphology as morph
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import watershed


class CellSplitter:
    """Class and methods for segmenting cells using watershedding from DAPI."""

    def __init__(self, multi_finder, channel=488, threshold='auto'):
        """Create a Nuclei object for segmentation."""
        self.filenames = multi_finder.filenames
        self.multi_finder = multi_finder
        self.threshold = threshold
        self.n_raw_nuclei = []
        self.segmented_nuclei = []
        self.nuclei_centers = []
        print('generating cell masks...')
        self.cell_masks = self.multi_finder.find_cells(channel, verbose=True)
        if 405 not in multi_finder.cell_channels:
            raise ValueError(
                'The MultiFinder object lacks nuclei fluorescence.'
            )
        self.seg_input = self.multi_finder.get_channel_arrays(405)[0]
        if len(self.seg_input.shape) == 4:
            self.nimgs = self.seg_input.shape[0]
            self.input_shape = 4
        else:
            if len(self.seg_input.shape) == 3:
                self.input_shape = 3
            else:
                raise ValueError('The 405 wl image is an unexpected shape.')

    def segment_nuclei(self, verbose=True):
        """Segment nuclei from 405 nm wavelength images."""
        if self.input_shape == 3:
            if verbose:
                print('input has 3 dimensions.')
                print('segmenting nuclei using PexSegmenter...')
            if self.threshold == 'auto':
                # use slice-by-slice scaled cutoffs with hard floor to make
                # thresholded image.
                if verbose:
                    print('Using thresholds set based on slice intensity')
                gaussian_im = gaussian_filter(self.seg_input, sigma=(0, 4, 4))
                thresh_floor = int(np.amax(gaussian_im.flatten())*0.2)
                maxima = np.amax(gaussian_im, axis=0)  # slice-by-slice maxima
                if verbose:
                    print('thresholding floor set to ' + str(thresh_floor))
                    print('slice-by-slice cutoffs:')
                    print(maxima*0.5)
                    print('generating threshold image...')
                threshold_im = np.copy(gaussian_im)
                threshold_im[self.cell_masks == 0] = 0
                for z in range(0, threshold_im.shape[0]):
                    if maxima[z]*0.5 < thresh_floor:
                        threshold_im[z, :, :][
                            threshold_im[z, :, :] < thresh_floor] = 0
                    else:
                        threshold_im[z, :, :][
                            threshold_im[z, :, :] < int(maxima[z]*0.5)] = 0
                threshold_im[threshold_im > 0] = 1
                segmenter = PexSegmenter(
                    src_data=threshold_im, seg_method='pre-thresholded')
            else:
                segmenter = PexSegmenter(
                    src_data=self.seg_input, threshold=self.threshold, g_xy=4,
                    g_z=0)
            # use empirically determined segmentation parameters
            seg_output = segmenter.segment(fill_holes=True,
                                           edt_sampling=(10, 1, 1),
                                           edt_smooth=[3, 50, 50])
            if verbose:
                print('passing segmented objs and seeds to Nuclei instance...')
            self.segmented_nuclei.append(seg_output.peroxisomes)
            self.nuclei_centers.append(seg_output.maxima)
            del seg_output  # clear up memory
        elif self.input_shape == 4:
            if verbose:
                print('input image has 4 dimensions (multiple imgs)')
            for i in range(0, self.seg_input.shape[0]):
                if verbose:
                    print('segmenting image #' + str(i+1) +
                          ' of ' + str(self.seg_input.shape[0]) + str('...'))
                if self.threshold == 'auto':
                    # use slice-by-slice scaled cutoffs with hard floor to make
                    # thresholded image.
                    if verbose:
                        print('Using thresholds set based on slice intensity')
                    gaussian_im = gaussian_filter(
                        self.seg_input[i, :, :, :], sigma=(0, 4, 4))
                    thresh_floor = int(np.amax(gaussian_im.flatten())*0.2)
                    # get slice-by-slice max
                    maxima = np.amax(gaussian_im, axis=(1, 2))
                    if verbose:
                        print('thresholding floor set to ' + str(thresh_floor))
                        print('slice-by-slice cutoffs:')
                        print(maxima*0.5)
                        print('generating threshold image...')
                    threshold_im = np.copy(gaussian_im)
                    threshold_im[self.cell_masks[i] == 0] = 0
                    for z in range(0, threshold_im.shape[0]):
                        if maxima[z]*0.5 < thresh_floor:
                            threshold_im[z, :, :][
                                threshold_im[z, :, :] < thresh_floor] = 0
                        else:
                            threshold_im[z, :, :][
                                threshold_im[z, :, :] < int(maxima[z]*0.5)] = 0
                    threshold_im[threshold_im > 0] = 1
                    segmenter = PexSegmenter(
                        src_data=threshold_im, seg_method='pre-thresholded')
                else:
                    if verbose:
                        print('performing segmentation with a threshold of ' +
                              str(self.threshold))
                    segmenter = PexSegmenter(
                        src_data=self.seg_input[i, :, :, :],
                        threshold=self.threshold, g_xy=4, g_z=0)
                seg_output = segmenter.segment(fill_holes=True,
                                               edt_sampling=(10, 1, 1),
                                               edt_smooth=[3, 50, 50])
                if verbose:
                    print('passing segmented objs and seeds' +
                          ' to Nuclei instance...')
                    print()
                self.nuclei_centers.append(seg_output.maxima)
                self.segmented_nuclei.append(seg_output.peroxisomes)
                self.n_raw_nuclei.append(seg_output.npexs)
        # remove perinuclear foci that are smaller than a true nucleus
        if verbose:
            print('removing small perinuclear foci...')
        for i in range(0, len(self.segmented_nuclei)):
            objs_w_cts = np.unique(self.segmented_nuclei[i],
                                   return_counts=True)
            if verbose:
                print(str(len(objs_w_cts[0]) - 1) + ' raw segmented DAPI foci')
                print('object volumes in pixels:')
                print(objs_w_cts[1][1:])
            objs_to_rm = objs_w_cts[0][objs_w_cts[1] < 10000]
            if verbose:
                print('removing objects volume < 10000 px, #s:')
                print(objs_to_rm)
            self.segmented_nuclei[i][np.reshape(
                np.in1d(self.segmented_nuclei[i], objs_to_rm),
                self.segmented_nuclei[i].shape)] = 0
            # remove corresponding nuclei_centers as well
            if verbose:
                print('removing nuclei_centers corresponding to small' +
                      ' DAPI foci...')
            self.nuclei_centers[i][self.segmented_nuclei[i] == 0] = 0

    def segment_cells(self, channel, verbose=True, rm_edge_cells=True):
        """Segment cells, identified using find_cells, based on nuclei."""
        strel = np.array([[[0,0,0],
                           [0,0,0],
                           [0,0,0]],
                          [[0,1,0],
                           [1,1,1],
                           [0,1,0]],
                          [[0,0,0],
                           [0,0,0],
                           [0,0,0]]])
        self.segmented_cells = []
        self.n_cells = []
        # test to make sure nuclei have already been segmented.
        if verbose:
            print('checking that nuclei have already been segmented...')
        if len(self.segmented_nuclei) == 0:
            if verbose:
                print('nuclei have not been segmented yet.')
                print('segmenting nuclei...')
            self.segment_nuclei()
        # convert segmented nuclei to an inverted mask for distance xform
        nuclei_masks = np.copy(self.segmented_nuclei)
        channel_ims = self.multi_finder.get_channel_arrays(
            channel, bg=False)
        if verbose:
            print('eroding cell edges...')
        for i in range(0, len(self.cell_masks)):
            if verbose:
                print('eroding mask ' + str(i) + '...')
            means = []
            for j in range(0, 10):
                curr_mask = morph.binary_erosion(
                    self.cell_masks[i], structure=strel, iterations=j)
                means.append(
                    np.mean(channel_ims[i, :, :, :][np.logical_and(
                        curr_mask != 0, nuclei_masks[i] == 0)]))
            means[0] = np.mean(channel_ims[i, :, :, :][np.logical_and(
                self.cell_masks[i] != 0, nuclei_masks[i] == 0)])
            if verbose:
                print('cell mask means:')
                print(means)
            slopes = []
            for j in range(0, len(means)-2):
                slopes.append(np.divide(means[j+2]-means[j], 2))
            if verbose:
                print('slopes:')
                print(slopes)
            delta_slopes = np.array([])
            iterations = np.arange(1.5, 8.5, 1)
            for j in range(0, len(slopes)-1):
                delta_slopes = np.append(
                    delta_slopes, np.absolute(slopes[j+1]-slopes[j]))
            delta_slopes[np.isnan(delta_slopes)] = 0
            if verbose:
                print('slope deltas:')
                print(delta_slopes)
            desired_erosions = int(
                iterations[np.argmax(delta_slopes)] + 0.5)
            print('desired erosions: ' + str(desired_erosions))
            self.cell_masks[i] = morph.binary_erosion(
                self.cell_masks[i], structure=strel,
                iterations=desired_erosions)
            if verbose:
                print('erosion of mask #' + str(i) + ' complete.')
        if verbose:
            print('converting nuclei to binary masks for distance xform...')
        for i in range(0, len(nuclei_masks)):
            nuclei_masks[i][nuclei_masks[i] > 0] = 1
            nuclei_masks[i] = np.invert(nuclei_masks[i].astype('bool'))
        # segment cells
        for j in range(0, len(self.cell_masks)):
            if verbose:
                print('segmenting cells in image #' + str(j + 1) +
                      ' out of ' + str(len(self.cell_masks)) + '...')
            # distance xform based on the distance to a nucleus
            if verbose:
                print('performing Euclidean distance xform...')
            dist_map = distance_transform_edt(nuclei_masks[j],
                                              sampling=(3, 1, 1))
            if verbose:
                print('smoothing the distance map...')
            dist_map = gaussian_filter(dist_map, [1, 2, 2])  # smooth the map
            # generate segmentation seeds from nuclei segmentation maxima
            if verbose:
                print('generating segmentation seeds from nuclei_centers...')
            labs = PexSegmenter.watershed_labels(self.nuclei_centers[j])
            # watershed segment
            if verbose:
                print('watershed segmenting cells from nuclei centers...')
                seg_cells = watershed(dist_map, labs, mask=self.cell_masks[j])
            if verbose:
                print('segmentation complete for image #' + str(j + 1) + '.')
            if rm_edge_cells:
                if verbose:
                    print('removing cells that contact the edge of the image.')
                seg_cells = self.rm_edge_objs(seg_cells)
            # remove foci that are smaller than a true cell
            if verbose:
                print('removing small foci...')
            objs_w_cts = np.unique(seg_cells, return_counts=True)
            if verbose:
                print(str(len(objs_w_cts[0]) - 1) + ' raw segmented cells')
            objs_to_rm = objs_w_cts[0][objs_w_cts[1] < 1000]
            print(str(len(objs_to_rm)) +
                  ' foci volume < 1000 px, removing...')
            seg_cells[np.reshape(
                np.in1d(seg_cells, objs_to_rm), seg_cells.shape)] = 0
            n = len(np.unique(seg_cells)) - 1  # number of cells
            self.segmented_cells.append(seg_cells)
            self.n_cells.append(n)
            if verbose:
                print()

    @staticmethod
    def rm_edge_objs(arr, z=False):
        """Remove segmented objects that contact the edge of an image.

        Arguments:
            arr (np.ndarray): a 3D NumPy array of ints produced by watershed
                segmentation.
            z (bool, optional): should objects that appear on the top or
                bottom slice of the image be removed (are the first and last
                slices part of the 'border'?) defaults to false.
        """
        border_arr = np.ones(shape=arr.shape)
        if z:  # if objects on the top and bottom slices should be removed:
            border_arr[1:-1, 1:-1, 1:-1] = 0  # leave 1s filling top + bottom
        else:
            border_arr[:, 1:-1, 1:-1] = 0  # remove 1s from center of all zs
        border_arr = border_arr.astype('bool')
        objs_on_edge = np.unique(arr[border_arr])
        output_arr = np.copy(arr)
        output_arr[np.reshape(np.in1d(arr, objs_on_edge), arr.shape)] = 0
        return output_arr
