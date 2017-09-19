#!/usr/bin/env
# -*- coding: utf-8 -*-
"""Classes and methods for finding cells in images based on background."""

import czifile
import numpy as np
from warnings import warn
from skimage import io, measure
from csth_analysis import czi_io
from scipy import stats
from scipy.ndimage import filters
import scipy.ndimage as nd
from sklearn import svm
import pickle
import os


class CellMask:
    """Container for cell mask and intermediates."""

    # NOTE: I DON'T USE THIS FOR ANYTHING RIGHT NOW!
    def __init__(self, raw_im, raw_bg, gaussian_im,
                 pvals_im, cell_labs, cell_mask):
        """Create an instance of a cell mask.

        NOTE: This class is not intended to be called directly, but should
        instead be initialized by the CellFinder.find_cells() method.
        """
        self.raw_im = raw_im
        self.raw_bg = raw_bg
        self.gaussian_im = gaussian_im
        self.pvals_im = pvals_im
        self.cell_labs = cell_labs
        self.cell_mask = cell_mask


class MultiFinder:
    """Distinguish cells from background in multi-position czi files."""

    def __init__(self, filename, bg_index=-1, bg_filename='', log_path=None,
                 oof_svm=None, optim=False):
        """Create a MultiFinder object.

        Arguments:
            filename (str): The path to an image file to initialize the object.
            bg_index (int, optional): If the initializing file is .czi format,
                this indicates the index within the czi file array that
                corresponds to the background image with no cells. Defaults to
                -1, which means the background image isn't contained within
                the initializing image file.
            bg_filename (str, optional): The path to an image file
                corresponding to the background image. This argument must be
                provided if the background stage position is not contained
                in the multi-image file provided by filename or an
                exception will occur. It should not be provided if the
                background image is contained within a multi-image file
                provided by filename.
            log_path (str, optional): The path to a log directory for saving
                images with defects in blur elimination. Images from blur
                elimination aren't saved unless this is provided.
            oof_svm (str, optional): Path to a trained scikit-learn svm.SVC()
                pickle to be used for eliminating out-of-focus planes of the
                stack.
            optim (bool, optional): Indicates whether or not this run is being
                performed on a subset of the data for threshold optimization
                purposes. Default is False (extract all images from the chosen
                file); if True, will only pull out the first three images.
        """
        self.filenames = [filename]
        if bg_index == -1:
            if bg_filename == '':
                warn('No background image provided during initialization.')
            self.bg_origin = 'separate'  # separate czi or tiff file
            self.bg_filename = bg_filename
            self.log_path = log_path
            if self.log_path is not None:
                if not os.path.isdir(self.log_path):
                    try:
                        os.makedirs(self.log_path)
                    except FileExistsError:
                        pass
            self.oof_svm = oof_svm
        else:
            self.bg_origin = 'slice'  # slice of a multi-czi
        if '.tif' in self.filenames[0]:
            self.cell_im = io.imread(self.filenames[0])
            # TODO: implement adding dimensions for multi-img/channels both
            # here and a method to add them later
        elif '.czi' in self.filenames[0]:
            cell_czi = czi_io.load_multi_czi(self.filenames[0])
            self.cell_im = cell_czi[0]
            # if cell_im has shape C-Z-X-Y, not IMG-C-Z-X-Y, add axis for img
            if len(self.cell_im.shape) == 4:
                self.cell_im = np.expand_dims(self.cell_im, axis=0)
            if optim:
                print('Optimization mode: only using first three images.')
                if self.cell_im.shape[0] > 2:
                    self.cell_im = self.cell_im[0:3, :, :, :, :]
            # add filename:slice #s dict to indicate which imgs came from where
            self.f_to_s = {self.filenames[0]: range(0, self.cell_im.shape[0])}
            self.cell_channels = cell_czi[1]
        if self.bg_filename != '':
            if '.tif' in self.bg_filename:
                self.bg_im = io.imread(self.bg_filename)
                # TODO: implement adding dimensions for multi-img/channels both
                # here and a method to add them later
            elif '.czi' in self.bg_filename:
                bg_czi = czi_io.load_single_czi(self.bg_filename)
                self.bg_im = np.expand_dims(bg_czi[0], axis=0)
                self.bg_channels = bg_czi[1]
        elif bg_index != -1:
            self.bg_im = self.cell_im[bg_index, :, :, :, :]
            # remove parts of cell_im that correspond to bg
            bg_mask = np.ones(shape=self.cell_im.shape, dtype=bool)
            bg_mask[bg_index, :, :, :, :] = False
            self.cell_im = self.cell_im[bg_mask]
            self.bg_channels = self.cell_channels
        if self.oof_svm is not None:
            self.flagged_oof_ims = np.zeros(self.cell_im.shape[0])
            self.flagged_z_ims = np.zeros(self.cell_im.shape[0])

    def add_czi(self, filename):
        """Add an additional czi file containing additional image(s).

        Arguments:
            filename (str): Path to the czi file to be added to the existing
                MultiFinder object.
        """
        new_czi = czi_io.load_multi_czi(filename)
        stripped_fname = filename.split('/')[1]
        if len(new_czi[0].shape) == 4:  # if it's a single img czi, not multi
            new_czi[0] = np.expand_dims(new_czi[0], axis=0)
        if new_czi[0].shape[1:] != self.cell_im.shape[1:]:  # if diff CZXY
            raise ValueError('The new czi has a different shape than cell_im.')
        if new_czi[1] != self.cell_channels:
            raise ValueError('The new czi uses non-matching channels.')
        self.filenames.append(stripped_fname)
        self.f_to_s[stripped_fname] = range(
            self.cell_im.shape[0], self.cell_im.shape[0] + new_czi.shape[0])
        self.cell_im = np.concatenate(self.cell_im, new_czi)

    def find_cells(self, channel, return_all=False, verbose=True):
        """Find cells within all images in the indicated channel."""
        # get channel images first
        im_arrs = self.get_channel_arrays(channel)
        # transform into log space, as bg is roughly log-normal
        # this requires adding 1 to each value to avoid NaN log-xform
        if verbose:
            print('log-transforming arrays...')
        log_f_im = np.log10(im_arrs[0] + 1)
        log_bg_im = np.log10(im_arrs[1] + 1)
        if verbose:
            print('applying gaussian filter...')
        log_gaussian_f = filters.gaussian_filter(log_f_im,
                                                 sigma=[0, 0, 3, 3])
        log_gaussian_bg = filters.gaussian_filter(log_bg_im,
                                                  sigma=[0, 0, 3, 3])
        bg_mean = np.mean(log_gaussian_bg)
        bg_sd = np.std(log_gaussian_bg)
        # get p-val that px intensity could be brighter than the value in each
        # array position in the "positive" im, which will indicate where
        # fluorescence is.
        if verbose:
            print('computing p-value transformation...')
        f_pvals = np.empty_like(log_gaussian_f)
        for s in range(0, f_pvals.shape[0]):
            print(' computing p-val xform for image ' + str(s + 1) +
                  ' out of ' + str(f_pvals.shape[0]))
            f_pvals[s, :, :, :] = 1-stats.norm.cdf(
                log_gaussian_f[s, :, :, :], bg_mean, bg_sd)
        # convert to binary using empirically tested cutoffs (p<0.5/65535)
        f_pvals = f_pvals*65535
        f_pvals = f_pvals.astype('uint16')
        f_pvals_binary = np.copy(f_pvals)
        if verbose:
            print('converting to binary...')
        f_pvals_binary[f_pvals > 0] = 0
        f_pvals_binary[f_pvals == 0] = 1
        # eliminate too-small regions that don't correspond to cells
        cell_masks = []
        if return_all:
            raw_labs_list = []
            labs_list = []
        im_for_clf = self.get_channel_arrays(561, bg=False)
        if verbose:
            print('')
            print('generating cell masks...')
            print('')
        for im in range(0, im_arrs[0].shape[0]):
            if verbose:
                print('generating mask #' + str(im + 1))
            curr_im = f_pvals_binary[im, :, :, :]
            if verbose:
                print('labeling contiguous objects...')
            r_labs = measure.label(curr_im, connectivity=2,
                                   background=0)
            # next command eliminates objects w vol < 100,000 px and
            # generates binary array indicating where cells are for output
            if verbose:
                print('eliminating objects w/volume < 100,000 px...')
            # don't count zeros in next line to avoid including background
            objs_w_cts = np.unique(r_labs[r_labs != 0], return_counts=True)
            cell_mask = np.reshape(np.in1d(
                r_labs, objs_w_cts[0][objs_w_cts[1] > 100000]),
                                   r_labs.shape)
            if verbose:
                print('pruning labels...')
            trim_labs = np.copy(r_labs)
            trim_labs[np.invert(cell_mask)] = 0  # eliminate small obj labels
            if self.oof_svm is not None:
                if verbose:
                    print('unlabeling out of focus slices...')
                with open(self.oof_svm, 'rb') as r:
                    clf = pickle.load(r)
                shrt_fname = self.filenames[0].split('/')[-1][:-4]
                if self.log_path is not None:
                    focus_slices = self.get_blur_slices(
                        im=im_for_clf[im, :, :, :], clf=clf, slc_no=im,
                        log_path=self.log_path+'/'+shrt_fname)
                else:
                    focus_slices = self.get_blur_slices(
                        im=im_for_clf[im, :, :, :], clf=clf, slc_no=im)
                if focus_slices[0] == 1 or focus_slices[-1] == 1:
                    self.flagged_z_ims[im] = 1
                    if verbose:
                        print(
                            'Warning: cells cut off by top or bottom of stack:'
                            )
                        print(focus_slices)
                    #if self.log_path is not None:
                    #    io.imsave(self.log_path + '/' + shrt_fname + '_' +
                    #              str(im) + '.tif', im_for_clf[im, :, :, :])
                cell_mask[np.where(focus_slices == 0)[0], :, :] = 0
            if verbose:
                print('appending outputs...')
            cell_masks.append(cell_mask)
            if return_all:
                raw_labs_list.append(r_labs)
                labs_list.append(trim_labs)
            if verbose:
                print('mask #' + str(im + 1) + ' complete.')
                print()
        if return_all:
            return({'input_ims': im_arrs[0],
                    'input_bg': im_arrs[1],
                    'gaussian_ims': np.power(10, log_gaussian_f),
                    'pvals_ims': f_pvals,
                    'raw_labs': raw_labs_list,
                    'trimmed_labs': labs_list,
                    'cell_masks': cell_masks})
        else:
            return cell_masks
    # helper methods #

    def get_channel_arrays(self, channel, fluorescence=True, bg=True,
                           mode='multi', ind=0):
        """Extract im arrays for specific channels."""
        channel = int(channel)  # Implicitly checks that channel is an int
        return_vals = []
        # return tuple of (fluorescence_array, bg_array) for the channel
        if fluorescence:
            if mode == 'multi':
                return_vals.append(self.cell_im[
                    :, self.cell_channels.index(channel), :, :, :])
            elif mode == 'single':
                return_vals.append(self.cell_im[
                    ind, self.cell_channels.index(channel), :, :, :])
        if bg:
            return_vals.append(self.bg_im[
                :, self.bg_channels.index(channel), :, :, :])
        if len(return_vals) == 1:
            return return_vals[0]
        else:
            return tuple(return_vals)

    def get_blur_slices(self, im, clf, slc_no, verbose=True, log_path=None):
        """Determine which slices are in and out of focus from 561 image."""
        if verbose:
            print('generating gradient image...')
        grad_im = MultiFinder.get_gradient_im(im)
        if verbose:
            print('generating gradient intensity histograms...')
        hist_arr = MultiFinder.get_grad_hists(grad_im)
        if verbose:
            print('predicting focus labels using SVM classifier...')
        labels = clf.predict(hist_arr)
        if verbose:
            print('calculating decision function values for labels...')
        dec_func = clf.decision_function(hist_arr)
        if verbose:
            print('raw focus labels:')
            print(labels)
        # test if there are any in-focus slices detected:
        if np.array_equal(np.unique(labels), np.array([0])):
            if verbose:
                print('Warning: no in-focus slices detected.')
            self.flagged_oof_ims[slc_no] = 1
#            if log_path is not None:
#                io.imsave(log_path + '_' + str(slc_no) + '.tif', im)
            return labels
        print('correcting slice labels...')
        corrected_labels = self.fix_interc_blur(labels, dec_func)
        if not np.array_equal(corrected_labels, labels):
            self.flagged_oof_ims[slc_no] = 1
            if verbose:
                print('Warning: blur detection resulted in intercalation:')
                print(corrected_labels)
#            if log_path is not None:
#                io.imsave(log_path + '_' + str(slc_no) + '.tif', im)
        else:
            if verbose:
                print('One continuous stretch defined as in focus.')
        return corrected_labels

    @staticmethod
    def fix_interc_blur(labels, dec_func):
        """Fix stacks with blur/focused slices intercalated."""
        one_inds = np.flatnonzero(labels)
        first_1 = one_inds[0]
        last_1 = one_inds[-1]
        if 0 in labels[first_1:last_1+1]:  # if there's a zero in the middle
            new_labels = np.copy(labels)
            sub_labels = new_labels[first_1:last_1+1]
            sub_dec = dec_func[first_1:last_1+1]
            # the number of ones before = the index of the first zero.
            ones_before = np.where(sub_labels == 0)[0][0]
            # count # of zeros
            n_zeros = 1
            still_zero = True
            while still_zero:
                if sub_labels[ones_before + n_zeros] == 0:
                    n_zeros += 1
                else:
                    still_zero = False
            # count # of consecutive ones after the zero
            ones_after = 1
            still_one = True
            while still_one:
                if ones_before + n_zeros + ones_after == sub_labels.shape[0]:
                    break  # if it hits the end of the sub-array
                if sub_labels[ones_before+n_zeros+ones_after] == 1:
                    ones_after += 1
                else:
                    still_one = False
            # figure out which to remove
            if ones_before < n_zeros:
                sub_labels[0:ones_before] = 0
            elif ones_after < n_zeros:
                sub_labels[ones_before+n_zeros:
                           ones_before+n_zeros+ones_after] = 0
            elif n_zeros < ones_before and n_zeros < ones_after:
                sub_labels[ones_before:ones_before+n_zeros] = 1
            elif ones_before == n_zeros and ones_after == n_zeros:
                sub_labels[ones_before:ones_before+n_zeros] = 1
            elif ones_before == n_zeros and ones_after > n_zeros:
                # if dec_func mean for ones before is less than zeros dec func
                if np.mean(sub_dec[0:ones_before]) < np.mean(
                        sub_dec[ones_before:ones_before+n_zeros]):
                    sub_labels[0:ones_before] = 0
                else:
                    sub_labels[ones_before:ones_before+n_zeros] = 1
            elif ones_before > n_zeros and ones_after == n_zeros:
                # if dec_func mean for ones after is less than zeros dec func
                if np.mean(sub_dec[
                        ones_before+n_zeros:ones_before+n_zeros+ones_after
                        ]) < np.mean(sub_dec[ones_before:ones_before+n_zeros]):
                    sub_labels[ones_before+n_zeros:
                               ones_before+n_zeros+ones_after] = 0
                else:
                    sub_labels[ones_before:ones_before+n_zeros] = 1
            # initiate additional recursive round with new labeled set
            return MultiFinder.fix_interc_blur(new_labels, dec_func)
        else:
            return labels

    @staticmethod
    def get_gradient_im(im, sigma=(0.25, 0.25)):
        """Get the stack of gradient magnitude slices for an input image."""
        if len(im.shape) != 3:
            raise ValueError("The input image is the incorrect shape.")
        grad_im = np.empty_like(im)
        for s in range(0, im.shape[0]):  # for each slice in the input image
            grad_slice = nd.gaussian_gradient_magnitude(im[s, :, :],
                                                        sigma=sigma)
            grad_slice = grad_slice.astype('float32')/np.amax(
                grad_slice.flatten())  # normalize to 1
            grad_slice = grad_slice*65535  # make 16 bit
            grad_slice = grad_slice.astype('uint16')
            grad_im[s, :, :] = grad_slice
        return grad_im

    @staticmethod
    def get_grad_hists(grad_im, logxform=True):
        """Get array of histograms for feeding into the classifier."""
        hist_arr = np.empty(shape=(grad_im.shape[0], 50))
        for s in range(0, grad_im.shape[0]):
            hist, bin_edges = np.histogram(grad_im[s, :, :].flatten(),
                                           bins=50, range=(0, 65536))
            hist_arr[s, :] = hist
        if logxform:
            hist_arr = np.log10(hist_arr+1)
        # when I first trained the classifier
        return hist_arr


class CellFinder:
    """Distinguish cells from background in fluorescence images."""

    def __init__(self, im_filename, bg_im_filename):
        """Create a CellFinder object."""
        # set attributes
        self.filename = im_filename
        self.bg_filename = bg_im_filename
        if '.tif' in self.filename:
            self.cell_im = io.imread(self.filename)
        elif '.czi' in self.filename:
            cell_czi = czi_io.load_single_czi(self.filename)
            self.cell_im = cell_czi[0]
            self.cell_channels = cell_czi[1]
        if '.tif' in self.bg_filename:
            self.bg_im = io.imread(self.bg_filename)
        elif '.czi' in self.bg_filename:
            bg_czi = czi_io.load_single_czi(self.bg_filename)
            self.bg_im = bg_czi[0]
            self.bg_channels = bg_czi[0]
        # check inputs
        if self.filename.shape[0] != self.bg_filename.shape[0]:
            warn('bg image and cell images have different #s of channels.')

    def find_cells(self, channel):
        """Find cells within image in the indicated channel."""
        # get channel images first
        im_arrs = self.get_channel_arrays(channel)
        # transform into log space, as bg is roughly log-normal
        log_f_im = np.log10(im_arrs[0])
        log_bg_im = np.log10(im_arrs[1])
        log_gaussian_f = filters.gaussian_filter(log_f_im, sigma=[0, 2, 2])
        log_gaussian_bg = filters.gaussian_filter(log_bg_im, sigma=[0, 2, 2])
        bg_mean = np.mean(log_gaussian_bg)
        bg_sd = np.std(log_gaussian_bg)
        # get p-val that px intensity could be brighter than the value in each
        # array position in the "positive" im, which will indicate where
        # fluorescence is.
        f_pvals = 1-stats.norm.cdf(log_gaussian_f, bg_mean, bg_sd)
        # convert to binary using empirically tested cutoffs (p<0.5/65535)
        f_pvals = f_pvals*65535
        f_pvals = f_pvals.astype('uint16')
        f_pvals_binary = np.copy(f_pvals)
        f_pvals_binary[f_pvals > 0] = 0
        f_pvals_binary[f_pvals == 0] = 1
        # eliminate too-small regions that don't correspond to cells
        r_labs = measure.label(f_pvals_binary, connectivity=2,
                               background=0)
        # next command eliminates objects w vol < 100,000 px
        trim_labs = np.reshape(np.in1d(
            r_labs, np.unique(r_labs)[np.unique(
                r_labs, return_counts=True)[1] < 100000]))
        # generate binary array indicating where cells are for output
        cell_mask = np.zeros(shape=f_pvals_binary.shape)
        cell_mask[trim_labs != 0] = 1

        return CellMask(im_arrs[0], im_arrs[1], np.power(10, log_gaussian_f),
                        f_pvals, trim_labs, cell_mask)

    # helper methods #

    def get_channel_arrays(self, channel, fluorescence=True, bg=True):
        """Extract im arrays for specific channels."""
        channel = int(channel)  # Implicitly checks that channel is an int
        return_vals = []
        # return tuple of (fluorescence_array, bg_array) for the channel
        if fluorescence:
            return_vals.append(self.cell_im[
                self.cell_channels.index(channel), :, :, :])
        if bg:
            return_vals.append(self.bg_im[
                self.bg_channels.index(channel), :, :, :])
        return tuple(return_vals)
