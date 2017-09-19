#!/usr/bin/env
# -*- coding: utf-8 -*-
"""Classes and methods for segmenting and assigning intracellular foci."""

from pyto_segmenter.PexSegment import PexSegmenter
import numpy as np
import pandas as pd
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage import filters


class Foci:
    """Container for segmented foci."""

    def __init__(self, CellSplitter, verbose=True):
        """Create a Foci instance from a CellSplitter instance."""
        if verbose:
            print('initializing attributes...')
        self.filenames = CellSplitter.filenames
        if len(self.filenames) == 1:
            self.filenames = self.filenames[0].split('/')[-1]
        self.segmented_nuclei = CellSplitter.segmented_nuclei
        self.segmented_cells = CellSplitter.segmented_cells
        self.n_raw_nuclei = CellSplitter.n_raw_nuclei
        self.cell_masks = CellSplitter.cell_masks
        self.n_cells = CellSplitter.n_cells
        if verbose:
            print('loading images...')
        self.imgs = {}
        for c in CellSplitter.multi_finder.cell_channels:
            self.imgs[c] = CellSplitter.multi_finder.get_channel_arrays(
                c, bg=False)
        self.channels = CellSplitter.multi_finder.cell_channels
        self.n_pos = self.imgs[self.channels[0]].shape[0]  # num of stage posns
        self.flagged_oof_ims = CellSplitter.multi_finder.flagged_oof_ims
        self.flagged_z_ims = CellSplitter.multi_finder.flagged_z_ims

    def segment(self, verbose=True, thresholds='auto'):
        """Identify foci in image."""
        self.foci = {}
        self.foci_df = pd.DataFrame(
                    {'id': [],
                     'intensity': [],
                     'volume': [],
                     'parent_cell': [],
                     'channel': [],
                     'scaling_factor': [],
                     'filename': [],
                     'im_number': []
                     }
                    )

        if verbose:
            print('beginning segmentation.')
        if thresholds == 'auto':
            self.thresholds = {488: (15000, 7500),
                               561: (8000, 4000)}  # default 488/561 thresh
        else:
            self.thresholds = thresholds
        self.erosion_struct = np.array(  # the strel for erosion of nuclei
            [[[False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, True,  False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False]],
             [[False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, True,  False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False]],
             [[False, False, False,  True, False, False, False],
              [False, False,  True,  True,  True, False, False],
              [False,  True,  True,  True,  True,  True, False],
              [True,   True,  True,  True,  True,  True,  True],
              [False,  True,  True,  True,  True,  True, False],
              [False, False,  True,  True,  True, False, False],
              [False, False, False,  True, False, False, False]],
             [[False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, True,  False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False]],
             [[False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, True,  False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False]]])
        for c in self.channels:
            if c == 405:
                continue  # don't segment foci from DAPI channel
            channel_foci = []
            if verbose:
                print('------------------------------------------------------')
                print('segmenting foci from channel ' + str(c))
                print('------------------------------------------------------')
                print('canny threshold for channel ' + str(c) + ': ' +
                      str(self.thresholds[c]))
            for i in range(0, self.n_pos):  # for each stage position
                # segment foci from this channel
                if verbose:
                    print('segmenting foci for position ' + str(i + 1) +
                          ' out of ' + str(self.n_pos))
                    print('generating normalized image for' +
                          ' segmentation...')
                norm_im, scaling_factor = self.normalize_im(
                    self.imgs[c][i, :, :, :],
                    mask=np.logical_and(self.cell_masks[i] != 0,
                                        self.segmented_nuclei[i] == 0))
                if verbose:
                    print('performing segmentation...')
                curr_segmenter = PexSegmenter(
                    src_data=norm_im, seg_method='canny',
                    high_threshold=self.thresholds[c][0],
                    low_threshold=self.thresholds[c][1], g_z=0)
                curr_seg = curr_segmenter.segment()
                c_foci = curr_seg.peroxisomes
                raw_img = curr_seg.raw_img
                # get #s and volumes for foci and make dict
                if verbose:
                    print('eliminating dim foci...')
                objs, vols = np.unique(c_foci, return_counts=True)
                if verbose:
                    print('before eliminating dim foci: ' +
                          str(len(objs) - 1) + ' foci in image')
                vols = dict(zip(objs, vols))
                # get mean intensity for each focus
                mean_intensity = {}
                for obj in objs:
                    mean_intensity[obj] = np.sum(
                        raw_img[c_foci == obj]).astype('float')/vols[obj]
                rev_dict = {v: k for k, v in mean_intensity.items()}
                eroded_nuclei = np.copy(self.segmented_nuclei[i])
                eroded_nuclei = binary_erosion(eroded_nuclei,
                                               structure=self.erosion_struct)
                cell_mean = np.nanmean(
                    raw_img[np.logical_and(self.segmented_cells[i] != 0,
                                           eroded_nuclei == 0)])
                cell_sd = np.nanstd(
                    raw_img[np.logical_and(self.segmented_cells[i] != 0,
                                           eroded_nuclei == 0)])
                if c == 488:
                    if verbose:
                        print('cell mean: ' + str(cell_mean))
                        print('cell standard deviation: ' + str(cell_sd))
                        print('intensity cutoff for removal: ' +
                              str(cell_mean + 4.5*cell_sd))
                        print('-----filtering foci-----')
                    for k in rev_dict.keys():
                        if verbose:
                            print('focus intensity: ' + str(k))
                        if k < cell_mean + 4.5*cell_sd:  # if mean intensity too low
                            if verbose:
                                print('removing focus')
                            c_foci[c_foci == rev_dict[k]] = 0  # eliminate focus
                if c == 561:
                    if verbose:
                        print('cell mean: ' + str(cell_mean))
                        print('cell standard deviation: ' + str(cell_sd))
                        print('intensity cutoff for removal: ' +
                              str(cell_mean + 3*cell_sd))
                        print('-----filtering foci-----')
                    for k in rev_dict.keys():
                        if verbose:
                            print('focus intensity: ' + str(k))
                        if k < cell_mean + 3*cell_sd:  # if mean intensity too low
                            if verbose:
                                print('removing focus')
                            c_foci[c_foci == rev_dict[k]] = 0  # eliminate focus
                if verbose:
                    print('after eliminating dim foci: ' +
                          str(len(np.unique(c_foci))-1) + ' foci in image')
                if verbose:
                    print('eliminating foci that reside outside of cells...')
                c_foci[self.cell_masks[i] == 0] = 0
                if verbose:
                    print('eliminating intranuclear foci...')
                c_foci[eroded_nuclei != 0] = 0
                if verbose:
                    print(str(np.unique(c_foci).size-1) + ' final foci')
                ids, vols = np.unique(c_foci, return_counts=True)
                vols = vols[ids != 0]
                ids = ids[ids != 0]
                intensities = np.empty_like(ids)
                parent_cells = np.empty_like(ids)
                channel_foci.append(c_foci)
                if ids.size != 0:  # if ids is not empty
                    for x in np.nditer(ids):  # iterate over foci IDs
                        # get parent cells
                        parent_cell, cell_cts = np.unique(
                            self.segmented_cells[i][c_foci == x],
                            return_counts=True
                            )
                        cell_cts = cell_cts[parent_cell != 0]  # rm bgrd
                        parent_cell = parent_cell[parent_cell != 0]  # rm bgrd
                        if parent_cell.shape[0] > 1:  # if part of >1 cell
                            # assign to cell containing more of the focus's px
                            parent_cell = parent_cell[np.argmax(cell_cts)]
                        elif parent_cell.shape[0] == 1:
                            parent_cell = parent_cell[0]  # extract value from arr
                        else:
                            parent_cell = -1
                        parent_cells[ids == x] = parent_cell
                        intensities[ids == x] = np.sum(
                            raw_img[c_foci == x])/vols[ids == x][0]
                    # create a temp pd df containing data
                    temp_df = pd.DataFrame(
                        {'id': pd.Series(ids, index=ids),
                         'intensity': pd.Series(intensities, index=ids),
                         'volume': pd.Series(vols, index=ids),
                         'parent_cell': pd.Series(parent_cells, index=ids),
                         'channel': c,
                         'scaling_factor': scaling_factor,
                         'filename': self.filenames,
                         'im_number': i
                         }
                        )
                    self.foci_df = self.foci_df.append(temp_df,
                                                       ignore_index=True)
                if verbose:
                    print('foci segmented from position ' + str(i + 1))
                    print()
            self.foci[c] = channel_foci
            if verbose:
                print('------------------------------------------------------')
                print('segmentation complete for channel ' + str(c))
                print('------------------------------------------------------')

    def count_foci(self, verbose=True):
        """Count the # of foci present in each segmented cell.

        Yields:
            a dict of dicts.
            Inner dict: a dict of image # (key): ndarray of #s of foci per cell
                (val) pairs.
            Outer dict: a dict of channel (key): Inner dict (val) pairs.
                channel is represented as a string to allow later addition of
                'overlap' as a key through self.measure_overlap().

        """
        if not hasattr(self, 'foci'):
            if verbose:
                print('foci not yet segmented. calling segment().')
            self.segment()
        self.foci_cts = {}
        for k in self.foci.keys():  # for each channel
            if verbose:
                print('------------------------------------------------------')
                print('counting foci/cell in channel ' + str(k) + ' images...')
                print('------------------------------------------------------')
            im_foci = {}
            for i in range(0, len(self.foci[k])):  # for each image
                print('counting foci/cell in image #' + str(i+1) + ' of ' +
                      str(len(self.foci[k])))
                foci_per_cell = []
                for cell in np.unique(self.segmented_cells[i]):
                    if cell == 0:
                        continue  # skip background
                    # get unique segmented foci IDs within the cell
                    cell_foci = np.unique(
                        self.foci[k][i][self.segmented_cells[i] == cell])
                    foci_per_cell.append(len(cell_foci) - 1)  # subtract bg
                im_foci[i] = np.asarray(foci_per_cell)
            self.foci_cts[str(k)] = im_foci

    def measure_overlap(self, verbose=True):
        """Determine how many foci overlap between two channels."""
        if not hasattr(self, 'foci'):
            if verbose:
                print('foci not yet segmented. calling segment().')
            self.segment()
        self.foci_df['overlap'] = 0  # will replace with 1s where appropriate
        if verbose:
            print('checking for overlap...')
        channels = list(self.foci.keys())
        n_ims = len(self.foci[channels[0]])
        overlap = {}
        for i in range(0, n_ims):
            if verbose:
                print('finding overlap in image #' + str(i + 1) + ' out of ' +
                      str(n_ims))
            overlap = np.logical_and(self.foci[channels[0]][i] > 0,
                                     self.foci[channels[1]][i] > 0)
            if verbose:
                print('getting IDs of overlapping foci...')
            overlap_IDs_ch0 = np.unique(self.foci[channels[0]][i][overlap])
            overlap_IDs_ch1 = np.unique(self.foci[channels[1]][i][overlap])
            if verbose:
                print('indicating overlap in foci_df...')
            # add 1s to overlap column where there was overlap
            self.foci_df.loc[(self.foci_df['channel'] == channels[0]) &
                             (self.foci_df['im_number'] == i) &
                             np.isin(self.foci_df['id'], overlap_IDs_ch0),
                             'overlap'] = 1
            self.foci_df.loc[(self.foci_df['channel'] == channels[1]) &
                             (self.foci_df['im_number'] == i) &
                             np.isin(self.foci_df['id'], overlap_IDs_ch1),
                             'overlap'] = 1

    def summarize_data(self, verbose=True):
        """Generate summary data frame for output."""
        if not hasattr(self, 'foci_df'):
            if verbose:
                print('foci data not yet calculated. calling measure_overlap.')
                # measure_overlap will call segmentation
            self.measure_overlap()
        else:
            if 'overlap' not in self.foci_df.columns:
                if verbose:
                    print('overlap not yet measured. calling measure_overlap.')
                self.measure_overlap()
        grouped = self.foci_df.groupby(['filename', 'im_number',
                                       'channel', 'parent_cell'])
        temp_vol_df = grouped['volume'].agg(['count', np.sum, np.mean, np.std])
        temp_vol_df.columns = ['count', 'vol_total', 'vol_mean', 'vol_sd']
        temp_intensity_df = grouped['intensity'].agg([np.mean, np.std])
        temp_intensity_df.columns = ['intensity_mean', 'intensity_sd']
        temp_overlap_df = grouped['overlap'].agg([np.sum])
        temp_overlap_df.columns = ['overlap_count']
        self.summary_df = pd.concat([temp_vol_df,
                                     temp_intensity_df,
                                     temp_overlap_df], axis=1)
        self.summary_df.reset_index(inplace=True)  # convert to normal df
        raw_cells_for_mapping = dict(
            zip(list(range(0, len(self.segmented_cells))), self.n_raw_nuclei))
        final_cells_for_mapping = {}
        flagged_oof_map = dict(zip(list(range(0,
                                              len(self.segmented_cells))),
                                   self.flagged_oof_ims.tolist()))
        flagged_z_map = dict(zip(list(range(0,
                                      len(self.segmented_cells))),
                                 self.flagged_z_ims.tolist()))
        for i in range(0, len(self.segmented_cells)):
            final_cells_for_mapping[i] = np.unique(
                self.segmented_cells[i]).shape[0]-1  # num diff vals minus bg
        self.summary_df['flagged_oof'] = self.summary_df['im_number'].map(
            flagged_oof_map
        )
        self.summary_df['flagged_z'] = self.summary_df['im_number'].map(
            flagged_z_map
        )
        self.summary_df['n_cells'] = self.summary_df['im_number'].map(
            final_cells_for_mapping
        )
        self.summary_df['raw_cells'] = self.summary_df['im_number'].map(
            raw_cells_for_mapping
        )

    def detailed_output(self, path, verbose=True):
        """Write raw and summarized DataFrame formatted data to a .csv."""
        if verbose:
            print('writing files...')
        if path.endswith('/'):
            path = path[:-1]
        self.foci_df.to_csv(
            path + '/' + self.filenames[:-4] + '_raw.csv')
        self.summary_df.to_csv(
            path + '/' + self.filenames[:-4] + '_summary.csv')

    def pandas_output(self, path, verbose=True):
        """**DEPRECATED! USE DETAILED_OUTPUT (AND RELEVANT DF METHODS)**.

        Write # of foci and overlap data to a .csv file.
        """
        if not hasattr(self, 'foci_cts'):
            raise AttributeError('# of foci not yet counted.')
        if str(self.channels[0])+'_overlap' not in list(self.foci_cts.keys()):
            raise AttributeError('overlap not measured between channels.')
        if verbose:
            print('initializing pd.DataFrame for output...')
        channels = [str(k) for k in self.foci.keys()]
        # initialize arrays that will populate the DataFrame
        tot_foci_1 = np.array([])
        tot_foci_2 = np.array([])
        overlap_foci_1 = np.array([])
        overlap_foci_2 = np.array([])
        cell_nums = np.array([])
        raw_cells = np.array([])
        total_cells = np.array([])
        im_nums = np.array([])
        flagged_oof = np.array([])
        flagged_z = np.array([])
        if verbose:
            print('populating output arrays...')
        for i in range(0, len(self.segmented_cells)):
            n_cells = len(self.foci_cts[channels[0]][i])
            tot_foci_1 = np.concatenate(
                (tot_foci_1, self.foci_cts[channels[0]][i]))
            tot_foci_2 = np.concatenate(
                (tot_foci_2, self.foci_cts[channels[1]][i]))
            overlap_foci_1 = np.concatenate(
                (overlap_foci_1, self.foci_cts[str(channels[0])+'_overlap'][i])
                )
            overlap_foci_2 = np.concatenate(
                (overlap_foci_2, self.foci_cts[str(channels[1])+'_overlap'][i])
                )
            cell_nums = np.concatenate(
                (cell_nums, np.arange(1, n_cells + 1)))
            im_nums = np.concatenate(
                (im_nums, np.repeat(i+1, n_cells)))
            raw_cells = np.concatenate(
                (raw_cells, np.repeat(self.n_raw_nuclei[i], n_cells)))
            flagged_oof = np.concatenate(
                (flagged_oof, np.repeat(self.flagged_oof_ims[i], n_cells)))
            flagged_z = np.concatenate(
                (flagged_z, np.repeat(self.flagged_z_ims[i], n_cells)))
            total_cells = np.concatenate(
                (total_cells, np.repeat(n_cells, n_cells)))
        non_olap_1 = tot_foci_1 - overlap_foci_1
        non_olap_2 = tot_foci_2 - overlap_foci_2
        output_df = pd.DataFrame({'filename': self.filenames,
                                  'image': im_nums,
                                  'cell': cell_nums,
                                  'im_n_cells': total_cells,
                                  'im_raw_cells': raw_cells,
                                  str(channels[0]) + '_total_foci': tot_foci_1,
                                  str(channels[1]) + '_total_foci': tot_foci_2,
                                  str(channels[0]) + '_only_foci': non_olap_1,
                                  str(channels[1]) + '_only_foci': non_olap_2,
                                  str(channels[0]) +
                                  '_overlap_foci': overlap_foci_1,
                                  str(channels[1]) +
                                  '_overlap_foci': overlap_foci_2,
                                  'flagged_oof': flagged_oof,
                                  'flagged_z': flagged_z
                                  })
        output_df.to_csv(path)

    @staticmethod
    def normalize_im(im, norm_mean=1000, mask=None, verbose=True):
        """Normalize an image intensity values to a set mean."""
        if len(im.shape) != 3:
            raise IndexError('normalize_im can only work on 3D ims.')
        if verbose:
            print('blurring image...')
        blurred_im = filters.gaussian_filter(im, sigma=(0, 3, 3))
        if verbose:
            print('calculating image mean...')
        if mask is not None:
            mean = np.mean(blurred_im[mask])
        else:
            mean = np.mean(blurred_im)
        if verbose:
            print('original cell mean: ' + str(mean))
            print('normalizing image...')
        return (im/(mean/norm_mean), mean/norm_mean)
