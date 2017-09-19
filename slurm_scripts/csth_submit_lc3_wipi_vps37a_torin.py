# steps for this script:
# 1. get arguments:
#   - path to .csv ref file
#   - array ID
#   - path to output directory
#
# 2. use pd.read_csv to read ref file
# 3. open .czi according to path
# 4. perform analysis
# 5. save output csv to output dir

# NOTE: THIS VERSION OF THIS SCRIPT WAS WRITTEN SPECIFICALLY TO SEGMENT
# NUCLEI BETTER IN THE dVPS37A TORIN SAMPLE. USE THE FOLLOWING SLURM
# SUBMISSION COMMAND LINE ENTRY TO EXECUTE:

# sbatch --array=10 csth_vps37a_torin_submit.sh /n/denic_lab/Users/nweir/python_packages/csth-imaging/im_list_wipi.csv /n/denic_lab/Lab/csth-output/lc3_wipi

import os
import numpy as np
import pandas as pd
import argparse
from skimage import io
import sys
sys.path.append('/n/denic_lab/Users/nweir/python_packages/')
sys.path.append(
    '/n/denic_lab/Users/nweir/python_packages/csth-imaging/dependencies/')
from csth_analysis import czi_io, find_cells, segment_cells, foci

parser = argparse.ArgumentParser(description='Process LC3/WIPI stain imgs.')
parser.add_argument('-r', '--ref_csv', required=True,
                    help='path to reference csv file')
parser.add_argument('-a', '--array_no', required=True,
                    help='SLURM job array ID')
parser.add_argument('-o', '--output_dir', required=True,
                    help='dir for CSV-formatted output')
args = parser.parse_args()
print('args:')
print(args)
ref_csv = args.ref_csv
array_no = int(args.array_no)
output_dir = args.output_dir
# read .czi file path from csv reference table
ref_df = pd.read_csv(ref_csv)
czi_path = ref_df['files'].iloc[array_no]
print('czi path: ' + czi_path)
# load .czi file into MultiFinder instance
finder = find_cells.MultiFinder(czi_path)
print('MultiFinder created.')
# load bg file from multi-image .czi and add to finder
bg_tif_im = io.imread('/n/denic_lab/Lab/TH_Imaging/WIPI_empty_control.tif')
bg_tif_im = np.moveaxis(bg_tif_im, -1, 0)  # move C axis to 1st position
bg_tif_im = np.expand_dims(bg_tif_im, axis=0)
finder.bg_im = bg_tif_im
finder.bg_channels = [488, 561, 405]
print('background image added to MultiFinder.')
# initialize a CellSplitter from finder
splitter = segment_cells.CellSplitter(finder, threshold=800)
print('CellSplitter instance created.')
splitter.segment_nuclei(verbose=True)  # segment nuclei
print('Nuclei segmented.')
splitter.segment_cells(488, verbose=True)  # segment cells using the 488 wl
print('Cells segmented.')
# initialize a Foci instance from splitter
foci_obj = foci.Foci(splitter, verbose=True)
print('Foci instance created.')
foci_obj.segment(verbose=True)  # segment foci using PexSegmenter
print('Foci segmented.')
foci_obj.count_foci(verbose=True)  # count foci
print('Foci counted.')
print('n foci:')
print(foci_obj.foci_cts['488'])
print(foci_obj.foci_cts['561'])
foci_obj.measure_overlap(verbose=True)  # measure # of overlapping foci
print('overlap between channels measured.')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
print('outputting to csv...')
foci_obj.pandas_output(output_dir + '/' + str(array_no) + '.csv',
                       verbose=True)
# output images to check quality of segmentation later
print('outputting images...')
im_fname = foci_obj.filenames.split('/')[-1]
im_output_dir = output_dir + '/' + im_fname[:-4]
if not os.path.isdir(im_output_dir):
    os.makedirs(im_output_dir)
os.chdir(im_output_dir)
for i in range(0, len(foci_obj.segmented_nuclei)):
    io.imsave(str(i)+'_nuclei.tif', foci_obj.segmented_nuclei[i].astype('uint16'))
    io.imsave(str(i)+'_cells.tif', foci_obj.segmented_cells[i].astype('uint16'))
    for c in foci_obj.foci.keys():  # keys are channel ints
        io.imsave(str(i)+'_'+str(c)+'_foci.tif', foci_obj.foci[c][i].astype('uint16'))
for c in finder.cell_channels:
    ch_ims = finder.get_channel_arrays(c, bg=False)
    for i in range(0, ch_ims.shape[0]):
        io.imsave(str(i)+'_'+str(c)+'_raw.tif', ch_ims[i, :, :, :].astype('uint16'))
