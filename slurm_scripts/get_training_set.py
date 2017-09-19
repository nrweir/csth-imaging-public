"""A script for getting training images for SVM classification of focus."""

import numpy as np
import pandas as pd
from skimage import io
import sys
import scipy.ndimage as nd
sys.path.append('/n/denic_lab/Users/nweir/python_packages/')
sys.path.append(
    '/n/denic_lab/Users/nweir/python_packages/csth-imaging/dependencies/')
from csth_analysis import czi_io

# steps to this script:
# 1. generate a set of images/planes to extract
# 2. extract those planes
# 3. generate the gradient histogram matrix for training
# 4. save the planes with a numerical identifier as the fname
print('generating random draws...')
# draw all from the set of 46 .czi files from the lc3/p62 expt
czi_vector = np.random.randint(0, 46, 500)
# draw from the 10 possible images in each .czi
im_vector = np.random.randint(0, 10, 500)
# draw slices from all 51 possible slices in each image
slice_vector = np.random.randint(0, 51, 500)
# reorder both vectors so that i can go through images one at a time
order_vect = np.argsort(czi_vector)
czi_vector = czi_vector[order_vect]
im_vector = im_vector[order_vect]
slice_vector = slice_vector[order_vect]
# open czi ref_csv
ref_df = pd.read_csv(
    '/n/denic_lab/Users/nweir/python_packages/csth-imaging/im_list_lc3_p62_2.csv')
ind = 0
grad_arr = np.empty(shape=(50, 500))
print('beginning image processing....')
print('----------------------------------------------------------------------')
for i in np.unique(czi_vector):
    print('Processing czi #' + str(i+1) + ' of 46')
    c_ims = im_vector[czi_vector == i]
    c_slices = slice_vector[czi_vector == i]
    c_czi = ref_df['files'].iloc[i]
    print('     opening czi...')
    (im_array, channels) = czi_io.load_multi_czi(c_czi)
    im_array = im_array[:, channels.index(561), :, :, :]  # get 561 sub-array
    for j in range(0, c_ims.shape[0]):
        print('         processing slice ' + str(j+1) +
              ' of ' + str(c_ims.shape[0]))
        curr_slice = im_array[c_ims[j], c_slices[j], :, :].astype('uint16')
        io.imsave('/n/denic_lab/Lab/csth-output/svm_train_1/' +
                  str(ind) + '.tif', curr_slice)
        print('         generating gradient...')
        grad_im = nd.gaussian_gradient_magnitude(
            curr_slice, sigma=(0.25, 0.25))
        grad_im = grad_im.astype('float32')/np.amax(grad_im.flatten())
        grad_im = grad_im*65535  # make 16 bit
        print('         calculating histogram...')
        hist, bin_edges = np.histogram(
            grad_im.flatten(), bins=50, range=(0, 65536))
        grad_arr[:, ind] = hist
        ind = ind + 1
np.save('/n/denic_lab/Lab/csth-output/svm_train_1/svm_training_set.npy',
        grad_arr)
