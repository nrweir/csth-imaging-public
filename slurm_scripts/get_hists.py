import numpy as np
from skimage import io
import os
import scipy.ndimage as nd

os.chdir('/n/denic_lab/Lab/csth-output/svm_train_1')
ims = [f for f in os.listdir() if '.tif' in f]
ind = 0
grad_arr = np.empty(shape=(50, 500))
for im in ims:
    c_im = io.imread(im)
    grad_im = nd.gaussian_gradient_magnitude(
            c_im, sigma=(0.25, 0.25))
    grad_im = grad_im.astype('float32')/np.amax(grad_im.flatten())
    grad_im = grad_im*65535  # make 16 bit
    grad_im = grad_im.astype('uint16')
    print('calculating histogram...')
    hist, bin_edges = np.histogram(
        grad_im.flatten(), bins=50, range=(0, 65536))
    grad_arr[:, ind] = hist
    ind = ind + 1
np.save('/n/denic_lab/Lab/csth-output/svm_train_1/svm_training_set.npy',
        grad_arr)
