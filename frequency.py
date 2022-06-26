from enum import unique
import numpy as np
from scipy.fftpack import dctn, fftn
from skimage.color import rgb2gray
from skimage.util import view_as_windows


def compute_dct_coefficients(
    img, patch_size=9, log_transform=True, as_channels=True, dft=False
):
    """Compute the module of the patch wise DCT coefficients of an image.
    Optionnaly store these coeffcients as patches at each pixels locations.
    The default output is an array of shape (image_width, image_height, patch_width * patch_height).
    """
    img_gray = rgb2gray(img)
    if patch_size % 2 == 0:
        pad_before = (patch_size - 1) // 2
        pad_after = pad_before + 1
    else:
        pad_before = pad_after = patch_size // 2
    img_pad = np.pad(img_gray, (pad_before, pad_after), mode="reflect")
    rolling_window_view = view_as_windows(img_pad, patch_size)
    if dft:
        dct_img = fftn(rolling_window_view, axes=(-2, -1))
    else:
        dct_img = dctn(rolling_window_view, axes=(-2, -1), norm="ortho")
    dct_img = np.abs(dct_img)
    if log_transform:
        dct_img = np.log(dct_img + 1)
    if as_channels:
        dct_img = dct_img.reshape(dct_img.shape[0], dct_img.shape[1], -1)
    return dct_img


def compute_dct_centers(dct_img, labels, reduce_func=np.median):
    """Compute the agrgegated DCT of a superpixels.
    Basically it computes a patch wise DCT of each pixel within the superpixel and aggregate it using the mean or the median for example."""
    unique_labels = np.unique(labels)

    dct_superpixels = np.zeros((unique_labels.shape[0], dct_img.shape[-1]))

    for i, label in enumerate(unique_labels):
        mask_label = labels == label
        dct_label = dct_img[mask_label]
        mean_dct = reduce_func(dct_label, axis=0)
        dct_superpixels[i] = mean_dct

    return dct_superpixels
