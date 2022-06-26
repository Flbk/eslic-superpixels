from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial import KDTree
from scipy.ndimage import convolve
from skimage.color import rgb2gray, rgb2lab
from skimage import measure
from skimage.segmentation import find_boundaries
from PIL import Image


def get_segmentation_object(gt_file_path):
    assert gt_file_path.is_file()
    return loadmat(gt_file_path)["groundTruth"]


def retrieve_segmentation_file(img_path, gt_path):
    """Loads a segmentation file."""
    image_id = img_path.stem
    for gt_file in gt_path.iterdir():
        if gt_file.stem == image_id:
            return loadmat(ground_truth_file)["groundTruth"]
    raise FileNotFoundError


def get_segmentation(ground_truth_file, n=0):
    """Extract one segmentation and the corresponding boundaries from a list of segmentations."""
    n = np.minimum(n, ground_truth_file.shape[0])
    segmentation, _ = ground_truth_file[0, n].item()
    return segmentation


def load_segmentations(gt_object):
    """Extract one segmentation and the corresponding boundaries from a list of segmentations."""
    n_segmentations = gt_object[0].shape[0]
    list_segmentations = []
    list_boundaries = []
    for i in range(n_segmentations):
        segmentation, _ = gt_object[0, i].item()
        list_segmentations.append(segmentation)
        boundaries = find_boundaries(segmentation)
        list_boundaries.append(boundaries)
    return list_segmentations, list_boundaries


def resize_to_block_shape(img_shape, block_shape):
    y_axis = img_shape[0]
    x_axis = img_shape[1]

    if isinstance(block_shape, int):
        block_shape = (block_shape, block_shape)

    new_y = np.ceil(y_axis / block_shape[0]) * block_shape[0]
    new_x = np.ceil(x_axis / block_shape[1]) * block_shape[1]

    return (int(new_y), int(new_x))


def get_pad_block(img, block_shape):
    new_img_shape = resize_to_block_shape(img.shape[0:2], block_shape)
    pad_y = new_img_shape[0] - img.shape[0]
    pad_x = new_img_shape[1] - img.shape[1]
    return pad_y, pad_x


def pad_block_size(img, block_shape):
    pad_y, pad_x = get_pad_block(img, block_shape)
    img_pad = np.pad(img, ((0, pad_y), (0, pad_x), (0, 0)))
    return img_pad


def create_grid(h, w):
    grid = np.empty((h, w, 2))
    grid[:, :, 0] = np.arange(h)[:, None]
    grid[:, :, 1] = np.arange(w)[None, :]
    return grid


def create_grid_like(img):
    h, w = img.shape[:2]
    return create_grid(h, w)


def get_sparse_distance_matrix(a, b=None, leafsize=30, *args, **kwargs):
    kd_tree_a = KDTree(a, leafsize=leafsize)
    if b is None:
        kd_tree_b = kd_tree_a
    else:
        kd_tree_b = KDTree(b, leafsize=leafsize)
    dist_matrix = kd_tree_a.sparse_distance_matrix(kd_tree_b, *args, **kwargs).tocoo()
    return dist_matrix


def compute_sobel():
    base_sobel = np.zeros((3, 3))
    sobel_col = np.array([1, 2, 1])
    base_sobel[:, 0] = sobel_col
    base_sobel[:, 2] = -sobel_col

    Gx = base_sobel
    Gy = base_sobel.T
    return Gx, Gy


def gradient(img):
    if len(img.shape) == 3:
        img = rgb2gray(img)
    Gx, Gy = compute_sobel()
    gx = convolve(img, Gx)
    gy = convolve(img, Gy)

    g_magnitude = np.hypot(gx, gy)
    g_dir = np.arctan2(gy, gx)

    return g_magnitude, g_dir


def get_main_orientation_image(img, q=0.99):
    g_magnitude, g_dir = gradient(img)
    quantile_magnitude = np.quantile(g_magnitude, q)
    main_gradients_mask = g_magnitude > quantile_magnitude
    abs_g_dir = np.abs(g_dir)

    return abs_g_dir[main_gradients_mask].mean()


def identify_biggest_connected_component(binary_img):
    """Identify the biggest connected component of a binary image.
    If two components are of the same size it will return on of the two components."""
    connected_components = measure.label(binary_img, background=0)
    unique_components, counts_components = np.unique(
        connected_components, return_counts=True
    )
    ind_biggest_component = np.argmax(counts_components[1:])
    return np.isin(
        connected_components, unique_components[1:][ind_biggest_component]
    ) & (connected_components != 0)


class WindowXY:
    """Class for generating windows around a pixel in an image"""

    def __init__(
        self, x: int, y: int, window_size: int, w_max: int, h_max: int
    ) -> None:
        """
        Generates a (2*window_size + 1)*(2*window_size + 1) window around pixel at
            position (x,y) but stay under the boundaries (0->w_max, 0->h_max)

        use w_size = 1 to get all candidates in a 3*3 window around (x,y) and
            update centers with gradients with another function

        use w_size = step to get all candidates in a (2*step)*(2*step) window
            around (x,y) and use K-means with another function

        window_xy(10,15,3) gives self.candidates_XY = [[7,12], [8,12], ..., [13,18]]
        """
        x, y = int(x), int(y)
        self.x_min = max(0, x - window_size)
        self.x_max = min(w_max - 1, x + window_size)
        self.y_min = max(0, y - window_size)
        self.y_max = min(h_max - 1, y + window_size)

        XX = np.arange(self.x_min, self.x_max + 1)
        YY = np.arange(self.y_min, self.y_max + 1)
        XX, YY = np.meshgrid(XX, YY)
        self.candidates_XY = np.array(list(zip(np.nditer(XX), np.nditer(YY))))


def get_XYLAB(img):
    h, w, c = img.shape
    assert c == 3, "Not a BGR image"

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    img_LAB = rgb2lab(img)

    img_XYLAB = np.dstack([Y, X, img_LAB])
    return img_XYLAB


def plot_median_color(img, segmentation):
    segmented_img = np.zeros_like(img)
    for label in np.unique(segmentation):
        mask = segmentation == label
        segmented_img[mask] = np.median(img[mask], axis=0)

    return segmented_img


def plot_boundary(img, boundaries):
    return np.where(boundaries[:, :, None], 255, img)


def rescale_segmentation(segmentation, ratio):
    new_size = int(max(segmentation.shape) * ratio)
    im = Image.fromarray(segmentation.astype(np.uint16))
    im.thumbnail((new_size, new_size), resample=Image.NEAREST)
    return np.array(im)
