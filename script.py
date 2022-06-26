from pixelisers import SLICPixeliser, EnergeticSLICPixeliser
from skimage.io import imread, imsave
from skimage.color import rgb2lab
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import numpy as np
from metrics import boundary_recall, undersegmentation_error
from utils import get_segmentation_file, get_segmentation
from pathlib import Path

img_path = Path("data/images/train/12003.jpg")
ground_truth_path = Path("data/groundTruth/train")
img = imread(img_path)
ground_truth = get_segmentation_file(img_path, ground_truth_path)
ground_truth_segmentation, ground_truth_boundaries = get_segmentation(ground_truth)
gt_boundaries_skimage = find_boundaries(ground_truth_segmentation)

slic_pixeliser = SLICPixeliser(
    connectivity=True, max_iter=10, m_spatial=30, step=30, clever_init=True
)
labels = slic_pixeliser.pixelate(img)
boundaries = find_boundaries(labels)
print(
    "undersegmentation error",
    undersegmentation_error(labels, ground_truth_segmentation),
)
print("boundary_recall", boundary_recall(boundaries, gt_boundaries_skimage))
img_boundaries = np.where(boundaries[:, :, None], 255, img)
imsave(
    "data/segmented_images/img_script.png",
    img_boundaries,
)

eslic_pixeliser = EnergeticSLICPixeliser(
    connectivity=True,
    max_iter=10,
    m_spatial=30,
    m_frequency=90,
    step=30,
    clever_init=True,
)
labels = eslic_pixeliser.pixelate(img)
boundaries = find_boundaries(labels)

img_boundaries = np.where(boundaries[:, :, None], 255, img)
print(
    "undersegmentation error energetic",
    undersegmentation_error(labels, ground_truth_segmentation),
)
print("boundary_recall energetic", boundary_recall(boundaries, gt_boundaries_skimage))
imsave(
    "data/segmented_images/img_script_dct.png",
    img_boundaries,
)
