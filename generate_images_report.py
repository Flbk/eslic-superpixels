from skimage.io import imread, imsave
from pathlib import Path
from utils import (
    load_segmentations,
    get_segmentation_object,
    plot_boundary,
    plot_median_color,
)
from pixelisers import SLICPixeliser
import argparse
from skimage.segmentation import find_boundaries
from metrics import undersegmentation_error, boundary_recall

FIGSIZE = (10, 7.5)
SAVE_PATH = Path("data/report/")


def fig_ax(figsize=FIGSIZE):
    return plt.subplots(1, figsize=figsize)


def imwrite(img, img_name):
    imsave(f"{SAVE_PATH / img_name}.png", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir_path", default="data/images/train")
    parser.add_argument("--gt_dir_path", default="data/groundTruth/train")

    args = parser.parse_args()
    img_dir_path = Path(args.img_dir_path)
    gt_dir_path = Path(args.gt_dir_path)
    img_ind = 0
    img_path = img_dir_path / "12003.jpg"
    img = imread(img_path)

    for gt_path in gt_dir_path.iterdir():
        if gt_path.stem == img_path.stem:
            gt_object = get_segmentation_object(gt_path)
            break

    list_segmentations, list_boundaries = load_segmentations(gt_object)

    # base img
    imwrite(img, "illustration-img")

    # illustrate two different segmentations
    segmented_img_1 = plot_median_color(img, list_segmentations[0])
    imwrite(segmented_img_1, "segmentation-illustration-1")
    segmented_img_2 = plot_median_color(img, list_segmentations[2])
    imwrite(segmented_img_2, "segmentation-illustration-2")

    # illustrate boundaries
    img_w_boundaries_1 = plot_boundary(img, list_boundaries[0])
    imwrite(img_w_boundaries_1, "boundaries-illustration-1")
    img_w_boundaries_2 = plot_boundary(img, list_boundaries[2])
    imwrite(img_w_boundaries_2, "boundaries-illustration-2")

    # illustrate undersegmentation error
    print("illustration undersegmentation error")
    slic = SLICPixeliser(
        m_spatial=90,
        step=30,
    )
    slic_segmentation_big_m = slic.pixelate(img)
    slic_boundaries_big_m = find_boundaries(slic_segmentation_big_m)
    print(
        "big undersegmentation error",
        undersegmentation_error(slic_segmentation_big_m, list_segmentations[0]),
    )
    boundaries_img_big_m = plot_boundary(img, slic_boundaries_big_m)
    print(
        "low boundary recall",
        boundary_recall(slic_segmentation_big_m, list_boundaries[0]),
    )
    imwrite(boundaries_img_big_m, "undersegmentation-big-m")

    slic = SLICPixeliser(
        m_spatial=30,
        step=30,
    )
    slic_segmentation_small_m = slic.pixelate(img)
    print(
        "small undersegmentation error",
        undersegmentation_error(slic_segmentation_small_m, list_segmentations[0]),
    )
    slic_boundaries_small_m = find_boundaries(slic_segmentation_small_m)
    boundaries_img_small_m = plot_boundary(img, slic_boundaries_small_m)
    print(
        "high boundary recall",
        boundary_recall(slic_boundaries_small_m, list_boundaries[0]),
    )
    imwrite(boundaries_img_small_m, "undersegmentation-small-m")
