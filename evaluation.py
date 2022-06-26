from utils import load_segmentations, get_segmentation_object
from metrics import undersegmentation_error, boundary_recall
from skimage.io import imread
from pixelisers import SLICPixeliser, EnergeticSLICPixeliser
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from skimage.segmentation import find_boundaries
import numpy as np
import json
from joblib import Parallel, delayed
import time


def evaluate_pixeliser(pixeliser, img_file_path, gt_file_path):
    gt_object = get_segmentation_object(gt_file_path)
    list_gt_segmentations, list_gt_boundaries = load_segmentations(gt_object)

    img = imread(img_file_path)

    segmentation = pixeliser.pixelate(img)
    boundaries = find_boundaries(segmentation)

    list_boundary_recall = []
    list_undersegmentation_error = []

    for gt_segmentation, gt_boundary in zip(list_gt_segmentations, list_gt_boundaries):
        list_undersegmentation_error.append(
            undersegmentation_error(segmentation, gt_segmentation)
        )
        list_boundary_recall.append(boundary_recall(boundaries, gt_boundary))

    return np.mean(list_undersegmentation_error), np.mean(list_boundary_recall)


def evaluate_img_pipeline(pixeliser, img_file_path, gt_dir_path):
    for gt_file_path in gt_dir_path.iterdir():
        if gt_file_path.stem == img_file_path.stem:
            break
    return evaluate_pixeliser(pixeliser, img_file_path, gt_file_path)


def evaluate_on_whole_directory(
    pixeliser, img_dir_path, gt_dir_path, max_evaluation=None
):
    list_img_file_path = [
        file_path
        for file_path in img_dir_path.iterdir()
        if file_path.suffix in [".png", ".jpg"]
    ]
    if max_evaluation is None:
        max_evaluation = len(list_img_file_path)

    list_undersegmentation_error = []
    list_boundary_recall = []
    for img_file_path in tqdm(list_img_file_path[:max_evaluation]):
        undersegmentation_error, boundary_recall = evaluate_img_pipeline(
            pixeliser, img_file_path, gt_dir_path
        )
        list_undersegmentation_error.append(undersegmentation_error)
        list_boundary_recall.append(boundary_recall)
    return list_undersegmentation_error, list_boundary_recall


def evaluate_on_whole_directory(
    pixeliser,
    img_dir_path,
    gt_dir_path,
    max_evaluation=None,
):
    list_img_file_path = [
        file_path
        for file_path in img_dir_path.iterdir()
        if file_path.suffix in [".png", ".jpg"]
    ]
    if max_evaluation is None:
        max_evaluation = len(list_img_file_path)

    list_undersegmentation_error = []
    list_boundary_recall = []

    parallel_eval = Parallel(n_jobs=-1)(
        delayed(evaluate_img_pipeline)(pixeliser, img_file_path, gt_dir_path)
        for img_file_path in list_img_file_path[:max_evaluation]
    )

    for undersegmentation_error, boundary_recall in parallel_eval:
        list_undersegmentation_error.append(undersegmentation_error)
        list_boundary_recall.append(boundary_recall)
    return list_undersegmentation_error, list_boundary_recall


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the evaluation of pixelisers on a set of images"
    )

    parser.add_argument("--img_dir_path", help="Image directory path", type=Path)
    parser.add_argument("--gt_dir_path", help="Ground truth data path", type=Path)
    parser.add_argument(
        "--max_evaluation", help="Maximum number of evaluation", default=None, type=int
    )
    parser.add_argument(
        "--pixeliser", help="The type of pixelisers to use.", default="slic+eslic"
    )
    parser.add_argument("--m_spatial", default=30.0, type=float)
    parser.add_argument("--m_frequency", default=50.0, type=float)
    parser.add_argument(
        "--patch_size", help="Patch size for energetic slic.", default=9, type=int
    )
    parser.add_argument("--disable_log_transform", action="store_true")
    parser.add_argument(
        "--dft", action="store_true", help="Whether to use DFT instead of DCT"
    )

    parser.add_argument("--step", default=50, type=int)
    parser.add_argument("--max_iter", default=10, type=float)
    parser.add_argument("--threshold_error", default=1.0, type=float)
    parser.add_argument("--disable_connectivity", action="store_true")
    parser.add_argument("--disable_clever_init", action="store_true")

    args = parser.parse_args()

    list_pixeliser_names = sorted(args.pixeliser.split("+"))
    dict_results = defaultdict(dict)

    start = time.time()
    for pixeliser_name in list_pixeliser_names:
        if pixeliser_name == "slic":
            pixeliser = SLICPixeliser(
                m_spatial=args.m_spatial,
                step=args.step,
                max_iter=args.max_iter,
                threshold_error=args.threshold_error,
                connectivity=not args.disable_connectivity,
                clever_init=not args.disable_clever_init,
            )
        elif pixeliser_name == "eslic":
            pixeliser = EnergeticSLICPixeliser(
                m_spatial=args.m_spatial,
                m_frequency=args.m_frequency,
                patch_size=args.patch_size,
                step=args.step,
                max_iter=args.max_iter,
                threshold_error=args.threshold_error,
                connectivity=not args.disable_connectivity,
                clever_init=not args.disable_clever_init,
                log_transform=not args.disable_log_transform,
                dft=args.dft,
            )

        else:
            raise NameError("Pixeliser not defined")
        print(f"Running {pixeliser_name}...")

        (
            list_undersegmentation_error,
            list_boundary_recall,
        ) = evaluate_on_whole_directory(
            pixeliser,
            args.img_dir_path,
            args.gt_dir_path,
            max_evaluation=args.max_evaluation,
        )
        dict_results[pixeliser_name]["boundary_recall"] = np.mean(list_boundary_recall)
        dict_results[pixeliser_name]["undersegmentation_error"] = np.mean(
            list_undersegmentation_error
        )

    print(f"Evaluation time: {time.time() - start:.3f}")
    print(json.dumps(dict_results, sort_keys=True, indent=4))
