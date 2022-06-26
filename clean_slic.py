"""Clean implementation of SLIC algorithm with classes for better modularity"""

from pathlib import Path
from typing import Literal
import logging
from tqdm import tqdm
import numpy as np
from skimage import io
from skimage.color import label2rgb, rgb2lab, rgb2gray
from connectivity import connectivity

logging.basicConfig(
    filename="slic.log", filemode="w", encoding="utf-8", level=logging.DEBUG
)


def load_img(img_path):
    """opens and load the image, converts it to LAB too"""
    img = io.imread(img_path)
    h, w, c = img.shape
    assert c == 3, "Not a BGR image"

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    img_LAB = rgb2lab(img)
    img_XYLAB = np.dstack([X, Y, img_LAB])

    return img, img_LAB, img_XYLAB


def get_XYLAB(img):
    h, w, c = img.shape
    assert c == 3, "Not a BGR image"

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    img_LAB = rgb2lab(img)
    img_XYLAB = np.dstack([X, Y, img_LAB])
    return img_XYLAB


TRAIN_GROUND_TRUTH_PATH = "data/groundTruth/train"
TEST_GROUND_TRUTH_PATH = "data/groundTruth/test"
VAL_GROUND_TRUTH_PATH = "data/groundTruth/val"
TRAIN_IMAGES_PATH = "data/images/train"
TEST_IMAGES_PATH = "data/images/test"
VAL_IMAGES_PATH = "data/images/val"


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


def init_centers(img, step, clever_init):
    """init centers as a list of array coordinates
    also gives them in XYLAB space
    init can be clever (see class init method)"""
    h, w, _ = img.shape
    centers_X = step // 2 + np.arange(0, w - step // 2, step)
    centers_Y = step // 2 + np.arange(0, h - step // 2, step)
    centers_X, centers_Y = np.meshgrid(centers_X, centers_Y)
    centers_XY = np.array(list(zip(np.nditer(centers_X), np.nditer(centers_Y))))

    # clever center initialisation
    if clever_init:
        img_gray = rgb2gray(img)
        grad_x, grad_y = np.gradient(img_gray)
        grad_img = (
            grad_x ** 2 + grad_y ** 2
        )  # no need to np.sqrt because only relative order matters
        for k, (X_center, Y_center) in enumerate(centers_XY):
            small_window = WindowXY(X_center, Y_center, 1, w, h)
            # WARNING should not be called if center is on the edges of the image :
            # the small window should be entirely in the image
            sub_grad = grad_img[
                small_window.y_min : small_window.y_max + 1,
                small_window.x_min : small_window.x_max + 1,
            ]
            Y_jitter, X_jitter = np.unravel_index(
                np.argmin(sub_grad, axis=None), sub_grad.shape
            )
            centers_XY[k] = np.array(
                [
                    min(X_center + X_jitter - 1, w),
                    min(Y_center + Y_jitter - 1, h),
                ]
            )

    return centers_XY


def distance_xylab(vect_1: np.ndarray, vect_2: np.ndarray, m: int, step: int):
    """computes the distance between points vect_1 and vect_2 in the XYLAB space"""
    A_xy, A_lab = vect_1[:2], vect_1[2:]
    B_xy, B_lab = vect_2[:2], vect_2[2:]
    d_color_squared = np.sum((A_lab - B_lab) ** 2)
    d_spatial_squared = np.sum((A_xy - B_xy) ** 2)
    D_squared = d_color_squared + (m / step) ** 2 * d_spatial_squared
    return np.sqrt(D_squared)


def one_iteration(img_LAB, m, step, centers_old_XYLAB):
    """executes one iteration of the SLIC algorithm
    updates labels and distances matrices
    """
    h, w, _ = img_LAB.shape
    list_dist = []
    distances = np.inf * np.ones((h, w), dtype=float)
    for k, center in tqdm(enumerate(centers_old_XYLAB)):
        list_dist_center = []
        x, y, *_ = center
        window = WindowXY(x, y, step, w, h)
        candidates_LAB = img_LAB[
            window.y_min : window.y_max + 1, window.x_min : window.x_max + 1, :
        ]
        candidates_LAB = candidates_LAB.reshape(-1, 3)
        candidates_XYLAB = np.column_stack((window.candidates_XY, candidates_LAB))
        # assign candidates to this cluster if distance is small enough
        for candidate_XYLAB in candidates_XYLAB:
            candidate_X, candidate_Y, *_ = candidate_XYLAB
            candidate_X, candidate_Y = int(candidate_X), int(candidate_Y)
            D = distance_xylab(center, candidate_XYLAB, m=m, step=step)
            current_distance = distances[candidate_Y, candidate_X]

            list_dist_center.append(D)
            if D < current_distance:
                distances[candidate_Y, candidate_X] = D
                labels[candidate_Y, candidate_X] = k
        list_dist.append(list_dist_center)
    print("sort distance first cluster", np.sort(list_dist[0])[:5])


def new_cluster_centers(img_XYLAB, centers_XYLAB, centers_old_XY, labels):
    """compute new cluster centers"""
    centers_new_XY = np.zeros_like(centers_old_XY)
    centers_new_XYLAB = np.zeros_like(centers_XYLAB)
    nb_centers = len(centers_XYLAB)
    for k in range(nb_centers):
        k_mask = np.ma.masked_not_equal(labels, k)
        k_mask_extended = np.repeat(np.expand_dims(k_mask.mask, 2), 5, 2)
        img_k_masked = np.ma.masked_array(img_XYLAB, mask=k_mask_extended)
        updated_center_k = np.mean(img_k_masked.reshape(-1, 5), axis=0).data
        centers_new_XYLAB[k] = updated_center_k.astype(np.int32)
        centers_new_XY[k] = updated_center_k[:2]
    return centers_new_XYLAB, centers_new_XY


def visualization(img, labels_connected, centers_XYLAB):
    """computes several output images for visualization purposes"""
    # do the rest in another function
    # -> find boundaries
    gradient_x, gradient_y = np.gradient(labels_connected)
    superpixel_map = np.logical_or(gradient_x, gradient_y)
    superpixel_map = 255 * np.repeat(np.expand_dims(superpixel_map, 2), 3, 2)
    assert superpixel_map.shape == img.shape

    # img with white slic boundaries. Filled with original content
    img_slicced = np.clip(img + superpixel_map, a_min=0, a_max=255).astype(np.uint8)

    # img with superpixels filled with flashy colors
    labels_colored = label2rgb(labels_connected)
    labels_colored = np.clip(
        255 * labels_colored + superpixel_map, a_min=0, a_max=255
    ).astype(np.uint8)

    # superpixels filled with mean color of superpixel
    labels_mean = np.zeros_like(img)
    for k in range(len(centers_XYLAB)):
        k_mask = np.ma.masked_not_equal(labels_connected, k)
        k_mask_extended = np.repeat(np.expand_dims(k_mask.mask, 2), 3, 2)  # 3 not 5
        img_k_masked = np.ma.masked_array(img, mask=k_mask_extended, fill_value=0.0)
        # unique BGR color of superpixel
        k_superpixel_color = (
            img_k_masked.reshape(-1, 3).mean(axis=0).astype(np.uint8).data
        )
        labels_mean += (1 - img_k_masked.mask.astype(np.uint8)) * k_superpixel_color
    labels_mean = np.clip(
        labels_mean + superpixel_map,
        a_min=0,
        a_max=255,
    ).astype(np.uint8)

    return {
        "img_slicced": img_slicced,
        "labels_colored": labels_colored,
        "labels_mean": labels_mean,
    }


def save_results(
    thing, output_repo, thing_basename, thing_type, kind=Literal["image", "array"]
):
    """save results as arrays and images"""
    output_repo.mkdir(exist_ok=True)
    if kind == "image":
        io.imsave(
            output_repo / (thing_basename + "_" + thing_type + ".jpg"),
            thing,
            quality=100,
        )
    elif kind == "array":
        np.save(output_repo / (thing_basename + "_" + thing_type), thing)


def SLIC(
    img_path: str,
    m: int = 30,
    step: int = 50,
    max_iter: int = 10,
    threshold_error: float = 1.0,
    clever_init: bool = True,
    do_connectivity: bool = True,
) -> None:
    """SLIC algorithm
    m: compactness
    step: distance between two adjacent clusters at the first iteration
        (nb_clusters ~= nb_pixels / step**2)
    max_iter: maximal number of iterations before stop. 10 should be fine
    threshold_error: threshold under which the algo stops iterating
        corresponds to mean movement of cluster centers in nb of pix
        between previous and current iterations
    save_result: save arrays and images to custom repo or not
    clever_init: look in 3*3 windows around regular-spaced initial cluster centers
        to not fall on an edge
    """
    basedir_path = img_path.parent.name
    output_repo = Path(
        f"data/{basedir_path}_m={m:d}_step={step:d}_max_iter={max_iter:d}_"
        f"threshold_error={threshold_error:.1f}_clever_init={clever_init}"
    )
    current_iteration = 1
    current_error = np.inf

    img, img_LAB, img_XYLAB = load_img(img_path)
    centers_XY = init_centers(img, step=step, clever_init=clever_init)
    centers_XYLAB = np.array(
        [[x, y, *img_LAB[y, x, :].flatten()] for (x, y) in centers_XY]
    )
    print(centers_XYLAB.shape)

    global labels
    distances = np.inf * np.ones(img.shape[:-1])
    labels = np.zeros(img.shape[:-1])
    print(centers_XYLAB[:5])
    while current_iteration <= max_iter and current_error > threshold_error:
        one_iteration(img_LAB, m, step, centers_XYLAB)
        centers_XYLAB, centers_new_XY = new_cluster_centers(
            img_XYLAB, centers_XYLAB, centers_XY, labels
        )
        current_error = np.mean((centers_new_XY - centers_XY) ** 2)
        centers_XY = centers_new_XY
        current_iteration += 1
        print("error", current_error)
        print("new centers", centers_XYLAB[:5])
        print("m", m)
    if do_connectivity:
        labels_connected = connectivity(labels, step)
    else:
        labels_connected = labels
    dict_imgs_to_save = visualization(img, labels_connected, centers_XYLAB)
    for img_to_save_type, img_to_save in dict_imgs_to_save.items():
        save_results(
            img_to_save,
            output_repo,
            img_path.stem,
            img_to_save_type,
            "image",
        )
    save_results(
        labels,
        output_repo,
        img_path.stem,
        "labels_raw",
        "array",
    )


if __name__ == "__main__":
    basedir_path = Path("data/France")
    for path in tqdm(basedir_path.iterdir()):
        SLIC(path, m=37, max_iter=10)
