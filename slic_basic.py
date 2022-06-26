from os import walk
from os.path import splitext
import logging
from time import time
from tqdm import tqdm
import numpy as np
from skimage.color import label2rgb
import cv2

from connectivity import connectivity

logging.basicConfig(
    filename="slic.log", filemode="w", encoding="utf-8", level=logging.DEBUG
)

DATAPATH = "data/images/train/"
nb_img = 10

for img_name in tqdm(list(walk(DATAPATH))[0][2][:nb_img]):
    img = cv2.imread(DATAPATH + img_name)
    img_name, _ = splitext(img_name)
    h, w, c = img.shape
    assert c == 3, "Not a BGR image"

    def window_xy(x: int, y: int, w_size: int):
        """use w_size = 1 to get all candidates in a 3*3 window around (x,y) and
        update centers with gradients with another function

        use w_size = step to get all candidates in a (2*step)*(2*step) window
        around (x,y) and use K-means with another function

        window_xy(10,15,3) -> [[7,12], [8,12], ..., [13,18]]
        """
        # x, y = x.astype(np.uint8), y.astype(np.uint8)
        x_min = max(0, x - w_size)
        x_max = min(w - 1, x + w_size)
        y_min = max(0, y - w_size)
        y_max = min(h - 1, y + w_size)

        XX = np.arange(x_min, x_max + 1)
        YY = np.arange(y_min, y_max + 1)
        XX, YY = np.meshgrid(XX, YY)
        candidates = np.array(list(zip(np.nditer(XX), np.nditer(YY))))
        return candidates, x_min, x_max, y_min, y_max

    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # m should be in range [1, 40]
    m = 30

    # step size
    step = 40

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    img_XYLAB = np.dstack([X, Y, img_LAB])

    # init centers as a list of array coordinates
    centers_X = step // 2 + np.arange(0, w - step // 2, step)
    centers_Y = step // 2 + np.arange(0, h - step // 2, step)
    centers_X, centers_Y = np.meshgrid(centers_X, centers_Y)
    centers_XY = np.array(list(zip(np.nditer(centers_X), np.nditer(centers_Y))))

    # clever center initialisation
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x, grad_y = np.gradient(img_gray)
    grad_img = (
        grad_x ** 2 + grad_y ** 2
    )  # no need to np.sqrt because only relative order matters
    for k, center in enumerate(centers_XY):
        small_window, *_ = window_xy(center[0], center[1], 1)
        x_min, y_min = small_window[0]
        x_max, y_max = small_window[-1]
        # WARNING should not be called if center is on the edges of the image :
        # the small window should be entirely in the image
        sub_grad = grad_img[y_min : y_max + 1, x_min : x_max + 1]
        new_center = np.unravel_index(np.argmin(sub_grad, axis=None), sub_grad.shape)
        centers_XY[k] = np.array(
            [center[0] + new_center[0] - 1, center[1] + new_center[1] - 1]
        )

    centers_XYLAB = np.array(
        [[x, y, *img_LAB[y, x, :].flatten()] for (x, y) in centers_XY]
    )

    # labels and distance init
    labels = -np.ones((h, w))
    distances = 1e8 * np.ones((h, w))

    def distance_xylab(A: np.ndarray, B: np.ndarray, m: int = m, S: int = step):
        """computes the distance between points A and B"""
        A_xy, A_lab = A[:2], A[2:]
        B_xy, B_lab = B[:2], B[2:]
        d_color_squared = np.sum((A_lab - B_lab) ** 2)
        d_spatial_squared = np.sum((A_xy - B_xy) ** 2)
        D_squared = d_color_squared + (m / S) ** 2 * d_spatial_squared
        return np.sqrt(D_squared)

    def show_img(iteration):
        # manual superpixel map
        gradient_x, gradient_y = np.gradient(labels)
        superpixel_map = np.logical_or(gradient_x, gradient_y)
        superpixel_map = 255 * np.repeat(np.expand_dims(superpixel_map, 2), 3, 2)
        assert superpixel_map.shape == img.shape

        img_slicced = np.clip(img + superpixel_map, a_min=0, a_max=255).astype(np.uint8)
        img_segmented = cv2.applyColorMap(
            labels.astype(np.uint8), colormap=cv2.COLORMAP_JET
        )
        img_segmented = np.clip(
            (img_segmented).astype(np.uint8) + superpixel_map, a_min=0, a_max=255
        ).astype(np.uint8)
        label_colored = label2rgb(labels)
        while True:
            cv2.imshow(f"image slicced {iteration+1}", img_slicced)
            cv2.imshow(f"image cmap {iteration+1}", label_colored)
            cv2.imwrite(f"img_slicced.jpg", img_slicced)
            cv2.imwrite(f"img cmap.jpg", (255 * label_colored).astype(int))
            cv2.waitKey(-1)
            cv2.destroyAllWindows()
            break

    max_iter = 5
    error_threshold = 1
    error = 10 * error_threshold
    for iteration in tqdm(range(max_iter)):
        t_it = time()
        if error >= error_threshold:
            centers_old_XY = centers_XYLAB[:, :2].copy()
            for k, center in tqdm(enumerate(centers_XYLAB)):
                t_center = time()
                x, y, *_ = center
                candidates_XY, x_min, x_max, y_min, y_max = window_xy(x, y, step)
                candidates_LAB = img_LAB[y_min : y_max + 1, x_min : x_max + 1, :]
                candidates_LAB = candidates_LAB.reshape(-1, 3)
                candidates_XYLAB = np.column_stack((candidates_XY, candidates_LAB))
                # assign candidates to this cluster if distance is small enough
                for candidate_XYLAB in candidates_XYLAB:
                    candidate_X, candidate_Y, *_ = candidate_XYLAB
                    D = distance_xylab(center, candidate_XYLAB, m=m, S=step)
                    current_distance = distances[candidate_Y, candidate_X]
                    if D < current_distance:
                        distances[candidate_Y, candidate_X] = D
                        labels[candidate_Y, candidate_X] = k
                logging.debug(
                    f"Did iteration {iteration+1} : center {k} in {time()-t_center} s"
                )

            # compute new cluster centers
            t_new_cluster_centers = time()
            centers_new_XY = np.zeros_like(centers_old_XY)
            for k, center in enumerate(centers_XYLAB):
                k_mask = np.ma.masked_not_equal(labels, k)
                k_mask_extended = np.repeat(np.expand_dims(k_mask.mask, 2), 5, 2)
                img_k_masked = np.ma.masked_array(img_XYLAB, mask=k_mask_extended)
                updated_center_k = np.mean(img_k_masked.reshape(-1, 5), axis=0).data
                centers_XYLAB[k] = updated_center_k.astype(np.uint32)
                centers_new_XY[k] = updated_center_k[:2]
            logging.debug(
                f"Did iteration {iteration+1} : new centers computation in {time()-t_new_cluster_centers} s"
            )

            error = np.mean((centers_new_XY - centers_old_XY) ** 2)
            logging.debug(f"Did iteration {iteration+1} in {time()-t_it} s")
            if iteration == max_iter:
                show_img(iteration)

    # np.save("labels", labels)

    # TODO enforce connectivity
    labels_connected = connectivity(labels, h, w, step)
    gradient_x, gradient_y = np.gradient(labels_connected)
    superpixel_map = np.logical_or(gradient_x, gradient_y)
    superpixel_map = 255 * np.repeat(np.expand_dims(superpixel_map, 2), 3, 2)
    assert superpixel_map.shape == img.shape

    img_slicced = np.clip(img + superpixel_map, a_min=0, a_max=255).astype(np.uint8)
    cv2.imwrite(f"res/{img_name}_slicced.jpg", img_slicced)
    cv2.imwrite(
        f"res/{img_name}_labels.jpg", (255 * label2rgb(labels_connected)).astype(int)
    )
    np.save(f"res/{img_name}_labels_connected", labels_connected)
