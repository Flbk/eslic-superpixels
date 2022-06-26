from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from utils import (
    resize_to_block_shape,
    identify_biggest_connected_component,
    get_XYLAB,
    WindowXY,
)
from scipy.sparse import csr_matrix
from sklearn.neighbors import (
    KDTree,
    RadiusNeighborsTransformer,
)
from sklearn.metrics import pairwise_distances
from frequency import compute_dct_centers, compute_dct_coefficients


def enforce_connectivity(img, cluster_centers, labels):
    img_flat = img.reshape(-1, img.shape[-1])
    unique_labels = np.unique(labels)
    labels_img = labels.reshape(*img.shape[:2])
    new_labels = labels_img.copy()
    for my_label in unique_labels:
        mask_label = labels_img == my_label
        non_maximal_components = identify_biggest_connected_component(mask_label)
        new_labels[non_maximal_components] = -1
    img_grid = img_flat[:, 3:]
    mask_outliers = new_labels.reshape(-1) == -1

    outliers = img_grid[mask_outliers]
    non_outliers = img_grid[~mask_outliers]
    cluster_grid = cluster_centers[:, 3:]
    kd_tree_non_outliers = KDTree(img_grid)

    nearest_neighbor_ind = kd_tree_non_outliers.query(
        outliers, 1, return_distance=False
    )
    new_labels.reshape(-1)[mask_outliers] = labels[nearest_neighbor_ind.reshape(-1)]
    return new_labels


class BasePixeliser:
    """
    Base class for the Pixelsiers. We are going to follow skimge convention.
    Each Pixelisser will have a `pixelate` method which returns an array of the same @ dimensions than the original image.
    Each pixel of this 2D array will be an integer representing the cluster it belongs to.
    """

    def __init__(self):
        pass

    def pixelate(
        self,
    ):
        pass

    def pixelate_file(self, img_file: Path, *args, **kwargs):
        return self.pixelate(imread(img_file, *args, **kwargs))


class BlockPixeliser(BasePixeliser):
    def __init__(self, patch_size: int = 21):
        super().__init__()
        self.patch_size = patch_size

    def pixelate(self, img: np.ndarray):
        """Pixelate an image by just dividing it into blocks."""

        n_height, n_width = resize_to_block_shape(img.shape, self.patch_size)
        labels = np.arange(n_height * n_width).reshape(n_height, n_width)

        heigh_block, width_block = (
            n_height // self.patch_size,
            n_width // self.patch_size,
        )
        labels = np.arange(heigh_block * width_block).reshape(heigh_block, width_block)
        labels = resize(labels, img.shape[:2], order=0, anti_aliasing=False)

        return labels


class SLICPixeliser(BasePixeliser):
    def __init__(
        self,
        m_spatial: int = 30,
        step: int = 50,
        max_iter: int = 10,
        threshold_error: float = 1.0,
        clever_init: bool = True,
        connectivity: bool = True,
    ):
        super().__init__()
        self.m_spatial = m_spatial
        self.step = step
        self.max_iter = max_iter
        self.threshold_error = threshold_error
        self.clever_init = clever_init
        self.connectivity = connectivity

    def init_centers(self, img: np.ndarray):
        """init centers as a list of array coordinates
        also gives them in XYLAB space
        init can be clever (see class init method)"""
        h, w, _ = img.shape
        centers_X = self.step // 2 + np.arange(0, w - self.step // 2, self.step)
        centers_Y = self.step // 2 + np.arange(0, h - self.step // 2, self.step)
        centers_X, centers_Y = np.meshgrid(centers_X, centers_Y)
        centers_XY = np.array(
            list(zip(np.nditer(centers_X), np.nditer(centers_Y))),
        )

        # clever center initialisation
        if self.clever_init:
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

        centers_XY[:, [0, 1]] = centers_XY[:, [1, 0]]
        return centers_XY

    def pixelate(self, img: np.ndarray):
        img_XYLAB = get_XYLAB(img)

        h, w, c = img_XYLAB.shape
        initial_centers = self.init_centers(img)

        centers_XY = np.ravel_multi_index(initial_centers.T, dims=(h, w))

        img_flat_XYLAB = img_XYLAB.reshape(-1, c)
        centers_XYLAB = img_flat_XYLAB[centers_XY].copy()

        current_error = np.inf
        current_iteration = 0

        radius_neighbors_transformer = RadiusNeighborsTransformer(
            mode="connectivity", metric="chebyshev", radius=self.step
        )
        radius_neighbors_transformer.fit(img_flat_XYLAB[:, :2])
        while (current_iteration < self.max_iter) and (
            current_error > self.threshold_error
        ):
            new_centers_XYLAB, labels = self.one_slic_iteration(
                img_flat_XYLAB, centers_XYLAB, radius_neighbors_transformer
            )

            current_error = np.mean(
                (new_centers_XYLAB[:, :2] - centers_XYLAB[:, :2]) ** 2
            )
            centers_XYLAB = new_centers_XYLAB
            current_iteration += 1

        if self.connectivity:
            labels = self.enforce_connectivity(img_XYLAB, centers_XYLAB, labels)
        return labels.reshape(h, w)

    def one_slic_iteration(
        self,
        img_flat_XYLAB: np.ndarray,
        centers_XYLAB: np.ndarray,
        radius_neighbors_transformer: RadiusNeighborsTransformer,
    ):
        labels = self.get_closest_center(
            img_flat_XYLAB,
            centers_XYLAB,
            radius_neighbors_transformer=radius_neighbors_transformer,
        )

        centers_XYLAB = self.compute_centers(img_flat_XYLAB, labels)
        return centers_XYLAB, labels

    def get_closest_center(
        self,
        img_flat_XYLAB: np.ndarray,
        centers_XYLAB: np.ndarray,
        radius_neighbors_transformer: RadiusNeighborsTransformer,
    ):

        centers_LAB = centers_XYLAB[:, 2:]
        centers_XY = centers_XYLAB[:, :2]
        img_LAB = img_flat_XYLAB[:, 2:]
        img_XY = img_flat_XYLAB[:, :2]

        connectivity = radius_neighbors_transformer.transform(centers_XY)

        data_distance_matrix = []

        for i, row in enumerate(connectivity):
            close_pixels_indices = row.indices
            close_pixels = img_flat_XYLAB[close_pixels_indices]
            current_center_XY = centers_XY[i : i + 1]
            current_center_LAB = centers_LAB[i : i + 1]
            distance_XY = pairwise_distances(
                current_center_XY, close_pixels[:, :2], metric="sqeuclidean"
            )
            distance_LAB = pairwise_distances(
                current_center_LAB, close_pixels[:, 2:], metric="sqeuclidean"
            )
            distance = distance_LAB + (self.m_spatial / self.step) ** 2 * distance_XY
            data_distance_matrix.extend(np.sqrt(distance.reshape(-1)))

        distance_matrix = csr_matrix(
            (data_distance_matrix, connectivity.indices, connectivity.indptr),
            connectivity.shape,
        )
        mask_distance_matrix = ~connectivity.todense(order="F").astype(bool)

        distance_matrix_ma = np.ma.array(
            distance_matrix.todense(order="F"), mask=mask_distance_matrix
        )

        closest_center_labels = distance_matrix_ma.argmin(axis=0, fill_value=np.inf)

        isolated_pixels_mask = ~mask_distance_matrix.any(axis=0)
        if isolated_pixels_mask.any():

            kd_tree_centers_XY = KDTree(centers_XY)
            indices = kd_tree_centers_XY.query(
                img_XY[isolated_pixels_mask], return_distance=False
            )
            closest_center_labels[isolated_pixels_mask] = indices

        return closest_center_labels

    def compute_centers(self, img_flat_XYLAB: np.ndarray, labels: np.ndarray):

        unique_labels = np.unique(labels)
        centers_XYLAB = np.zeros((len(unique_labels), img_flat_XYLAB.shape[-1]))
        for i, current_label in enumerate(unique_labels):
            mask_label = labels == current_label
            centers_XYLAB[i] = np.mean(img_flat_XYLAB[mask_label], axis=0)
        return centers_XYLAB.astype(np.int32)

    def enforce_connectivity(
        self, img_XYLAB: np.ndarray, centers_XYLAB: np.ndarray, labels: np.ndarray
    ):
        unique_labels = np.unique(labels)
        labels_img = labels.reshape(img_XYLAB.shape[:2])
        new_labels = labels_img.copy()
        img_flat_XYLAB = img_XYLAB.reshape(-1, img_XYLAB.shape[-1])

        maximal_components_mask = np.zeros_like(new_labels, dtype=bool)
        for my_label in unique_labels:
            mask_label = labels_img == my_label
            maximal_components_mask = np.logical_or(
                maximal_components_mask,
                identify_biggest_connected_component(mask_label),
            )

        new_labels[~maximal_components_mask] = -1

        img_XY = img_flat_XYLAB[:, :2]
        mask_outliers = ~maximal_components_mask.reshape(-1)
        centers_XY = centers_XYLAB[:, :2]
        if mask_outliers.any():
            outliers = img_XY[mask_outliers]
            kd_tree_clusters = KDTree(centers_XY)
            indices = kd_tree_clusters.query(outliers, return_distance=False)
            new_labels[~maximal_components_mask] = indices.squeeze()
            return new_labels
        else:
            return labels


class EnergeticSLICPixeliser(SLICPixeliser):
    def __init__(
        self,
        m_spatial: float = 30.0,
        m_frequency: float = 50.0,
        step: int = 50,
        max_iter: int = 10,
        threshold_error: float = 1.0,
        clever_init: bool = True,
        connectivity: bool = True,
        reduce_func=np.median,
        patch_size=9,
        log_transform=True,
        dft=False,
    ):
        super().__init__(
            m_spatial=m_spatial,
            step=step,
            max_iter=max_iter,
            threshold_error=threshold_error,
            clever_init=clever_init,
            connectivity=connectivity,
        )

        self.m_frequency = m_frequency
        self.patch_size = patch_size
        self.log_transform = log_transform
        self.reduce_func = reduce_func
        self.dft = dft

    def pixelate(self, img: np.ndarray):
        img_XYLAB = get_XYLAB(img)

        h, w, c = img_XYLAB.shape
        initial_centers = self.init_centers(img)

        centers_XY = np.ravel_multi_index(initial_centers.T, dims=(h, w))

        img_flat_XYLAB = img_XYLAB.reshape(-1, c)
        centers_XYLAB = img_flat_XYLAB[centers_XY].copy()

        radius_neighbors_transformer = RadiusNeighborsTransformer(
            mode="connectivity", metric="chebyshev", radius=self.step
        )
        radius_neighbors_transformer.fit(img_flat_XYLAB[:, :2])

        kdtree_centers = KDTree(centers_XYLAB[:, :2], metric="chebyshev")

        initial_labels = kdtree_centers.query(
            img_flat_XYLAB[:, :2], return_distance=False, k=1
        ).reshape(-1)

        img_dct = compute_dct_coefficients(
            img,
            patch_size=self.patch_size,
            log_transform=self.log_transform,
            as_channels=True,
            dft=self.dft,
        )

        img_XYLABF = np.concatenate([img_XYLAB, img_dct], axis=-1)

        img_flat_XYLABF = img_XYLABF.reshape(-1, img_XYLABF.shape[-1])

        centers_dct = compute_dct_centers(
            img_flat_XYLABF[:, 5:], initial_labels, reduce_func=self.reduce_func
        ).reshape(-1, self.patch_size ** 2)
        centers_XYLABF = np.concatenate([centers_XYLAB, centers_dct], axis=-1)

        current_error = np.inf
        current_iteration = 0
        while (current_iteration < self.max_iter) and (
            current_error > self.threshold_error
        ):
            new_centers_XYLABF, labels = self.one_slic_iteration(
                img_flat_XYLABF, centers_XYLABF, radius_neighbors_transformer
            )

            current_error = np.mean(
                (new_centers_XYLABF[:, :2] - centers_XYLABF[:, :2]) ** 2
            )
            centers_XYLABF = new_centers_XYLABF
            current_iteration += 1

        if self.connectivity:
            labels = self.enforce_connectivity(img_XYLABF, centers_XYLABF, labels)
        return labels.reshape(h, w)

    def one_slic_iteration(
        self,
        img_flat_XYLABF: np.ndarray,
        centers_XYLABF: np.ndarray,
        radius_neighbors_transformer: RadiusNeighborsTransformer,
    ):
        labels = self.get_closest_center(
            img_flat_XYLABF,
            centers_XYLABF,
            radius_neighbors_transformer=radius_neighbors_transformer,
        )

        centers_XYLABF = self.compute_centers(img_flat_XYLABF, labels)
        return centers_XYLABF, labels

    def get_closest_center(
        self,
        img_flat_XYLABF: np.ndarray,
        centers_XYLABF: np.ndarray,
        radius_neighbors_transformer: RadiusNeighborsTransformer,
    ):

        img_XY = img_flat_XYLABF[:, :2]

        connectivity = radius_neighbors_transformer.transform(centers_XYLABF[:, :2])

        data_distance_matrix = []

        for i, row in enumerate(connectivity):

            close_pixels_indices = row.indices
            close_pixels = img_flat_XYLABF[close_pixels_indices]
            current_center = centers_XYLABF[i : i + 1]

            distance_XY = pairwise_distances(
                current_center[:, :2], close_pixels[:, :2], metric="sqeuclidean"
            )
            distance_LAB = pairwise_distances(
                current_center[:, 2:5], close_pixels[:, 2:5], metric="sqeuclidean"
            )

            distance_F = pairwise_distances(
                current_center[:, 5:], close_pixels[:, 5:], metric="sqeuclidean"
            )
            distance = (
                distance_LAB
                + (self.m_spatial / self.step) ** 2 * distance_XY
                + (self.m_frequency / self.step) ** 2 * distance_F
            )
            data_distance_matrix.extend(np.sqrt(distance.reshape(-1)))

        distance_matrix = csr_matrix(
            (data_distance_matrix, connectivity.indices, connectivity.indptr),
            connectivity.shape,
        )
        mask_distance_matrix = ~connectivity.todense(order="F").astype(bool)

        distance_matrix_ma = np.ma.array(
            distance_matrix.todense(order="F"), mask=mask_distance_matrix
        )

        closest_center_labels = distance_matrix_ma.argmin(axis=0, fill_value=np.inf)

        isolated_pixels_mask = ~mask_distance_matrix.any(axis=0)
        if isolated_pixels_mask.any():

            kd_tree_centers_XY = KDTree(centers_XYLABF[:, :2])
            indices = kd_tree_centers_XY.query(
                img_XY[isolated_pixels_mask], return_distance=False
            )
            closest_center_labels[isolated_pixels_mask] = indices

        return closest_center_labels
