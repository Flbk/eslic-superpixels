import numpy as np
from scipy.spatial import KDTree


def max_distance(X, Y):
    """Build a distance matrix D[i, j] = max(|X[i, 0] - Y[j, 0]|, |X[i, 1] - Y[j, 1]|).
    Useful to compute the boundary recall.
    """
    return abs(X[:, :, None] - Y.T[None, :, :]).max(axis=1)


def boundary_recall(boundaries, gt_boundaries):
    """Compute the fraction of computed boundaries that are close to a ground truth boundary (close means it has a neighbor within a 3x3 square)."""
    coordinates = np.indices(boundaries.shape)
    coordinates_flat = coordinates.reshape(2, -1).T

    boundaries_coord = coordinates_flat[boundaries.reshape(-1)]
    gt_boundaries_coord = coordinates_flat[gt_boundaries.reshape(-1)]
    gt_kd_tree = KDTree(gt_boundaries_coord)
    kd_tree = KDTree(boundaries_coord)
    dist_matrix = gt_kd_tree.sparse_distance_matrix(
        kd_tree, max_distance=2.0, p=np.inf
    ).tocoo()

    nnz = dist_matrix.getnnz(axis=1)
    return (nnz > 0.0).sum() / nnz.shape[0]


def undersegmentation_error(segmentation, gt_segmentation, B=0.05):
    """Compute the undersegmentation error.
    Compute the error only on the superpixels that are covered by more than 100*B% by the ground truth segmentation (default to 5%)."""
    gt_labels = np.unique(gt_segmentation)
    all_segmentation_labels, n_segmentation = np.unique(
        segmentation, return_counts=True
    )
    sum_error = 0
    for label in gt_labels:
        mask_gt_labels = gt_segmentation == label

        covering_segmentation = segmentation[mask_gt_labels]
        covering_labels, n_covering = np.unique(
            covering_segmentation, return_counts=True
        )

        mask_present_segmentation = np.isin(all_segmentation_labels, covering_labels)
        n_present_segmentation = n_segmentation[mask_present_segmentation]
        n_ratio = n_covering / n_present_segmentation

        mask_ratio = n_ratio >= B

        sum_error += n_present_segmentation[mask_ratio].sum()
    return sum_error / gt_segmentation.size - 1.0


def compactness(segmentation, boundaries):
    """Compactness of a segmentation.
    Correspond to the mean ratio between the area of the superpixels and the area of a circle with the same perimeter."""
    _, n_segmentation = np.unique(segmentation, return_counts=True)
    _, n_boundaries = np.unique(segmentation[boundaries], return_counts=True)

    N = segmentation.size

    return ((n_segmentation / n_boundaries) ** 2 * 4 * np.pi).sum() / N
