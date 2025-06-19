import numpy as np


def xyxy_reduce_to_points_matrix(ratio_x: float, ratio_y: float) -> np.ndarray:
    return np.array([
        [(1 - ratio_x), 0, ratio_x, 0],
        [0, (1 - ratio_y), 0, ratio_y]
    ])


def reduce_xyxy_to_points(xyxy: np.ndarray, ratio_x: float, ratio_y: float) -> np.ndarray:

    if xyxy.size == 0:
        return xyxy

    if len(xyxy.shape) > 2:
        raise ValueError('xyxy must be a 2D array or a 1D array')

    if len(xyxy.shape) == 1:
        return reduce_xyxy_vector_to_point(xyxy, ratio_x, ratio_y)

    return reduce_xyxy_matrix_to_points(xyxy, ratio_x, ratio_y)


def reduce_xyxy_vector_to_point(
    xyxy: np.ndarray,
    ratio_x: float,
    ratio_y: float
) -> np.ndarray:

    if len(xyxy.shape) != 1:
        raise ValueError('xyxy must be a 1D array')

    if xyxy.shape[0] != 4:
        raise ValueError('xyxy must be a 1D array with 4 elements')

    reduce_matrix = xyxy_reduce_to_points_matrix(ratio_x, ratio_y)
    return np.dot(reduce_matrix, xyxy)


def reduce_xyxy_matrix_to_points(
    xyxy: np.ndarray,
    ratio_x: float,
    ratio_y: float
) -> np.ndarray:

    if len(xyxy.shape) != 2:
        raise ValueError('xyxy must be a 2D array')

    if xyxy.shape[1] != 4:
        raise ValueError('xyxy must be a 2D array with 4 columns')

    reduce_matrix = xyxy_reduce_to_points_matrix(ratio_x, ratio_y)
    return np.dot(xyxy, reduce_matrix.T)
