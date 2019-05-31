import numpy as np
import config


def generate(image):
    im_array = np.array(image.convert('L')) / 255

    image_limits = crop_image(im_array, config.crop_percentiles[0], config.crop_percentiles[1])

    x_coords, y_coords = compute_grid_points(im_array, image_limits)

    avg_grey = compute_mean_level(im_array, x_coords, y_coords)

    diff_mat = compute_differentials(avg_grey)

    diff_array = normalize_and_threshold(diff_mat, identical_tolerance=config.identical_tolerance,
                                         n_levels=config.n_levels)

    return np.ravel(diff_array).astype('int8')


def crop_image(image, lower_percentile, upper_percentile):
    rw = np.cumsum(np.sum(np.abs(np.diff(image, axis=1)), axis=1))
    cw = np.cumsum(np.sum(np.abs(np.diff(image, axis=0)), axis=0))

    upper_column_limit = np.searchsorted(cw,
                                         np.percentile(cw, upper_percentile),
                                         side='left')
    lower_column_limit = np.searchsorted(cw,
                                         np.percentile(cw, lower_percentile),
                                         side='right')
    upper_row_limit = np.searchsorted(rw,
                                      np.percentile(rw, upper_percentile),
                                      side='left')
    lower_row_limit = np.searchsorted(rw,
                                      np.percentile(rw, lower_percentile),
                                      side='right')

    if lower_row_limit > upper_row_limit:
        lower_row_limit = int(lower_percentile / 100. * image.shape[0])
        upper_row_limit = int(upper_percentile / 100. * image.shape[0])
    if lower_column_limit > upper_column_limit:
        lower_column_limit = int(lower_percentile / 100. * image.shape[1])
        upper_column_limit = int(upper_percentile / 100. * image.shape[1])

    return [(lower_row_limit, upper_row_limit),
            (lower_column_limit, upper_column_limit)]


def compute_grid_points(image, window):
    n = config.n

    if window is None:
        window = [(0, image.shape[0]), (0, image.shape[1])]

    x_coords = np.linspace(window[0][0], window[0][1], n + 2, dtype=int)[1:-1]
    y_coords = np.linspace(window[1][0], window[1][1], n + 2, dtype=int)[1:-1]

    return x_coords, y_coords


def compute_mean_level(image, x_coords, y_coords):
    P = max([2.0, int(0.5 + min(image.shape) / 20.)])

    avg_grey = np.zeros((x_coords.shape[0], y_coords.shape[0]))

    for i, x in enumerate(x_coords):
        lower_x_lim = int(max([x - P / 2, 0]))
        upper_x_lim = int(min([lower_x_lim + P, image.shape[0]]))

        for j, y in enumerate(y_coords):
            lower_y_lim = int(max([y - P / 2, 0]))
            upper_y_lim = int(min([lower_y_lim + P, image.shape[1]]))

            avg_grey[i, j] = np.mean(image[lower_x_lim:upper_x_lim,
                                     lower_y_lim:upper_y_lim])

    return avg_grey


def compute_differentials(grey_level_matrix):
    right_neighbors = -np.concatenate((np.diff(grey_level_matrix),
                                       np.zeros(grey_level_matrix.shape[0]).
                                       reshape((grey_level_matrix.shape[0], 1))),
                                      axis=1)
    left_neighbors = -np.concatenate((right_neighbors[:, -1:],
                                      right_neighbors[:, :-1]),
                                     axis=1)

    down_neighbors = -np.concatenate((np.diff(grey_level_matrix, axis=0),
                                      np.zeros(grey_level_matrix.shape[1]).
                                      reshape((1, grey_level_matrix.shape[1]))))

    up_neighbors = -np.concatenate((down_neighbors[-1:], down_neighbors[:-1]))

    diagonals = np.arange(-grey_level_matrix.shape[0] + 1,
                          grey_level_matrix.shape[0])

    upper_left_neighbors = sum(
        [np.diagflat(np.insert(np.diff(np.diag(grey_level_matrix, i)), 0, 0), i)
         for i in diagonals])
    lower_right_neighbors = -np.pad(upper_left_neighbors[1:, 1:],
                                    (0, 1), mode='constant')

    flipped = np.fliplr(grey_level_matrix)
    upper_right_neighbors = sum([np.diagflat(np.insert(
        np.diff(np.diag(flipped, i)), 0, 0), i) for i in diagonals])
    lower_left_neighbors = -np.pad(upper_right_neighbors[1:, 1:],
                                   (0, 1), mode='constant')

    return np.dstack(np.array([
        upper_left_neighbors,
        up_neighbors,
        np.fliplr(upper_right_neighbors),
        left_neighbors,
        right_neighbors,
        np.fliplr(lower_left_neighbors),
        down_neighbors,
        lower_right_neighbors]))


def normalize_and_threshold(difference_array, identical_tolerance=2 / 255., n_levels=2):
    mask = np.abs(difference_array) < identical_tolerance
    difference_array[mask] = 0.

    if np.all(mask):
        return None

    positive_cutoffs = np.percentile(difference_array[difference_array > 0.],
                                     np.linspace(0, 100, n_levels + 1))
    negative_cutoffs = np.percentile(difference_array[difference_array < 0.],
                                     np.linspace(100, 0, n_levels + 1))

    for level, interval in enumerate([positive_cutoffs[i:i + 2]
                                      for i in range(positive_cutoffs.shape[0] - 1)]):
        difference_array[(difference_array >= interval[0]) &
                         (difference_array <= interval[1])] = level + 1

    for level, interval in enumerate([negative_cutoffs[i:i + 2]
                                      for i in range(negative_cutoffs.shape[0] - 1)]):
        difference_array[(difference_array <= interval[0]) &
                         (difference_array >= interval[1])] = -(level + 1)

    return difference_array
