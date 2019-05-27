import numpy as np
import constant

def generate(image):
    """

    :param image: image
    :return: The image signature: A rank 1 numpy array of length n x n x 8
    """
    # Step 1:    Load image as array of grey-levels

    im_array = np.array(image.convert('L'))/255

    # Step 2b:   Generate grid centers
    x_coords, y_coords = compute_grid_points(im_array)

    # Step 3:    Compute grey level mean of each P x P square centered at each grid point
    avg_grey = compute_mean_level(im_array, x_coords, y_coords)

    # Step 4a:   Compute array of differences for each grid point vis-a-vis each neighbor
    diff_mat = compute_differentials(avg_grey)

    # Step 4b: Bin differences to only 2n+1 values
    diff_array = normalize_and_threshold(diff_mat, identical_tolerance=constant.identical_tolerance,
                                         n_levels=constant.n_levels)

    return np.ravel(diff_array).astype('int8')


def compute_grid_points(image, n=constant.n, window=None):
        """Computes grid points for image analysis.
        Corresponds to the second part of 'step 2' in the paper
        Args:
            image (numpy.ndarray): n x m array of floats -- the greyscale image. Typically,
                the output of preprocess_image
            n (Optional[int]): number of gridpoints in each direction (default 9)
            window (Optional[List[Tuple[int]]]): limiting coordinates [(t, b), (l, r)], typically the
                output of (default None)
        Returns:
            tuple of arrays indicating the vertical and horizontal locations of the grid points
        Examples:
            (array([100, 165, 230, 295, 360, 424, 489, 554, 619]),
             array([ 66, 109, 152, 195, 238, 280, 323, 366, 409]))
        """

        # if no limits are provided, use the entire image
        if window is None:
            window = [(0, image.shape[0]), (0, image.shape[1])]

        x_coords = np.linspace(window[0][0], window[0][1], n + 2, dtype=int)[1:-1]
        y_coords = np.linspace(window[1][0], window[1][1], n + 2, dtype=int)[1:-1]

        return x_coords, y_coords


def compute_mean_level(image, x_coords, y_coords, P=None):

    """Computes array of greyness means.
    Corresponds to 'step 3'
    Args:
        image (numpy.ndarray): n x m array of floats -- the greyscale image. Typically,
            the output of preprocess_image
        x_coords (numpy.ndarray): array of row numbers
        y_coords (numpy.ndarray): array of column numbers
        P (Optional[int]): size of boxes in pixels (default None)
    Returns:
        an N x N array of average greyscale around the gridpoint, where N is the
            number of grid points """

    if P is None:
        P = max([2.0, int(0.5 + min(image.shape)/20.)])

    avg_grey = np.zeros((x_coords.shape[0], y_coords.shape[0]))

    for i, x in enumerate(x_coords):        # not the fastest implementation
        lower_x_lim = int(max([x - P/2, 0]))
        upper_x_lim = int(min([lower_x_lim + P, image.shape[0]]))
        for j, y in enumerate(y_coords):
            lower_y_lim = int(max([y - P/2, 0]))
            upper_y_lim = int(min([lower_y_lim + P, image.shape[1]]))

            avg_grey[i, j] = np.mean(image[lower_x_lim:upper_x_lim,
                                        lower_y_lim:upper_y_lim])  # no smoothing here as in the paper

    return avg_grey

def compute_differentials(grey_level_matrix,  diagonal_neighbors=True):

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

    # flip for anti-diagonal differences
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
    """Normalizes difference matrix in place.
            'Step 4' of the paper.  The flattened version of this array is the image signature.
            Args:
                difference_array (numpy.ndarray): n x n x l array, where l are the differences between
                    the grid point and its neighbors. Typically the output of compute_differentials
                identical_tolerance (Optional[float]): maximum amount two gray values can differ and
                    still be considered equivalent (default 2/255)
                n_levels (Optional[int]): bin differences into 2 n + 1 bins (e.g. n_levels=2 -> [-2, -1,
                    0, 1, 2]) """

    mask = np.abs(difference_array) < identical_tolerance
    difference_array[mask] = 0.

    # if image is essentially featureless, exit here
    if np.all(mask):
        return None

    # bin so that size of bins on each side of zero are equivalent
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



