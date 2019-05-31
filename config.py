"""
            n : size of grid imposed on image (default 9)
            identical_tolerance: cutoff difference for declaring two adjacent
                grid points identical (default 2/255)
            n_levels: number of positive and negative groups to stratify neighbor
                differences into. n = 2 -> [-2, -1, 0, 1, 2] (default 2)
            crop_percentiles: lower and upper bounds when considering how much
                variance to keep in the image (default (5, 95))
"""
n = 9
identical_tolerance = 2/255
n_levels = 10
crop_percentiles = (5, 95)
lower_percentile = 0
upper_percentile = 100
threshold = 0.4

