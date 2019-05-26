import os


def run_magic(path):
    if not os.path.exists(path):
        raise Exception('Path is not exists.')

    output = [('1', '1_duplicate'), ('2', '2_modification')]
    return output

