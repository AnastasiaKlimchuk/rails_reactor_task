import os
import numpy
from PIL import Image
import glob
import imageComparator

def run_magic(path):
    if not os.path.exists(path):
        raise Exception('Path is not exists.')

    images = []
    for filename in glob.glob(path + '/*.jpg'):
        im = Image.open(filename)
        images.append(im)

    output = []
    for i in range(len(images)-1):
        for j in range(i+1, len(images)-1):
            res = imageComparator.compare(images[i], images[j])
            if res:
                output.append((images[i].filename, images[j].filename))
            # images[i].filename
            # print(res)
    print(output)
    return output

