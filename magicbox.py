import os
import numpy
from PIL import Image
import glob
import imageComparator
import imageSignature


def run_magic(path):
    if not os.path.exists(path):
        raise Exception('Path is not exists.')

    images = []
    for filename in glob.glob(path + '/*.jpg'):
        im = Image.open(filename)
        images.append(im)

    signatures = []
    for image in images:
        signatures.append(imageSignature.generate(image))

    print(signatures)
    output = []
    for i in range(len(signatures)-1):
        for j in range(i+1, len(signatures)-1):
            res = imageComparator.compare(signatures[i], signatures[j])
            if res:
                output.append((images[i].filename, images[j].filename))
    return output

