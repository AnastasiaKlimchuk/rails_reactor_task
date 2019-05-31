import os
from PIL import Image
import glob
import vector_comparator
import image_signature


def run_magic(path):
    if not os.path.exists(path):
        raise Exception('Path is not exists.')

    images = get_images(path)
    signatures = get_signatures(images)

    return compare_images(images, signatures)


def compare_images(images, signatures):
    res = []

    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):

            similar = vector_comparator.compare(signatures[i], signatures[j])

            if similar:
                res.append((images[i].filename.split(os.sep)[-1], images[j].filename.split(os.sep)[-1]))

    return res


def get_signatures(images):
    signatures = []

    for image in images:
        signatures.append(image_signature.generate(image))

    return signatures


def get_images(path):
    images = []

    for filename in glob.glob(path + '/*.jpg'):
        im = Image.open(filename)
        images.append(im)

    return images

