import argparse
import image_comparator


parser = argparse.ArgumentParser(description='First test task on images similarity.')
parser.add_argument('--path', help='folder with image', required=True)
args = parser.parse_args()


try:

    result = image_comparator.run_magic(args.path)

    for item in result:
        print(item[0], item[1])

except Exception as error:
    print(error)

