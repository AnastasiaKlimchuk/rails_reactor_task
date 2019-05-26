import argparse
import magicbox


parser = argparse.ArgumentParser(description='First test task on images similarity.')
parser.add_argument('--path', help='folder with image',required=True)
args = parser.parse_args()

result = magicbox.run_magic(args.path)

print(result)