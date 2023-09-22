import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir", nargs='+', required=True, help="Path to the main training files."
)
parser.add_argument('--output_dir', type=str, default=None, help='the filtered data')

args = parser.parse_args()

output_data = []

for path in args.input_dir:
    texts = open(path, 'r').readlines()
    output_data.extend(texts)


f = open(args.output_dir, 'w')
f.writelines(output_data)
f.close()