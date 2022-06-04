from argparse import ArgumentParser
import os
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

parser = ArgumentParser(description="Transcribes binarized images of Hebrew texts into machine text.")

parser.add_argument("input_dir", metavar="I", nargs=1, type=str, help="input directory containing Hebrew text images.")
parser.add_argument("output_dir", metavar="O", nargs=1, help="output directory for transcribed texts.")
parser.add_argument("--only-binarized-imgs", default=False, action="store_true", dest="only_binarized",
                    help="only transcribe images with 'binarized' in their file names.")


args = parser.parse_args()
input_dir = args.input_dir[0]
output_dir = args.output_dir[0]
binarized_check = (lambda x: "binarized" in x) if args.only_binarized else lambda x: True

cwd = os.getcwd()
print(os.path.join(input_dir, os.listdir(input_dir)[0]))
print(binarized_check(os.listdir(input_dir)[0]))
file_names = [os.path.join(input_dir, fn) for fn in os.listdir(input_dir) if binarized_check(fn)]
  
input_imgs = list(map(cv.imread, file_names))

plt.imshow(input_imgs[0])
print(np.shape(input_imgs[0]))
plt.show()

print(list(file_names))