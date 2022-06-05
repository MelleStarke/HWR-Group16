import os
import cv2 as cv
import numpy as np
import util
import preprocess as pp

from argparse import ArgumentParser

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
  
input_imgs = [cv.threshold(cv.imread(fn, 0), 127, 255, cv.THRESH_BINARY_INV)[1] for fn in file_names]

# plt.imshow(input_imgs[0])
# print(np.shape(input_imgs[0]))
# plt.show()

# print(list(file_names))

line_segmenter = pp.HHLineSegmenter()

lines = [line_segmenter(img) for img in input_imgs]

## Check line segmentation performance
# for img in input_imgs:
#   plt.imshow(img)
#   plt.show()
#   new_img = line_segmenter.preprocess(img)
#   row_hist = line_segmenter.get_row_hist(new_img)
#   valleys = line_segmenter.find_valleys(row_hist)
#   plt.imshow(new_img)
#   for v in valleys:
#     plt.axhline(v, color='lime')
#   plt.show()
  
#   lines = line_segmenter(img)
  
#   fig, axes = plt.subplots(len(lines), 1, figsize=(20, 20))
#   for ax, line in zip(axes, lines):
#     ax.imshow(line)
#   plt.tight_layout()
#   plt.show()

word_segmenter = pp.HHWordSegmenter()

words = [[word_segmenter(line) for line in img_lines] for img_lines in lines]

restoration_model = util.load_word_restoration_model()
restoration_model.eval()

for img_idx in range(len(words)):
  img_words = words[img_idx]
  for line_idx in range(len(img_words)):
    line_words = img_words[line_idx]
    for word_idx in range(len(line_words)):
      word = line_words[word_idx]
      fn = f"./segmented/base/img{img_idx}/line{line_idx}"
      
      try:
        os.makedirs(fn)
      except FileExistsError:
        pass
      
      fn += f"/word{word_idx}.png"
      cv.imwrite(fn, word)
      
      # Restored word saving
      fn = f"./segmented/restored/img{img_idx}/line{line_idx}"
      
      try:
        os.makedirs(fn)
      except FileExistsError:
        pass
      
      fn += f"/word{word_idx}.png"
      word = util.pad_and_resize(util.to_norm_tensor(word), (64, 64*4)).reshape(1, 64*64*4, 1, 1)
      plt.imshow(word.reshape(64, 64*4))
      plt.show()
      word = restoration_model.forward(word).reshape((64, 64*4, 1)).detach().numpy()
      plt.imshow(word.reshape(64, 64*4))
      plt.show()
      cv.imwrite(fn, word)
      # print('.')