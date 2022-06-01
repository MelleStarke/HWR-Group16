import json
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torchvision.transforms.functional as VF
import torch
import torch.nn as nn
from torch import Tensor
import cv2
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from copy import deepcopy
from math import prod

# torch.cuda.set_device(0)

DATA_DIR = "../../data/dss/"
CHAR_DATA_DIR = DATA_DIR + "monkbrill/"
IMG_DATA_DIR = DATA_DIR + "train-imgs/"


def load_char_restoration_model():
  gen = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(64**2, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 1 x 64 x 64
  )

  gen.load_state_dict(torch.load('./trained/char/generator'))
  return gen

def load_dataset(dataset_name, equal_shapes=None, image_size=64):
  if any(map(lambda x: x in dataset_name.lower(), ['char', 'monk'])):
    dataset_name = CHAR_DATA_DIR
  
  elif any(map(lambda x: x in dataset_name.lower(), ['img', 'image', 'test'])):
    dataset_name = IMG_DATA_DIR
    
  else:
    raise ValueError(f"Invalid value for 'dataset_name' parameter. Expected 'char' or 'img', but got '{dataset_name}'")
  
  # Only if equal_shapes isn't specified: defaults equal_shapes to True if character data set is queried, else False
  equal_shapes = dataset_name == CHAR_DATA_DIR if equal_shapes is None else equal_shapes
  
  stats = (0.5,), (0.5,)
  
  if equal_shapes:
    transform = tt.Compose([tt.Grayscale(num_output_channels=1),
                            tt.RandomInvert(p=1),
                            tt.Resize(image_size),
                            tt.CenterCrop(image_size),
                            tt.ToTensor(),
                            # Reshape(1, '-2', '-1'),
                            tt.Normalize(*stats),
                           ])
  else:
    transform = tt.Compose([tt.Grayscale(num_output_channels=1),
                            tt.RandomInvert(p=1),
                            tt.ToTensor(),
                            # Reshape(1, '-2', '-1'),
                            tt.Normalize(*stats),
                           ])

  return ImageFolder(dataset_name, transform=transform)

_std_ds_label_dict = {v: k for k, v in load_dataset('char').class_to_idx.items()}
_std_output_label_dict = {}
with open('./output_dictionary.json') as file:
  std_output_label_dict = json.load(file)

def transcribe_label(label, ds_label_dict=_std_ds_label_dict, output_label_dict=_std_output_label_dict):
  if isinstance(label, (list, tuple)):
    return "".join([output_label_dict[ds_label_dict[l]] for l in label])
  return output_label_dict[ds_label_dict[label]]

def img_subtract(img1, img2):
  # img1 = img1 if isinstance(img1, torch.Tensor) else tt.ToTensor()(img1)
  # img2 = img2 if isinstance(img2, torch.Tensor) else tt.ToTensor()(img2)
  # print(f"img 1 range: {(torch.min(img1), torch.max(img1))}")
  # print(f"img 2 range: {(torch.min(img2), torch.max(img2))}")
  subtr_img = img1 - (img2 + 1) # img1.subtract(img2 + 1)
  # print(f"pre-range: {(torch.min(subtr_img), torch.max(subtr_img))}")
  subtr_img = torch.maximum(subtr_img, -torch.ones(*subtr_img.shape))
  # print(f"post-range: {(torch.min(subtr_img), torch.max(subtr_img))}")
  return subtr_img

def img_add(img1, img2):
  # print(f"img 1 range: {(torch.min(img1), torch.max(img1))}")
  # print(f"img 2 range: {(torch.min(img2), torch.max(img2))}")
  added_img = img1 + img2 + 1
  added_img = torch.minimum(added_img, torch.ones(*added_img.shape))
  return added_img

def randint(low, *args, **kwargs):
  if low == 0:
    low = 1
  return np.random.randint(low, *args, **kwargs)

def just_give_me_the_goddamned_image_size(x):
  if callable(x):
    x = x()
  try:
    if callable(x.size):
      return x.size()
    elif isinstance(x.size, tuple):
      return x.size
  
  except:
    return x

def working_imshow(img, ax=None):
  if len(np.shape(img)) == 2:
    if ax is None:
      plt.imshow(img)
    else:
      ax.imshow(img)
  else:
    if ax is None:
      plt.imshow(img[0,:,:])
    else:
      ax.imshow(img[0,:,:])

def equalize_heights(chars):
  max_height = sorted([char.shape[0] for char in chars])[-1]
  
  for i in range(len(chars)):
    char = chars[i]
    chars[i] = VF.pad(char, [0, max_height - char.shape[0], 0, 0], fill = -1)[None, :, :]
    
  return chars

def glue_chars(chars, padding=0):
  if not callable(padding) and padding == 0:
      return torch.cat(tuple(chars), dim=2)
    
  output = chars[0]
  
  for char in chars[1:]:
    pad = padding() if callable(padding) else padding
    char_w = np.shape(char)[-1]
    output = VF.pad(output, (0, 0, int(pad + char_w), 0), fill = -1)
    output_w = np.shape(output)[-1]
    char = VF.pad(char, (output_w - char_w, 0, 0, 0), fill = -1)
    output = img_add(output, char)
    
  return output
  
  
def pad_and_resize(img, output_shape):
  output_h, output_w = output_shape[-2], output_shape[-1]
  output_proportion = output_h / output_w
  
  img_h, img_w = np.shape(img)[-2], np.shape(img)[-1]
  img_proportion = img_h / img_w
  
  padding = [int(((output_w * img_h / output_h) - img_w) / 2),
              int((output_proportion - img_proportion) * img_w / 2)]
  # print(f"padding: {padding}")
  
  transformation = tt.Compose([#tt.ToPILImage(),
                                tt.Pad(list(map(lambda x: max(x, 0), padding)), fill = -1),
                                tt.Resize(output_shape),
                              #  tt.ToTensor(),
                              #  tt.Normalize((0.5,), (0.5,))
                              ])
  # base_word = VF.pad(base_word, list(map(lambda x: max(x, 0), padding)), fill = -1)
  # base_word = VF.resize(base_word, self.img_shape)
  # print(f"gen word shape: {base_word.shape}")
  return transformation(img)
  


class ConsistentImageFolder(ImageFolder):
  
  def __getitem__(self, *args, **kwargs):
    sample, target = super(ConsistentImageFolder, self).__getitem__(*args, **kwargs)
    
    if not isinstance(sample, torch.Tensor):
      sample = tt.ToTensor()(sample)
      
    if len(np.shape(sample)) == 2:
      sample = torch.reshape(sample, (1, *np.shape(sample)))
      
    return sample, target


class Reshape(nn.Module):
  """
  nn.Module that reshapes tensor ro specified size, mainly for use in nn.Sequential.
  Takes a tuple specifying the shape.
  Use int for absolute size, and str for index of the shape of the input tensor.
  
  Can also multiply relative sizes, for example:
    Shape of the input tensor: (1, 1, 64, 64).
    Then Reshape(1, 1, "2*3")(input) will reshape into a tensor of shape (1, 1, 64*64) = (1, 1, 4096).
  """
  def __init__(self, *shape):
    super(Reshape, self).__init__()
    
    if isinstance(shape[0], tuple):
      shape = shape[0]
      
    self.shape = shape
    
  def forward(self, input):
    relative_dim = lambda x: input.shape[int(x)]
    shape = tuple([prod(map(relative_dim, d.split("*"))) if isinstance(d, str) else d for d in self.shape])
    print(f"Reshaping tensor of shape {input.shape} to {shape}")
    input = torch.reshape(input, shape)
    print(f"result: {input.shape}")
    return input


class RandomPad(tt.Pad):
  
  def __init__(self, pad_args, *args, **kwargs):
    super(RandomPad, self).__init__(0, *args, **kwargs)
    self.pad_args = pad_args
  
  def forward(self, input):
    padding = [randint(pad_arg) for pad_arg in self.pad_args]
    
    if len(padding) == 2:
      for i in range(len(padding)):
        pad = padding[i]
        opposite_pad = randint(pad)
        padding[i] -= opposite_pad
        padding.append(opposite_pad)
    
    self.padding = tuple(padding)
    
    return super().forward(input)
    

# class RandomPad(tt.Pad):
  
#   def __init__(self, pad_args, *args, **kwargs):
#     super(RandomPad, self).__init__(0, *args, **kwargs)
    
#     assert isinstance(pad_args, (int, float)) or \
#            isinstance(pad_args, (list, tuple)) and len(pad_args) in [2,4],\
#       "Invalid padding. Either use float or int, or a list/tuple of 2 or 4 floats or ints."
      
#     self.pad_args = pad_args
    
#   def forward(self, input):
#     pad_args = self.pad_args
#     if len(np.shape(input)) == 2:
#       input = np.array(input)[None,:,:]
#     print(np.shape(input))
#     print(np.shape(input[0]))
#     padding = []

#     if isinstance(pad_args, (int, float)):
#       pad_args = [pad_args] * 2
    
#     input_h, input_w = np.shape(input)[1:] if any(map(lambda x: isinstance(x, float), pad_args)) else [1] * 4
      
#     for pad_arg, dim_size in zip(pad_args, [input_w, input_h] * 2):
#       if isinstance(pad_arg, float):
#         pad_arg *= dim_size
      
#       padding.append(randint(pad_arg))
    
#     if len(pad_args) == 2:
#       print(f"pad arg len is 2. Len padding is {len(padding)}")
#       for i, pad in enumerate(padding):
#         opposite_pad = randint(pad)
#         padding[i] -= opposite_pad
#         padding.append(opposite_pad)
    
#     self.padding = tuple(padding)
    
#     return super(RandomPad, self).forward(input)
    
# class Segmenter():
  
#   def __init__(self, *args, **kwargs):
#     raise NotImplementedError()
  
#   def __call__(self, img):
#     """
#     Turns 2D grayscale image into list of sub-images.
#     """
#     raise NotImplementedError()
  
#   def preprocess_image(self, img):
#     return img


# class HHLineSegmenter(Segmenter):
  
#   def __init__(self):
#     self.row_hist = None
    
#   def __call__(self, img, **kwargs):
#     img = self.preprocess_image(img, **kwargs)
#     row_hist = self.get_row_hist(img)
#     valleys = self.find_valleys(row_hist, **kwargs)
#     valleys = valleys + [len(img) - 1]
#     lines = [img[t:b,:] for t, b in zip(valleys, valleys[1:])]
#     return lines

  
#   def preprocess_image(self, img, pad=128, thresh=256):
#     row_hist = np.sum(img, axis=1)
#     # Record which rows and cols have a total pixel value above the threshold.
#     inked_rows = np.array(range(img.shape[0]))[row_hist > thresh]
#     inked_cols = np.array(range(img.shape[1]))[np.sum(img, axis=0) > thresh]
#     # Define new img range as the first and last inked rows and cols, + padding.
#     row_range = inked_rows[0] - pad, inked_rows[-1] + pad
#     col_range = inked_cols[0] - pad, inked_cols[-1] + pad
#     row_hist = row_hist[row_range[0]:row_range[1]]
#     # Resize image.
#     img = img[row_range[0]:row_range[1], col_range[0]:col_range[1]]
#     return img
  
#   def get_row_hist(self, img):
#     return np.sum(img, axis=1)
  
#   def find_valleys(self, row_hist, window_size=50, height_diff=20000):
#     valleys = []
#     valley_height = np.inf
#     valley_idx = 0
#     peak_height = 0
#     in_valley = True
    
#     for row in range(len(row_hist) - window_size):
#       height = np.mean(row_hist[row : row + window_size])
#       # print(row_hist[row])
#       if in_valley and height > valley_height + height_diff:
#         # print(1)
#         in_valley = False
#         valley_height = np.inf
#         valleys.append(valley_idx + int(window_size / 2))
      
#       elif not in_valley and height < peak_height - height_diff:
#         # print(2)
#         in_valley = True
#         peak_height = 0
      
#       if in_valley and height < valley_height:
#         # print(3)
#         valley_height = height
#         valley_idx = row
      
#       elif not in_valley and height > peak_height:
#         # print(4)
#         peak_height = height

#     return valleys
  
  
# class HHWordSegmenter(Segmenter):
  
#   def __init__(self):
#     pass
  
#   def __call__(self, line, **kwargs):
#     seqs = self.get_seqs(line, **kwargs)
#     # Column indices at which to separate the words
#     word_seps = list(map(np.median, seqs))
#     words = [line[:, int(l):int(r)] for l, r in zip(word_seps, word_seps[1:])]
#     return words
    
#   def find_empty_col_seqs(self, line, min_col_sum=0.01, min_seq_len=0.005):
#     """
#     Gives list of lists, where each sub-list contains indices of subsequent empty columns.
#     Only sub-lists of lengths at or above the threshold value are returned.
    
#     Args:
#       min_col_sum (int or float): maximum total value of the pixels in a column for it to be considered empty.
#                                   Functions as absolute value if int, and as proportion
#                                   of the average total pixel value in the true non-empty columns if float.
                                  
#       min_seq_len (int or float): minimum length of the sequence for it to be returned.
#                                   Functions as absolute length if int, and as proportion
#                                   of the total number of cols if float.
#     """
#     col_hist = np.sum(line, axis=0)
#     n_cols = len(col_hist)
    
#     if isinstance(min_col_sum, float):
#       mean_col_sum = np.mean(list(filter(lambda x: x > 0, col_hist)))
#       min_col_sum = min_col_sum * mean_col_sum
    
#     if isinstance(min_seq_len, float):
#       min_seq_len = int(n_cols * 0.005)

#     prev_i = -1
#     new_seq = True
#     seqs = []

#     for i, _ in filter(lambda x: x[1] < min_col_sum, zip(range(len(col_hist)), col_hist)):
#       if prev_i + 1 == i:
#         if new_seq:
#           new_seq = False
#           seqs.append([])
          
#         seqs[-1].append(i)
        
#       else:
#         new_seq = True
      
#       prev_i = i

#     seqs = list(filter(lambda x: len(x) > min_seq_len, seqs))
#     return seqs
  
    
# class CorruptCharGen():
  
#   def __init__(self, *args, latent_size=64**2, max_iter=2048, **kwargs):
#     self.dl_args = args
#     self.dl_kwargs = {**kwargs, "batch_size": 1}
#     self.latent_size = latent_size
#     self.n_iter = 0
#     self.max_iter = max_iter
#     self.data_loader = None
  
#   def __iter__(self):
#     self.n_iter = 0
#     self.data_loader = None
#     return self
  
#   def __next__(self):
#     if self.n_iter > self.max_iter:
#       raise StopIteration
    
#     if self.data_loader is None:
#       self.data_loader = iter(DataLoader(*self.dl_args, **self.dl_kwargs))
    
#     try:
#       base_img = next(self.data_loader)
#       base_img_lab = base_img[1]
#       base_img = base_img[0][0][0]
#       subtr_img = next(self.data_loader)[0][0][0]
#       crpt_img = base_img - (subtr_img + 1)
#       crpt_img = torch.maximum(crpt_img, -torch.ones(*crpt_img.shape))
      
#       self.n_iter += 1
      
#       return crpt_img.reshape((self.latent_size, 1, 1)), base_img.reshape((self.latent_size, 1, 1)), base_img_lab
      
#     except StopIteration:
#       self.data_loader = None
#       return next(self)
    
#   def gen_chars(self, num=1):
#     crpt_imgs = [next(self) for _ in range(num)]
#     return tuple([torch.stack([img[i] for img in crpt_imgs]) for i in range(len(crpt_imgs[0]))])
 

# class WordAugmenter(tt.AutoAugment):
  
#   def __init__(self, *args, **kwargs):
#     super(WordAugmenter, self).__init__(*args, **kwargs)
#     self.policies = list(filter(
#       lambda xs:
#         all(map(lambda x: "Equalize" not in x and "Posterize" not in x, xs)),
#       self.policies
#     ))
    


# class CorruptWordGen():
  
#   def __init__(self, *args, char_size=64, n_char_range=(2,10), max_iter=2048, **kwargs):
#     self.dl_args = args
#     self.dl_kwargs = {**kwargs, "batch_size": 1}
#     self.n_iter = 0
#     self.max_iter = max_iter
#     self.img_shape = (char_size, int(char_size * (n_char_range[1] + n_char_range[0]) / 3))
#     self.n_char_range = n_char_range
#     self.data_loader = None
  
#   def __iter__(self):
#     self.n_iter = 0
#     self.data_loader = None
#     return self
  
#   def __next__(self):
#     if self.n_iter > self.max_iter:
#       raise StopIteration
    
#     if self.data_loader is None:
#       self.data_loader = iter(DataLoader(*self.dl_args, **self.dl_kwargs))
    
#     try:
#       base_img = next(self.data_loader)
#       base_img_lab = base_img[1]
#       base_img = base_img[0][0][0]
#       # subtr_img = next(self.data_loader)[0][0][0]
#       # crpt_img = base_img - (subtr_img + 1)
#       # crpt_img = torch.maximum(crpt_img, -torch.ones(*crpt_img.shape))
      
#       self.n_iter += 1
      
#       # return char_gen(base_img.reshape((1, latent_size, 1, 1))).reshape((image_size, image_size)), base_img_lab
#       return base_img, base_img_lab
      
#     except StopIteration:
#       self.data_loader = None
#       return next(self)
    
#   def gen_chars(self, num=1):
#     crpt_imgs = [next(self) for _ in range(num)]
#     return tuple([[img[i] for img in crpt_imgs] for i in range(len(crpt_imgs[0]))])
  
#   def gen_words(self, num=1):
#     chars, labs = self.gen_chars(randint(*self.n_char_range))
#     n_chars = len(chars)
#     max_height = sorted([char.shape[0] for char in chars])[-1]
#     output_img_proportion = self.img_shape[0] / self.img_shape[1]
#     # chars[0] = F.pad(chars[0], [0, 24, 0, 0])
#     # h = chars[0].shape[0]
    
#     for i in range(n_chars):
#       char = chars[i]
#       chars[i] = VF.pad(char, [0, max_height - char.shape[0], 0, 0], fill = -1)[None, :, :]
    
#     # print([x.shape for x in chars])
    
#     # base_word = VF.resize(torch.cat(tuple(chars), dim=2), size=self.img_size)[0, :, :]
#     base_word = torch.cat(tuple(chars), dim=2)
    
#     _, base_word_h, base_word_w = base_word.shape
#     base_word_proportion = base_word_h / base_word_w
#     y_pad = base_word_proportion < output_img_proportion
#     # padding = [int((base_word_proportion - output_img_proportion) / (2 * base_word_w)),
#     #            int((base_word_h - (base_word_h / base_word_proportion * output_img_proportion)) / 2) ]
#     padding = [int(((self.img_shape[1] * base_word_h / self.img_shape[0]) - base_word_w) / 2),
#                int((output_img_proportion - base_word_proportion) * base_word_w / 2)]
#     # print(f"padding: {padding}")
#     base_word = VF.pad(base_word, list(map(lambda x: max(x, 0), padding)), fill = -1)
#     base_word = VF.resize(base_word, self.img_shape)
    
#     crpt_word = WordAugmenter(fill = -1)(base_word)
    
#     # print(chars.shape)
#     # return torch.cat(tuple(chars), dim=1).detach().numpy()
#     # return word.detach().numpy()
#     return base_word, crpt_word, labs