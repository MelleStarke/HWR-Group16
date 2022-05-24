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

torch.cuda.set_device(0)

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

def get_char_dataset(equal_shapes=True, image_size=64):
  image_size = 64
  stats = (0.5,), (0.5,)
  
  if equal_shapes:
    transform = tt.Compose([tt.Grayscale(num_output_channels=1),
                            tt.Resize(image_size),
                            tt.CenterCrop(image_size),
                            tt.ToTensor(),
                            tt.Normalize(*stats)])
  else:
    transform = tt.Compose([tt.Grayscale(num_output_channels=1),
                            tt.ToTensor(),
                            tt.Normalize(*stats)])

  return ImageFolder(CHAR_DATA_DIR, transform=transform)


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
    return input.reshape(shape)


class Segmenter():
  
  def __init__(self, *args, **kwargs):
    raise NotImplementedError()
  
  def __call__(img):
    """
    Turns 2D grayscale image into list of sub-images.
    """
    raise NotImplementedError()
  
  def preprocess_image(self, img):
    return img


class HHLineSegmenter(Segmenter):
  
  def __init__(self):
    self.row_hist = None
    
  def __call__(self, img, **kwargs):
    img = self.preprocess_image(img, **kwargs)
    row_hist = self.get_row_hist(img)
    valleys = self.find_valleys(row_hist, **kwargs)
    valleys = valleys + [len(img) - 1]
    lines = [img[t:b,:] for t, b in zip(valleys, valleys[1:])]
    return lines

  
  def preprocess_image(self, img, pad=128, thresh=256):
    row_hist = np.sum(img, axis=1)
    # Record which rows and cols have a total pixel value above the threshold.
    inked_rows = np.array(range(img.shape[0]))[row_hist > thresh]
    inked_cols = np.array(range(img.shape[1]))[np.sum(img, axis=0) > thresh]
    # Define new img range as the first and last inked rows and cols, + padding.
    row_range = inked_rows[0] - pad, inked_rows[-1] + pad
    col_range = inked_cols[0] - pad, inked_cols[-1] + pad
    row_hist = row_hist[row_range[0]:row_range[1]]
    # Resize image.
    img = img[row_range[0]:row_range[1], col_range[0]:col_range[1]]
    return img
  
  def get_row_hist(self, img):
    return np.sum(img, axis=1)
  
  def find_valleys(self, row_hist, window_size=50, height_diff=20000):
    valleys = []
    valley_height = np.inf
    valley_idx = 0
    peak_height = 0
    in_valley = True
    
    for row in range(len(row_hist) - window_size):
      height = np.mean(row_hist[row : row + window_size])
      # print(row_hist[row])
      if in_valley and height > valley_height + height_diff:
        # print(1)
        in_valley = False
        valley_height = np.inf
        valleys.append(valley_idx + int(window_size / 2))
      
      elif not in_valley and height < peak_height - height_diff:
        # print(2)
        in_valley = True
        peak_height = 0
      
      if in_valley and height < valley_height:
        # print(3)
        valley_height = height
        valley_idx = row
      
      elif not in_valley and height > peak_height:
        # print(4)
        peak_height = height

    return valleys
  
  
class HHWordSegmenter(Segmenter):
  
  def __init__(self):
    pass
  
  def __call__(self, line, **kwargs):
    seqs = self.get_seqs(line, **kwargs)
    # Column indices at which to separate the words
    word_seps = list(map(np.median, seqs))
    words = [line[:, int(l):int(r)] for l, r in zip(word_seps, word_seps[1:])]
    return words
    
  def find_empty_col_seqs(self, line, min_col_sum=0.01, min_seq_len=0.005):
    """
    Gives list of lists, where each sub-list contains indices of subsequent empty columns.
    Only sub-lists of lengths at or above the threshold value are returned.
    
    Args:
      min_col_sum (int or float): maximum total value of the pixels in a column for it to be considered empty.
                                  Functions as absolute value if int, and as proportion
                                  of the average total pixel value in the true non-empty columns if float.
                                  
      min_seq_len (int or float): minimum length of the sequence for it to be returned.
                                  Functions as absolute length if int, and as proportion
                                  of the total number of cols if float.
    """
    col_hist = np.sum(line, axis=0)
    n_cols = len(col_hist)
    
    if isinstance(min_col_sum, float):
      mean_col_sum = np.mean(list(filter(lambda x: x > 0, col_hist)))
      min_col_sum = min_col_sum * mean_col_sum
    
    if isinstance(min_seq_len, float):
      min_seq_len = int(n_cols * 0.005)

    prev_i = -1
    new_seq = True
    seqs = []

    for i, _ in filter(lambda x: x[1] < min_col_sum, zip(range(len(col_hist)), col_hist)):
      if prev_i + 1 == i:
        if new_seq:
          new_seq = False
          seqs.append([])
          
        seqs[-1].append(i)
        
      else:
        new_seq = True
      
      prev_i = i

    seqs = list(filter(lambda x: len(x) > min_seq_len, seqs))
    return seqs
  
    
class CorruptCharGen():
  
  def __init__(self, *args, latent_size=64**2, max_iter=2048, **kwargs):
    self.dl_args = args
    self.dl_kwargs = {**kwargs, "batch_size": 1}
    self.latent_size = latent_size
    self.n_iter = 0
    self.max_iter = max_iter
    self.data_loader = None
  
  def __iter__(self):
    self.n_iter = 0
    self.data_loader = None
    return self
  
  def __next__(self):
    if self.n_iter > self.max_iter:
      raise StopIteration
    
    if self.data_loader is None:
      self.data_loader = iter(DataLoader(*self.dl_args, **self.dl_kwargs))
    
    try:
      base_img = next(self.data_loader)
      base_img_lab = base_img[1]
      base_img = base_img[0][0][0]
      subtr_img = next(self.data_loader)[0][0][0]
      crpt_img = base_img - (subtr_img + 1)
      crpt_img = torch.maximum(crpt_img, -torch.ones(*crpt_img.shape))
      
      self.n_iter += 1
      
      return crpt_img.reshape((self.latent_size, 1, 1)), base_img.reshape((self.latent_size, 1, 1)), base_img_lab
      
    except StopIteration:
      self.data_loader = None
      return next(self)
    
  def gen_chars(self, num=1):
    crpt_imgs = [next(self) for _ in range(num)]
    return tuple([torch.stack([img[i] for img in crpt_imgs]) for i in range(len(crpt_imgs[0]))])