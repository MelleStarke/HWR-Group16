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

DATA_DIR = "../../data/dss/"
CHAR_DATA_DIR = DATA_DIR + "monkbrill/"
IMG_DATA_DIR = DATA_DIR + "train-imgs/"


def load_char_restoration_model():
  gen = nn.Sequential(
    # in: 4096 x 1 x 1

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

def load_word_restoration_model():
  generator = nn.Sequential(
    # in: 16.384 x 1 x 1

    nn.ConvTranspose2d(64 * 64 * 4, 1024, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(1024),
    nn.ReLU(True),
    # out: 1024 x 4 x 4

    nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 8 x 8

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 16 x 16

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 32 x 32
    
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 64 x 64

    nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh(),
    # out: 1 x 128 x 128
    
    Reshape('0', 1, '2*3')
  )
  generator.load_state_dict(torch.load('./trained/word/generator'))
  
  return generator

def load_dataset(dataset_name, equal_shapes=None, image_size=64):
  """
  Returns a DataLoader of either character imahes or Hebrew text images.
  
  Args:
    dataset_name (str): "char" or "img". Determines which data the returned dataset contains.
    
    equal_shapes (bool): Whether or not to crop and resize images to have the same shape.
                         True by default for character data, false by default for text image data.
    
    image_size (int or (int, int)): Shape for images to be resized to, if equal_shapes is true.
  """
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
                            tt.Normalize(*stats),
                           ])
  else:
    transform = tt.Compose([tt.Grayscale(num_output_channels=1),
                            tt.RandomInvert(p=1),
                            tt.ToTensor(),
                            tt.Normalize(*stats),
                           ])

  return ImageFolder(dataset_name, transform=transform)

# Standard dictionaries for transcription.To ensure transcribe_label doesn't need to constantly
# make new objects every time the function is called without kwargs.
_std_ds_label_dict = {v: k for k, v in load_dataset('char').class_to_idx.items()}
_std_output_label_dict = {}
with open('./output_dictionary.json', encoding='utf-8') as file:
  _std_output_label_dict = json.load(file)

def transcribe_label(label, ds_label_dict=_std_ds_label_dict, output_label_dict=_std_output_label_dict):
  """
  Turns a numeric label into the corresponding Hebrew character, according to output_dictionary.json.
  """
  if isinstance(label, (list, tuple)):
    return "".join([output_label_dict[ds_label_dict[l]] for l in label])
  return output_label_dict[ds_label_dict[label]]

def img_subtract(img1, img2):
  """
  Subtract two images and clip all values below -1.
  Assumes "ink" is encoded as 1 and whitespace as -1.
  """
  subtr_img = img1 - (img2 + 1)
  subtr_img = torch.maximum(subtr_img, -torch.ones(*subtr_img.shape))
  return subtr_img

def img_add(img1, img2):
  """
  Adds two images and clip all values above 1.
  Assumes "ink" is encoded as 1 and whitespace as -1.
  """
  added_img = img1 + img2 + 1
  added_img = torch.minimum(added_img, torch.ones(*added_img.shape))
  return added_img

def randint(low, *args, **kwargs):
  """
  Wrapper function around np.random.randint to ensure 0 is a valid arg.
  """
  if low == 0:
    low = 1
  return np.random.randint(low, *args, **kwargs)

def working_imshow(img, ax=None):
  """
  plt.imshow wrapper that synamically corrects for 2 or 3-dimensional arrays.
  """
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
  """
  Pads the tops of each image in the list of chars to have the same height as the tallest char.
  
  Args:
    chars ([chars])
  """
  max_height = sorted([char.shape[0] for char in chars])[-1]
  
  for i in range(len(chars)):
    char = chars[i]
    chars[i] = VF.pad(char, [0, max_height - char.shape[0], 0, 0], fill = -1)[None, :, :]
    
  return chars

def glue_chars(chars, padding=0):
  """
  Glues list of characters to a word, with specified padding.
  Negative padding results in overlapping characters.
  Padding can also be a callable returning an int.
  """
  if not callable(padding):
    if padding == 0:
      return torch.cat(tuple(chars), dim=2)
    
    if padding > 0:
      pad_shape = (1, np.shape(chars[0])[-2], int(padding))
      interlaced_pads = [-torch.ones(pad_shape) for _ in range(len(chars) - 1)]
      output = [None] * (len(chars) + len(interlaced_pads))
      output[::2] = chars
      output[1::2] = interlaced_pads
      return torch.cat(tuple(output), dim=2)
    
  output = chars[0]
  
  for char in chars[1:]:
    pad = padding() if callable(padding) else padding
    char_w = np.shape(char)[-1]
    output = VF.pad(output, (0, 0, int(pad + char_w), 0), fill = -1)
    output_w = np.shape(output)[-1]
    char = VF.pad(char, (output_w - char_w, 0, 0, 0), fill = -1)
    output = img_add(output, char)
    
  return output
  
def to_norm_tensor(img):
  """
  Turns img into Torch tensor, with values ranging from -1 to 1.
  """
  return tt.Compose([tt.ToTensor(), tt.Normalize((0.5,), (0.5,))])(img)
  
def to_pil_image(img):
  """
  Turns tensor or ndarray to pillow img.
  """
  return tt.ToPILImage()(img)  

def pad_and_resize(img, output_shape):
  """
  Pads img to have the same proportion as output_shape, then resize to output_shape.
  """
  output_h, output_w = output_shape[-2], output_shape[-1]
  output_proportion = output_h / output_w
  
  img_h, img_w = np.shape(img)[-2], np.shape(img)[-1]
  img_proportion = img_h / img_w
  
  padding = [int(((output_w * img_h / output_h) - img_w) / 2),
              int((output_proportion - img_proportion) * img_w / 2)]
  
  transformation = tt.Compose([tt.Pad(list(map(lambda x: max(x, 0), padding)), fill = -1),
                               tt.Resize(output_shape),
                              ])
  return transformation(img)


class Reshape(nn.Module):
  """
  nn.Module that reshapes tensor to specified size, mainly for use in nn.Sequential.
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
    input = torch.reshape(input, shape)
    return input


class RandomPad(tt.Pad):
  """
  Adds random padding to the image, sampled uniformly between 0 and the contents of pad_args.
  """
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