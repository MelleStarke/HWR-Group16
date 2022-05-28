from regex import D
from util import *


######################
###  SEGMENTATION  ###
######################

class Segmenter():
  
  def __init__(self, *args, **kwargs):
    raise NotImplementedError()
  
  def __call__(self, img):
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


#####################
###  RESTORATION  ###
#####################
    
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
    # self.data_loader = None
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
 

class RandomCorruptor(nn.Module):
  
  def __init__(self, char_gen, transforms=None):
    super(RandomCorruptor, self).__init__()
    self.char_gen = char_gen


class WordAugmenter(nn.Module):
  
  def __init__(self, transforms=None, *args, **kwargs):
    super(WordAugmenter, self).__init__(*args, **kwargs)
    self.transforms = self._default_transforms if transforms is None else transforms
  
  @property
  def _default_transforms(self):
    return [
      (1.0, tt.ToPILImage()),
      (0.2, tt.RandomAffine(0, scale=(0.7, 1))),
      (0.2, tt.RandomPerspective(p=1, distortion_scale=0.2)),
      (0.2, tt.RandomAffine(0, shear=(-10, 10, -10, 10))),
      (0.2, tt.RandomAffine(8)),
      (0.2, tt.RandomAffine(0, translate=(0.1, 0.3))),
      (0.2, tt.RandomAffine(0, scale=(1, 1.3))),
      (0.2, tt.ColorJitter(contrast=.5)),
      (1.0, tt.ToTensor())
    ]
    
  def forward(self, input):
    transes = [trans for p, trans in self.transforms if p >= np.random.rand()]
    transformation = tt.Compose(transes)
    
    print(transes)
    
    return transformation(input)

class CorruptWordGen():
  
  def __init__(self, *args, char_size=64, n_char_range=(2,10), max_iter=2048, **kwargs):
    self.dl_args = args
    self.dl_kwargs = {**kwargs, "batch_size": 1}
    self.n_iter = 0
    self.max_iter = max_iter
    self.img_shape = (char_size, int(char_size * (n_char_range[1] + n_char_range[0]) / 3))
    self.n_char_range = n_char_range
    self.data_loader = None
  
  def __iter__(self):
    self.n_iter = 0
    # self.data_loader = None
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
      # subtr_img = next(self.data_loader)[0][0][0]
      # crpt_img = base_img - (subtr_img + 1)
      # crpt_img = torch.maximum(crpt_img, -torch.ones(*crpt_img.shape))
      
      self.n_iter += 1
      
      # return char_gen(base_img.reshape((1, latent_size, 1, 1))).reshape((image_size, image_size)), base_img_lab
      return base_img, base_img_lab
      
    except StopIteration:
      self.data_loader = None
      return next(self)
    
  def gen_chars(self, num=1):
    crpt_imgs = [next(self) for _ in range(num)]
    return tuple([[img[i] for img in crpt_imgs] for i in range(len(crpt_imgs[0]))])
  
  def gen_words(self, num=1):
    chars, labs = self.gen_chars(randint(*self.n_char_range))
    n_chars = len(chars)
    max_height = sorted([char.shape[0] for char in chars])[-1]
    output_img_proportion = self.img_shape[0] / self.img_shape[1]
    # chars[0] = F.pad(chars[0], [0, 24, 0, 0])
    # h = chars[0].shape[0]
    
    for i in range(n_chars):
      char = chars[i]
      chars[i] = VF.pad(char, [0, max_height - char.shape[0], 0, 0], fill = -1)[None, :, :]
    
    # print([x.shape for x in chars])
    
    # base_word = VF.resize(torch.cat(tuple(chars), dim=2), size=self.img_size)[0, :, :]
    base_word = torch.cat(tuple(chars), dim=2)
    
    _, base_word_h, base_word_w = base_word.shape
    base_word_proportion = base_word_h / base_word_w
    y_pad = base_word_proportion < output_img_proportion
    # padding = [int((base_word_proportion - output_img_proportion) / (2 * base_word_w)),
    #            int((base_word_h - (base_word_h / base_word_proportion * output_img_proportion)) / 2) ]
    padding = [int(((self.img_shape[1] * base_word_h / self.img_shape[0]) - base_word_w) / 2),
               int((output_img_proportion - base_word_proportion) * base_word_w / 2)]
    # print(f"padding: {padding}")
    base_word = VF.pad(base_word, list(map(lambda x: max(x, 0), padding)), fill = -1)
    base_word = VF.resize(base_word, self.img_shape)
    
    crpt_word = WordAugmenter()(base_word)
    
    # print(chars.shape)
    # return torch.cat(tuple(chars), dim=1).detach().numpy()
    # return word.detach().numpy()
    return base_word, crpt_word, labs
  