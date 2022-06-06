from util import *
from torch.utils.data import DataLoader


######################
###  SEGMENTATION  ###
######################

class Segmenter():
  """
  Base Segmenter class.
  All word and line segmentation classes should inherit from this to encourage extendibility.
  Though currently serves little purpose other than sharing the postprocess function.
  """
  def __init__(self):
    raise NotImplementedError()
  
  def __call__(self, img):
    """
    Turns 2D grayscale image into list of sub-images.
    """
    raise NotImplementedError()
  
  def preprocess(self, img):
    return img
  
  def postprocess(self, units):
    """
    Crops all sub-images to remove the outer uninked rows and columns.
    """
    for i in range(len(units)):
      unit = units[i]
      
      inked_rows = np.array(range(unit.shape[0]))[np.sum(unit, axis=1) > 0]
      inked_cols = np.array(range(unit.shape[1]))[np.sum(unit, axis=0) > 0]
      
      row_range = inked_rows[0], inked_rows[-1] + 1
      col_range = inked_cols[0], inked_cols[-1] + 1
      
      units[i] = unit[row_range[0]:row_range[1], col_range[0]:col_range[1]]

    return units


class HHLineSegmenter(Segmenter):
  """
  Line segmenter using horizontal histogram projection.
  The __call__() implementation includes the full line segmentation pipeline,
  including pre- and postprocessing.
  """
  def __init__(self, pp_pad=128, pp_thresh=256, window_size=50, height_diff=20000):
    """
    Args:
      pp_pad (int): Nr. of rows or columns padded to the cropped image during preprocessing.
      
      pp_thresh (int): In preprocessing, images are cropped from the first outermost row and column
                       that exceeds this value. Corrects for any stray pixels before the actual text.
                       
      window_size (int): Nr. of bins to average the hight of the histogram over.
      
      height_diff (int): Difference between the current height and either the lowest height in the last
                         valley, or highest height in the last peak, for the window to be considered to
                         have left the peak or valley.
    """
    self.row_hist = None
    self.pp_pad = pp_pad
    self.pp_thresh = pp_thresh
    self.window_size = window_size
    self.height_diff = height_diff
    
  def __call__(self, img):
    """
    Full line segmentation function. Includes pre- and postprocessing.
    """
    img = self.preprocess(img)
    row_hist = self.get_row_hist(img)
    valleys = self.find_valleys(row_hist)
    valleys = valleys + [len(img) - 1]
    lines = [img[t:b,:] for t, b in zip(valleys, valleys[1:])]
    lines = self.postprocess(lines)
    return lines

  
  def preprocess(self, img, pad=None, thresh=None):
    """
    Crops and pads around the binarized text.
    
    Args:
    pad (int): Nr. of rows or columns padded to the cropped image during preprocessing.
               Uses self.pp_pad if unspecified.
    
    thresh (int): In preprocessing, images are cropped from the first outermost row and column
                  that exceeds this value. Corrects for any stray pixels before the actual text.
                  Uses self.pp_thresh if unspecified.
    """
    pad = self.pp_pad if pad is None else pad
    thresh = self.pp_thresh if thresh is None else thresh
    
    row_hist = np.sum(img, axis=1)
    # Record which rows and cols have a total pixel value above the threshold.
    inked_rows = np.array(range(img.shape[0]))[row_hist > thresh]
    inked_cols = np.array(range(img.shape[1]))[np.sum(img, axis=0) > thresh]
    # Define new img range as the first and last inked rows and cols + padding.
    row_range = max(inked_rows[0] - pad, 0), min(inked_rows[-1] + pad, len(img))
    col_range = max(inked_cols[0] - pad, 0), min(inked_cols[-1] + pad, len(img[0]))
    row_hist = row_hist[row_range[0]:row_range[1]]
    # Crop image.
    img = img[row_range[0]:row_range[1], col_range[0]:col_range[1]]
    return img
  
  def get_row_hist(self, img):
    """
    Sums all pixel values over the column axis.
    """
    return np.sum(img, axis=1)
  
  def find_valleys(self, row_hist, window_size=None, height_diff=None):
    """
    Uses a sliding window to find the lowest part (averaged over the window_size) for every valley.
    Initializes and being in a valley.
    
    Args:
      window_size (int): Nr. of bins to average the hight of the histogram over.
                         Uses self.window_size if unspecified.
      
      height_diff (int): Difference between the current height and either the lowest height in the last
                          valley, or highest height in the last peak, for the window to be considered to
                          have left the peak or valley. Uses self.height_diff if unspecified.
    """
    window_size = self.window_size if window_size is None else window_size
    height_diff = self.height_diff if height_diff is None else height_diff
    
    # Init valley list
    valleys = []
    # Init heights of the previous peak and valley
    valley_height = np.inf
    peak_height = 0
    # Index of the lowest point in the current valley
    valley_idx = 0
    # Whether the window is in a valley or peak (determined by height_diff)
    in_valley = True
    
    # Slide window over rows.
    for row in range(len(row_hist) - window_size):
      # Current averaged height.
      height = np.mean(row_hist[row : row + window_size])
      
      # Determine the window to have moved from a valley into a peak.
      if in_valley and height > valley_height + height_diff:
        in_valley = False
        valley_height = np.inf
        # Append the lowest point in the last valley to the list of valleys.
        valleys.append(valley_idx + int(window_size / 2))
      
      # Determine the window to have moved from a peak into a valley.
      elif not in_valley and height < peak_height - height_diff:
        in_valley = True
        peak_height = 0
      
      # Compare and save the lowest height encountered in the valley.
      if in_valley and height < valley_height:
        # print(3)
        valley_height = height
        valley_idx = row
      
      # Compare and save the highest height encountered in the peak.
      elif not in_valley and height > peak_height:
        # print(4)
        peak_height = height

    return valleys
  
  
class HHWordSegmenter(Segmenter):
  
  def __init__(self):
    pass
  
  def __call__(self, line, **kwargs):
    seqs = self.find_empty_col_seqs(line, **kwargs)
    # Column indices at which to separate the words
    word_seps = list(map(np.median, seqs))
    words = [line[:, int(l):int(r)] for l, r in zip(word_seps, word_seps[1:])]
    words = self.postprocess(words)
    return words
    
  def find_empty_col_seqs(self, line, min_col_sum=0.01, min_seq_len=0.1):
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
    n_rows = len(line)
    
    if isinstance(min_col_sum, float):
      mean_col_sum = np.mean(list(filter(lambda x: x > 0, col_hist)))
      min_col_sum = min_col_sum * mean_col_sum
    
    if isinstance(min_seq_len, float):
      min_seq_len = int(n_rows * min_seq_len)

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
    
class PrettyCorruptCharGen(DataLoader):
  """
  Prettier version of CorruptCharGen, though untested.
  Should now actually function as an iterator that stops when the char dataset is expended, 
  instead of after a maximum amount of iterations.
  """
  def __init__(self, *args, n_chars=1, is_base_gen=True, **kwargs):
    """
    Args:
      n_chars (int or callable): Nr. of chars to generate per __next__() call.
                                 Can also be a callable that returns an int.
      
      is_base_gen (bool): Boolean used to determine whether this object generates base images
                          or images used as corruptors. Admittedly a bit of a strange way to
                          cycle through two dataset copies.
    """
    super(PrettyCorruptCharGen, self).__init(*args, **kwargs)
    
    self.n_chars = n_chars if callable(n_chars) else lambda: n_chars
    
    self.is_base_gen = is_base_gen
    self.crpt_char_gen = None
    if is_base_gen:
      # Instance of the same class, used as a source for characters used as corruptors.
      self.crpt_char_gen = PrettyCorruptCharGen(*args, n_chars=1, is_base_gen=False)
  
  def __iter__(self):
    new_iter = super(PrettyCorruptCharGen, self).__iter__()
    if self.crpt_char_gen is not None:
      self.crpt_char_gen = iter(self.crpt_char_gen)
    return new_iter
  
  def __next__(self):
    """
    Returns triplet of a list of uncorrupted chars, a list of corrupted chars (based on the uncorrupted
    chars), and corresponding labels.
    """
    base_chars = []
    crpt_chars = []
    labels = []
    
    for _ in range(self.n_chars()):
      base_char, lab = next(super())
      base_chars.append(base_char)
      
      if self.is_base_gen:
        labels.append(lab)
        
        subtr_char = next(self.crpt_char_gen)[0]
        crpt_char = img_subtract(base_char, subtr_char)
        crpt_chars.append(crpt_char)
    
    if self.is_base_gen:
      return base_chars, crpt_chars, labels
    
    return base_chars
    
  def gen_chars(self, num=1):
    """
    Generate specific amount of characters, or with a callable that returns an int.
    """
    old_n_chars = self.n_chars
    self.n_chars = num if callable(num) else lambda: num
    
    chars_tuple = next(self)
    
    self.n_chars = old_n_chars
    return chars_tuple

    
class CorruptCharGen():
  """
  Corrupt character generator.
  Takes base and corruptor images from the same dataset, and reinitializes said dataset when empty.
  Only stops iterating when the amount of generated corrupted characters exceeds max_iter.
  """
  def __init__(self, *args, latent_size=64**2, max_iter=2048, **kwargs):
    """
    Args:
      latent_size (int): Length of the vectorized output image. Probably an unnecessary addition.
      
      max_iter (int): Maximum amount of corrupted characters to be generated before stopping iteration.
    """
    # Args and kwargs used for DataLoader initialization.
    self.dl_args = args
    self.dl_kwargs = {**kwargs, "batch_size": 1}
    self.latent_size = latent_size
    self.n_iter = 0
    self.max_iter = max_iter
    self.data_loader = None
  
  def __iter__(self):
    self.n_iter = 0
    return self
  
  def __next__(self):
    if self.n_iter > self.max_iter:
      raise StopIteration
    
    if self.data_loader is None:
      # Re-init contained dataloader.
      self.data_loader = iter(DataLoader(*self.dl_args, **self.dl_kwargs))
    
    try:
      # Randomly take base char image, without replacement.
      base_img = next(self.data_loader)
      # Character label.
      base_img_lab = base_img[1]
      base_img = base_img[0][0][0]
      # Subtractor image, from the same data set as base_char
      subtr_img = next(self.data_loader)[0][0][0]
      # Corrupted image.
      crpt_img = base_img - (subtr_img + 1)
      # Clip all values below -1
      crpt_img = torch.maximum(crpt_img, -torch.ones(*crpt_img.shape))
      
      self.n_iter += 1
      
      # Reshape to (latent_size, 1, 1)
      return crpt_img.reshape((self.latent_size, 1, 1)), base_img.reshape((self.latent_size, 1, 1)), base_img_lab
      
    except StopIteration:
      self.data_loader = None
      return next(self)
    
  def gen_chars(self, num=1):
    """
    Generate specific amount of characters, or with a callable that returns an int.
    """
    crpt_imgs = [next(self) for _ in range(num)]
    return tuple([torch.stack([img[i] for img in crpt_imgs]) for i in range(len(crpt_imgs[0]))])
 

class RandomCorrupt(nn.Module):
  """
  Torch Module to be used in word augmentation.
  Creates an image from one or more randomly transformed characters taken from the char_loader.
  Then subtracts this image from the passed image in the forward call, to synthetically erode the input image.
  """
  def __init__(self, char_loader, n_chars=(1, 4), transforms=None):
    """
    Args:
      char_loader (DataLoader): Data set containing characters used for subtraction.
      
      n_chars ((int, int)): Range of characters to be used for subtraction. Sampled from a uniform distribution.
      
      transforms ([(float, nn.Module)]): List of tuples of transformations to be applied to the characters,
                                         and a float indication the application probability.
    """
    super(RandomCorrupt, self).__init__()
    self.transforms = self._default_transforms if transforms is None else transforms
    self.char_list = char_loader
    self.rand_n_chars = lambda : randint(n_chars[0], n_chars[1] + 1)

  def gen_rand_chars(self):
    """
    Returns list of randomly generated characters.
    """
    return [self.char_list[i][0] for i in randint(len(self.char_list), size=self.rand_n_chars())]

  @property
  def _default_transforms(self):
    """
    Default list of transforms and their probabilities.
    """
    return [
      (0.3, tt.RandomAffine(360, fill=-1)),
      (1.0, RandomPad((300, 100), fill=-1)),
      (0.3, tt.RandomAffine(0, scale=(0.7, 1), fill=-1)),
      (0.3, tt.RandomPerspective(p=1, distortion_scale=0.5, fill=-1)),
      (0.3, tt.RandomAffine(0, shear=(-20, 20, -20, 20), fill=-1)),
      (0.3, tt.RandomAffine(0, translate=(0.7, 0.4), fill=-1)),
      (None, None),
    ]
  
  def forward(self, input):
    input_shape = np.shape(input)
    # Include resize transformation to the input shape.
    self.transforms[-1] = (1.0, tt.Resize(input_shape[-2:]))
    
    added_chars = None
    
    # Save an image of the uncorrupted word.
    save_image(input, './generated/word/clean_word.png')
    
    for char in self.gen_rand_chars():
      # Construct list of included transformations.
      transforms = [trans for p, trans in self.transforms if p >= np.random.rand()]
      transformation = tt.Compose(transforms)
      
      char = transformation(char).reshape(np.shape(input))
      
      # Subtract the transformed char from the input.
      input = img_subtract(input, char)
      
      if added_chars is None:
        added_chars = char
      else:
        added_chars = img_add(added_chars, char)
    
    if added_chars is not None:
      save_image(added_chars, './generated/word/added_chars.png')
      
    save_image(input, './generated/word/corrupted_word.png')

    return input
    
    
class WordAugmenter(nn.Module):
  """
  Augmentation (and corruption) class for synthetic words.
  Includes erosion via RandomCorrupt, as well as trandom translations, rotations, scaling,
  perspective shifts, and shears.
  """
  def __init__(self,  *args, transforms=None, **kwargs):
    """
    Args:
      transforms ([(float, nn.Module)]): List of tuples of transformations to be applied to the word,
                                         and a float indication the application probability.
    """
    super(WordAugmenter, self).__init__(*args, **kwargs)
    self.transforms = self._default_transforms if transforms is None else transforms
  
  @property
  def _default_transforms(self):
    """
    Default list of transforms and their probabilities.
    """
    return [
      (0.4, RandomCorrupt(load_dataset('char', equal_shapes=False))),
      (1.0, tt.ToPILImage()),
      (0.2, tt.RandomAffine(0, scale=(0.7, 1))),
      (0.2, tt.RandomPerspective(p=1, distortion_scale=0.2)),
      (0.2, tt.RandomAffine(0, shear=(-10, 10, -10, 10))),
      (0.2, tt.RandomAffine(8)),
      (0.2, tt.RandomAffine(0, translate=(0.1, 0.4))),
      (0.2, tt.RandomAffine(0, scale=(1, 1.3))),
      (1.0, tt.ToTensor()),
      (1.0, tt.Normalize((0.5,), (0.5,))),
    ]
    
  def forward(self, input):
    # Construct list of included transformations.
    transforms = [trans for p, trans in self.transforms if p >= np.random.rand()]
    transformation = tt.Compose(transforms)
    
    return transformation(input)


class OldCorruptWordGen():
  """
  Older, uglier version of CorruptWordGen.
  Structured similarly to CorruptCharGen.
  """
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
      
      self.n_iter += 1
      
      return base_img, base_img_lab
      
    except StopIteration:
      self.data_loader = None
      return next(self)
    
  def gen_chars(self, num=1):
    crpt_imgs = [next(self) for _ in range(num)]
    return tuple([[img[i] for img in crpt_imgs] for i in range(len(crpt_imgs[0]))])
  
  def gen_words(self, num=1):
    chars, labs = self.gen_chars(randint(*self.n_char_range))
    
    chars = equalize_heights(chars)
    base_word = pad_and_resize(glue_chars(chars, 8), self.img_shape)
    crpt_word = pad_and_resize(glue_chars(chars, padding = lambda: np.random.uniform(-18, 6)), self.img_shape)
    crpt_word = WordAugmenter().forward(crpt_word)
    
    return base_word, crpt_word, labs
  
  
class CorruptWordGen():
  """
  Corrupt word generator class.
  Generates triples of uncorrupted synthetic words, their corrupted versions, and labels.
  Functions as an iterator.
  """
  def __init__(self, data_loader, img_shape=(64, 64*4), n_char_range=(2,10), batch_size=1, base_char_pad=8, crpt_char_pad=None):
    """
    Args:
      data_loader (DataLoader): Data set containing character images.
      
      img_shape (int or (int, int)): Shape of the output word images.
      
      n_char_range ((int, int)): Range of characters in a synthetic word. Sampled uniformly.
      
      batch_size (int): Nr. of words per __next__() call.
      
      base_char_pad (int or callable): Padding between the characters in the uncorrupted word.
                                       Can also be a callable returning an int.
      
      crpt_char_pad (int or callable): Padding between the characters in the corrupted word.
                                       Can also be a callable returning an int.
                                       Negative values cause overlapping characters.
    """
    self.batch_size = batch_size
    # Reshape function for after concatenating the words to a tensor of words.
    # Simply just puts an extra dimension in there, in accordance with grayscale image format.
    self.cat_reshape = Reshape('0', 1, '1', '2').forward
    
    if isinstance(img_shape, tuple):
      self.img_shape = img_shape
    else:
      self.img_shape = (img_shape, int(img_shape * (n_char_range[1] + n_char_range[0]) / 3))
      
    self.n_char_range = n_char_range
    self.data_loader = iter(data_loader)
    self.empty_data_loader = False
    
    if crpt_char_pad is None:
      crpt_char_pad = lambda: np.random.uniform(-12, 6)
      
    self.crpt_char_pad = crpt_char_pad
    self.base_char_pad = base_char_pad
  
  def __iter__(self):
    self.data_loader = iter(self.data_loader)
    self.empty_data_loader = False
    return self
  
  def __next__(self):
    if self.empty_data_loader:
      raise StopIteration

    base_words = []
    crpt_words = []
    labels = []

    for _ in range(self.batch_size):

      try:
        chars = []
        labs = []
        for _ in range(randint(*self.n_char_range)):
          item = next(self.data_loader)
          chars.append(item[0][0][0])
          labs.append(item[1].item())

      except StopIteration:
        self.empty_data_loader = True

        if len(base_words) == 0:
          raise StopIteration  

        break

      # Pad the tops of the chars to have the same height as the tallest char.
      chars = equalize_heights(chars)
      # Glue the characters into a word (using the character padding), then pad and resize to the output image shape.
      base_word = pad_and_resize(glue_chars(chars, padding = self.base_char_pad), self.img_shape)
      crpt_word = pad_and_resize(glue_chars(chars, padding = self.crpt_char_pad), self.img_shape)
      # Augment (i.e. corrupt) the corrupted word.
      crpt_word = WordAugmenter().forward(crpt_word)

      base_words.append(base_word)
      crpt_words.append(crpt_word)
      labels.append(labs)
      
    base_words = self.cat_reshape(torch.cat(base_words))
    crpt_words = self.cat_reshape(torch.cat(crpt_words))
    return base_words, crpt_words, labels