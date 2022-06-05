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
    
class PrettyCorruptCharGen(DataLoader):
  """
  Prettier version of CorruptCharGen, though untested.
  Should now actually function as an iterator that stops when the char dataset is expended, 
  instead of after a maximum amount of iterations.
  """
  def __init__(self, *args, n_chars=1, is_base_gen=True, **kwargs):
    """
    Args:
      latent_size (int): length of the input vector 
    """
    super(PrettyCorruptCharGen, self).__init(*args, **kwargs)
    self.latent_size = latent_size
    
    self.n_chars = n_chars if callable(n_chars) else lambda: n_chars
    
    self.is_base_gen = is_base_gen
    self.crpt_char_gen = None
    if is_base_gen:
      self.crpt_gen = PrettyCorruptCharGen(*args, latent_size, n_chars, is_base_gen=False)
  
  def __iter__(self):
    new_iter = super(PrettyCorruptCharGen, self).__iter__()
    self.crpt_char_gen = iter(self.crpt_char_gen)
    return new_iter
  
  def __next__(self):
    base_chars = []
    crpt_chars = []
    labels = []
    
    for _ in range(self.n_chars()):
      base_char, lab = next(self)
      base_chars.append(base_char)
      
      if self.is_base_gen:
        labels.append(lab)
        
        subtr_char = next(self.crpt_char_gen)[0]
        crpt_char = img_subtract(base_char, subtr_char)
        crpt_chars.append(crpt_char)
    
    if self.is_base_gen:
      return base_chars, crpt_chars, labels
    
    return base_chars
    
    if self.n_iter > self.max_iter:
      raise StopIteration
    
  def gen_chars(self, num=1):
    old_n_chars = self.n_chars
    self.n_chars = num if callable(num) else lambda: num
    
    chars_tuple = next(self)
    
    self.n_chars = old_n_chars
    return chars_tuple

    
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
 

class RandomCorrupt(nn.Module):
  
  def __init__(self, char_loader, n_chars=(1, 4), transforms=None):
    super(RandomCorrupt, self).__init__()
    self.transforms = self._default_transforms if transforms is None else transforms
    self.char_list = char_loader
    self.rand_n_chars = lambda : randint(n_chars[0], n_chars[1] + 1)
    # self.rand_n_chars = lambda : 3
    # self.rand_char_list = lambda : np.array(char_list)[randint(len(char_list), size=rand_n_chars())]

  def gen_rand_chars(self):
    # print(f"char list len: {len(self.char_list)}\nrand index: {randint(len(self.char_list))}\n" +\
    #       f"size: {self.rand_n_chars()}\nchar list shape: {np.shape(self.char_list)}")
    # return np.choose(randint(len(self.char_list), size=self.rand_n_chars()), self.char_list)
    # return self.char_list[torch.randint(len(self.char_list), size=(self.rand_n_chars(),))]
    # return np.random.choice(self.char_list, self.rand_n_chars, replace=True)
    return [self.char_list[i][0] for i in randint(len(self.char_list), size=self.rand_n_chars())]

  @property
  def _default_transforms(self):
    # total_pad = randint(200)
    # l_pad = randint(total_pad)
    # r_pad = total_pad - l_pad
    return [
      # (1.0, tt.ToPILImage()),
      # (1.0, tt.Pad((l_pad, 0, r_pad, 0), fill=-1)),
      (0.3, tt.RandomAffine(360, fill=-1)),
      (1.0, RandomPad((300, 100), fill=-1)),
      (0.3, tt.RandomAffine(0, scale=(0.7, 1), fill=-1)),
      (0.3, tt.RandomPerspective(p=1, distortion_scale=0.5, fill=-1)),
      (0.3, tt.RandomAffine(0, shear=(-20, 20, -20, 20), fill=-1)),
      (0.3, tt.RandomAffine(0, translate=(0.7, 0.4), fill=-1)),
      (None, None),
      # (1.0, tt.ToTensor()),
      # (1.0, tt.Normalize((0.5,), (0.5,))),
    ]
  
  def forward(self, input):
    # input_shape = just_give_me_the_goddamned_image_size(input)[-2:]
    input_shape = np.shape(input)
      
    # input = deepcopy(input)
    # input_shape = input.shape
    self.transforms[-1] = (1.0, tt.Resize(input_shape[-2:]))
    
    # working_imshow(input)
    # plt.show()
    
    added_chars = None
    
    save_image(input, './generated/word/clean_word.png')
    
    for char in self.gen_rand_chars():
      # # char = char.detach().numpy()
      # try:
      #   print(np.shape(char[0]))
      # except TypeError:
      #   print(f"somehow int: {char}\n")
        
      # print(len(np.shape(char[0])))
      # print("reshaped")
      # char = char[0][0] if len(np.shape(char[0])) > 2 else char[0]
      # print(np.shape(char))
      # print(np.shape(char[0]))
      # working_imshow(char)
      # plt.show()
      # print(np.shape(char))
      
      transforms = [trans for p, trans in self.transforms if p >= np.random.rand()]
      transformation = tt.Compose(transforms)
      # print(f"input shape: {input_shape}")
      char = transformation(char).reshape(np.shape(input))
      
      input = img_subtract(input, char)
      
      if added_chars is None:
        added_chars = char
      else:
        added_chars = img_add(added_chars, char)
      
      # print(f"subtracted word range: {(torch.min(input), torch.max(input))}\n subtracting char range: {(torch.min(char), torch.max(char))}")
      # working_imshow(char)
      # plt.show()
    
    if added_chars is not None:
      save_image(added_chars, './generated/word/added_chars.png')
    save_image(input, './generated/word/corrupted_word.png')
    
    # working_imshow(input)
    # plt.show()
    # print(torch.mean(input))
    return input
    
    
class WordAugmenter(nn.Module):
  
  def __init__(self,  *args, transforms=None, **kwargs):
    super(WordAugmenter, self).__init__(*args, **kwargs)
    self.transforms = self._default_transforms if transforms is None else transforms
  
  @property
  def _default_transforms(self):
    return [
      (0.4, RandomCorrupt(load_dataset('char', equal_shapes=False))),
      (1.0, tt.ToPILImage()),
      (0.2, tt.RandomAffine(0, scale=(0.7, 1))),
      (0.2, tt.RandomPerspective(p=1, distortion_scale=0.2)),
      (0.2, tt.RandomAffine(0, shear=(-10, 10, -10, 10))),
      (0.2, tt.RandomAffine(8)),
      (0.2, tt.RandomAffine(0, translate=(0.1, 0.4))),
      (0.2, tt.RandomAffine(0, scale=(1, 1.3))),
      # (0.2, tt.ColorJitter(contrast=.5)),
      (1.0, tt.ToTensor()),
      (1.0, tt.Normalize((0.5,), (0.5,))),
    ]
    
  def forward(self, input):
    transforms = [trans for p, trans in self.transforms if p >= np.random.rand()]
    transformation = tt.Compose(transforms)
    
    # print(f"word augmenter transforms: {transforms}")
    
    # print(f"to be augmented word shape: {input.shape}")
    
    return transformation(input)


class OldCorruptWordGen():
  
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
    # n_chars = len(chars)
    # max_height = sorted([char.shape[0] for char in chars])[-1]
    # output_img_proportion = self.img_shape[0] / self.img_shape[1]
    # # chars[0] = F.pad(chars[0], [0, 24, 0, 0])
    # # h = chars[0].shape[0]
    
    # for i in range(n_chars):
    #   char = chars[i]
    #   chars[i] = VF.pad(char, [0, max_height - char.shape[0], 0, 0], fill = -1)[None, :, :]
    
    # # print([x.shape for x in chars])
    
    # # base_word = VF.resize(torch.cat(tuple(chars), dim=2), size=self.img_size)[0, :, :]
    # base_word = torch.cat(tuple(chars), dim=2)
    
    # _, base_word_h, base_word_w = base_word.shape
    # base_word_proportion = base_word_h / base_word_w
    # y_pad = base_word_proportion < output_img_proportion
    # # padding = [int((base_word_proportion - output_img_proportion) / (2 * base_word_w)),
    # #            int((base_word_h - (base_word_h / base_word_proportion * output_img_proportion)) / 2) ]
    # padding = [int(((self.img_shape[1] * base_word_h / self.img_shape[0]) - base_word_w) / 2),
    #            int((output_img_proportion - base_word_proportion) * base_word_w / 2)]
    # # print(f"padding: {padding}")
    
    # transformation = tt.Compose([#tt.ToPILImage(),
    #                              tt.Pad(list(map(lambda x: max(x, 0), padding)), fill = -1),
    #                              tt.Resize(self.img_shape),
    #                             #  tt.ToTensor(),
    #                             #  tt.Normalize((0.5,), (0.5,))
    #                             ])
    # # base_word = VF.pad(base_word, list(map(lambda x: max(x, 0), padding)), fill = -1)
    # # base_word = VF.resize(base_word, self.img_shape)
    # # print(f"gen word shape: {base_word.shape}")
    # base_word = transformation(base_word)
    
    # # print(type(base_word))
    
    # crpt_word = WordAugmenter().forward(base_word[0,:,:])
    
    chars = equalize_heights(chars)
    base_word = pad_and_resize(glue_chars(chars, 8), self.img_shape)
    crpt_word = pad_and_resize(glue_chars(chars, padding = lambda: np.random.uniform(-18, 6)), self.img_shape)
    crpt_word = WordAugmenter().forward(crpt_word)
    
    
    # print(chars.shape)
    # return torch.cat(tuple(chars), dim=1).detach().numpy()
    # return word.detach().numpy()
    return base_word, crpt_word, labs
  
  
class CorruptWordGen():
  
  def __init__(self, data_loader, img_shape=64, n_char_range=(2,10), batch_size=1, base_char_pad=8, crpt_char_pad=None):
    self.batch_size = batch_size
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
  
  def __len__(self):
    # lb, ub = self.n_char_range
    # return int(len(self.data_loader) * 2 / (lb + ub))
    return 35
  
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

      chars = equalize_heights(chars)
      base_word = pad_and_resize(glue_chars(chars, padding = self.base_char_pad), self.img_shape)
      crpt_word = pad_and_resize(glue_chars(chars, padding = self.crpt_char_pad), self.img_shape)
      crpt_word = WordAugmenter().forward(crpt_word)

      base_words.append(base_word)
      crpt_words.append(crpt_word)
      labels.append(labs)

    # print(chars.shape)
    # return torch.cat(tuple(chars), dim=1).detach().numpy()
    # return word.detach().numpy()
    # print(labels)
    base_words = self.cat_reshape(torch.cat(base_words))
    crpt_words = self.cat_reshape(torch.cat(crpt_words))
    return base_words, crpt_words, labels