from util import *
from preprocess import *

rc = RandomCorruptor(load_dataset('char', equal_shapes=False))
rc.forward(0)
