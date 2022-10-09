import os
from tkinter.filedialog import Directory
import json
import pandas as pd
import cv2 as cv
import numpy as np
import util
import preprocess as pp

from argparse import ArgumentParser
import shutil

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Flatten, Input, add, BatchNormalization, Conv2D, AveragePooling2D

import matplotlib.pyplot as plt

parser = ArgumentParser(description="Transcribes binarized images of Hebrew texts into machine text.")

parser.add_argument("input_dir", metavar="I", nargs=1, type=str, help="input directory containing Hebrew text images.")
parser.add_argument("--output_dir", metavar="-o", nargs=1, type=str, default="./results/",
                    help="output directory for transcribed texts.")
parser.add_argument("--only-binarized-imgs", default=False, action="store_true", dest="only_binarized",
                    help="only transcribe images with 'binarized' in their file names.")
parser.add_argument("--train-classifier", default=False, action="store_true", dest="train_classifier",
                    help="train the classifier, instead of testing it on the data.")


# python transcribe_images.py ../../data/dss/train-imgs/ --only-binarized-imgs

args = parser.parse_args()
input_dir = args.input_dir[0]
output_dir = args.output_dir
binarized_check = (lambda x: "binarized" in x) if args.only_binarized else lambda x: True

cwd = os.getcwd()
# print(os.path.join(input_dir, os.listdir(input_dir)[0]))
# print(binarized_check(os.listdir(input_dir)[0]))
file_names = [os.path.join(input_dir, fn) for fn in os.listdir(input_dir) if binarized_check(fn)]
img_names = [fn.split('.')[0] for fn in os.listdir(input_dir) if binarized_check(fn)]
  
input_imgs = [cv.threshold(cv.imread(fn, 0), 127, 255, cv.THRESH_BINARY_INV)[1] for fn in file_names]

# plt.imshow(input_imgs[0])
# print(np.shape(input_imgs[0]))
# plt.show()

# print(list(file_names))

line_segmenter = pp.HHLineSegmenter()

print("Segmenting lines...")

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

print("Segmenting words...")

words = [[word_segmenter(line) for line in img_lines] for img_lines in lines]

restoration_model = util.load_word_restoration_model()
restoration_model.eval()

try:
  shutil.rmtree("./segmented/")
except FileNotFoundError:
  pass

print("Saving words...")

for img_idx in range(len(words)):
  img_words = words[img_idx]
  for line_idx in range(len(img_words)):
    line_words = img_words[line_idx]
    for word_idx in range(len(line_words)):
      word = line_words[word_idx]
      fn = f"./segmented/{img_names[img_idx]}/line{line_idx}"
      
      try:
        os.makedirs(fn)
      except FileExistsError:
        pass
      
      fn += f"/word{word_idx}_base.png"
      cv.imwrite(fn, word)
      
      # # Restored word saving
      # fn = f"./segmented/{file_names[img_idx]}/line{line_idx}"
      
      # fn += f"/word{word_idx}_restored.png"
      # word = util.pad_and_resize(util.to_norm_tensor(word), (64, 64*4)).reshape(1, 64*64*4, 1, 1)
      # plt.imshow(word.reshape(64, 64*4))
      # plt.show()
      # word = restoration_model.forward(word).reshape((64, 64*4, 1)).detach().numpy()
      # plt.imshow(word.reshape(64, 64*4))
      # plt.show()
      # cv.imwrite(fn, word)

#-------------------------------------------------------------------------------------------------#

img_folder = input_dir
seg_folder = 'segmented'
char_folder = './../../data/dss/monkbrill'

print('Segmenting characters...')
#Function to calculate row histograms and split the characters using it.
def split(imgg_path):
    img = cv.imread(imgg_path, 0)
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)[1]
    row_hist = np.sum(img, axis = 1)
    inked_rows = np.array(range(img.shape[0]))[row_hist > 256]
    inked_cols = np.array(range(img.shape[1]))[np.sum(img, axis=0) > 256]
    row_range = inked_rows[0], inked_rows[-1]
    col_range = inked_cols[0], inked_cols[-1]
    row_hist = row_hist[row_range[0]:row_range[1]]
    img = img[row_range[0]:row_range[1], col_range[0]:col_range[1]]

    idx = np.argpartition(row_hist[int(len(row_hist)/4):int(4*len(row_hist)/4)], int(img.shape[1]/50))
    idx = idx[:int(img.shape[1]/50)+1]
    idx = idx.tolist()
    idx.sort()

    for i in range(len(idx)):
        if i == 0:
            roi = result[y:y+h, x:x+idx[i]]
        else:
            roi = result[y:y+h, x+idx[i-1]:x+idx[i]]
        if roi.size > 0:
            cv.imwrite(imgg_path[:-4] + '_' + str(i) + '.jpg', roi)
        
    roi2 = result[y:y+h, x+max(idx):x+w]
    cv.imwrite(imgg_path[:-4] + '_' + str(i+1) + '.jpg', roi2)
    
n_chars = []

#Splitting the characters from the segmented words
for im_folder in os.listdir(os.path.join(seg_folder)):
    if im_folder != '.DS_Store':
        n_chars.append([])
        for line_folder in os.listdir(os.path.join(seg_folder, im_folder)):
            if line_folder != '.DS_Store':
                n_chars[-1].append([])
                for file in os.listdir(os.path.join(seg_folder, im_folder, line_folder)):
                    if file != '.DS_Store':
                        image_path = os.path.join(seg_folder, im_folder, line_folder, file)
                        gray = cv.imread(image_path, 0)
                        # gray = cv.bitwise_not(gray)                              
                        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]    

                        kernel = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
                        morph = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
                        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
                        morph = cv.morphologyEx(morph, cv.MORPH_ERODE, kernel)

                        cntrs = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

                        result = gray.copy()
                        
                        n_chars[-1][-1].append(len(cntrs))
                        
                        for c in cntrs:
                            x, y, w, h = cv.boundingRect(c)
                            cv.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            roi = result[y:y+h, x:x+w]
                            cv.imwrite(image_path[:-4] + '_' + str(x)+ '.jpg', roi)
                            if w > 70:
                                img_path = image_path[:-4] + '_' + str(x)+ '.jpg'
                                split(img_path)
                                os.remove(img_path)
                        os.remove(image_path)
                                
                        cv.destroyAllWindows()
# print(n_chars)

for im_folder in os.listdir(seg_folder):
    if im_folder != '.DS_Store':
        for line_folder in os.listdir(os.path.join(seg_folder, im_folder)):
            if line_folder != '.DS_Store':
                for file in os.listdir(os.path.join(seg_folder, im_folder, line_folder)):
                    if file != '.DS_Store':
                        image_path = os.path.join(seg_folder, im_folder, line_folder, file)
                        imm = cv.imread(image_path, 0)
                        if imm.shape[0] < 4 or imm.shape[1] < 4:
                            os.remove(image_path)

#-------------------------------------------------------------------------------------------------#

#ResNet module for classification
class ResNet:
    def residual_module(data, K, stride, chanDim, red = False, reg = 0.0001, bnEps = 2e-5, bnMom = 0.9):
        shortcut = data

        Lay1 = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(data)
        Lay1 = Activation("relu")(Lay1)
        Lay1 = Conv2D(int(K * 0.25), (1, 1), use_bias = False, kernel_regularizer = l2(reg))(Lay1)


        Lay2 = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(Lay1)
        Lay2 = Activation("relu")(Lay2)
        Lay2 = Conv2D(int(K * 0.25), (3, 3), strides = stride, padding = "same", use_bias = False, kernel_regularizer = l2(reg))(Lay2)


        Lay3 = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(Lay2)
        Lay3 = Activation("relu")(Lay3)
        Lay3 = Conv2D(K, (1, 1), use_bias = False, kernel_regularizer = l2(reg))(Lay3)

        if red:
            shortcut = Conv2D(K, (1, 1), strides = stride, use_bias = False, kernel_regularizer = l2(reg))(Lay1)

        x = add([Lay3, shortcut])

        return x

    def build(classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        inputShape = (32, 32, 1)
        chanDim = -1

        inputs = Input(shape = inputShape)
        x = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(inputs)
        x = Conv2D(filters[0], (3, 3), use_bias = False, padding = "same", kernel_regularizer = l2(reg))(x)


        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, red = True, bnEps = bnEps, bnMom = bnMom)

            for j in range(0, stages[i] - 1):
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim, bnEps = bnEps, bnMom = bnMom)

        x = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer = l2(reg))(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name = "resnet")


        return model
#-------------------------------------------------------------------------------------------------#

#Function to create a dataset from the image folder
def create_dataset(folder):
    img_data_array = []
    class_name = []
   
    for dir1 in os.listdir(folder):
        for file in os.listdir(os.path.join(folder, dir1)):
       
            image_path = os.path.join(folder, dir1, file)
            image = cv.imread(image_path, 0)
            image = cv.threshold(image, 63, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
            image = cv.resize(image, (32, 32), interpolation = cv.INTER_NEAREST)                    
            image = np.array(image, dtype = 'float32')                                                
            image = np.expand_dims(image, axis = -1)
            image /= 255.0                                                         

            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name


img_data, class_data = create_dataset(char_folder)
img_data = np.asarray(img_data)

lb = LabelBinarizer()
labels = lb.fit_transform(class_data)
ounts = labels.sum(axis = 0)

(X_train, X_test, y_train, y_test) = train_test_split(img_data, labels, test_size = 0.01, stratify = None, random_state = 42)

# tr_te = int(input('Enter 1 to train and 0 to test: '))
tr_te = args.train_classifier

#Training the model
if tr_te == True:
    EPOCHS = 50
    INIT_LR = 5e-2
    BS = 64

    classTotals = labels.sum(axis = 0)
    classWeight = {}

    for i in range(0, len(classTotals)):
        classWeight[i] = classTotals.max() / classTotals[i]


    aug = ImageDataGenerator(rotation_range = 10, zoom_range = 0.01, width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.1, horizontal_flip = False, fill_mode = "nearest")

    print("[UPDATE] Compiling model...")
    opt = SGD(learning_rate = INIT_LR, decay = INIT_LR / EPOCHS)
    model = ResNet.build(len(lb.classes_), (3, 3, 3), (64, 64, 128, 256), reg = 0.0005)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

    print("[UPDATE] Training network...")
    H = model.fit(aug.flow(X_train, y_train, batch_size = BS), validation_data = (X_test, y_test), steps_per_epoch = len(X_train) // BS, epochs = EPOCHS, class_weight = classWeight, verbose = 1)

#-------------------------------------------------------------------------------------------------#

#Testing the model
if tr_te == False:
    model = ResNet.build(len(lb.classes_), (3, 3, 3), (64, 64, 128, 256), reg = 0.0005)
    model.load_weights('trained/classifier/classification_model.h5')
    lines, images, all_ims = [], [], []

    for im_folder in os.listdir(seg_folder):
        if im_folder != '.DS_Store':
            for line_folder in os.listdir(os.path.join(seg_folder, im_folder)):
                if line_folder != '.DS_Store':
                    for file in os.listdir(os.path.join(seg_folder, im_folder, line_folder)):
                        if file != '.DS_Store':
                            charac = os.path.join(seg_folder, im_folder, line_folder, file)
                            charac = cv.imread(charac, 0)
                            charac = cv.bitwise_not(charac)
                            charac = cv.resize(charac, (32, 32), interpolation = cv.INTER_NEAREST)
                            charac = np.array(charac, dtype = 'float32')                                               
                            charac = np.expand_dims(charac, axis = -1)
                            charac = np.asarray(charac)
                            lines.append(charac)
                    images.append(lines)
                    lines = []
            all_ims.append(images)
            images = []
    
    # f = open('output_dictionary.json')
    # chars = json.load(f, encoding='utf-16')
    # f.close()
    for ii, im in enumerate(all_ims):
        line_output = []
        for line in im:  
            line = np.asarray(line)
            line /= 255.0
            if len(line) > 0:
              preds = model.predict(line)
              preds = lb.inverse_transform(preds)
            else:
              preds = []
            word_output = [util.transcribe_label(preds[jj], numeric=False) for jj in range(len(preds))]
            # for jj in range(len(preds)):
            #     f.write(util.transcribe_label(preds[jj], numeric=False) + ' ')
            # f.write('\n')
            line_output.append(" ".join(word_output))
        f = open( output_dir + '{}.txt'.format(img_names[ii]), 'w+', encoding='utf-16')
        f.write("\n".join(line_output)) 
        f.close()

