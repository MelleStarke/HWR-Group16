import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Flatten, Input, add, BatchNormalization, Conv2D, AveragePooling2D
from sklearn.preprocessing import LabelBinarizer

img_folder = './../../data/dss/train-imgs'
seg_folder = 'segmented'
char_folder = './../../data/dss/monkbrill'

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
      
# lb = LabelBinarizer()
      
model = ResNet.build(len(lb.classes_), (3, 3, 3), (64, 64, 128, 256), reg = 0.0005)
model.load_weights('trained/classifier/classification_model.h5')