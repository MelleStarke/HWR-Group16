import transformers
import datasets

# Import pandas to allow working with data frames
import pandas as pd

# Make a data frame of the .txt data
TRAIN_DATA_DIR = "data/"
raw_txt_df = pd.read_fwf(TRAIN_DATA_DIR + 'iam_lines_gt.txt', header=None)
first_column = raw_txt_df.iloc[::2]
second_column = raw_txt_df.iloc[1::2]
txt_df = pd.concat([first_column.reset_index(drop=True), second_column.reset_index(drop=True)], axis=1)
txt_df = pd.DataFrame(txt_df)
txt_df = txt_df.set_axis(['file_name', 'text'], axis=1)

# Import a train_test_split function to allow for validation
from sklearn.model_selection import train_test_split

# Use a 80-20 train-validation split
train_txt_df, val_txt_df = train_test_split(txt_df, test_size=0.2)
train_txt_df.reset_index(drop=True, inplace=True)
val_txt_df.reset_index(drop=True, inplace=True)

import os
import cv2
import matplotlib.pyplot as plt

IMG_TRAIN_DATA_DIR = 'data/img/'

cnt = 0

# Iterate over the IAM data, generate binary images and show these
# While the images are not used, for know, this provides an illustration
for filename in os.listdir(IMG_TRAIN_DATA_DIR):
    # Get an image
    f = os.path.join(IMG_TRAIN_DATA_DIR, filename)
    img = cv2.imread(f)
    
    # Present 10 images in total
    cnt = cnt + 1
    if cnt == 10:
      break

    # Threshold the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, threshold_img) = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
    window_name = "img"
    
    # Show the resulting binary image
    plt.imshow(threshold_img, cmap='gray')
    plt.imshow(gray_img, cmap='gray')


from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# from keras.preprocessing.image import img_to_array
from matplotlib import pyplot

# Create an Image Data Generator for augmentation
# Rotate, zoom and shear the images
augmentator = ImageDataGenerator(
	rotation_range = 0.5,
	shear_range = 0.5,
	fill_mode = "constant",
    cval = 255)

# Create new data frames in which the augmented image names are included
new_train_txt_df = train_txt_df.copy()
new_val_txt_df = val_txt_df.copy()

cnt = 0

# Loop through all images
for filename in os.listdir(IMG_TRAIN_DATA_DIR):
    cnt = cnt + 1
    
    # Get the image
    f = os.path.join(IMG_TRAIN_DATA_DIR, filename)
    img = cv2.imread(f)  
    
    # Reshape the image so that it can be used
    x = np.array(img)
    x = x.reshape((1, ) + x.shape)  
    
    # Create the folder in which the augmented images are saved
    aug_image_folder = "data/img_aug/"
    
    i = 0
    # Generate 5 augmented images per image
    for batch in augmentator.flow(x, batch_size = 1):
        # Save the augmented image
        name = str(i) + "_" + filename
        cv2.imwrite(f'{aug_image_folder}/{name}',batch[0,:,:,:])
        
        # Add the augmented image name to a txt dataframe
        df_filename = new_train_txt_df[new_train_txt_df['file_name'] == filename]
        df_filename['file_name'] = name
        new_train_txt_df = new_train_txt_df.append([df_filename],ignore_index=True)
        df_filename = new_val_txt_df[new_val_txt_df['file_name'] == filename]
        df_filename['file_name'] = name
        new_val_txt_df = new_val_txt_df.append([df_filename],ignore_index=True)
        
        i += 1
        if i > 5: 
            # Remove the original image from the txt dataframes
            new_train_txt_df = new_train_txt_df.drop(new_train_txt_df[new_train_txt_df['file_name'] == filename].index)
            new_val_txt_df = new_val_txt_df.drop(new_val_txt_df[new_val_txt_df['file_name'] == filename].index)
            break
new_train_txt_df.reset_index(drop=True, inplace=True)
new_val_txt_df.reset_index(drop=True, inplace=True)


import torch
from torch.utils.data import Dataset
from PIL import Image

# Create a class for the IAM data (largely based on https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb)
class IAMDataset(Dataset):
    
    # Initialise 
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
    
    # Return the length
    def __len__(self):
        return len(self.df)
    
    # Get an item from the data
    def __getitem__(self, idx):
        # Get the file name and corresponding text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]

        # Get the image
        image = cv2.imread(self.root_dir + file_name)

        # Create a binary image from the original image (to get rid of background noise)
        (thresh, threshold_img) = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
        threshold_img = np.img_to_array(threshold_img)

        # Get the pixel values
        pixel_values = self.processor(threshold_img, return_tensors="pt").pixel_values

        # Get labels by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        
        # Ensure that the PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        
        return encoding
      
      
from transformers import TrOCRProcessor

# Use the base TROCR model as a processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Create IAM datasets using the IAMDataset class
train_dataset = IAMDataset(root_dir='data/img_aug/',
                           df=new_train_txt_df,
                           processor=processor)
val_dataset = IAMDataset(root_dir='data/img_aug/',
                           df=new_val_txt_df,
                           processor=processor)


from torch.utils.data import DataLoader

# Create data loaders for the training and validation data sets
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_dataloader = DataLoader(val_dataset, batch_size=4)


from transformers import VisionEncoderDecoderModel
import torch

# Use a GPU if that is possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the base TROCR model
model = VisionEncoderDecoderModel.from_pretrained("/microsoft/trocr-base-stage1")
model.to(device)


