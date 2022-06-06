import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import he_normal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import batch_normalization
from keras.utils.data_utils import get_file
import tensorflow as tf

from constants.credentials import credentials as CR

num_classes  = len(CR.CATEGORIES)
batch_size   = 10
num_epochs   = 10
img_size = CR.IMGSIZE

model = keras.Sequential(
    [
        keras.layers.Input(shape= (img_size,img_size,1)),                                          
        keras.layers.Conv2D(32, 3, padding = "valid", activation = "relu"),             
        keras.layers.MaxPooling2D(pool_size = (2,2)),                                  
        keras.layers.Conv2D(64, 3, activation = "relu"),                                                            
        keras.layers.MaxPooling2D(pool_size = (2,2)),                                  
        keras.layers.Conv2D(128, 3, activation = "relu"),                             
        keras.layers.Flatten(),                                                        
        keras.layers.Dense(64, activation = "relu"),                                   
        keras.layers.Dense(10),                                                        
    
    ]
)

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
  CR.FOLDR[1],
  labels = 'inferred',
  label_mode = 'int',
  class_names = CR.CATEGORIES,
  color_mode = 'grayscale',
  batch_size = batch_size,
  image_size = (CR.IMGSIZE,CR.IMGSIZE),
  shuffle = True,
  seed = 1,
  validation_split = 0.1,
  subset = "training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
  CR.FOLDR[1],
  labels = 'inferred',
  label_mode = 'int',
  class_names = CR.CATEGORIES,
  color_mode = 'grayscale',
  batch_size = batch_size,
  image_size = (CR.IMGSIZE,CR.IMGSIZE),
  shuffle = True,
  seed = 1,
  validation_split = 0.1,
  subset = "validation",
)

def augment(x ,y) :
  image = tf.image.random_brightness(x, max_delta = 0.05)
  return image,y

ds_train = ds_train.map(augment)

  
model.compile(
  optimizer='adam',
  loss=[
    keras.losses.SparseCategoricalCrossentropy(from_logits= True)
    ],
  metrics=["accuracy"]
)

model.fit(ds_train, epochs=num_epochs)