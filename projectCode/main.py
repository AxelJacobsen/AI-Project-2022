import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import keras
from keras.layers import Dropout
from keras.models import Model
import tensorflow as tf
from constants.credentials import credentials as CR

num_classes  = len(CR.CATEGORIES)
batch_size   = 64
num_epochs   = 10
img_size = CR.IMGSIZE




ds_train = tf.keras.preprocessing.image_dataset_from_directory(
  CR.FOLDR,
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
  CR.FOLDR,
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
 


model_resnet = tf.keras.applications.ResNet152(
    include_top=True,
    weights = None,
    input_shape=(img_size, img_size, 1),
    # pooling='avg',
    classes=num_classes,
)

model_vgg19 = tf.keras.applications.VGG19(
    include_top=True,
    weights = None,
    input_shape=(img_size, img_size, 1),
    # pooling='avg',
    classes=num_classes,
    classifier_activation="softmax",
)

# Set eval model to use specific model: model_resnet, model_vgg19
eval_model = model_vgg19

# Store the fully connected layers
fc1 = eval_model.layers[-3]
fc2 = eval_model.layers[-2]
predictions = eval_model.layers[-1]

# Create the dropout layers
dropout1 = Dropout(0.1)
dropout2 = Dropout(0.1)

# Reconnect the layers
x = dropout1(fc1.output)
x = fc2(x)
x = dropout2(x)
predictors = predictions(x)

# Create a new model
train_model = Model(inputs=eval_model.input, outputs=predictors)

train_model.compile(
  optimizer=tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.01, nesterov=False, name="SGD"
),
  loss=[
    keras.losses.SparseCategoricalCrossentropy(from_logits= False)
    ],
  metrics=["accuracy"]
)

history = train_model.fit(ds_train, validation_data = ds_validation, epochs=num_epochs, batch_size = batch_size)

(loss, accuracy) = eval_model.evaluate(ds_train, batch_size=128, verbose=1)
print("accuracy: {:.2f}%".format(accuracy * 100))
