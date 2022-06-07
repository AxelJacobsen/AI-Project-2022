import os

import numpy as np
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import keras
from keras.layers import Dropout
from keras.models import Model
import tensorflow as tf
from constants.credentials import credentials as CR
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

num_classes  = len(CR.CATEGORIES)
batch_size   = 64
num_epochs   = 20
img_size = CR.IMGSIZE


ds_train = tf.keras.preprocessing.image_dataset_from_directory(
  CR.DIR,
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
  CR.DIR,
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

# Store the fully connected layers
fc1 = model_resnet.layers[-3]
fc2 = model_resnet.layers[-2]
predictions = model_resnet.layers[-1]

# Create the dropout layers
dropout1 = Dropout(0.6)
dropout2 = Dropout(0.6)

# Reconnect the layers
x = dropout1(fc1.output)
x = fc2(x)
x = dropout2(x)
predictors = predictions(x)

# Create a new model
resNet_model = Model(inputs=model_resnet.input, outputs=predictors)

model_vgg19 = tf.keras.applications.VGG19(
    include_top=True,
    weights = None,
    input_shape=(img_size, img_size, 1),
    # pooling='avg',
    classes=num_classes,
    classifier_activation="softmax",
)

# Set eval model to use specific model: model_resnet, model_vgg19
train_model = model_vgg19    ##resNet_model

train_model.compile(
  optimizer=tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.01, nesterov=False, name="SGD"
),
  loss=[
    keras.losses.SparseCategoricalCrossentropy(from_logits= False)
    ],
  metrics=["accuracy"]
)


predictions = np.array([])
labels =  np.array([])

for x, y in ds_train:
  predictions = np.concatenate([predictions, np.argmax(train_model.predict(x), axis = -1)])
  labels = np.concatenate([labels, y.numpy()])

history = train_model.fit(ds_train, validation_data = ds_validation, epochs=num_epochs, batch_size = batch_size)
train_model.save('savedVGG19.h5')

# Plot accuracy per iteration
plt.plot(history.history['accuracy'], label='accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='val_accuracy', color='green')
plt.legend(['accuracy','val_accuracy'])
plt.title('Accuracy History')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# Plot loss per iteration
plt.plot(history.history['loss'], label='loss', color='red')
plt.plot(history.history['val_loss'], label='val_loss', color='orange')
plt.legend(['loss','val_loss'])
plt.title('Loss History')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

print("f1-score: ",f1_score(labels, predictions, average='weighted'))
print("accuracy-score: ",accuracy_score(labels, predictions))
print("Confusion matrix:\n",tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())

(loss, accuracy) = train_model.evaluate(ds_train, batch_size=128, verbose=1)
print('accuracy: {:.2f}%'.format(accuracy * 100)) 
