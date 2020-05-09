# -*- coding: utf-8 -*-
# 10 Monkey classification with custom DNN Architecture

"""
Setting Operating System Variable 'KAGGLE_USERNAME' as theroyakash. 
You can download your 'KAGGLE_KEY' from kaggle's accounts settngs.
"""
import os
os.environ['KAGGLE_USERNAME'] = "theroyakash"
os.environ['KAGGLE_KEY'] = "##########CONFIDENTIAL##########"

# Run this command on command line !kaggle datasets download -d slothkong/10-monkey-species


from zipfile import ZipFile

with ZipFile('10-monkey-species.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, activations
from matplotlib import pyplot as plt

# print("Tensorflow version " + tf.__version__)

# try:
#   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
#   print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
# except ValueError:
#   raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
,
training_datadir = '/content/training/training/'
validation_datadir = '/content/validation/validation/'
labels_path = '/content/monkey_labels.txt'

f = open("monkey_labels.txt", "r")
print(f.read())

labels_latin = ['alouatta_palliata', 
          'erythrocebus_patas', 
          'cacajao_calvus', 
          'macaca_fuscata', 
          'cebuella_pygmea', 
          'cebus_capucinus', 
          'mico_argentatus', 
          'saimiri_sciureus', 
          'aotus_nigriceps', 
          'trachypithecus_johnii']

labels_common = ['mantled_howler', 
                 'patas_monkey', 
                 'bald_uakari', 
                 'japanese_macaque',
                 'pygmy_marmoset',
                 'white_headed_capuchin',
                 'silvery_marmoset',
                 'common_squirrel_monkey',
                 'black_headed_night_monkey',
                 'nilgiri_langur']

len(labels_common)

len(labels_latin)

import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = training_datagen.flow_from_directory(
	training_datadir,
	target_size=(200,200),
	class_mode='categorical',
    batch_size = 32
)

validation_datagen = ImageDataGenerator(
    rescale = 1./255
)

validation_generator = validation_datagen.flow_from_directory(
    validation_datadir,
    target_size = (200,200),
    class_mode='categorical',
    batch_size=32
)

# model = tf.keras.models.Sequential([
#     # Note the input shape is the desired size of the image 150x150 with 3 bytes color
#     # This is the first convolution
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(200, 200, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     # The second convolution
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     # The third convolution
#     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     # The fourth convolution
#     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     # Flatten the results to feed into a DNN
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.5),
#     # 512 neuron hidden layer
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.summary()

# # Create a resolver
# # Distribution strategies
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)

# strategy = tf.distribute.experimental.TPUStrategy(resolver)

img_input = Input(shape=(200,200,3))

conv2d_1 = Conv2D(64, (3,3), activation='relu', padding='valid', name='conv2d_1')(img_input)
maxpool1 = MaxPooling2D(pool_size=(2,2))(conv2d_1)

conv2d_2 = Conv2D(128, (3,3), activation='relu', padding='valid', name='conv2d_2')(maxpool1)

conv2d_3 = Conv2D(128, (3,3), activation='relu', padding='valid', name='conv2d_3')(conv2d_2)
maxpool1 = MaxPooling2D(pool_size=(2,2))(conv2d_3)

conv2d_5 = Conv2D(128, (3,3), activation='relu', padding='valid', name='conv2d_5')(maxpool1)

branch0 = Conv2D(64, (1,1), padding='same', name='Branch_Zero_1_by_1_Conv2D')(conv2d_5)

branch1 = Conv2D(64, (1,1), activation='relu', padding='same', name='BranchOne3By3Conv2D1')(conv2d_5)
branch1 = Conv2D(64, (3,3), activation='relu', padding='same', name='BranchOne3By3Conv2D2')(branch1)
branch1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='BranchOne3By3Conv2D3')(branch1)

concatenated_branchA = Concatenate()([branch0, branch1])

pool0 = MaxPooling2D(pool_size=(2, 2))(concatenated_branchA)

branch00 = Conv2D(64, (1,1), padding='same', name='BranchZeroZero1By1Conv2D')(pool0)

branch11 = Conv2D(64, (1,1), activation='relu', padding='same', name='BranchOneOne3By3Conv2D1')(pool0)
branch11 = Conv2D(64, (3,3), activation='relu', padding='same', name='BranchOneOne3By3Conv2D2')(branch11)
branch11 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='BranchOneOne3By3Conv2D3')(branch11)

concatenated_branchB = Concatenate()([branch00, branch11])

flattened_before_dense = Flatten()(concatenated_branchB)
dense1 = Dense(1024, activation='relu', name='firstDenseLayer')(flattened_before_dense)
dense2 = Dense(512, activation='relu', name='SecondDenseLayer')(dense1)
dense3 = Dense(128, activation='relu', name='ThirdDenseLayer')(dense2)

prediction_branch = Dense(10,activation='softmax', name='FinalSoftmaxLayer')(dense3)

model = Model(inputs=img_input, outputs=prediction_branch)
model.summary()

learning_rate, epochs = 0.001, 30

# compile our model
print("compiling model...")
model.compile(loss="categorical_crossentropy", 
              		   optimizer=Adam(lr=learning_rate, decay=learning_rate / epochs), 
			           metrics=["accuracy"])

print("Model Compiled Successfully")
print("[SUMMARY]:")
print(model.summary())

history = model.fit(train_generator, 
                    epochs=epochs, 
                    steps_per_epoch=35, 
                    validation_data = validation_generator, 
                    verbose = 1, 
                    validation_steps=35)

from tensorflow.keras.utils import plot_model
plot_model(model , 'MonkeyNet.png' , show_shapes=True)

model.save('MonkeyNet.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()
