 # Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Convolution2D

from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.layers import Dense
import os
os.chdir('F:\\DataSets\\P14-Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset')

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('F:\\DataSets\\P14-Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('F:\\DataSets\\P14-Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)



os.getcwd()

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('F:\\DataSets\\P14-Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset\\training_set\\nnn\\xyz.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
