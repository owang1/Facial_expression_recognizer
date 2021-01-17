# Olivia Wang Fall 2020
# Credits to: Adam Czajka, November 2019 mnist_sample.py
# To run: python face_exp_trainer.py --training images/training --testing images/validation

import cv2
import tensorflow as tf
import numpy as np
import itertools as it
import sys

from sklearn import preprocessing
from sklearn.metrics import classification_report
from imutils import paths
from keras.models import model_from_json
from keras.layers import BatchNormalization
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to the training images")
ap.add_argument("-e", "--testing", required=True,
	help="path to the testing images")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(16, 8)


# Redirect print to out.txt file
f = open('out.txt', 'w')
sys.stdout = f

y_train = []
x_train = []
input_shape = (48, 48, 1)

count = 0
# loop over the training images
for imagePath in paths.list_images(args["training"]):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	print(imagePath)
	print(count)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = np.asarray(gray).reshape((48, 48, 1))
	y_train.append(imagePath.split(os.path.sep)[-2])
	x_train.append(gray)
	count += 1
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_test = []
y_test = []
count = 0
for imagePath in paths.list_images(args["testing"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.asarray(gray).reshape((48, 48, 1))
    y_test.append(imagePath.split(os.path.sep)[-2])
    x_test.append(gray)
    count += 1

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255.0
x_test /= 255.0

# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()

model.add(Conv2D(16, kernel_size=(5,5), strides = (1,1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

# You can now have some fun with modifying the layers. For instance add extra convolutional layer with 32 kernels:
model.add(Conv2D(32, kernel_size=(3,3), strides = (1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation='relu')) # the number of 128 neurons selected for this hidden layer is sort of arbitrary

model.add(Dropout(0.2)) # Randomly drop 20% of connections when training (equivalent to an ensamble model learning)
model.add(Dense(7,activation='softmax')) # 7 output neurons since we have 7 classes in this task

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print('-TRAINING----------------------------')
print('Input shape:', x_train.shape)
print('Number of training images: ', x_train.shape[0])

# Encode y labels as numerical values
le = preprocessing.LabelEncoder()
y_train_num = le.fit_transform(y_train)
y_test_num = le.fit_transform(y_test)
print(y_train_num)
print(y_test_num)

# Reducing batch_size from 32 (default) to 10
model.fit(x=x_train,y=y_train_num, epochs=100, batch_size=10)

# That's all! Let's see now how our model recognizes all test faces
print('-TESTING-----------------------------')
print('Number of test images:', x_test.shape[0])
score = model.evaluate(x_test, y_test_num)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Do it for each classes

y_pred = model.predict_classes(x_test)
print(classification_report(y_test_num, y_pred))

# Save the model

model_json = model.to_json()

with open("model_full.json","w") as json_file:
    json_file.write(model_json)
json_file.close()
model.save_weights("model_full.h5")
print("Saved model to disk")
f.close()
