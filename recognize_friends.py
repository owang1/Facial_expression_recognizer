# Program that takes precomputing model and outputs the classification report on friends faces (stored in images/friends folder)
# To run: python recognize_friends.py --testing images/friends

import cv2
import os
import numpy as np
from keras.models import model_from_json
from sklearn import preprocessing
from sklearn.metrics import classification_report

from imutils import paths
import argparse

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--testing", required=True,
	help="path to the testing images")
args = vars(ap.parse_args())

def load_model():
    # Load json file of model
    json_file = open('model_full.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights("model_full.h5")
    print("Loaded model from disk")
    return model

def recognize_expression(path):
    test = []
    image = cv2.imread(path)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test.append(image)
    test = np.asarray(test)
    test = test.reshape(test.shape[0], 48, 48, 1)
    test = test.astype('float32')
    test /= 255
    print(prediction)
    return (int(np.argmax(prediction)), round(max(prediction[0])*100, 2))


model = load_model()
print(model)

x_test = []
y_test = []
for imagePath in paths.list_images(args["testing"]):
    image = cv2.imread(imagePath)
    print(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (16, 16),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cropped = gray[ y: y+h, x : x+w ]
    resized_img = cv2.resize(cropped, (48, 48))
    
    # cv2.imshow("Preprocessed image", resized_img)
    # cv2.waitKey(0)
    resized_img = np.asarray(resized_img).reshape((48, 48, 1))
    y_test.append(imagePath.split(os.path.sep)[-2])
    x_test.append(resized_img)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

x_test = x_test.astype('float32')
x_test /= 255.0


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Encode y labels as numerical values
le = preprocessing.LabelEncoder()
y_test_num = le.fit_transform(y_test)
print(y_test_num)

# That's all! Let's see now how our model recognizes all test faces
print('-TESTING-----------------------------')
print('Number of test images:', x_test.shape[0])
score = model.evaluate(x_test, y_test_num)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Do it for each classes

y_pred = model.predict_classes(x_test)
print(classification_report(y_test_num, y_pred))