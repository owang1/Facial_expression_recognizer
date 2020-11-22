# Olivia Wang Fall 2020
# To run: python webcam.py
# Press spacebar to get the predicted expression
import cv2
import os
import numpy as np
from keras.models import model_from_json

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

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
    prediction = model.predict(test)
    return (int(np.argmax(prediction)), round(max(prediction[0])*100, 2))


model = load_model()
print(model)
emotion_dict = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness", 4: "neutral", 5: "sadness", 6: "surprise"}

while (True):
    retval, img = cam.read()
    res_scale = .5             # rescale the input image if it's too large
    img = cv2.resize(img, (0,0), fx=res_scale, fy=res_scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (16, 16),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cropped = gray[ y: y+h, x : x+w ]
    resized_img = cv2.resize(cropped, (48, 48))

    cv2.imshow("Face detector", img)
    # cv2.imshow("Gray", resized_img)
    if cv2.waitKey(1) == 32:		# Hit the space key to save image
        cap_image = resized_img
        cv2.imwrite("test.jpg", cap_image)
        path = "test.jpg"
        emotion_index, confidence = recognize_expression(path)
        prediction = emotion_dict[emotion_index]
        cap_image_resized = cv2.resize(cap_image, (400, 400))
        cv2.putText(cap_image_resized, prediction, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), lineType=cv2.LINE_AA)
        cv2.imshow("Captured Image", cap_image_resized) 
        print("Prediction: " + prediction)
        print("Confidence Level: " + str(confidence) + "%")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()