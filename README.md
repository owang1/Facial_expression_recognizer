# Facial_expression_recognizer
Computer vision final project: recognizing faces from 7 different expressions

## Environment Needed
64-bit Anaconda. I used version 4.9.0 and Python 3.8.3
Install the following packages:
- imutils (conda install -c conda-forge imutils)
- scikit-image (conda install scikit-image)
- numpy (conda install numpy)
- opencv (conda install -c conda-forge opencv=3.4.2)
- scikit-learn (conda install -c anaconda scikit-learn)
- h5py (conda install -c anaconda h5py)
- tensorflow (conda install -c conda-forge tensorflow)
- keras (conda install -c conda-forge keras)

## Instructions to Run Programs
- To run the real-time webcam version, use the command "python webcam.py".
- To run the program that recognizes friends' faces, run the command "python recognize_friends.py --testing images/friends". The images folder will have a folder inside called friends, and inside the friends folder make 7 folders corresponding to each of the 7 expressions. The images can be color and of any size.
- Too run the program that trains and validates large datasets, run the command "python face_exp_trainer.py --training images/training --testing images/validation". The structure of these folders is the same as in the previous bullet point; however, the images must be pre-cropped to 48x48 pixels and grayscale. At the end of processing, it will output the classification report showing each expression's accuracy as well as the overall accuracy. This program can take a few hours to run on a full dataset of thousands of images, so I saved the model_full.json and model_full.h5 files for use in webcam.py and recognize_friends.py.

## Overview of Project
This was my semester project for Computer Vision CSE 40535 at Notre Dame. The goal of the project was to recognize a person's expression from 7 different categories: anger, disgust, fear, happiness, neutral, sadness, and surprise.
For a demo of the project, click <a href="https://www.youtube.com/watch?v=ca6z_Kn5nM4">here</a>
