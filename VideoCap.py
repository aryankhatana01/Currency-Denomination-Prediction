# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
import numpy as np
import imutils
import time
import cv2
import os
import pyttsx3

frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.95   # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# FOR SPEAKING OUT LOUD
engine = pyttsx3.init('sapi5')
voices= engine.getProperty('voices') #getting details of current voice
engine.setProperty('voice', voices[0].id)

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)


model = tf.keras.models.load_model('keras_model11.h5')

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = (img.astype(np.float32) / 127.0) - 1    #NORMALIZATION
    return img

def speak(message):
    engine.say(message)
    engine.runAndWait()

classes = ["10", "20", "50", "100", "200", "500", "2000"]

while True:

    # READ IMAGE
    _, imgOrignal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (224, 224))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 224, 224, 3)
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    result = np.argmax(predictions)
    classIndex = classes[result]
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        #print(getCalssName(classIndex))
        cv2.putText(imgOrignal,classIndex, (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        speak(classIndex + 'rupees')
    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
