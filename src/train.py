import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from PIL import Image

x_train = []
y_train = []
labelId = {}
currentLabel = 0

face_classifier = cv2.CascadeClassifier(
    'C:/Users/carlo/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

INIT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(INIT_DIR, "dataset")

for root, dirs, files in os.walk(IMG_DIR):
    for file in files:
        if file.endswith("PNG") or file.endswith("png") or file.endswith(
                "jpg") or file.endswith("JPG"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            if label in labelId:
                pass
            else:
                labelId[label] = currentLabel
                currentLabel += 1
            id_ = labelId[label]

            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            faces = face_classifier.detectMultiScale(image_array,
                                                     scaleFactor=1.5,
                                                     minNeighbors=5)

            for (x, y, w, h) in faces:
                rol = image_array[y:y + h, x:x + w]
                x_train.append(rol)
                y_train.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(labelId, f)

recognizer.train(x_train, np.array(y_train))
recognizer.save("model.yml")