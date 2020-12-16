# Para hacer la red neuronal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_yaml

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from PIL import Image

x_train = []  # Data de entrenamiento
y_train = []  # Y de entrenamiento
labelId = {}  # ID de los labels
currentLabel = 0

face_classifier = cv2.CascadeClassifier(
    'C:/Users/carlo/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
)
INIT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(INIT_DIR, "datasets/training")

# Modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3),
                           activation='relu',
                           input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

for root, dirs, files in os.walk(IMG_DIR):
    for file in files:
        if file.endswith("PNG") or file.endswith("png") or file.endswith(
                "jpg") or file.endswith("JPG"):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            # label = os.path.basename(root).replace(" ", "-").lower()

            # Guarda los labels en un arreglo
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

print(x_train)
#model_fit = model.fit(x_train, np.array(y_train), steps_per_epoch=3, epochs=30)

#model_yaml = model.to_yaml()

#with open("model.yaml", "w") as f:
#   f.write(model_yaml)

#model.save_weights("weights.h5")