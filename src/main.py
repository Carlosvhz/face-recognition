import numpy as np
import cv2
import pickle

face = cv2.CascadeClassifier(
    'C:/Users/carlo/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model.yml')

labels = {}
with open("labels.pickle", 'rb') as f:
    temp_labels = pickle.load(f)
    labels = {v: k for k, v in temp_labels.items()}

camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise IOError("Cannot open webcam")

while (True):

    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x, y, w, h)

        # Para guardar la imagen
        img_gray = gray[y:y + h, x:x + h]
        color = frame[y:y + h, x:x + h]

        id_, conf = recognizer.predict(img_gray)
        if conf >= 45 and conf <= 85:
            print(labels[id_])

        img = "image.png"
        #cv2.imwrite(img, img_gray)

        # Para el rectángulo al detectar el rostro
        rec_color = (255, 0, 0)
        stroke = 3
        cv2.rectangle(frame, (x, y), (x + w, y + h), rec_color, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()