import numpy as np
import cv2
import pickle
import os

INIT_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_DIR = os.path.join(INIT_DIR,
                           "data/haarcascade_frontalface_default.xml")

face = cv2.CascadeClassifier(CASCADE_DIR)


def detectar():
    # Crea el recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('model.yml')
    # Abrir las labels
    labels = {}
    with open("labels.pickle", 'rb') as f:
        temp_labels = pickle.load(f)
        labels = {v: k for k, v in temp_labels.items()}

    # Abre la camara
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise IOError("No se reconoce o no se puede abrir la camara")

    while (True):

        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            # print(x, y, w, h)

            # Para guardar la imagen
            img_gray = gray[y:y + h, x:x + h]
            # color = frame[y:y + h, x:x + h]

            id_, conf = recognizer.predict(img_gray)
            if conf >= 45 and conf <= 85:
                print(labels[id_])

            # Para el rectángulo al detectar el rostro
            rec_color = (255, 0, 0)
            stroke = 3
            cv2.rectangle(frame, (x, y), (x + w, y + h), rec_color, stroke)
            cv2.putText(frame, labels[id_], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()