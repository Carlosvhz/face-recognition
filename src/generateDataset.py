import cv2
import os
from pathlib import Path

INIT_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_DIR = os.path.join(INIT_DIR,
                           "data/haarcascade_frontalface_default.xml")

SAVE_PATH_DIR = os.path.join(INIT_DIR, "dataset/{}")
SAVE_IMG_DIR = os.path.join(INIT_DIR, "dataset/{}/{}_{}.jpg")


def saveImage(image, userName, userId, imgId):
    Path(SAVE_PATH_DIR.format(userName)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(SAVE_IMG_DIR.format(userName, userId, imgId), image)
    print("Imagen {} se salvo en el dataset de: {}".format(imgId, userName))


def initiateCamera(userName, userId, count):
    faceCascade = cv2.CascadeClassifier(CASCADE_DIR)
    vc = cv2.VideoCapture(0)
    print("Espere mientras se inicia la camara")

    while True:
        _, img = vc.read()
        original_img = img.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray_img,
                                             scaleFactor=1.2,
                                             minNeighbors=5,
                                             minSize=(50, 50))

        cv2.putText(img, "S: para salvar imagen", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(img, "Q: Cerrar el creador de dataset", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Creación del dataset", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if count <= 10:
                saveImage(original_img, userName, userId, count)
                cv2.putText(img, str(count) + "/ 10", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                count += 1
            else:
                break
        elif key == ord('q'):
            break

    print("Se ha creado el dataset {}".format(userName))

    vc.release()
    cv2.destroyAllWindows()
