import cv2
import os
from pathlib import Path

faceCascade = cv2.CascadeClassifier(
    'C:/Users/carlo/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
)
vc = cv2.VideoCapture(0)


def saveImage(image, userName, userId, imgId):
    Path("dataset/{}".format(userName)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite("dataset/{}/{}_{}.jpg".format(userName, userId, imgId), image)
    print("Imagen {} se salvo en el dataset de: {}".format(imgId, userName))


def initiateCamera(userName, userId, count):
    print("Espere mientras se inicia la camara")

    while True:
        _, img = vc.read()
        original_img = img.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray_img,
                                             scaleFactor=1.2,
                                             minNeighbors=5,
                                             minSize=(50, 50))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Creación del dataset", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if count <= 10:
                saveImage(original_img, userName, userId, count)
                count += 1
            else:
                break
        elif key == ord('q'):
            break

    print("Se ha creado el dataset {}".format(userName))

    vc.release()
    cv2.destroyAllWindows()


def main():
    print("Ingrese un id: ")
    userId = input()
    print("Ingrese el username de la persona: ")
    userName = input()
    count = 1
    initiateCamera(userName, userId, count)


main()
