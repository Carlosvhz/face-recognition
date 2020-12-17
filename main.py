import tkinter as tk
from tkinter import simpledialog
from src import generateDataset, trainer, recognizer


def windowsgeneratedataset():
    USER_INP = simpledialog.askstring(title="Generar dataset",
                                      prompt="ingrese el nombre del usuario:")
    generateDataset.initiateCamera(USER_INP, 1, 1)


def windowtraindataset():
    print("llamar el train")
    trainer.train()


def windowrecognizedataset():
    print("llamar el recognizer")
    recognizer.detectar()


def main():
    window = tk.Tk()
    frame_a = tk.Frame()
    frame_b = tk.Frame()
    frame_c = tk.Frame()

    button_gendataset = tk.Button(frame_a,
                                  text="Generar dataset",
                                  width=25,
                                  height=5,
                                  bg="blue",
                                  fg="yellow",
                                  command=windowsgeneratedataset)
    button_gendataset.pack()

    button_train = tk.Button(frame_b,
                             text="Entrenar el dataset",
                             width=25,
                             height=5,
                             bg="blue",
                             fg="yellow",
                             command=windowtraindataset)
    button_train.pack()

    button_recognize = tk.Button(frame_c,
                                 text="Reconocer",
                                 width=25,
                                 height=5,
                                 bg="blue",
                                 fg="yellow",
                                 command=windowrecognizedataset)
    button_recognize.pack()

    frame_a.pack()
    frame_b.pack()
    frame_c.pack()

    window.mainloop()


main()