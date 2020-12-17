# Import de ventanas
from datasetWindow import datasetWindow

import tkinter as tk
from tkinter import messagebox

# Creación de ventanas
window_d = datasetWindow()


# Actions
def genDataset():
    window_d.show()


# Muestra la ventana para crear el dataset

window = tk.Tk()
window.title("Proyecto")

# Divs
frame_1 = tk.Frame()
button_genDataset = tk.Button(frame_1,
                              text="Button",
                              width="15",
                              height="5",
                              command=genDataset)
"""
button_train = tk.Button(frame_2,
                         text="Button",
                         width="15",
                         height="5",
                         command=awa)
button_recognize = tk.Button(frame_2,
                             text="Button",
                             width="15",
                             height="5",
                             command=awa)
"""
frame_1.pack()
button_genDataset.pack()

window.geometry("1200x600")
window.resizable(False, False)
window.mainloop()
