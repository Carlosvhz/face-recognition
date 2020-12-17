import tkinter as tk
from tkinter import messagebox


class datasetWindow:

    window = tk.Tk()

    def show(self):
        self.window.title("Generar dataset")
        self.window.geometry("1200x600")
        self.window.resizable(False, False)
        self.window.mainloop()