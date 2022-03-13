import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from tkinter import NORMAL, DISABLED, messagebox, Tk
from src.UI import render_main_window
from src.MLP import MLP

class Plotter:
    def __init__(self):
        self.ax_max = 5
        self.ax_min = -5
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(111)
        # establecemos los limites de la gráfica
        self.ax.set_xlim([self.ax_min, self.ax_max])
        self.ax.set_ylim([self.ax_min, self.ax_max])

        # definimos la que será una instancia de la clase mlp
        self.mlp = None

        self.Y = [] # guarda los clusters
        self.X = [] # guarda los puntos de entrenamiento
        self.test_data = [] # guarda los puntos para evaluar

        # bandera para evaluar datos despues del entrenamiento
        self.is_training = True

        # Parámetros para el algoritmo
        self.epochs = 0 # epocas o generaciones máximas
        self.lr = 0.0 # tasa de aprendizaje

        # Error acumulado
        self.acum_error = []
        self.default_epoch = 25

        # inicializamos la ventana principal
        self.window = Tk()
        render_main_window(self)
        self.window.mainloop()

    def set_point(self, event):
        print(event.xdata, event.ydata)

    def init_weights(self):
        if self.mlp != None:
            self.mlp.randomize_weights()
    def run(self):
        pass
    
    def plot_point(self, point: tuple, cluster=None):
        """Toma un array de tuplas y las añade los puntos en la figura con el
        color de su cluster"""
        plt.figure(1)
        if (cluster == None):
            plt.plot(point[0], point[1], 'o', color='k')
        else:
            color = 'b' if cluster == 1 else 'r'
            shape = 'o' if cluster == 1 else 'x'
            plt.plot(point[0], point[1], shape, color=color)

    def plot_training_data(self):
        """Grafica los datos de entrenamiento"""
        plt.figure(1)
        for i in range(len(self.Y)):
            self.plot_point(self.X[i], self.Y[i])

    def clear_plot(self, figure = 1):
        """Borra los puntos del canvas"""
        plt.figure(figure)
        plt.cla()
        if figure == 1:
            self.ax.set_xlim([-5, 5])
            self.ax.set_ylim([-5, 5])
        self.fig.canvas.draw()

    def run(self):
        """es ejecutada cuando el botón de «entrenar» es presionado"""
        pass

    def restart(self):
        """devuelve los valores y elementos gráficos a su estado inicial"""
        pass