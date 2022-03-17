import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.cm import get_cmap
from tkinter import NORMAL, DISABLED, messagebox, Tk
from src.UI import render_main_window
from src.MLP import MLP
import numpy as np

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
        self.inputs = None

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
        if event.xdata != None or event.ydata != None:
            self.X.append((event.xdata, event.ydata))
            self.Y.append(int(self.input_class.get()))
            self.plot_point((event.xdata, event.ydata), self.input_class.get())
            self.fig.canvas.draw()

        self.outputs_class = len(np.unique(self.Y))
        if (self.outputs_class > 1):
            self.weight_btn["state"] = NORMAL

    def mlp_can_randomize(self):
        if self.mlp == None:
            return False

        return not (self.mlp.output_neurons != self.outputs_class
               or self.mlp.hidden_layers != int(self.layers.get())-1
               or self.mlp.hidden_neurons != int(self.neurons.get()))

    def init_weights(self):
        if not self.mlp_can_randomize():
            self.mlp = MLP(len(self.X[0]),
                           int(self.layers.get()),
                           int(self.neurons.get()),
                           self.outputs_class)

        self.mlp.randomize_weights()
        self.mlp.print_weights()
        self.run_btn["state"] = NORMAL

    def plot_point(self, point: tuple, cluster=None):
        """Toma un array de tuplas y las añade los puntos en la figura con el
        color de su cluster"""
        plt.figure(1)
        if (cluster == None):
            plt.plot(point[0], point[1], 'o', color='k')
        else:
            cmap = get_cmap('flag')
            color = cmap(float(int(cluster)/100))
            plt.plot(point[0], point[1], 'o', color=color)

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
        self.mlp.train(self.X, self.Y, int(self.max_epoch.get()), float(self.min_error.get()))
        self.params_container.grid_remove()
        self.reset_container.grid(row=2, column=0, columnspan=6, sticky="we")

    def evaluate(self):
        print("evaluando")

    def restart(self):
        """devuelve los valores y elementos gráficos a su estado inicial"""
        self.clear_plot()
        self.input_class.set(0)
        self.learning_rate.set(0.1)
        self.max_epoch.set(self.default_epoch)
        self.min_error.set(0.1)
        self.layers.set(1)
        self.neurons.set(1)
        self.outputs.set(1)
        self.reset_container.grid_remove()
        self.params_container.grid(row=2, column=0, columnspan=6, sticky="we")
