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
        self.fig2 = plt.figure(2)
        self.ax = self.fig.add_subplot(111)
        self.ax2 = self.fig2.add_subplot(111)
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

        # Error acumulado
        self.acum_error = []
        self.default_epoch = 250
        self.default_min_error = 0.1
        self.default_lr = 0.3
        self.default_layers = 1
        self.default_neurons = 4

        # inicializamos la ventana principal
        self.window = Tk()
        render_main_window(self)
        self.window.mainloop()

    def set_point(self, event):
        if event.xdata != None or event.ydata != None:
            if self.is_training:
                self.X.append((event.xdata, event.ydata))
                self.Y.append(int(self.input_class.get()))
                self.plot_point((event.xdata, event.ydata), self.input_class.get())
                self.fig.canvas.draw()
            else:
                self.test_data.append((event.xdata, event.ydata))
                self.plot_point((event.xdata, event.ydata))
                self.fig.canvas.draw()

        self.outputs_class = len(np.unique(self.Y))
        if (self.outputs_class > 2):
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

    def clear_plot(self, fig, fig_index = 1):
        """Borra los puntos del canvas"""
        plt.figure(fig_index)
        plt.cla()
        if fig_index == 1:
            self.ax.set_xlim([-5, 5])
            self.ax.set_ylim([-5, 5])
        fig.canvas.draw()

    def plot_mse(self, x: list):
        """Imprime la gráfica de convergencia del error cuadrático medio"""
        plt.figure(2)
        self.clear_plot(self.fig2, 2)
        plt.plot(x)
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()
        plt.figure(1)

    def run(self):
        """es ejecutada cuando el botón de «entrenar» es presionado"""
        # entrenamos la red con los datos ingresados
        self.mlp.lr = float(self.learning_rate.get())
        self.mlp.error_figure = self.fig2
        self.mlp.plot_mse = self.plot_mse
        self.mlp.train(self.X,
                self.Y,
                int(self.max_epoch.get()),
                float(self.min_error.get()))
        # establecemos el modo de evaluación
        self.is_training = not self.is_training
        self.params_container.grid_remove()
        self.reset_container.grid(row=2, column=0, columnspan=6, sticky="we")

    def evaluate(self):
        print("evaluando")
        for i in self.test_data:
            res = self.mlp.feed_forward(i)
            print(res)

    def restart(self):
        """devuelve los valores y elementos gráficos a su estado inicial"""
        self.clear_plot(self.fig2, 2)
        plt.figure(1)
        self.clear_plot(self.fig)
        self.is_training = not self.is_training
        self.test_data.clear()
        self.input_class.set(0)
        self.learning_rate.set(self.default_lr)
        self.max_epoch.set(self.default_epoch)
        self.min_error.set(self.default_min_error)
        self.layers.set(self.default_layers)
        self.neurons.set(self.default_neurons)
        self.reset_container.grid_remove()
        self.params_container.grid(row=2, column=0, columnspan=6, sticky="we")
