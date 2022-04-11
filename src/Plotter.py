import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from tkinter import NORMAL, DISABLED, Tk, Toplevel
from src.UI import render_main_window, render_table
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
        self.ax.set_facecolor("#dedede")
        self.ax2.set_facecolor("#dedede")

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
        self.default_algorithm = "Gradiente estocastico"

        # Línea de puntos para obtener x2 al imprimir pesos
        self.x1_density = 100
        self.x1_line = np.linspace(self.ax_min, self.ax_max, self.x1_density)

        # inicializamos la ventana principal
        self.window = Tk()
        render_main_window(self)
        self.table_window = None
        self.window.mainloop()

    def set_window_to_none(self):
        self.table_window.destroy()
        self.table_window = None

    def set_point(self, event):
        """Añade las coordenadas tanto de los puntos de entrenamiento como los de prueba"""
        # validación para no ingresar clases salteadas
        if int(self.input_class.get()) != 0:
            last_class = int(self.input_class.get()) - 1
            if len(np.where(np.array(self.Y)==last_class)[0]) == 0:
                self.default_class.set(last_class)
                return

        if event.xdata != None or event.ydata != None:
            if self.is_training:
                self.X.append((event.xdata, event.ydata))
                self.Y.append(int(self.input_class.get()))
                self.plot_point((event.xdata, event.ydata), None, self.input_class.get())
                self.fig.canvas.draw()
            else:
                self.test_data.append((event.xdata, event.ydata))
                self.plot_point((event.xdata, event.ydata), None, None)
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

        self.mlp.show_weights = self.show_weights
        self.mlp.randomize_weights()
        self.plot_weights(self.mlp.W_inputs)
        self.run_btn["state"] = NORMAL

    def plot_point(self, point: tuple, alpha=None, cluster=None):
        """Toma un array de tuplas y las añade los puntos en la figura con el
        color de su cluster"""
        plt.figure(1)
        cmap = get_cmap('flag')
        if (cluster == None): # clase desconocida
            plt.plot(point[0], point[1], 'o', color='k')
        elif alpha != None: # degradado
            color = cmap(float(int(cluster)/100))
            offset = 0.25
            alpha = (alpha - offset) if (alpha - offset) > 0 else 0
            plt.plot(point[0], point[1], 'o', color=color, markersize=7, markeredgewidth=0, alpha=alpha)
        else: # entrenamiento
            color = cmap(float(int(cluster)/100))
            plt.plot(point[0], point[1], 'o', markeredgecolor='k', markeredgewidth=1.5, color=color)


    def plot_gradient(self):
        """gráfica el degradado de las clases"""
        x = np.linspace(self.ax_min, self.ax_max, 40)
        y = np.linspace(self.ax_min, self.ax_max, 30)

        for i in range(len(x)):
            for j in range(len(y)):
                res = self.mlp.feed_forward((x[i], y[j]))
                cluster = self.mlp.encode_guess(res)
                self.plot_point((x[i], y[j]), res[cluster], cluster)

    def plot_training_data(self):
        """Grafica los datos de entrenamiento"""
        plt.figure(1)
        for i in range(len(self.Y)):
            self.plot_point(self.X[i], None,  self.Y[i])

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

    def plot_weights(self, W, color = 'r'):
        """Imprime las rectas de pesos de la primer capa del MLP"""
        self.clear_plot(self.fig)
        self.plot_training_data()

        m, _ = W.shape
        for i in range(m):
            x2 = (-W[i, 1] * self.x1_line - W[i, 0]) / W[i, -1]
            plt.plot(self.x1_line, x2, color=color)
        self.fig.canvas.draw()

    def show_weights(self):
        if self.checkbox_value.get():
            if self.table_window == None:
                self.table_window = Toplevel(self.window)
            elif self.table_window.state() != 'normal':
                self.table_window = Toplevel(self.window)
            render_table(self)

    def run(self):
        """es ejecutada cuando el botón de «entrenar» es presionado"""
        # entrenamos la red con los datos ingresados
        self.mlp.lr = float(self.learning_rate.get())
        self.mlp.error_figure = self.fig2
        self.mlp.plot_mse = self.plot_mse
        self.mlp.plot_weights = self.plot_weights

        iter = 0
        if (self.algorithms.get() == "Gradiente estocastico"):
            iter, err = self.mlp.train(self.X,
                    self.Y,
                    int(self.max_epoch.get()),
                    float(self.min_error.get()), "Stochastic")
        elif (self.algorithms.get() == "Lotes"):
            iter, err = self.mlp.train(self.X,
                    self.Y,
                    int(self.max_epoch.get()),
                    float(self.min_error.get()), "Batch")
        else:
            iter, err = self.mlp.train(self.X,
                    self.Y,
                    int(self.max_epoch.get()),
                    float(self.min_error.get()), "Quickprop")
            if (self.algorithms.get() == "QP + BP"):
                iter, err = self.mlp.train(self.X,
                    self.Y,
                    int(self.max_epoch.get()) + iter,
                    float(self.min_error.get()), "Stochastic")

        self.mlp.epoch = 0
        self.mlp.mse_list = []

        err = round(err, 4)
        if iter == int(self.max_epoch.get()):
            self.converged_text['text'] = f'Número máximo de epocas alcazada, con un error de {err}'
        else:
            self.converged_text['text'] = f'El set de datos convergió en {iter} epocas, con un error de {err}'
        self.converged_text.grid(row=5, column=0, columnspan=8, sticky="we")
        # establecemos el modo de evaluación
        self.is_training = not self.is_training
        self.params_container.grid_remove()
        self.reset_container.grid(row=6, column=0, columnspan=8)

    def evaluate(self):
        """Obtiene la predicción de las clases correctas de los datos de prueba"""
        self.clear_plot(self.fig)
        self.plot_gradient()
        self.plot_training_data()
        for i in self.test_data:
            res = self.mlp.feed_forward(i)
            self.plot_point(i, None, self.mlp.encode_guess(res))
        self.fig.canvas.draw()

    def restart(self):
        """devuelve los valores y elementos gráficos a su estado inicial"""
        self.init_weights()
        if self.table_window != None:
            self.table_window.destroy()
        self.checkbox_value.set(False)
        self.clear_plot(self.fig2, 2)
        plt.figure(1)
        self.clear_plot(self.fig)
        self.is_training = not self.is_training
        self.test_data.clear()
        self.X.clear()
        self.Y.clear()
        self.default_class.set(0)
        self.learning_rate.set(self.default_lr)
        self.max_epoch.set(self.default_epoch)
        self.min_error.set(self.default_min_error)
        self.layers.set(self.default_layers)
        self.neurons.set(self.default_neurons)
        self.algorithms.set(self.default_algorithm)
        self.reset_container.grid_remove()
        self.run_btn['state'] = DISABLED
        self.weight_btn['state'] = DISABLED
        self.converged_text['text'] = ''
        self.converged_text.grid_remove()
        self.params_container.grid(row=5, column=0, columnspan=8, sticky="we")
