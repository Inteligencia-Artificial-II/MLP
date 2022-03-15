from tkinter import Tk, Frame, Label, Entry, Button, DISABLED, Toplevel, ttk, CENTER, NO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sys import exit
import numpy as np

def render_error_plot(self):
    """Definimos la ventana donde se imprime la gráfica de convergencia"""
    pass

def render_confusion_matrix(self):
    """Definimos la ventana donde se imprime la gráfica de convergencia"""
    pass
    
def render_main_window(self):
    """Definimos la interfaz gráfica de usuario"""
    self.window.title('Multilayer Perceptron')

    Label(self.window, text="MLP", bg="white", font=("Arial", 20)).grid(row=0, column=0, columnspan=6, sticky="we")
    # añade el gráfico de matplotlib a la interfaz de tkinter
    FigureCanvasTkAgg(self.fig, self.window).get_tk_widget().grid(row=1, column=0, columnspan=6, sticky="we")

    # contendrá un segmento de la interfaz donde se establecerán los parametros de inicio
    self.params_container = Frame(self.window, bg="white", padx=20)
    
    # Nos permitirá elegir la clase a dibujar en el plano
    Label(self.params_container, bg="white", text="Clase: ").grid(row=0, column=1, columnspan=2, sticky="e")
    self.input_class = ttk.Combobox(self.params_container, state="readonly")
    self.input_class["values"] = list(range(0, 100))
    self.input_class.set(0)

    # contendrá la columna izquierda de parámetros
    self.left_container = Frame(self.params_container, bg="white", padx=10, pady=20)

    # Tasa de aprendizaje, Epocas máximas y Error mínimo deseado
    Label(self.left_container, bg="white", text="Tasa de aprendizaje: ").grid(row=0, column=0, sticky="e")
    self.learning_rate = ttk.Combobox(self.left_container)
    self.learning_rate["values"] = list(np.arange(0.1, 1, 0.1).round(2))
    self.learning_rate.set(0.1)

    Label(self.left_container, bg="white", text="Epocas máximas: ").grid(row=1, column=0, sticky="e")
    self.max_epoch= ttk.Combobox(self.left_container)
    self.max_epoch["values"] = list(range(0, 1000))
    self.max_epoch.set(self.default_epoch)

    Label(self.left_container, bg="white", text="Error mínimo deseado: ").grid(row=2, column=0, sticky="e")
    self.min_error = ttk.Combobox(self.left_container)
    self.min_error["values"] = list(np.arange(0, 1, 0.01).round(3))
    self.min_error.set(0.1)

    # contendrá la columna derecha de parámetros
    self.center_container = Frame(self.params_container, bg="white", padx=10, pady=20)

    # arquitectura de la red
    Label(self.center_container, bg="white", text="Número de capas ocultas: ").grid(row=0, column=2, sticky="e")
    self.layers = ttk.Combobox(self.center_container, state="readonly")
    self.layers["values"] = [1, 2] # solo podemos tener una o dos capas ocultas
    self.layers.set(1) # usamos como valor inicial "1"

    Label(self.center_container, bg="white", text="Número de neuronas por capa: ").grid(row=1, column=2, sticky="e")
    self.neurons = ttk.Combobox(self.center_container, state="readonly")
    self.neurons["values"] = list(range(1, 15)) # podemos elegir cualquier número de neuronas
    self.neurons.set(1) # usamos como valor inicial "1"

    self.right_container = Frame(self.params_container, bg="white", padx=10, pady=20)
    self.weight_btn = Button(self.right_container, bg="white",text="Inicializar pesos", command=self.init_weights, state=DISABLED)
    self.run_btn = Button(self.right_container, text="Entrenar", command=self.run, state=DISABLED)

    self.params_container.grid(row=2, column=0, columnspan=6, sticky="we")
    self.left_container.grid(row=1, column=0, columnspan=2, sticky="we")
    self.center_container.grid(row=1, column=3, columnspan=2, sticky="we")
    self.right_container.grid(row=1, column=5, sticky="nswe")
    self.learning_rate.grid(row=0, column=1, sticky="w")
    self.max_epoch.grid(row=1, column=1, sticky="w")
    self.min_error.grid(row=2, column=1, sticky="w")
    self.layers.grid(row=0, column=3, sticky="w")
    self.neurons.grid(row=1, column=3, sticky="w")
    self.weight_btn.grid(row=0, column=0, sticky="we")
    self.run_btn.grid(row=1, column=0, sticky="we")
    self.input_class.grid(row=0, column=3, sticky="w")

    # contenedor de la interfaz después de entrenar
    self.reset_container = Frame(self.window, bg="white", padx=260, pady=20)
    self.evaluate_btn = Button(self.reset_container, text="Evaluar", command=self.evaluate)
    self.reset_btn = Button(self.reset_container, text="Reiniciar", command=self.restart)
    self.evaluate_btn.grid(row=0, column=3, padx=15)
    self.reset_btn.grid(row=0, column=4, padx=15)
    # escucha los eventos del mouse sobre el gráfico
    self.fig.canvas.mpl_connect('button_press_event', self.set_point)

    # termina el programa al hacer click en la X roja de la ventana
    self.window.protocol('WM_DELETE_WINDOW', lambda: exit())
