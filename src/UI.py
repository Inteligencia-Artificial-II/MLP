from tkinter import Frame, Label, Button, DISABLED, CENTER, NO, ttk, Spinbox, IntVar, Toplevel, BooleanVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sys import exit
import numpy as np

def render_table(self):
    if self.mlp != None:
        self.table = ttk.Treeview(self.table_window)
        Label(self.table_window, text="Capa final", bg="white", font=("Arial", 15)).grid(row=0, column=0, columnspan=len(self.mlp.W_outputs[0]), sticky="we")
        get_table(self.table, self.mlp.W_outputs)
        self.table.grid(row=1, column=0)
        if self.mlp.hidden_layers == 1:
            self.table1 = ttk.Treeview(self.table_window)
            Label(self.table_window, text="Primer capa oculta", bg="white", font=("Arial", 15)).grid(row=2, column=0, columnspan=len(self.mlp.W_hiddens[0,0]), sticky="we")
            get_table(self.table1, self.mlp.W_hiddens[0])
            self.table1.grid(row=3, column=0)
    self.table_window.protocol('WM_DELETE_WINDOW', self.set_window_to_none)

def get_table(table, W):
    table['columns'] = [f'W{i}' for i in range(len(W[0]))]
    table.column("#0", width=0,  stretch=NO)
    table.heading("#0", text="",anchor=CENTER)
    for i in table['columns']:
        table.column(i, anchor=CENTER, width=120)
        table.heading(i, text=i, anchor=CENTER)

    for i in range(len(W)):
        values = [j for j in W[i]]
        table.insert(parent='',index='end',iid=i,text='',
        values=values)

def render_main_window(self):
    """Definimos la interfaz gráfica de usuario"""
    self.window.title('Multilayer Perceptron')
    self.window.configure(bg='white')

    Label(self.window, text="MLP", bg="white", font=("Arial", 20)).grid(row=0, column=0, columnspan=8, sticky="we")
    # añade el gráfico de matplotlib a la interfaz de tkinter
    FigureCanvasTkAgg(self.fig, self.window).get_tk_widget().grid(row=1, rowspan=4, column=0, columnspan=4, sticky="we")
    # Añade la gráfica de convergencia del error cuadrático medio
    FigureCanvasTkAgg(self.fig2, self.window).get_tk_widget().grid(row=1, rowspan=4, column=4, columnspan=4, sticky="we")

    # contendrá un segmento de la interfaz donde se establecerán los parametros de inicio
    self.params_container = Frame(self.window, bg="white", padx=40)

    # Nos permitirá elegir la clase a dibujar en el plano
    Label(self.params_container, bg="white", text="Clase: ").grid(row=0, column=1, columnspan=2, sticky="e")
    self.default_class = IntVar(self.params_container)
    self.input_class = Spinbox(self.params_container, state="readonly", textvariable=self.default_class, from_=0, to=100)

    # contendrá la columna izquierda de parámetros
    self.left_container = Frame(self.params_container, bg="white", padx=40, pady=20)

    # Tasa de aprendizaje, Epocas máximas y Error mínimo deseado
    Label(self.left_container, bg="white", text="Tasa de aprendizaje: ").grid(row=0, column=0, sticky="e")
    self.learning_rate = ttk.Combobox(self.left_container)
    self.learning_rate["values"] = list(np.arange(0.1, 1, 0.1).round(2))
    self.learning_rate.set(self.default_lr)

    Label(self.left_container, bg="white", text="Epocas máximas: ").grid(row=1, column=0, sticky="e")
    self.max_epoch= ttk.Combobox(self.left_container)
    self.max_epoch["values"] = list(range(0, 1000))
    self.max_epoch.set(self.default_epoch)

    Label(self.left_container, bg="white", text="Error mínimo deseado: ").grid(row=2, column=0, sticky="e")
    self.min_error = ttk.Combobox(self.left_container)
    self.min_error["values"] = list(np.arange(0, 1, 0.01).round(3))
    self.min_error.set(self.default_min_error)

    # contendrá la columna derecha de parámetros
    self.center_container = Frame(self.params_container, bg="white", padx=40, pady=20)

    # arquitectura de la red
    Label(self.center_container, bg="white", text="Número de capas ocultas: ").grid(row=0, column=2, sticky="e")
    self.layers = ttk.Combobox(self.center_container, state="readonly")
    self.layers.bind('<<ComboboxSelected>>', self.check_layers)
    self.layers["values"] = [1, 2] # solo podemos tener una o dos capas ocultas
    self.layers.set(self.default_layers) # usamos como valor inicial "1"

    Label(self.center_container, bg="white", text="Número de neuronas por capa: ").grid(row=1, column=2, sticky="e")

    self.neurons_container = Frame(self.center_container, bg="white")
    self.neurons1 = ttk.Combobox(self.neurons_container, state="readonly")
    self.neurons1["values"] = list(range(1, 10)) # podemos elegir cualquier número de neuronas
    self.neurons1.set(self.default_neurons) # usamos como valor inicial "1"
    
    self.neurons2 = ttk.Combobox(self.neurons_container, state="readonly")
    self.neurons2["values"] = list(range(1, 10)) # podemos elegir cualquier número de neuronas
    self.neurons2.set(self.default_neurons) # usamos como valor inicial "1"

    self.neurons_container.grid(row=1, column=3, sticky="w")
    self.neurons1.grid(row=0, column=0, sticky="w")

    Label(self.center_container, bg="white", text="Algoritmo: ").grid(row=2, column=2, sticky="e")
    self.algorithms = ttk.Combobox(self.center_container, state="readonly")
    self.algorithms["values"] = ["Gradiente estocastico", "Lotes", "Quickprop"]
    self.algorithms.set(self.default_algorithm)


    self.right_container = Frame(self.params_container, bg="white", padx=40, pady=20)
    self.weight_btn = Button(self.right_container, bg="white",text="Inicializar pesos", command=self.init_weights, state=DISABLED)
    self.quickprop_btn = Button(self.right_container, bg="white", text="Inicializar con QP", command=self.QP_initializer, state=DISABLED)
    self.init_err_label = Label(self.right_container, bg="white", text="", font=("Arial", 15))
    self.init_epoch_label = Label(self.right_container, bg="white", text="", font=("Arial", 15))
    self.run_btn = Button(self.right_container, text="Entrenar", command=self.run, state=DISABLED)
    self.checkbox_value = BooleanVar(self.right_container)
    self.checkbox = ttk.Checkbutton(self.right_container, text="Mostar pesos", variable=self.checkbox_value)

    self.params_container.grid(row=5, column=0, columnspan=8, sticky="we")
    self.left_container.grid(row=1, column=0, columnspan=2, sticky="we")
    self.center_container.grid(row=1, column=3, columnspan=2, sticky="we")
    self.right_container.grid(row=1, column=5, sticky="e")
    self.learning_rate.grid(row=0, column=1, sticky="w")
    self.max_epoch.grid(row=1, column=1, sticky="w")
    self.min_error.grid(row=2, column=1, sticky="w")
    self.layers.grid(row=0, column=3, sticky="w")
    self.algorithms.grid(row=2, column=3, sticky="w")
    self.weight_btn.grid(row=0, column=0, sticky="we")
    self.quickprop_btn.grid(row=1, column=1, sticky="we")
    self.init_err_label.grid(row=0, column=2, sticky="we")
    self.init_epoch_label.grid(row=1, column=2, sticky="we")
    self.run_btn.grid(row=1, column=0, sticky="we")
    self.checkbox.grid(row=2, column=0, sticky="we")
    self.input_class.grid(row=0, column=3, sticky="w")

    # contenedor de la interfaz después de entrenar
    self.init_conv_text = Label(self.window, text="", bg="white", font=("Arial", 14))
    self.converged_text = Label(self.window, text="", bg="white", font=("Arial", 14))
    self.reset_container = Frame(self.window, bg="white", padx=20, pady=15)
    self.evaluate_btn = Button(self.reset_container, text="Evaluar", command=self.evaluate).grid(row=1, column=0, padx=15)
    self.reset_btn = Button(self.reset_container, text="Reiniciar", command=self.restart).grid(row=1, column=1, padx=15)
    self.reset_container.grid(row=6, column=0, columnspan=8)

    self.reset_container.grid_remove()
    # escucha los eventos del mouse sobre el gráfico
    self.fig.canvas.mpl_connect('button_press_event', self.set_point)

    # termina el programa al hacer click en la X roja de la ventana
    self.window.protocol('WM_DELETE_WINDOW', lambda: exit())
