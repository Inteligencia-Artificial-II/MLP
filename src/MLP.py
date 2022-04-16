import numpy as np

class MLP:
    def __init__(self, i_neurons, h_layers, h_neurons, o_neurons):
        # número de neuronas por capa
        self.input_neurons = i_neurons
        self.hidden_neurons = h_neurons
        self.output_neurons = o_neurons

        # número de capas ocultas
        self.hidden_layers = h_layers - 1

        # tasa de aprendizaje
        self.lr = 0.1

        # Función para imprimir el error cuadrático medio
        self.error_figure = None

        # guarda los valores de activación de todas las capas
        self.sigmoids = [0 for _ in range(2 + self.hidden_layers)]

        # guarda las sensibilidades = input layer + output layer + hidden layers
        self.sensitivities = [0 for _ in range(h_layers + 1)]

        # Función del plotter para imprimir el mse
        self.plot_mse = None

        # Función del plotter para imprimir los pesos de la primer capa oculta
        self.plot_weights = None

        self.show_weights = None

        # Error significativo para QP + BP
        self.qp_min_error = 0.1
        self.iter_stuck = 10
        self.best_mse = 0

        # Valor previo del quickprop
        self.gradients_prev = []
        self.is_st_prev_init = False

        # Acumulador del gradiente estocástico para calcular lotes
        self.W_inputs_batch = np.empty((self.hidden_neurons, self.input_neurons + 1))
        # pesos de las capas ocultas menos la última
        self.W_hiddens_batch = np.empty((self.hidden_layers, self.hidden_neurons, self.hidden_neurons + 1))
        # pesos de la última capa oculta y la capa final
        self.W_outputs_batch = np.empty((self.output_neurons, self.hidden_neurons + 1))

        self.gradients = []
        # Número de épocas
        self.epoch = 0
        # Lista de errores cuadráticos medios
        self.mse_list = []
        # matrices de pesos
        # pesos de la entrada a la primer capa oculta
        self.W_inputs = np.empty((self.hidden_neurons, self.input_neurons + 1))
        # pesos de las capas ocultas menos la última
        self.W_hiddens = np.empty((self.hidden_layers, self.hidden_neurons, self.hidden_neurons + 1))
        # pesos de la última capa oculta y la capa final
        self.W_outputs = np.empty((self.output_neurons, self.hidden_neurons + 1))
        self.randomize_weights()

    def randomize_weights(self):
        """Inicializa los pesos con valores aleatorios"""
        self.W_inputs = np.random.uniform(-1, 1, self.W_inputs.shape)
        self.W_hiddens = np.random.uniform(-1, 1, self.W_hiddens.shape)
        self.W_outputs = np.random.uniform(-1, 1, self.W_outputs.shape)
        self.W_inputs_batch = self.W_inputs.copy()
        self.W_hiddens_batch = self.W_hiddens.copy()
        self.W_outputs_batch = self.W_outputs.copy()
        if self.show_weights != None:
            self.show_weights()

    def sigmoid(self, y):
        """Función de activación"""
        return 1 / (1 + np.exp(-y))

    def derivative_sigmoid(self, y):
        """Derivada de la función de activación (sigmoide)"""
        return y * (1.0 - y)

    def feed_forward(self, inputs):
        """
        Toma un vector de entradas para calcular la net y su valor en una función
        de activación
        """
        # capa de entrada
        inputs = np.insert(inputs, 0, -1)
        net = np.dot(self.W_inputs, inputs) # Wx + bias
        output = self.sigmoid(net) # función de activación
        self.sigmoids[0] = output

        # capas intermedias
        for i in range(self.hidden_layers):
            output = np.insert(output, 0, -1)
            net = np.dot(self.W_hiddens[i,:,:], np.array(output)) # Wx + bias
            output = self.sigmoid(net) # función de activación
            self.sigmoids[i + 1] = output

        # capa de salida
        output = np.insert(output, 0, -1)
        net = np.dot(self.W_outputs, output) # Wx + bias
        output = self.sigmoid(net) # función de activación
        self.sigmoids[-1] = output

        return np.array(output)

    def jacobian(self, x):
        """Convierte un vector en una matriz diagonal"""
        derivative = self.derivative_sigmoid(x)
        new_jacobian = np.zeros((derivative.shape[0], derivative.shape[0]))
        for index in range(new_jacobian.shape[0]):
            new_jacobian[index, index] = derivative[index]
        return new_jacobian

    def get_sensitivity(self, error):
        """Obtiene las sensibilidades de cada capa"""
        # Sensibilidad de la última capa
        s = np.array([-2 * np.dot(self.jacobian(self.sigmoids[-1]), error)]).T
        self.sensitivities[-1] = s.copy()

        # Sensibilidad de la última capa oculta (penúltima capa)
        s = np.dot(np.dot(self.jacobian(self.sigmoids[-2]), self.W_outputs[:,1:].T), s)
        self.sensitivities[-2] = s.copy()

        layer = -3
        # Sensibilidad para el restos de capas ocultas
        for i in reversed(range((self.hidden_layers))):
            s = np.dot(np.dot(self.jacobian(self.sigmoids[layer]), self.W_hiddens[i, :, 1:].T), s)
            self.sensitivities[layer] = s.copy()
            layer -= 1

    def backpropagation(self, inputs, algorithm):
        """Realiza el método de retropropagación"""
        # actualiza los pesos de la capa oculta final con la capa de salida
        a = np.insert(self.sigmoids[-2], 0, -1) # añade el bias al sigmoide
        gradient = np.multiply(np.array(self.sensitivities[-1]), a.T)
        if algorithm == "Stochastic":
            self.W_outputs += -self.lr * gradient
        elif algorithm == "Batch":
            self.W_outputs_batch += -self.lr * gradient
        else:
            self.gradients.append(gradient)

        # actualiza los pesos de las capas ocultas
        layer = -2
        for i in reversed(range((self.hidden_layers))):
            a = np.insert(self.sigmoids[layer-1], 0, -1) # añade el bias al sigmoide
            gradient = np.multiply(np.array(self.sensitivities[layer]), a.T)
            if algorithm == "Stochastic":
                self.W_hiddens[i] += -self.lr * gradient
            elif algorithm == "Batch":
                self.W_hiddens_batch[i] += -self.lr * gradient
            else:
                self.gradients.append(gradient)
            layer -= 1

        # actualiza los pesos de la capa de entrada con la primer capa oculta
        a = np.array(inputs)
        a = np.insert(a, 0, -1)
        gradient = np.multiply(np.array(self.sensitivities[0]), a.T)
        if algorithm == "Stochastic":
            self.W_inputs += -self.lr * gradient
        elif algorithm == "Batch":
            self.W_inputs_batch += -self.lr * gradient
        else:
            self.gradients.append(gradient)

        if self.show_weights != None:
            self.show_weights()

    def encode_desired_output(self, Y: list):
        """Retorna una matriz de valores codificados para representar los valores
        del vector de valores deseados Y"""
        D = np.zeros((len(Y), len(np.unique(Y))))
        for i in range(len(Y)):
            # TODO: Cambiar esto para que se puedan poner clases no consecutivas
            D[i, Y[i]] = 1
        return D

    def get_confusion_matrix(self, X: list, Y: list, D:list):
        """Se genera la matriz de confusion"""
        # numero de clases
        n = len(np.unique(Y))
        # matriz de confusion
        conf_matrix = np.zeros((n + 1, n + 1))

        # por cada elemento del set de datos de entrenamiento obtenemos su clasificación
        count = 0
        for i in X:
            evaluate = self.feed_forward(i)
            guessed = self.encode_guess(evaluate)
            actual = self.encode_guess(D[count])
            conf_matrix[guessed][actual] += 1
            count += 1

        column_sum = conf_matrix.sum(axis=0)
        row_sum = conf_matrix.sum(axis=1)

        for i in range(n):
            conf_matrix[i][n] = row_sum[i]
            conf_matrix[n][i] = column_sum[i]

        conf_matrix[n][n] = column_sum.sum()

        print("\nMatriz de confusión:")
        print("Filas = clases obtenidas por el MLP")
        print("Columnas = clases reales")
        print("clase|", end="")
        for i in range(n):
            print(f"{i}|", end="")
        print("total|")
        for i in range(n):
            print(f"{i}    |{conf_matrix[i]}")
        print(f"total|{conf_matrix[n]}")

    def quickprop(self):
        epsilon = 1e-6
        if not self.is_st_prev_init:
            self.gradients_prev = [np.random.uniform(-1, 1, self.gradients[i].shape) for i in range(len(self.gradients))]
            self.is_st_prev_init = True
        for i, st in enumerate(self.gradients):
            miu = 1.75
            # st - 1
            st_prev = self.gradients_prev[i]
            # incremento de W-1 
            w_prev = -self.lr * st_prev
            div = (st_prev - st)
            div[div == 0] = epsilon
            temp = (st / div) * w_prev

            if np.linalg.norm(temp) > np.linalg.norm(miu * w_prev):
                temp = miu * w_prev

            if np.linalg.norm(st_prev * st) < 0:
                wt_increment = temp
            else:
                wt_increment = temp + epsilon * st


            if i == len(self.gradients) - 1:
                self.W_inputs += wt_increment
            elif len(self.gradients) > 2 and i == 1:
                self.W_hiddens[0] += wt_increment
            else:
                self.W_outputs += wt_increment

        self.gradients_prev = self.gradients[:]


    def train(self, X: list, Y: list, max_epoch: int, min_error: float, algorithm: str):
        """Se entrena el mlp usando feed_forward y backpropagation"""
        # Se calcula la cantidad de filas m
        m = len(X)
        # vector de datos correctos y codificado
        D = self.encode_desired_output(Y)

        # Error cuadrático medio (mse)
        mean_sqr_error = 0
        # Error cuadrático acumulado por época
        epoch_sqr_error = 0
        self.is_st_prev_init = False

        while True:
            # Se itera por cada fila de X
            for i in range(m):
                y = self.feed_forward(X[i])
                error = np.subtract(D[i], y)
                epoch_sqr_error += np.sum(error ** 2)

                # Se calculan las sensibilidades
                self.get_sensitivity(error)

                # Se ajustan los pesos
                self.backpropagation(X[i], algorithm)

                if algorithm == "Quickprop" or algorithm == "QP + BP":
                    self.quickprop()
                    self.gradients.clear()
                    self.gradients = []

            # Se imprimen los pesos de la capa oculta por época
            if self.plot_weights != None:
                self.plot_weights(self.W_inputs, 'g')

            if algorithm == "Batch":
                self.W_inputs = np.divide(self.W_inputs_batch, m)
                self.W_hiddens = np.divide(self.W_hiddens_batch, m)
                self.W_outputs = np.divide(self.W_outputs_batch, m)

            # Se obtiene la media del error cuadrático y hacemos que el mse sea cero
            mean_sqr_error = epoch_sqr_error / m

            # Se guarda el mejor mse hasta el momento
            if self.best_mse == 0:
                self.best_mse = mean_sqr_error
            elif mean_sqr_error < self.best_mse:
                self.best_mse = mean_sqr_error

            print(f'Epoca: {self.epoch} | Error cuadrático: {mean_sqr_error}')
            if len(self.mse_list) > 0:
                print("error actual: ", mean_sqr_error - self.best_mse)
                print("algoritmo: ", algorithm)
                # input()
                if mean_sqr_error - self.best_mse > self.qp_min_error and algorithm == "QP + BP":
                    break
            self.mse_list.append(mean_sqr_error)
            epoch_sqr_error = 0
            self.epoch += 1

            if self.plot_mse != None:
                self.plot_mse(self.mse_list)

            # Si se llegó al número máximo de épocas o si el mse es menor al error mínimo deseado
            if self.epoch == max_epoch or mean_sqr_error < min_error:
                break

        self.get_confusion_matrix(X, y, D)
        self.sigmoids = [0 for _ in range(2 + self.hidden_layers)]
        self.sensitivities = [0 for _ in range(2 + self.hidden_layers)]
        return self.epoch, mean_sqr_error


    def encode_guess(self, y):
        """Devuelve el valor más alto de una salida de la red"""
        return np.where(y == np.amax(y))[0][0]

    def print_weights(self):
        print("inputs: ", self.W_inputs)
        print("hiddens: ", self.W_hiddens)
        print("outputs: ", self.W_outputs)
