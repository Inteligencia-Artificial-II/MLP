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
        self.sigmoids = [0 for i in range(2 + self.hidden_layers)]

        # guarda las sensibilidades = input layer + output layer + hidden layers
        self.sensitivities = [0 for i in range(h_layers + 1)]

        # Función del plotter para imprimir el mse
        self.plot_mse = None

        # Función del plotter para imprimir los pesos de la primer capa oculta
        self.plot_weights = None

        self.show_weights = None

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
    
    def backpropagation(self, inputs):
        """Realiza el método de retropropagación"""
        # actualiza los pesos de la capa oculta final con la capa de salida
        a = np.insert(self.sigmoids[-2], 0, -1) # añade el bias al sigmoide
        self.W_outputs += -self.lr * np.multiply(np.array(self.sensitivities[-1]), a.T)

        # actualiza los pesos de las capas ocultas
        layer = -2
        for i in reversed(range((self.hidden_layers))):
            a = np.insert(self.sigmoids[layer-1], 0, -1) # añade el bias al sigmoide
            self.W_hiddens[i] += -self.lr * np.multiply(np.array(self.sensitivities[layer]), a.T)
            layer -= 1

        # actualiza los pesos de la capa de entrada con la primer capa oculta
        a = np.array(inputs)
        a = np.insert(a, 0, -1)
        self.W_inputs += -self.lr * np.multiply(np.array(self.sensitivities[0]), a.T)

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

    def train(self, X: list, Y: list, max_epoch: int, min_error: float, batch: bool):
        """Se entrena el mlp usando feed_forward y backpropagation"""
        # Se calcula la cantidad de filas m
        m = len(X)
        # vector de datos correctos y codificado
        D = self.encode_desired_output(Y)

        # Error cuadrático medio (mse)
        mean_sqr_error = 0
        # Lista de errores cuadráticos medios
        mse_list = []
        # Error cuadrático acumulado por época
        epoch_sqr_error = 0
        # Número de épocas
        epoch = 0
        # acumulador de sensibilidades
        sens = [0 for i in range(len(self.sensitivities))]

        while True:
            # Se itera por cada fila de X
            for i in range(m):
                y = self.feed_forward(X[i])
                error = np.subtract(D[i], y)
                epoch_sqr_error += np.sum(error ** 2)    

                # Se calculan las sensibilidades
                self.get_sensitivity(error)

                # Se ajustan los pesos
                if not batch:
                    self.backpropagation(X[i])
                else:
                    for i in range(len(self.sensitivities)):
                        sens[i] = np.add(self.sensitivities[i], sens[i])

            # Se imprimen los pesos de la capa oculta por época
            if self.plot_weights != None:
                self.plot_weights(self.W_inputs, 'g')

            if batch:
                for i in range(len(self.sensitivities)):
                    self.sensitivities[i] = np.divide(sens[i], m)
                sens = [0 for i in range(len(self.sensitivities))]
                for i in range(m):
                    self.backpropagation(X[i])
                self.sensitivities = [0 for i in range(len(self.sensitivities))]

            # Se obtiene la media del error cuadrático y hacemos que el mse sea cero
            mean_sqr_error = epoch_sqr_error / m
            print(f'Epoca: {epoch} | Error cuadrático: {mean_sqr_error}')
            mse_list.append(mean_sqr_error)
            epoch_sqr_error = 0
            epoch += 1

            if self.plot_mse != None:
                self.plot_mse(mse_list)

            # Si se llegó al número máximo de épocas o si el mse es menor al error mínimo deseado
            if epoch == max_epoch or mean_sqr_error < min_error:
                break
        
        self.get_confusion_matrix(X, y, D)
        return epoch, mean_sqr_error


    def encode_guess(self, y):
        """Devuelve el valor más alto de una salida de la red"""
        return np.where(y == np.amax(y))[0][0]

    def print_weights(self):
        print("inputs: ", self.W_inputs)
        print("hiddens: ", self.W_hiddens)
        print("outputs: ", self.W_outputs)