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

        # guarda los valores de activación de todas las capas
        self.sigmoids = list(range(2 + self.hidden_layers))

        # guarda las sensibilidades = input layer + output layer + hidden layers
        self.sensitivities = []

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
        self.sensitivities.append(s)

        # Sensibilidad de la última capa oculta (penúltima capa)
        s = np.dot(np.dot(self.jacobian(self.sigmoids[-2]), self.W_outputs[:,1:].T), s)
        self.sensitivities.append(s)

        layer = -3
        # Sensibilidad para el restos de capas ocultas
        for i in reversed(range((self.hidden_layers))):
            s = np.dot(np.dot(self.jacobian(self.sigmoids[layer]), self.W_hiddens[i, :, 1:].T).T, s)
            self.sensitivities.append(s)
            layer -= 1

    def backpropagation(self):
        """Realiza el método de retropropagación"""
        # actualiza los pesos de la capa oculta final con la capa de salida
        self.W_outputs += -self.lr * np.array(self.sensitivities[0]) * self.W_outputs

        # actualiza los pesos de las capas ocultas
        layer = 1
        for i in reversed(range((self.hidden_layers))):
            self.W_hiddens[i] += -self.lr * np.array(self.sensitivities[layer]) * self.W_hiddens[i]
            layer += 1
        
        # actualiza los pesos de la capa de entrada con la primer capa oculta
        self.W_inputs += -self.lr * np.array(self.sensitivities[-1]) * self.W_inputs


    def encode_desired_output(self, Y: list):
        """Retorna una matriz de valores codificados para representar los valores
        del vector de valores deseados Y"""
        D = np.zeros((len(Y), len(np.unique(Y))))
        for i in range(len(Y)):
            D[i, Y[i]] = 1
        return D

    def train(self, X: list, Y: list, max_epoch: int, min_error: float):
        """Se entrena el mlp usando feed_forward y backpropagation"""
        # Se calcula la cantidad de filas m y la cantidad de columnas n de X
        m, n = len(X), len(X[0])
        # vector de datos correctos y codificado
        D = self.encode_desired_output(Y)

        # Error cuadrático medio (mse)
        mean_sqr_error = 0
        # Error cuadrático acumulado por época
        epoch_sqr_error = 0
        # Número de épocas
        epoch = 0

        while True:
            # Se itera por cada fila de X
            for i in range(m):
                y = self.feed_forward(X[i])
                error = D[i] - y
                epoch_sqr_error += np.sum(error ** 2)

                # Se calculan las sensibilidades
                self.get_sensitivity(error)

                # Se ajustan los pesos
                self.backpropagation()

            # Se obtiene la media del error cuadrático y hacemos que el mse sea cero
            mean_sqr_error += epoch_sqr_error / m
            epoch_sqr_error = 0

            epoch += 1
            # Si se llegó al número máximo de épocas o si el mse es menor al error mínimo deseado
            if epoch == max_epoch or mean_sqr_error < min_error:
                break

    def print_weights(self):
        print("inputs: ", self.W_inputs)
        print("hiddens: ", self.W_hiddens)
        print("outputs: ", self.W_outputs)
