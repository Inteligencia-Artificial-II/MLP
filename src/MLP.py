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
        self.W_inputs = np.random.uniform(-1, 1, self.W_inputs.shape)
        self.W_hiddens = np.random.uniform(-1, 1, self.W_hiddens.shape)
        self.W_outputs = np.random.uniform(-1, 1, self.W_outputs.shape)

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def derivative_sigmoid(self, y):
        return y * (1.0 - y)

    def feed_forward(self, inputs):
        """
        Toma un vector de entradas para calcular la net y su valor en una función
        de activación
        """
        inputs = np.insert(inputs, 0, -1)

        # capa de entrada
        net = np.dot(self.W_inputs, inputs) # Wx + bias
        output = self.sigmoid(net) # función de activación
        output = np.insert(output, 0, -1)
        self.sigmoids[0] = output

        # capas intermedias
        for i in range(self.hidden_layers):
            net = np.dot(self.W_hiddens[i,:,:], np.array(output)) # Wx + bias
            output = self.sigmoid(net) # función de activación
            output = np.insert(output, 0, -1)
            self.sigmoids[i + 1] = output

        # capa de salida
        net = np.dot(self.W_outputs, output) # Wx + bias
        output = self.sigmoid(net) # función de activación
        self.sigmoids[-1] = output

        return np.array(output)

    def jacobian(self, x):
        derivative = self.derivative_sigmoid(x)
        new_jacobian = np.zeros((derivative.shape[0], derivative.shape[0]))
        for index in range(new_jacobian.shape[0]):
            new_jacobian[index, index] = derivative[index]
        return new_jacobian

    def get_sensitivity(self, error):

        # Sensibilidad de la última capa
        s = np.array([-2 * np.dot(self.jacobian(self.sigmoids[-1]), error)]).T
        self.sensitivities.append(s)
        print("s copy: ", s, s.shape)

        print("sigmoids: ", self.sigmoids[-2])
        print("W outputs: ", self.W_outputs, self.W_outputs.shape)
        # Sensibilidad de la última capa oculta (penúltima capa)
        s = np.dot(np.dot(self.jacobian(self.sigmoids[-2]), self.W_outputs.T), s)
        self.sensitivities.append(s)
        print("s copy: ", s, s.shape)

        layer = -3
        # Sensibilidad para el restos de capas ocultas
        for i in reversed(range((self.hidden_layers))):
            print("sensitivity: ", s, s.shape)
            print("jacobian: ", self.jacobian(self.sigmoids[layer]), self.jacobian(self.sigmoids[layer]).shape)
            print("sigmoids: ", self.sigmoids[layer])
            print("W hiddens: ", self.W_hiddens[i, :], self.W_hiddens[i, :].shape)
            s = np.dot(np.dot(self.jacobian(self.sigmoids[layer]), self.W_hiddens[i, :].T).T, s)
            print("sensitivity hidden: ", s, s.shape)
            print("s copy: ", s.copy(), s.copy().shape)
            self.sensitivities.append(np.array([s]).T)
            layer -= 1
        input()

    def backpropagation(self, x, error):
        """Realiza el método de retropropagación"""
        pass

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

                # TODO: Se calculan las sensibilidades
                self.get_sensitivity(error)

                # TODO: Se ajustan los pesos

            # Se obtiene la media del error cuadrático y hacemos que el mse sea cero
            mean_sqr_error += epoch_sqr_error / m
            epoch_sqr_error = 0

            epoch += 1
            # Si se llegó al número máximo de épocas o si el mse es menor al error mínimo deseado
            if epoch == max_epoch or mean_sqr_error < min_error:
                break

        # TODO: Se tiene que cambiar esto para que no se actualicen los pesos hasta
        # obtener todas las sensibilidades
        # actulización de pesos en la capa final
        f_y = self.sigmoids[-1] * (1 - self.sigmoids[-1])
        self.W_outputs += np.dot(self.W_outputs, np.dot((self.lr * error), f_y))

        # error retropropagado de la capa de salida a la última capa oculta
        hidden_error = np.dot(self.W_outputs[:,:-1].T, error)

        # actulización de pesos en la última capa
        if (self.hidden_layers == 0):
            f_y = self.sigmoids[-2] * (1 - self.sigmoids[-2])
            # si solo existe una capa oculta, se actualizan los pesos de entrada
            self.W_inputs += np.dot(self.W_inputs, np.dot((self.lr * hidden_error), f_y))
            return

        # si existe más de una capa oculta, se itera inversamente 
        layer = -2 # iniciamos un indice que cuente desde la última capa oculta
        for i in reversed(range(self.hidden_layers)):
            # actualizamos pesos entre las capas ocultas
            f_y = self.sigmoids[layer] * (1 - self.sigmoids[layer])
            self.W_hiddens[i] += np.dot(self.W_hiddens[i], np.dot((self.lr * hidden_error), f_y))
            # error retropropagado entre capas ocultas
            hidden_error = np.dot(self.W_hiddens[i,:,:-1].T, hidden_error)
            layer -= 1

        # se actualizan los pesos de entrada
        self.W_inputs += np.dot(self.W_inputs, np.dot((self.lr * hidden_error), f_y))

    def print_weights(self):
        print("inputs: ", self.W_inputs)
        print("hiddens: ", self.W_hiddens)
        print("outputs: ", self.W_outputs)
