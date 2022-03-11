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

    def feed_forward(self, inputs):
        # capa de entrada
        net = np.dot(self.W_inputs[:,:-1], inputs[:-1]) # Wx
        net = np.add(net, self.W_inputs[:,-1]) # Wx + bias
        output = self.sigmoid(net) # función de activación
        self.sigmoids[0] = output

        # capas intermedias
        for i in range(self.hidden_layers):
            net = np.dot(self.W_hiddens[i,:,:-1], np.array(output)) # Wx
            net = np.add(net, self.W_hiddens[i,:,-1]) # Wx + bias
            output = self.sigmoid(net) # función de activación
            self.sigmoids[i+1] = output

        # capa de salida
        net = np.dot(self.W_outputs[:,:-1], output) # Wx
        net = np.add(net, self.W_outputs[:,-1]) # Wx + b
        output = self.sigmoid(net) # función de activación
        self.sigmoids[-1] = output
        
        return np.array(output)

    def train(self, X, Y):
        # obtenemos la predicción de la red
        D = self.feed_forward(X)

        # obtenemos el error de la capa de salida
        error = np.subtract(Y, D) # deseada - obtenida

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
            
