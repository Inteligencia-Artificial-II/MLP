import numpy as np

class MLP:
    def __init__(self, i_neurons, h_layers, h_neurons, o_neurons):
        # número de neuronas por capa
        self.input_neurons = i_neurons
        self.hidden_neurons = h_neurons
        self.output_neurons = o_neurons

        # número de capas ocultas
        self.hidden_layers = h_layers - 1

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

        # capas intermedias
        for i in range(self.hidden_layers):
            net = np.dot(self.W_hiddens[i,:,:-1], np.array(output)) # Wx
            net = np.add(net, self.W_hiddens[i,:,-1]) # Wx + bias
            output = self.sigmoid(net) # función de activación

        # capa de salida
        net = np.dot(self.W_outputs[:,:-1], output) # Wx
        net = np.add(net, self.W_outputs[:,-1]) # Wx + b
        output = self.sigmoid(net) # función de activación
        
        return np.array(output)
            
