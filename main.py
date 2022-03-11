from src.MLP import MLP
import numpy as np

if __name__ == "__main__":
    # parámetros
    inputs = 3
    hidden_layers = 2
    hidden_neurons = 4
    outputs = 2

    mlp = MLP(inputs, hidden_layers, hidden_neurons, outputs)
    result = mlp.feed_forward(np.array([1,2,3,1]))

    print(result)