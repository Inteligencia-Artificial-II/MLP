from src.MLP import MLP
import numpy as np

if __name__ == "__main__":
    # parámetros
    inputs = 3
    hidden_layers = 2
    hidden_neurons = 4
    outputs = 2

    input_data = np.array([1,2,3,1])
    output_data = np.array([1,0])


    mlp = MLP(inputs, hidden_layers, hidden_neurons, outputs)
    print("primera predicción: ", mlp.feed_forward(input_data))
    mlp.print_weights()
    for i in range(200):
        mlp.train(input_data, output_data)
    print(".-.-.-.-.-.-.-.-.-.-.-.-.-")
    mlp.print_weights()
    print("segunda predicción: ", mlp.feed_forward(input_data))