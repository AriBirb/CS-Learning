import csv
import random
import math

def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))
def dsigmoid(x: float):
    s = sigmoid(x)
    return s * (1 - s)

def linear(x: float):
    return x
def dlinear(x:float):
    return 1

def relu(x):
    return max(0, x)
def drelu(x):
    if x > 0:
        return 1
    return 0


class Node:
    def __init__(self, input_size: int):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(input_size)]
        self.bias = random.uniform(-0.1, 0.1)

    def __str__(self):
        return (f"class Node: "
                f"Weights={self.weights}, "
                f"Bias={self.bias}")

    def node_compute_output(self, inputs: list[float]):
        assert len(inputs) == len(self.weights)
        return sum(self.weights[i] * inputs[i] for i in range(len(self.weights))) + self.bias


class Layer:
    def __init__(self, num_nodes: int, input_size: int, activation_function, dactivation_function):
        self.nodes = []
        self.activation_function = activation_function
        self.dactivation_function = dactivation_function
        for _ in range(num_nodes):
            self.nodes.append(Node(input_size))

    def __str__(self):
        return (f"class Layer: "
                f"Weights={[(node.weights) for node in self.nodes]}, "
                f"Biases={[(node.bias) for node in self.nodes]}")

    def layer_compute_output(self, inputs: list[float]):
        return [self.activation_function(node.node_compute_output(inputs)) for node in self.nodes]

    def layer_compute_output_no_activation(self, inputs: list[float]):
        return [node.node_compute_output(inputs) for node in self.nodes]


class Neural_Network_Regression:
    def __init__(self, num_inputs: int, num_outputs: int, hidden_layers: list[int], learning_rate: float,
                 activation_function, dactivation_function, exit_activation_function, dexit_activation_function):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = []
        self.hidden_layers.append(Layer(hidden_layers[0], num_inputs, activation_function, dactivation_function))
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.append(Layer(hidden_layers[i], hidden_layers[i-1], activation_function, dactivation_function))
        self.output_layer = Layer(num_outputs, hidden_layers[-1], exit_activation_function, dexit_activation_function)
        self.learning_rate = learning_rate

    def forward_propagation_single_observation(self, inputs: list[float]):
        current_layer_value = inputs
        for layer in self.hidden_layers:
            current_layer_value = layer.layer_compute_output(current_layer_value)
        return self.output_layer.layer_compute_output(current_layer_value)

    def forward_propagation_many_observations(self, inputs: list[list[float]]):
        return [self.forward_propagation_single_observation(observation) for observation in inputs]

    def mean_squared_error(self, predicted_outputs: list[float], true_outputs: list[float]):
        assert len(predicted_outputs) == len(true_outputs)
        return 0.5 * sum((p - t) ** 2 for p, t in zip(predicted_outputs, true_outputs)) / len(predicted_outputs)

    def back_propagation_single_observation(self, inputs: list[float], true_outputs: list[float]):
        assert self.num_outputs == len(true_outputs)
        z = []
        a = []
        current_layer_value = inputs
        for layer in self.hidden_layers:
            z.append(layer.layer_compute_output_no_activation(current_layer_value))
            current_layer_value = layer.layer_compute_output(current_layer_value)
            a.append(current_layer_value)
        z.append(self.output_layer.layer_compute_output_no_activation(current_layer_value))
        a.append(self.output_layer.layer_compute_output(current_layer_value))

        delta = [[(a[-1][i] - true_outputs[i]) * self.output_layer.dactivation_function(z[-1][i])
                        for i in range(len(true_outputs))]]
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            weights_of_next_layer = []
            if i == len(self.hidden_layers) - 1:
                weights_of_next_layer = [node.weights for node in self.output_layer.nodes]
            else:
                weights_of_next_layer = [node.weights for node in self.hidden_layers[i + 1].nodes]
            weights_transposed = [[weights_of_next_layer[k][j]
                                   for k in range(len(weights_of_next_layer))
                                   ] for j in range(len(weights_of_next_layer[0]))]
            delta_of_next_layer = delta[0]
            product_of_weights_and_delta = [sum([
                current_to_next_weights[j] * delta_of_next_layer[j] for j in range(len(current_to_next_weights))
            ]) for current_to_next_weights in weights_transposed]
            delta_current_layer = [product_of_weights_and_delta[j] * self.hidden_layers[i].dactivation_function(z[i][j])
                                   for j in range(len(product_of_weights_and_delta))]
            delta.insert(0, delta_current_layer)

        grad_W = []
        grad_B = [d[:] for d in delta]

        for i in range(len(delta)-1, 0, -1):
            grad_W.insert(0, [[delta[i][j] * a[i-1][k]
                               for k in range(len(a[i-1]))]
                              for j in range(len(delta[i]))])
        grad_W.insert(0, [[delta[0][j] * inputs[k]
                           for k in range(len(inputs))]
                          for j in range(len(delta[0]))])

        for i in range(len(self.hidden_layers)):
            for j in range(len(self.hidden_layers[i].nodes)):
                self.hidden_layers[i].nodes[j].weights = \
                    [self.hidden_layers[i].nodes[j].weights[k] - self.learning_rate * grad_W[i][j][k]
                     for k in range(len(grad_W[i][j]))]
                self.hidden_layers[i].nodes[j].bias = (
                        self.hidden_layers[i].nodes[j].bias - self.learning_rate * grad_B[i][j])
        for i in range(len(self.output_layer.nodes)):
            self.output_layer.nodes[i].weights = \
                [self.output_layer.nodes[i].weights[j] - self.learning_rate * grad_W[-1][i][j]
                 for j in range(len(grad_W[-1][i]))]
            self.output_layer.nodes[i].bias = (
                    self.output_layer.nodes[i].bias - self.learning_rate * grad_B[-1][i])

    def back_propagation_many_observations(self, inputs: list[list[float]], true_outputs: list[list[float]]):
        [self.back_propagation_single_observation(observation, true_output)
         for observation, true_output in zip(inputs, true_outputs)]


# --- Simple synthetic dataset: y = 2x + 3 ---
X_train = [[x] for x in range(-10, 11)]  # Inputs from -10 to 10
Y_train = [[2 * x[0] + 3 + random.uniform(-0.5, 0.5)] for x in X_train]  # True outputs

# Create a network: 1 input, 1 output, 2 hidden layers (3 and 2 neurons)
network = Neural_Network_Regression(
    num_inputs=1,
    num_outputs=1,
    hidden_layers=[4, 4, 2],
    learning_rate=0.002,
    activation_function=sigmoid,
    dactivation_function=dsigmoid,
    exit_activation_function=linear,
    dexit_activation_function=dlinear
)

# Train for 1000 epochs
for epoch in range(10000):
    # Train on each sample (SGD)
    for x, y in zip(X_train, Y_train):
        network.back_propagation_single_observation(x, y)

    # Compute loss every 100 epochs
    if epoch % 100 == 0:
        predictions = network.forward_propagation_many_observations(X_train)
        preds = [p[0] for p in predictions]
        targets = [t[0] for t in Y_train]
        loss = network.mean_squared_error(preds, targets)
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

# --- Test predictions after training ---
test_inputs = [[-5], [0], [5], [10]]
print("\nFinal predictions:")
for inp in test_inputs:
    pred = network.forward_propagation_single_observation(inp)
    print(f"x = {inp[0]:>2} -> predicted y = {pred[0]:.3f}")



import matplotlib.pyplot as plt

# Predictions on the training data
predictions = network.forward_propagation_many_observations(X_train)
preds = [p[0] for p in predictions]
targets = [t[0] for t in Y_train]
x_vals = [x[0] for x in X_train]

plt.figure(figsize=(8,5))
plt.scatter(x_vals, targets, color='blue', label='True data')
plt.plot(x_vals, preds, color='red', label='Model predictions')
plt.title("Neural Network Regression: y = 2x + 3")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# def get_data_from_csv(file_name: str) -> list:
#
#     header = []
#     data = []
#
#     with open(file_name, mode='r', encoding='utf-8') as file:
#         csv_reader = csv.reader(file)
#         first = True
#         for row in csv_reader:
#             if first:
#                 first = False
#                 header = row
#                 continue
#             data.append(row)
#
#     return data
#
# def convert_dataset_str_to_float (data: list[list[str]]) -> list[list[float]]:
#     return [[float(value) for value in row] for row in data]
#
# data = get_data_from_csv('1.01. Simple linear regression.csv')
# data = convert_dataset_str_to_float(data)
#
# network = Neural_Network_Regression(1, 1, [2, 3, 2], 0.05, relu, drelu, linear, dlinear)




