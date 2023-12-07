import numpy as np

# Initialize weights and input
weights = np.array([0.1, 0.2])
input_data = np.array([1.0, 2.0])
true_output = 0.6

# Learning rate
learning_rate = 0.01

# Forward Propagation
def forward_propagation(input_data, weights):
    return np.dot(input_data, weights)


# Loss Calculation (Mean Squared Error)
def loss(prediction, true_output):
    return (prediction - true_output) ** 2


# Backpropagation
def backpropagation(input_data, weights, true_output):
    prediction = forward_propagation(input_data, weights)
    d_loss_d_pred = 2 * (prediction - true_output)
    d_pred_d_weights = input_data
    gradient = d_loss_d_pred * d_pred_d_weights
    return gradient


# Weight Update
def update_weights(weights, gradient, learning_rate):
    return weights - learning_rate * gradient

# Forward Propagation: Compute the output of the network for a given input.
# Loss Calculation: Compute the loss (difference) between the network output and the true value.
# Backpropagation: Compute the gradients of the loss with respect to the weights.
# Weight Update: Adjust the weights using the gradients.

# Training loop
for epoch in range(100):  # Train for 100 epochs
    prediction = forward_propagation(input_data, weights)
    current_loss = loss(prediction, true_output)
    gradient = backpropagation(input_data, weights, true_output)
    weights = update_weights(weights, gradient, learning_rate)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {current_loss}")

print("Final weights:", weights)
