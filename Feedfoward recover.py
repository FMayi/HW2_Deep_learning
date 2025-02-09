
from turtle import width
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

font = {'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


######################################################
# Q1 Implement Init, Forward, and Backward For Layers
######################################################


class SigmoidCrossEntropy:
    def __init__(self):
        self.labels = None
        self.sigmoid = None
        self.loss = None
        self.batch_size = None
        self.logits = None
    # Compute the cross entropy loss after sigmoid. The reason they are put into the same layer is because the gradient has a simpler form
    # logits -- batch_size x num_classes set of scores, logits[i,j] is score of class j for batch element i
    # labels -- batch_size x 1 vector of integer label id (0,1) where labels[i] is the label for batch element i
    # TODO: Output should be a positive scalar value equal to the average cross entropy loss after sigmoid
    def forward(self, logits, labels):
        # TODO: Compute the gradient of the cross entropy loss with respect to the the input logits
        eps = 1e-9
        self.labels = labels
        self.logits = logits
        self.batch_size = labels.shape[0]
        self.sigmoid = 1 / (1 + np.exp(-logits))

        if np.isnan(self.sigmoid).any():
            print("ERROR: NaN detected in sigmoid output!")
            exit()
        self.loss = - ((labels * np.log(self.sigmoid + eps)) + (1 - labels) * np.log(1 - self.sigmoid + eps))
        return np.sum(self.loss) / self.batch_size

    def backward(self):
        gradient_of_loss = self.sigmoid - self.labels
        norm = np.linalg.norm(gradient_of_loss)
        return gradient_of_loss


class ReLU:

    # TODO: Compute ReLU(input) element-wise
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
    # TODO: Given dL/doutput, return dL/dinput
    def backward(self, grad):
        self.grad = grad
        grad_base = np.where(self.input > 0, 1, 0)
        return grad_base * self.grad
    # No parameters so nothing to do during a gradient descent step
    def step(self, step_size, momentum=0, weight_decay=0):
        return


class LinearLayer:

    # TODO: Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
    def __init__(self, input_dim, output_dim):
        # FMA
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fan_in = input_dim
        self.fan_out = output_dim
        self.W = np.random.randn(self.fan_in, self.fan_out) * np.sqrt(2.0 / self.fan_in)
        self.b = np.zeros((1, output_dim))

    # TODO: During the forward pass, we simply compute XW+b
    def forward(self, input):
        # print(f"Linear forward: input shape {input.shape}, weight shape {self.W.shape}, bias shape {self.b.shape}")
        self.input = input
        output = np.dot(input, self.W) + self.b
        # print(f"Linear forward output shape: {output.shape}")
        return output

    # TODO: Backward pass inputs:
    #
    # grad dL/dZ -- For a batch size of n, grad is a (n x output_dim) matrix where
    #         the i'th row is the gradient of the loss of example i with respect
    #         to z_i (the output of this layer for example i)

    # Computes and stores:
    #
    # self.grad_weights dL/dW --  A (input_dim x output_dim) matrix storing the gradient
    #                       of the loss with respect to the weights of this layer.
    #                       This is an summation over the gradient of the loss of
    #                       each example with respect to the weights.
    #
    # self.grad_bias dL/dZ--     A (1 x output_dim) matrix storing the gradient
    #                       of the loss with respect to the bias of this layer.
    #                       This is an summation over the gradient of the loss of
    #                       each example with respect to the bias.

    # Return Value:
    #
    # grad_input dL/dX -- For a batch size of n, grad_input is a (n x input_dim) matrix where
    #               the i'th row is the gradient of the loss of example i with respect
    #               to x_i (the input of this layer for example i)

    def backward(self, grad):
        # grad dL/dZ
        self.grad = grad
        # self.grad_weights dL/dW
        self.grad_weights = np.dot(self.input.T, grad)
        # self.grad_bias dL/dZ--
        self.grad_bias = np.sum(grad, axis=0, keepdims=True)
        # grad_input dL/dX
        # Check FMA
        max_norm = 5
        norm = np.linalg.norm(self.grad_weights)
        if norm > max_norm:
            self.grad_weights = (self.grad_weights / norm) * max_norm
        return np.dot(grad, self.W.T)

        # Return Value:
        # grad_input dL/dX
        # raise Exception('Student error: You haven\'t implemented the backward pass for LinearLayer yet.')

    ######################################################
    # Q2 Implement SGD with Weight Decay
    ######################################################
    def step(self, step_size, momentum=0.8, weight_decay=0.0):
        # TODO: Implement the step
        self.step_size = step_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_weights += weight_decay * self.W

        if not hasattr(self, "velocity_w"):
            self.velocity_w = np.zeros_like(self.W)  # Momentum for weights
            self.velocity_b = np.zeros_like(self.b)  # Momentum for biases

            # Apply momentum update
        self.velocity_w = momentum * self.velocity_w + step_size * self.grad_weights
        self.velocity_b = momentum * self.velocity_b + step_size * self.grad_bias

        # Update weights and biases
        self.W -= self.velocity_w
        self.b -= self.velocity_b


######################################################
# Q4 Implement Evaluation for Monitoring Training
######################################################

# TODO: Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
    num_examples = X_val.shape[0]
    total_loss = 0
    total_accuracy = 0
    eps = 1e-9
    for i in range(0, num_examples, batch_size):
        x_batch = X_val[i:i + batch_size]
        y_batch = Y_val[i:i + batch_size]
        logits = model.forward(x_batch)
        probs = 1/(1+np.exp(-logits))
        prediction = np.clip(probs, 1e-7, 1 - 1e-7)

        batch_loss = -np.mean(y_batch * np.log(prediction + eps) + (1 - y_batch) * np.log(1 - prediction + eps))
        total_loss += batch_loss * x_batch.shape[0]

        prediction_b = (prediction > 0.5).astype(int)
        total_accuracy += np.sum(prediction_b == y_batch)

    ave_loss = total_loss / num_examples
    ave_accuracy = total_accuracy / num_examples
    return ave_loss, ave_accuracy

    # raise Exception('Student error: You haven\'t implemented the step for evaluate function.')


def main():
    # TODO: Set optimization parameters (NEED TO SUPPLY THESE)
    batch_size = 64
    max_epochs = 50
    step_size = 0.0001

    number_of_layers = 2
    width_of_layers = 256
    weight_decay = 0.01
    momentum = 0.9

    # Load data
    data = pickle.load(open('cifar_2class_py3.p', 'rb'))
    X_train = data['train_data']
    Y_train = data['train_labels']
    X_test = data['test_data']
    Y_test = data['test_labels']

    eps = 1e-9
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    X_train = (X_train-train_mean)/(train_std + eps)
    X_test = (X_test - train_mean)/(train_std + eps)
    # Some helpful dimensions
    num_examples, input_dim = X_train.shape
    output_dim = 1  # number of class labels -1 for sigmoid loss

    # Build a network with input feature dimensions, output feature dimension,
    # hidden dimension, and number of layers as specified below. You can edit this as you please.
    net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)

    # Some lists for book-keeping for plotting later
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    loss_funct = SigmoidCrossEntropy()

    # For plotting norms
    #batch_gradient_norms = []
    # Open a CSV file to log results
    with open("C:/Users/FMayi/PycharmProjects/AI535/A2/hw2/training_results.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Write header row
        writer.writerow(["Epoch", "Train Objective", "Train Acc", "Train misclassification ER", "Testing Objective", "Testing Acc", "Testing misclassification ER"])
    # Q2 TODO: For each epoch below max epochs
        for epoch in range(max_epochs):
            # Scramble order of examples
            indices = np.random.permutation(num_examples)
            X_train, Y_train = X_train[indices], Y_train[indices]
            epoch_loss = 0
            correct_pred = 0
            total_s = 0


            num_batches = num_examples // batch_size
            # for each batch in data:
            # Gather batch
            for i in range(0, num_examples, batch_size):
                X_batch = X_train[i:i + batch_size]
                Y_batch = Y_train[i:i + batch_size]

                # Compute forward pass
                predictions = net.forward(X_batch)
                # Compute loss
                if predictions is None:
                    print("Error: predictions is None!")
                    exit()
                loss = loss_funct.forward(predictions, Y_batch)
                epoch_loss += loss *X_batch.shape[0]

                correct_batch = np.sum((predictions > 0.5).astype(int) == Y_batch.reshape(-1,1))
                correct_pred += correct_batch
                total_s += Y_batch.shape[0]
                # Backward loss and networks
                grad = loss_funct.backward()
                net.backward(grad)

                # Take optimizer step
                net.step(step_size, momentum, weight_decay)
            # Book-keeping for loss / accuracy
            losses.append(epoch_loss / total_s)  # Average loss per epoch
            accs.append(correct_pred / total_s)

            val_loss, val_acc = evaluate(net, X_test, Y_test, batch_size)
            val_losses.append(val_loss)
            val_accs.append(val_acc)


            error = 1 - accs[-1]
            val_error = 1 - val_accs[-1]
            # Log the current epoch's results to CSV

            row =[epoch + 1, f"{losses[-1]:.4f}", f"{accs[-1]:.4f}", f"{error:.4f}", f"{val_losses[-1]:.4f}", f"{val_accs[-1]:.4f}", f"{val_error:.4f}"]
            writer.writerow(row)
            csv_file.flush()

            # Evaluate performance on test.

            # val_loss, tacc = evaluate(net, X_test, Y_test, batch_size)
            # print(tacc)

            ###############################################################
            # Print some stats about the optimization process after each epoch
            ###############################################################
            # epoch_avg_loss -- average training loss across batches this epoch
            print(f"Epoch {epoch + 1}/{max_epochs}: ")
            print(f"Train _Loss: {losses[-1]:.4f}\n")
            # epoch_avg_acc -- average accuracy across batches this epoch
            print(f"Train _Acc: {accs[-1]:.4f}, \n")
            print(f"Error:  {error}")
            # vacc -- testing accuracy this epoch
            print(f"Val_Acc: {val_accs[-1]:.4f}\n")
            print(f"Val_error: {val_error}")
            # Val_losses
            print(f"Val_Loss: {val_losses[-1]:.4f}\n")
        ###############################################################
        # plt.plot(batch_gradient_norms, label="Batch Gradient Norms (SigmoidCrossEntropy)")
        # plt.xlabel("Batch Iterations")
        # plt.ylabel("Gradient Norm")
        # plt.legend()
        # plt.show()

        # logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(i,epoch_avg_loss, epoch_avg_acc, vacc*100))

        ###############################################################
        # Code for producing output plot requires
        ###############################################################
        # losses -- a list of average loss per batch in training
        # accs -- a list of accuracies per batch in training
        # val_losses -- a list of average testing loss at each epoch
        # val_acc -- a list of testing accuracy at each epoch
        # batch_size -- the batch size
        ################################################################

        # Plot training and testing curves
        fig, ax1 = plt.subplots(figsize=(16, 9))
        color = 'tab:red'
       # ax1.plot(range(len(losses)), losses, c=color, alpha=0.85, label="Training Objective")
        ax1.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_losses))],
                 val_losses, c="red", label="Testing Objective")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
        ax1.tick_params(axis='y', labelcolor=color)
        # ax1.set_ylim(-0.01,3)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        #ax2.plot(range(len(losses)), accs, c=color, label="Training Acc.", alpha=0.85)
        ax2.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_accs))], val_accs,
                 c="blue", label="Testing Acc.")
        ax2.set_ylabel(" Accuracy", c=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.01, 1.01)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.legend(loc="center")
        ax2.legend(loc="center right")
        plt.show()


#####################################################
# Feedforward Neural Network Structure
# -- Feel free to edit when tuning
#####################################################

class FeedForwardNeuralNetwork:

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        self.layers = []
        if num_layers == 1:
            self.layers = [LinearLayer(input_dim, output_dim)]
        else:
            # FMA momentary code
            self.layers = [LinearLayer(input_dim, hidden_dim)]
            self.layers.append(ReLU())
            for i in range(num_layers - 2):
                self.layers.append(LinearLayer(hidden_dim, hidden_dim))
                self.layers.append(ReLU())
            self.layers.append(LinearLayer(hidden_dim, output_dim))

    # TODO: Please create a network with hidden layers based on the parameters

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, step_size, momentum, weight_decay):
        for layer in self.layers:
            layer.step(step_size, momentum, weight_decay)


def displayExample(x):
    r = x[:1024].reshape(32, 32)
    g = x[1024:2048].reshape(32, 32)
    b = x[2048:].reshape(32, 32)

    plt.imshow(np.stack([r, g, b], axis=2))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
