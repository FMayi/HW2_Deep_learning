import numpy as np
from FeedFoward import SigmoidCrossEntropy  # Import only the loss class
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
        #raise Exception('Student error: You haven\'t implemented the forward pass for SigmoidCrossEntropy yet.')

    # TODO: Compute the gradient of the cross entropy loss with respect to the the input logits
        self.labels = labels
        self.logits = logits
        self.batch_size = labels.shape[0]
        self.sigmoid = 1/(1 + np.exp(-logits))
        self.loss = - (labels * np.log(self.sigmoid)) + (1-labels) *np.log(1- self.sigmoid)
        return np.sum(self.loss)/self.batch_size

    def backward(self):
        print("sigmoid shape:", self.sigmoid.shape)
        print("labels shape:", self.labels.shape)
        gradient_of_loss = self.sigmoid - self.labels
        return gradient_of_loss
        print("Gradient shape:", gradient_of_loss.shape)
       # raise Exception('Student error: You haven\'t implemented the backward pass for SigmoidCrossEntropy yet.')


# Create dummy data
logits = np.random.randn(3, 1)  # Simulate batch_size=3, num_classes=1
labels = np.random.randint(0, 2, size=(3, 1))  # Random 0/1 labels of same shape

# Initialize loss function
loss_fn = SigmoidCrossEntropy()

# Forward pass
loss_value = loss_fn.forward(logits, labels)
print("Loss:", loss_value)

# Backward pass (gradient check)
gradient = loss_fn.backward()
print("Gradient shape:", gradient.shape)
print("Gradient values:\n", gradient)

