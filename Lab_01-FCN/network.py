from .layer import *
alpha = 0.01  # L1正则化的超参数
beta = 0.01   # L2正则化的超参数

class Network(object):
    def __init__(self, weight_decay=0.0):
        self.fc1 = FullyConnected(28*28, 128, weight_decay=weight_decay)
        self.act1 = ActivationReLU()
        self.fc2 = FullyConnected(128, 64, weight_decay=weight_decay)
        self.act2 = ActivationReLU()
        self.fc3 = FullyConnected(64, 10, weight_decay=weight_decay)
        self.loss_layer = SoftmaxWithLoss()

    def forward(self, input, target):
        h1 = self.fc1.forward(input)
        h1_activated = self.act1.forward(h1)
        h2 = self.fc2.forward(h1_activated)
        h2_activated = self.act2.forward(h2)
        pred, loss = self.loss_layer.forward(self.fc3.forward(h2_activated), target)
        return pred, loss

    def backward(self):
        loss_grad = self.loss_layer.backward()
        h2_grad = self.act2.backward(self.fc3.backward(loss_grad))
        h1_grad = self.act1.backward(self.fc2.backward(h2_grad))
        _ = self.fc1.backward(h1_grad)

    def update(self, lr):
        self.sgd_optimizer(lr)

    def sgd_optimizer(self, lr):
        self.fc1.weight -= lr * (self.fc1.weight_grad + alpha * np.sign(self.fc1.weight) + beta * 2 * self.fc1.weight_decay * self.fc1.weight)
        self.fc1.bias -= lr * self.fc1.bias_grad
        self.fc2.weight -= lr * (self.fc2.weight_grad + alpha * np.sign(self.fc2.weight) + beta * 2 * self.fc2.weight_decay * self.fc2.weight)
        self.fc2.bias -= lr * self.fc2.bias_grad
        self.fc3.weight -= lr * (self.fc3.weight_grad + alpha * np.sign(self.fc3.weight) + beta * 2 * self.fc3.weight_decay * self.fc3.weight)
        self.fc3.bias -= lr * self.fc3.bias_grad
