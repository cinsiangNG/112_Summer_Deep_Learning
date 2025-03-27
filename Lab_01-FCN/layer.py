import numpy as np

class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *output_grad):
        raise NotImplementedError

class FullyConnected(_Layer):
    def __init__(self, in_features, out_features, weight_decay=0.0):  # 添加weight_decay参数
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))
        self.input = None
        self.weight_decay = weight_decay  # 正则化系数

    def forward(self, input):
        self.input = input
        output = np.dot(input, self.weight) + self.bias
        return output

    def backward(self, output_grad):
        input_grad = np.dot(output_grad, self.weight.T)
        self.weight_grad = np.dot(self.input.T, output_grad)
        self.bias_grad = np.sum(output_grad, axis=0, keepdims=True)

        # 添加正则化项的梯度
        self.weight_grad += 2 * self.weight_decay * self.weight

        return input_grad
        return input_grad

class ActivationReLU(_Layer):
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        output = np.maximum(0, input)
        return output

    def backward(self, output_grad):
        input_grad = output_grad * (self.input > 0)
        return input_grad

class SoftmaxWithLoss(_Layer):
    def __init__(self):
        self.softmax_output = None
        self.target = None

    def forward(self, input, target):
        self.target = target
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        softmax_output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        self.softmax_output = softmax_output
        batch_size = input.shape[0]
        loss = -np.sum(np.log(softmax_output[np.arange(batch_size), target])) / batch_size
        return softmax_output, loss

    def backward(self):
        batch_size = self.target.shape[0]
        input_grad = self.softmax_output.copy()
        input_grad[np.arange(batch_size), self.target] -= 1
        input_grad /= batch_size
        return input_grad


