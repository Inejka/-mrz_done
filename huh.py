"""
////////////////////////////////////////////////////////////////////////////////////
// Лабораторная работа 4 по дисциплине МРЗвИС
// Выполнена студентом группы 9217023
// БГУИР Павлов Даниил Иванович
// Вариант 2 - модель сети Элмана с
// логарифмической функией активации
// 08.12.2021
// Использованные материалы:
// https://numpy.org/doc/stable/index.html - методические материалы по numpy
// https://www.learnpython.org/ - методические материалы по python
// https://ru.wikipedia.org/wiki/%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C_%D0%AD%D0%BB%D0%BC%D0%B0%D0%BD%D0%B0 - описание сети
"""
import numpy as np


def activation(x):
    return np.arcsinh(x)
    # return x


def d_activation(x):
    return 1.0 / np.sqrt(1 + x ** 2)
    # return 1


class net:
    def __init__(self, input_size, hidden_size, output_size, lr, epoch, different_examples):
        self.different_examples = different_examples
        self.scale_k = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.epoch = epoch
        n = 3

        self.layers = []
        self.layers.append(np.ones(self.input_size + 1 + self.output_size))
        self.layers.append(np.ones(hidden_size))
        self.layers.append(np.ones(output_size))
        self.weights = []
        for i in range(n - 1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i + 1].size)))

        self.dw = [0, ] * len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        self.dw = [0, ] * len(self.weights)
        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size, self.layers[i + 1].size))
            self.weights[i][...] = (2 * Z - 1) * 0.25

    def propagate_forward(self, data):
        data = data / self.scale_k
        self.layers[0][0:self.input_size] = data
        self.layers[0][self.input_size:-1] = self.layers[-1]
        for i in range(1, 3):
            self.layers[i][...] = activation(np.dot(self.layers[i - 1], self.weights[i - 1]))
        return self.layers[-1] * self.scale_k

    def propagate_backward(self, target, momentum=0.1):
        target = target / self.scale_k
        deltas = []
        error = target - self.layers[-1]
        delta = self.lr * error * d_activation(self.layers[-1])
        deltas.append(delta)
        for i in range(1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * d_activation(self.layers[i])
            deltas.insert(0, delta)
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += dw + momentum * self.dw[i]
            self.dw[i] = dw
        return (error ** 2).sum()

    def build_matrix(self, function, shift=0):
        self.meta_input_m = []
        self.meta_output = []
        for i in range(self.different_examples):
            self.meta_input_m.append(self.generate_train_matrix(function, i + shift))
            self.meta_output.append(
                self.generate_train_matrix_result(function, i + shift))

    def train(self):
        for j in range(self.epoch):
            self.input_m = self.meta_input_m[j % self.different_examples]
            self.output = self.meta_output[j % self.different_examples]
            self.layers[-1] = np.zeros(self.output_size)
            for i in range(self.input_size):

                self.propagate_forward(self.input_m[i])
                current_error = self.propagate_backward(self.output[i])
                if np.isnan(current_error):
                    self.lr /= 10.0
                    j = 0
                    i = 0
                    self.__init__(self.input_size, self.hidden_size, self.output_size, self.lr, self.epoch,
                                  self.different_examples)

                # print("error ", current_error)

    def generate_train_matrix(self, sequence_function, shift):
        train_matrix = np.zeros((self.hidden_size, self.input_size))
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                train_matrix[i][j] = sequence_function(j + i + shift)
        return train_matrix

    def generate_train_matrix_result(self, sequence_function, shift):
        result_matrix = np.zeros((self.hidden_size, self.output_size))
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                result_matrix[i][j] = sequence_function(j + i + self.input_size + shift)
        return result_matrix

    def show_test(self):
        self.layers[-1] = np.zeros(self.output_size)
        for i in range(self.different_examples):
            o = self.propagate_forward(self.meta_input_m[i][0])
            print('Sample %d: %s -> %s' % (i, self.meta_input_m[i][0], self.meta_output[i][0]))
            print('               Network output: " ', o)
            print()

    def get_errors(self):
        to_return = []
        self.layers[-1] = np.zeros(self.output_size)
        for i in range(self.different_examples):
            o = self.propagate_forward(self.meta_input_m[i][0])
            temp = self.meta_output[i][0]
            to_return.append((temp - o) / o)
        return to_return

    def get_mean_error(self):
        to_return = []
        self.layers[-1] = np.zeros(self.output_size)
        for i in range(self.different_examples):
            o = self.propagate_forward(self.meta_input_m[i][0])
            temp = self.meta_output[i][0]
            to_return.append((temp - o) / o)
        return np.abs(np.mean(to_return))
