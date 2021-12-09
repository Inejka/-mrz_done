import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class NeuroMath:
    learning_set_size = 30
    seq = []
    steps_count = 5
    model = 0
    epoch_count = 500

    def __init__(self, training_set_size=30, window_size=3, sequence='degree', epoch_count=500):
        self.features_count = 1
        self.learning_set_size = training_set_size
        self.steps_count = window_size
        self.epoch_count = epoch_count
        if sequence == 'degree':
            self.degree_filling()
        elif sequence == 'fib':
            self.fibonacci_filling()
        elif sequence == 'one_zero':
            self.one_zero_filling()
        elif sequence == 'mod':
            self.mod_filling()
        elif sequence == 'arithmetic':
            self.arithmetic_filling()

    def one_zero_filling(self):
        for i in range(0, self.learning_set_size):
            self.seq.append(i % 2)

    def fibonacci_filling(self):
        for i in range(self.learning_set_size):
            self.seq.append(self.fib_recursion(i))

    def fib_recursion(self, n):
        if n <= 1:
            return n
        else:
            return self.fib_recursion(n - 1) + self.fib_recursion(n - 2)

    def degree_filling(self):
        for i in range(0, self.learning_set_size):
            self.seq.append(2 ** i)

    def mod_filling(self):
        for i in range(0, self.learning_set_size):
            self.seq.append(i % 10)

    def arithmetic_filling(self):
        for i in range(0, self.learning_set_size):
            self.seq.append(i)

    def split_sequence(self):
        x = []
        y = []

        for i in range(len(self.seq)):
            last_index = i + self.steps_count
            if last_index > len(self.seq) - 1:
                break
            seq_x, seq_y = self.seq[i:last_index], self.seq[last_index]
            x.append(seq_x)
            y.append(seq_y)
            pass
        x = np.array(x)
        y = np.array(y)
        return x, y

        pass

    def training(self):
        # print(self.seq)
        x, y = self.split_sequence()
        # print(x)
        # print(y)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        x = x.reshape((x.shape[0], x.shape[1], self.features_count))
        self.model = tf.keras.Sequential()
        self.model.add(layers.LSTM(50, activation='relu', input_shape=(self.steps_count, self.features_count)))
        self.model.add(layers.Dense(1))
        # model.layers
        self.model.summary()

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError())
        self.model.fit(x, y, epochs=self.epoch_count, verbose=0)

    def guessing(self, test_data):
        test_data = test_data.reshape((1, self.steps_count, self.features_count))
        predict_next_number = self.model.predict(test_data, verbose=0)
        return predict_next_number
