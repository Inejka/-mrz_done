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
import matplotlib.pyplot as plt
from huh import net
import numpy as np


def func1(x):
    return x


def func2(x):
    return x ** 2


def func3(x):
    temp = [0, 1, 1, 2]
    while (len(temp) <= x):
        temp.append(temp[len(temp) - 1] + temp[len(temp) - 2])
    return temp[x]


func = {1: func1, 2: func2, 3: func3}
func_names = {1: "y=x", 2: "y=x^2", 3: "y=fibb(x)"}
func_color = {1: 'r', 2: 'g', 3: 'b'}


def first_plot():
    plt_max_x = -1
    plt_max_y = -1
    for j in range(1, 4):
        x_plot = []
        y_plot = []
        for i in range(2, 10):
            network = net(1 + i, 20, 1, 1.0, 1000, 10)
            network.build_matrix(func[j])
            network.train()
            network.build_matrix(func[j], 10)
            x_plot.append(1 + i)
            y_plot.append(network.get_mean_error())
        plt.plot(x_plot, y_plot, func_color[j], label=func_names[j])
        x_plot.append(plt_max_x)
        y_plot.append(plt_max_y)
        plt_max_x = np.max(x_plot)
        plt_max_y = np.max(y_plot)
    plt.xlabel('Размер скользящего окна')
    plt.ylabel('Средний процент ошибки')
    plt.xlim([0, plt_max_x * 1.05])
    plt.ylim([0, plt_max_y * 1.1])
    plt.legend()
    plt.savefig('first.png')
    plt.show()


def second_plot():
    plt_max_x = -1
    plt_max_y = -1
    for j in range(1, 4):
        x_plot = []
        y_plot = []
        for i in range(-5, 5, 1):
            network = net(4, 20 + i, 1, 1.0, 10000, 10)
            network.build_matrix(func[j])
            network.train()
            network.build_matrix(func[j], 10)
            x_plot.append(20 + i)
            y_plot.append(network.get_mean_error())
        plt.plot(x_plot, y_plot, func_color[j], label=func_names[j])
        x_plot.append(plt_max_x)
        y_plot.append(plt_max_y)
        plt_max_x = np.max(x_plot)
        plt_max_y = np.max(y_plot)
    plt.xlabel('Количество контекстных нейронов')
    plt.ylabel('Средний процент ошибки')
    plt.xlim([0, plt_max_x * 1.05])
    plt.ylim([0, plt_max_y * 1.1])
    plt.legend()
    plt.savefig('second.png')
    plt.show()


def third_plot():
    plt_max_x = -1
    plt_max_y = -1
    for j in range(1, 4):
        x_plot = []
        y_plot = []
        for i in range(0, 4000, 500):
            network = net(4, 20, 1, 1.0, 1000 + i, 10)
            network.build_matrix(func[j])
            network.train()
            network.build_matrix(func[j], 10)
            x_plot.append(1000 + i)
            y_plot.append(network.get_mean_error())
        plt.plot(x_plot, y_plot, func_color[j], label=func_names[j])
        x_plot.append(plt_max_x)
        y_plot.append(plt_max_y)
        plt_max_x = np.max(x_plot)
        plt_max_y = np.max(y_plot)
    plt.xlabel('Количество эпох')
    plt.ylabel('Средний процент ошибки')
    plt.xlim([0, plt_max_x * 1.05])
    plt.ylim([0, plt_max_y * 1.1])
    plt.legend()
    plt.savefig('third.png')
    plt.show()
