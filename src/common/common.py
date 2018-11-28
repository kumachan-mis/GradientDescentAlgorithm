# coding=utf-8
import time
import random
import numpy
from matplotlib import pyplot


def sigmoid_func(z):
    return 1.0 / (1.0 + numpy.exp(-z))


def get_input(value_name, type, check_condition):
    while True:
        try:
            value = type(input('{}: '.format(value_name)))
            if not check_condition(value):
                raise ValueError()
            break
        except ValueError:
            print(ValueError)

    return value


paired_data = (
        (1.20, 0.0), (1.25, 0.0), (1.30, 0.0),
        (1.35, 0.0), (1.40, 1.0), (1.45, 0.0),
        (1.50, 1.0), (1.55, 0.0), (1.60, 1.0),
        (1.65, 1.0), (1.70, 1.0), (1.75, 1.0)
    )


def _get_data():
    return [
        (numpy.array((x, 1.0)), y) for (x, y) in paired_data
    ]


def gradient_descent(
        repeat_num, gradient_descent_func, get_updated_param, graph_name
):
    figure, (error_plot, result_plot) = pyplot.subplots(nrows=2)
    numpy.set_printoptions(formatter={'float': '{:.6f}'.format})

    error_plot.set_xlabel('iteration count')
    error_plot.set_ylabel('E(w)')
    result_plot.grid(True)

    count_limit = 100000
    min_error = 10.0
    ideal_param = None
    ideal_count = None

    start_sec = time.process_time()

    for index in range(0, repeat_num):
        try:
            initial_param = numpy.array(
                (random.uniform(-50.0, 50.0), random.uniform(-50.0, 50.0))
            )
            error, param, count = gradient_descent_func(
                initial_param, get_updated_param, _get_data(), count_limit,
                error_plot.plot, 100, 'bo'
            )
            print(
                '[{:03}] error:{:.6f}, param:{}, count: {}'
                .format(index+1, error, param, count)
            )
            if error < min_error:
                min_error = error
                ideal_param = param
                ideal_count = count
        except TimeoutError:
            print(
                '[{:03}] count is over {}. give up.'
                .format(index+1, count_limit)
            )

    end_sec = time.process_time()

    print(
        ('========== result ==========\n'
         'minimum error: {:.6f}\n'
         'ideal param:   {}\n'
         'time per trial:{:.6f}[s]\n')
        .format(min_error, ideal_param, (end_sec - start_sec) / repeat_num)
    )
    error_plot.plot(ideal_count, min_error, 'ro')
    error_plot.set_xlim(0, count_limit)

    result_plot.set_xlabel('x')
    result_plot.set_ylabel('y')
    result_plot.grid(True)
    x = numpy.linspace(1, 2, 1000)
    y = sigmoid_func(ideal_param[0]*x + ideal_param[1])
    result_plot.plot(x, y)
    for (x, y) in paired_data:
        result_plot.plot(x, y, 'ro')
    result_plot.set_xlim(1, 2)

    figure.savefig(graph_name, figsize=(20, 30))
