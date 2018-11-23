# coding=utf-8
import sys
import time
import random
import numpy
from matplotlib import pyplot


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


'''
􏰢(1.20,0), 􏰨􏰢(1.25,0􏰨􏰢), (1.30,0), (􏰨􏰢1.35􏰙,0), (􏰨􏰢1.40􏰙,1), (􏰨􏰢1.45,􏰙0􏰨),
(􏰢1.50,1), (􏰨􏰢1.55􏰙,0􏰨􏰢), (1.60􏰙,1􏰨􏰢), (1.65􏰙,1), (􏰨􏰢1.70􏰙,1), (􏰨􏰢1.75􏰙,1􏰨)
'''


def _get_data():
    paired_data = (
        (1.20, 0.0), (1.25, 0.0), (1.30, 0.0),
        (1.35, 0.0), (1.40, 1.0), (1.45, 0.0),
        (1.50, 1.0), (1.55, 0.0), (1.60, 1.0),
        (1.65, 1.0), (1.70, 1.0), (1.75, 1.0)
    )
    return [
        (numpy.array((x, 1.0)), y) for (x, y) in paired_data
    ]


def gradient_descent(repeat_num, gradient_descent_func, get_updated_param):
    numpy.set_printoptions(formatter={'float': '{:.6f}'.format})
    min_error = sys.float_info.max
    ideal_param = None
    ideal_count = None

    start_sec = time.process_time()

    for index in range(0, repeat_num):
        initial_param = numpy.array(
            (random.uniform(-50.0, 50.0), random.uniform(-50.0, 50.0))
        )
        error, param, count = gradient_descent_func(
            initial_param, get_updated_param, _get_data(),
            pyplot.plot, 100, 'bo'
        )
        print(
            '[{:03}] error:{:.3f}, param:{}, count: {}'
            .format(index+1, error, param, count)
        )
        if error < min_error:
            min_error = error
            ideal_param = param
            ideal_count = count

    end_sec = time.process_time()

    print(
        'minimum error:{:.6f}, ideal parameter:{}, time per try:{:.6f}'
        .format(min_error, ideal_param, (end_sec - start_sec) / repeat_num)
    )
    pyplot.plot(ideal_count, numpy.log10(min_error), 'ro')
    pyplot.show()
