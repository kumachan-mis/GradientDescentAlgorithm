# coding=utf-8
import numpy
import random
from common import common


def _error_func(w, data):
    ret = 0.0
    for (x, y) in data:
        err = common.sigmoid_func(numpy.inner(w, x)) - y
        ret += 0.5*err*err

    return ret


def _grad_single_error_func(w, pair):
    (x, y) = pair
    sigmoid = common.sigmoid_func(numpy.inner(w, x))
    return (sigmoid - y)*sigmoid*(1 - sigmoid)*x


def run(
        initial_w, get_updated_param, data, count_limit,
        plot_func, plot_frequency, plot_setting
):
    prev_error = 10.0
    w = initial_w
    grad = _grad_single_error_func(w, data[0])
    error = _error_func(w, data)

    count = 0
    plot_func(count, error, plot_setting)

    while True:
        error_diff = prev_error - error
        if error_diff * error_diff < 1e-14:
            break

        for pair in data:
            prev_error = error
            w = get_updated_param(w, grad)
            grad = _grad_single_error_func(w, pair)
            error = _error_func(w, data)

        count += 1
        if count % plot_frequency == 0:
            plot_func(count, error, plot_setting)

        if count == count_limit:
            raise TimeoutError()

        random.shuffle(data)

    error = _error_func(w, data)
    return error, w, count
