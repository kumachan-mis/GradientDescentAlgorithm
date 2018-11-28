# coding=utf-8
import numpy
from common import common


def _error_func(w, data):
    ret = 0.0
    for (x, y) in data:
        err = common.sigmoid_func(numpy.inner(w, x)) - y
        ret += 0.5*err*err

    return ret


def _grad_error_func(w, data):
    ret = numpy.array((0.0, 0.0))
    for (x, y) in data:
        sigmoid = common.sigmoid_func(numpy.inner(w, x))
        ret += (sigmoid - y)*sigmoid*(1 - sigmoid)*x

    return ret


def run(
        initial_w, get_updated_param, data, count_limit,
        plot_func, plot_frequency, plot_setting
):
    prev_error = 10.0
    w = initial_w
    grad = _grad_error_func(w, data)
    error = _error_func(w, data)

    count = 0
    plot_func(count, error, plot_setting)

    while True:
        error_diff = prev_error - error
        if error_diff * error_diff < 1e-14:
            break

        prev_error = error
        w = get_updated_param(w, grad)
        grad = _grad_error_func(w, data)
        error = _error_func(w, data)

        count += 1
        if count % plot_frequency == 0:
            plot_func(count, error, plot_setting)

        if count == count_limit:
            raise TimeoutError()

    error = _error_func(w, data)
    return error, w, count
