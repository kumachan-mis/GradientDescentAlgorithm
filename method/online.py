# coding=utf-8
import numpy
import random


def _sigmoid_func(z):
    return 1.0 / (1.0 + numpy.exp(-z))


def _error_func(w, data):
    ret = 0.0
    for (x, y) in data:
        err = _sigmoid_func(numpy.inner(w, x)) - y
        ret += 0.5*err*err

    return ret


def _grad_single_error_func(w, pair):
    (x, y) = pair
    sigmoid = _sigmoid_func(numpy.inner(w, x))
    return (sigmoid - y)*sigmoid*(1 - sigmoid)*x


def run(
        initial_w, get_updated_param, data, limit_count,
        plot_func, plot_frequency, plot_setting
):
    prev_error = 10.0
    w = initial_w
    grad = _grad_single_error_func(w, data[0])
    error = _error_func(w, data)

    count = 0
    plot_func(count, numpy.log10(error), plot_setting)

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
        if count == limit_count:
            raise TimeoutError()
        if count % plot_frequency == 0:
            plot_func(count, numpy.log10(error), plot_setting)

        random.shuffle(data)

    error = _error_func(w, data)
    return error, w, count
