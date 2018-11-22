# coding=utf-8
import numpy


def _sigmoid_func(z):
    return 1.0 / (1.0 + numpy.exp(-z))


def _error_func(w, data):
    ret = 0.0
    for (x, y) in data:
        err = _sigmoid_func(numpy.inner(w, x)) - y
        ret += 0.5*err*err

    return ret


def _grad_error_func(w, data):
    ret = numpy.array((0.0, 0.0))
    for (x, y) in data:
        sigmoid = _sigmoid_func(numpy.inner(w, x))
        ret += (sigmoid - y)*sigmoid*(1 - sigmoid)*x

    return ret


def run(
        initial_w, get_updated_param, data, limit_count,
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
        if error_diff*error_diff < 1e-12:
            break

        prev_error = error
        w = get_updated_param(w, grad)
        grad = _grad_error_func(w, data)
        error = _error_func(w, data)

        count += 1
        if count == limit_count:
            raise TimeoutError()
        if count % plot_frequency == 0:
            plot_func(count, error, plot_setting)

    error = _error_func(w, data)
    return error, w, count