# coding=utf-8
from common import common
import numpy
from method import batch, online


def get_updated_param(param, grad):
    epsilon = 1e-6
    weight = 0.95

    get_updated_param.r \
        = weight * get_updated_param.r + (1.0 - weight) * grad * grad
    v \
        = (numpy.sqrt(get_updated_param.s) + epsilon) \
        / (numpy.sqrt(get_updated_param.r) + epsilon) \
        * grad
    get_updated_param.s \
        = weight * get_updated_param.s + (1 - weight) * v * v

    return param - v


repeat_num = common.get_input(
    'repeat num', int, lambda value: value > 0
)

print('===== batch learning =====')
get_updated_param.r = numpy.array((0.0, 0.0))
get_updated_param.s = numpy.array((0.0, 0.0))
common.gradient_descent(repeat_num, batch.run, get_updated_param)

print('===== online learning =====')
get_updated_param.r = numpy.array((0.0, 0.0))
get_updated_param.s = numpy.array((0.0, 0.0))
common.gradient_descent(repeat_num, online.run, get_updated_param)
