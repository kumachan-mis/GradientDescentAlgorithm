# coding=utf-8
from common import common
import numpy
from method import batch, online

epsilon = 1e-8


def get_updated_param(param, grad):
    get_updated_param.r += grad * grad
    learning_late \
        = get_updated_param.learning_rate_param \
        / numpy.sqrt(get_updated_param.r)

    return param - learning_late * grad


repeat_num = common.get_input(
    'repeat num', int, lambda value: value > 0
)

print('===== batch learning =====')
get_updated_param.r = numpy.array((epsilon, epsilon))
get_updated_param.learning_rate_param = common.get_input(
    'learning rate param', float, lambda value: value > 0.0
)
common.gradient_descent(repeat_num, batch.run, get_updated_param)

print('===== online learning =====')
get_updated_param.r = numpy.array((epsilon, epsilon))
get_updated_param.learning_rate_param = common.get_input(
    'learning rate param', float, lambda value: value > 0.0
)
common.gradient_descent(repeat_num, online.run, get_updated_param)
