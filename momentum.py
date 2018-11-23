# coding=utf-8
from common import common
import numpy
from method import batch, online


def get_updated_param(param, grad):
    attenuation_rate = 0.9
    updated_param\
        = param + attenuation_rate * get_updated_param.param_diff\
        - get_updated_param.learning_rate*grad
    get_updated_param.param_diff\
        = updated_param - param
    return updated_param


repeat_num = common.get_input(
    'repeat num', int, lambda value: value > 0
)

print('===== batch learning =====')
get_updated_param.param_diff = numpy.array((0.0, 0.0))
get_updated_param.learning_rate = common.get_input(
    'learning rate', float, lambda value: value > 0.0
)
common.gradient_descent(repeat_num, batch.run, get_updated_param)

print('===== online learning =====')
get_updated_param.param_diff = numpy.array((0.0, 0.0))
get_updated_param.learning_rate = common.get_input(
    'learning rate', float, lambda value: value > 0.0
)
common.gradient_descent(repeat_num, online.run, get_updated_param)
