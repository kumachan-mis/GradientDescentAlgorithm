# coding=utf-8
from common import common
import numpy
from method import batch, online


def get_updated_param(param, grad):
    smallness = 0.001
    weight = 0.9

    get_updated_param.attenuation\
        = weight*get_updated_param.attenuation + (1.0 - weight)*grad*grad
    learning_rate\
        = get_updated_param.learning_rate_param\
        / numpy.sqrt(get_updated_param.attenuation + smallness)

    return param - learning_rate*grad


repeat_num = common.get_input(
    'repeat num', int, lambda value: value > 0
)

print('===== batch learning =====')
get_updated_param.attenuation = numpy.array((0.0, 0.0))
get_updated_param.learning_rate_param = common.get_input(
    'learning rate param', float, lambda value: value > 0.0
)
common.gradient_descent(repeat_num, batch.run, get_updated_param)

print('===== online learning =====')
get_updated_param.attenuation = numpy.array((0.0, 0.0))
get_updated_param.learning_rate_param = common.get_input(
    'learning rate param', float, lambda value: value > 0.0
)
common.gradient_descent(repeat_num, online.run, get_updated_param)
