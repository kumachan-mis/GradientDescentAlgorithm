# coding=utf-8
from common import common, batch, online
import numpy


def get_updated_param(param, grad):
    attenuation_rate = 0.9
    get_updated_param.v\
        = attenuation_rate * get_updated_param.v\
        + get_updated_param.learning_rate * grad
    return param - get_updated_param.v


repeat_num = 300

print('===== batch learning =====')
get_updated_param.v = numpy.array((0.0, 0.0))
get_updated_param.learning_rate = 0.02
common.gradient_descent(
    repeat_num, batch.run, get_updated_param, '../out/momentum_batch.png'
)

print('===== online learning =====')
get_updated_param.v = numpy.array((0.0, 0.0))
get_updated_param.learning_rate = 0.2
common.gradient_descent(
    repeat_num, online.run, get_updated_param, '../out/momentum_online.png'
)
