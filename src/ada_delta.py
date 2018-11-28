# coding=utf-8
from common import common, batch, online
import numpy


def get_updated_param(param, grad):
    epsilon = 0.001
    weight = 0.9

    get_updated_param.g \
        = weight * get_updated_param.g + (1.0 - weight) * grad * grad
    v \
        = numpy.sqrt(get_updated_param.r + epsilon) \
        / numpy.sqrt(get_updated_param.g + epsilon) \
        * grad
    get_updated_param.r \
        = weight * get_updated_param.r + (1.0 - weight) * v * v

    return param - v


repeat_num = 300

print('===== batch learning =====')
get_updated_param.g = numpy.array((0.0, 0.0))
get_updated_param.r = numpy.array((0.0, 0.0))
common.gradient_descent(
    repeat_num, batch.run, get_updated_param, '../out/ada_delta_batch.png'
)

print('===== online learning =====')
get_updated_param.g = numpy.array((0.0, 0.0))
get_updated_param.r = numpy.array((0.0, 0.0))
common.gradient_descent(
    repeat_num, online.run, get_updated_param, '../out/ada_delta_online.png'
)
