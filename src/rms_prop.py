# coding=utf-8
from common import common, batch, online
import numpy


def get_updated_param(param, grad):
    epsilon = 0.001
    weight = 0.90

    get_updated_param.g \
        = weight * get_updated_param.g + (1.0 - weight) * grad * grad
    learning_rate \
        = get_updated_param.learning_rate_param \
        / numpy.sqrt(get_updated_param.g + epsilon)

    return param - learning_rate*grad


repeat_num = 300

print('===== batch learning =====')
get_updated_param.g = numpy.array((0.0, 0.0))
get_updated_param.learning_rate_param = 0.05
common.gradient_descent(
    repeat_num, batch.run, get_updated_param, '../out/rms_prop_batch.png'
)

print('===== online learning =====')
get_updated_param.g = numpy.array((0.0, 0.0))
get_updated_param.learning_rate_param = 0.02
common.gradient_descent(
    repeat_num, online.run, get_updated_param, '../out/rms_prop_online.png'
)
