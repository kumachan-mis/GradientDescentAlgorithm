# coding=utf-8
from common import common, batch, online
import numpy


def get_updated_param(param, grad):
    epsilon = 0.001
    get_updated_param.g += grad * grad
    learning_late \
        = get_updated_param.learning_rate_param \
        / numpy.sqrt(get_updated_param.g + epsilon)

    return param - learning_late * grad


repeat_num = 300

print('===== batch learning =====')
get_updated_param.g = numpy.array((0.0, 0.0))
get_updated_param.learning_rate_param = 0.2
common.gradient_descent(
    repeat_num, batch.run, get_updated_param, '../out/ada_grad_batch.png'
)

print('===== online learning =====')
get_updated_param.g = numpy.array((0.0, 0.0))
get_updated_param.learning_rate_param = 0.01
common.gradient_descent(
    repeat_num, online.run, get_updated_param, '../out/ada_grad_online.png'
)
