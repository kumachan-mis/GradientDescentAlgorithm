# coding=utf-8
from common import common, batch, online
import numpy


def get_updated_param(param, grad):
    m_weight = 0.9
    g_weight = 0.99
    epsilon = 0.001

    get_updated_param.m \
        = m_weight * get_updated_param.m + (1 - m_weight) * grad
    get_updated_param.g \
        = g_weight * get_updated_param.g + (1 - g_weight) * grad * grad

    get_updated_param.m_weight_pow *= m_weight
    get_updated_param.g_weight_pow *= g_weight

    m_hat = get_updated_param.m / (1 - get_updated_param.m_weight_pow)
    g_hat = get_updated_param.g / (1 - get_updated_param.g_weight_pow)

    return param\
        - get_updated_param.learning_rate_param\
        * m_hat / (numpy.sqrt(g_hat) + epsilon)


repeat_num = 300

print('===== batch learning =====')
get_updated_param.m_weight_pow = 1.0
get_updated_param.m = numpy.array((0.0, 0.0))
get_updated_param.g_weight_pow = 1.0
get_updated_param.g = numpy.array((0.0, 0.0))
get_updated_param.learning_rate_param = 0.2
common.gradient_descent(
    repeat_num, batch.run, get_updated_param, '../out/adam_batch.png'
)

print('===== online learning =====')
get_updated_param.m_weight_pow = 1.0
get_updated_param.m = numpy.array((0.0, 0.0))
get_updated_param.g_weight_pow = 1.0
get_updated_param.g = numpy.array((0.0, 0.0))
get_updated_param.learning_rate_param = 0.02
common.gradient_descent(
    repeat_num, online.run, get_updated_param, '../out/adam_online.png'
)
