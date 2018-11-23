# coding=utf-8
from common import common
import numpy
from method import batch, online


def get_updated_param(param, grad):
    update_coef = 0.001
    m_weight = 0.9
    v_weight = 0.999
    epsilon = 1e-8

    get_updated_param.m \
        = m_weight * get_updated_param.m + (1 - m_weight) * grad
    get_updated_param.v \
        = v_weight * get_updated_param.v + (1 - v_weight) * grad * grad

    get_updated_param.m_weight_pow *= m_weight
    get_updated_param.v_weight_pow *= v_weight

    m_hat = get_updated_param.m / (1 - get_updated_param.m_weight_pow)
    v_hat = get_updated_param.v / (1 - get_updated_param.v_weight_pow)

    return param - update_coef * m_hat / (numpy.sqrt(v_hat) + epsilon)


repeat_num = common.get_input(
    'repeat num', int, lambda value: value > 0
)

print('===== batch learning =====')
get_updated_param.m_weight_pow = 1.0
get_updated_param.m = numpy.array((0.0, 0.0))
get_updated_param.v_weight_pow = 1.0
get_updated_param.v = numpy.array((0.0, 0.0))
common.gradient_descent(repeat_num, batch.run, get_updated_param)

print('===== online learning =====')
get_updated_param.m_weight_pow = 1.0
get_updated_param.m = numpy.array((0.0, 0.0))
get_updated_param.v_weight_pow = 1.0
get_updated_param.v = numpy.array((0.0, 0.0))
common.gradient_descent(repeat_num, online.run, get_updated_param)
