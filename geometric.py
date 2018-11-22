# coding=utf-8
from common import common
from method import batch, online


def get_updated_param(param, grad):
    get_updated_param.count += 1

    if get_updated_param.count == 10:
        get_updated_param.count = 0
        get_updated_param.learning_rate *= get_updated_param.geometric_ratio

    return param - get_updated_param.learning_rate*grad


repeat_num = common.get_input(
    'repeat num', int, lambda value: value > 0
)

print('===== batch learning =====')
get_updated_param.count = 0
get_updated_param.learning_rate = common.get_input(
    'learning rate', float, lambda value: value > 0.0
)
get_updated_param.geometric_ratio = common.get_input(
    'geometric ratio', float, lambda value: 0.0 < value <= 1.0
)
common.gradient_descent(repeat_num, batch.run, get_updated_param)

print('===== online learning =====')
get_updated_param.count = 0
get_updated_param.learning_rate = common.get_input(
    'learning rate', float, lambda value: value > 0.0
)
get_updated_param.geometric_ratio = common.get_input(
    'geometric ratio', float, lambda value: 0.0 < value <= 1.0
)
common.gradient_descent(repeat_num, online.run, get_updated_param)
