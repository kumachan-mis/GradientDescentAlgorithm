# coding=utf-8
from common import common
from method import batch, online


def get_updated_param(param, grad):
    return param - get_updated_param.learning_rate*grad


repeat_num = common.get_input(
    'repeat num', int, lambda value: value > 0
)

print('===== batch learning =====')
get_updated_param.learning_rate = common.get_input(
    'learning rate', float, lambda value: value > 0.0
)
common.gradient_descent(repeat_num, batch.run, get_updated_param)

print('===== online learning =====')
get_updated_param.learning_rate = common.get_input(
    'learning rate', float, lambda value: value > 0.0
)
common.gradient_descent(repeat_num, online.run, get_updated_param)
