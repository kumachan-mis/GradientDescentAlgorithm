# coding=utf-8
from common import common, batch, online


def get_updated_param(param, grad):
    return param - get_updated_param.learning_rate*grad


repeat_num = 300

print('===== batch learning =====')
get_updated_param.learning_rate = 0.1
common.gradient_descent(
    repeat_num, batch.run, get_updated_param, '../out/const_batch.png'
)

print('===== online learning =====')
get_updated_param.learning_rate = 0.2
common.gradient_descent(
    repeat_num, online.run, get_updated_param, '../out/const_online.png'
)
