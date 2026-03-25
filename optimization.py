# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_


def warmup_cosine(x, warmup=0.002):  # cos 余弦
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))  # warmup之后 cos(0.1*pi)到cos(pi) 即cos(0.1 * pi)到 0 再到 -1


# warmup = 0.1 作为一个比例 即当前步数与总次数的比例 即前10%的步数的学习率会比10步之后低，10步之后学习率恢复正常
def warmup_constant(x, warmup=0.002):  # 固定
    if x < warmup:
        return x / warmup
    return 1.0  # warmup之后固定lr


def warmup_linear(x, warmup=0.002):  # 线性
    if x < warmup:
        return x / warmup
    return 1.0 - x  # warmup之后线性减小lr


# dict存放对应的函数名称
SCHEDULES = {
    'warmup_cosine': warmup_cosine,  # cos
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,  # 线性
}


# Optimizer：继承语此类
class BERTAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion比例 of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        # 判断传入数据的有效性
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,  # 优化选项
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(BERTAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    # warmup = 0.1 假如总次数是100次  前10步 schedule_fct 返回的是0.1 0.2 0.3 ... 0.9
                    # 11步之后返回 0.89 0.88 0.87 0.86 ... 0.00
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

        # 梯度下降

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 闭包运算 针对需要多次计算的优化算法
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:  # 迭代params内的全部p
                if p.grad is None:  # 需要loss的反向传播计算才会生成grad
                    continue
                grad = p.grad.data  # 具体的梯度值
                if grad.is_sparse:  # 如果是稀疏梯度
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]  # 参数的缓存，如momentum的缓存；
                # State initialization
                if len(state) == 0:
                    state['step'] = 0  # 步数为0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)  # p.data:具体的参数值tensor
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']  # 更新next_m 等于更新state['next_m']
                beta1, beta2 = group['b1'], group['b2']  # 衰减率 0.9 0.999

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    # NN参数，最大梯度范数，范数类型=2(默认)
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)  # 先乘后加 梯度的指数移动平均
                # next_v.mul_(beta2)+(1 - beta2)×grad×grad
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)  # 逐元素计算 梯度的平方的指数移动平均
                update = next_m / (next_v.sqrt() + group['e'])  # 梯度的指数移动平均 / (梯度的平方的指数移动平均开根号 + 很小的数)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam, 直接将将L2或者weight_decay加到loss是不对的
                # since that will interact with the m and v parameters in strange ways. 因为会与m和v发生相互作用
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                # 动态lr
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)  # 加一个负数 梯度下降 更新参数

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # bias_correction1 = 1 - beta1 ** state['step'] 偏差修正
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


# 初始化bert优化器
def init_bert_adam_optimizer(model, training_data_len, epoch, batch_size,
                             gradient_accumulation_steps, init_lr, warmup_proportion):
    no_decay = ["bias", "gamma", "beta"]
    optimizer_parameters = [
        {"params": [p for name, p in model.named_parameters() \
                    if name not in no_decay], "weight_decay_rate": 0.01},
        {"params": [p for name, p in model.named_parameters() \
                    if name in no_decay], "weight_decay_rate": 0.0}
    ]

    # 总的梯度下降数 用于计算学习率的变化
    # 样本数量 / 批处理大小 / 梯度累积步数(常为1)  --> 一个epoch需要多少次梯度下降
    # 样本数量 / 批处理大小 / 梯度累积步数 * epoch

    num_train_steps = int(training_data_len / batch_size / \
                          gradient_accumulation_steps * epoch)
    # 可迭代的torch.tensor
    # 字典(dict), 字典中的键(key)为"params"那一项的值(value)必须为可迭代的torch.tensor
    # 需要强调的是如果有多个dict要传入, 则要放入一个列表容器中.
    optimizer = BERTAdam(optimizer_parameters,
                         lr=init_lr,
                         warmup=warmup_proportion,
                         t_total=num_train_steps)
    return optimizer
