import numpy as np
import torch
from torch import nn
from torch.nn import init


class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)  # 线性层mk
        self.mv = nn.Linear(S, d_model, bias=False)  # 线性层mv
        self.softmax = nn.Softmax(dim=1)  # dim = 1
        self.init_weights()  # 随机初始化

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        """
        :param queries: 输入特征 [bs, n, d_model]
        :return:
        """
        # [d_model, S]*[bs, n, d_model]*->[bs, n, S] S为超参数
        # [64, 200, 768]->linear(768, 64) = [64, 200, 64]
        attn = self.mk(queries)  # bs,n,S 类似于计算相似度
        # double-normalization
        # 在dim = 1上进行归一化 列归一化
        # tensor([[[0.3871, 0.7303, 0.3730, 0.2802],
        #          [0.0565, 0.2628, 0.3335, 0.3268],
        #          [0.5565, 0.0069, 0.2936, 0.3930]],
        # 理解:bs表示有多少个句子、n表示一个句子的长度 即字的个数、d_model表示一个字由一个维度为d_model的向量表示
        #
        attn = self.softmax(attn)  # bs,n,S
        # 求和：在dim = 2的维度上进行累加 每个表示字的向量自身进行归一化
        # 分子/分母
        # 行归一化
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        # 再一次线性转换 类似于根据注意力得分与value相乘得到注意力输出结果
        out = self.mv(attn)  # bs,n,d_model
        return out


if __name__ == '__main__':
    input = torch.randn(64, 200, 768)
    ea = ExternalAttention(d_model=768, S=32)
    output = ea(input)
    print(output.shape)
