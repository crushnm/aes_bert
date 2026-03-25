import torch
import torch.nn as nn

"""
输出层
"""


class Output(nn.Module):
    def __init__(self, hidden_size=1536, linear_high=256, num_classes=2):
        super(Output, self).__init__()
        self.linear_high = nn.Linear(2 * hidden_size, linear_high)
        self.linear_num_classes = nn.Linear(linear_high, num_classes)
        self.ce_loss_fct = nn.CrossEntropyLoss()  # Log_softmax() + NLLLoss() 组成

    def forward(self, x, labels):
        logits_high = self.linear_high(x)
        logits_num_classes = self.linear_num_classes(logits_high)
        loss = self.ce_loss_fct(logits_num_classes, labels)  # 计算loss
        return loss, logits_num_classes
