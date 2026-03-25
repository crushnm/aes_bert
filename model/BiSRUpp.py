from sru import SRUpp
import torch
import torch.nn as nn

"""
BiSRUpp模型定义
"""


class BiSRUpp(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, proj_size=1024, num_layers=1, dropout=0.5, attn_dropout=0.1,
                 num_heads=12, bidirectional=True, layer_norm=True, normalize_after=True, highway_bias=-2.0,
                 attention_every_n_layers=1, rescale=True, nn_rnn_compatible_return=False,
                 proj_input_to_hidden_first=True, weight_c_init=1.0):
        super(BiSRUpp, self).__init__()
        self.BiSRUpp = SRUpp(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_layers=num_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            num_heads=num_heads,
            bidirectional=bidirectional,
            layer_norm=layer_norm,
            normalize_after=normalize_after,
            highway_bias=highway_bias,
            attention_every_n_layers=attention_every_n_layers,
            rescale=rescale,
            nn_rnn_compatible_return=nn_rnn_compatible_return,
            proj_input_to_hidden_first=proj_input_to_hidden_first,
            weight_c_init=weight_c_init)

    def forward(self, x):
        x = self.BiSRUpp(x)
        return x
