import json
import logging
import torch
from transformers import BertTokenizer
import numpy as np
np.set_printoptions(suppress=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

from pathlib import Path

model_name = Path("bert-base-uncased").resolve()
class Model_Predictor:
    def __init__(self, max_seq_len, path):
        self.max_seq_len = max_seq_len
        self.model = self.init_model(path)

    def init_model(self, path):
        device = torch.device("cpu")  # cpu上加载
        model = torch.load(path, map_location=device)
        model.eval()  # 验证模式s
        return model

    # 处理输入文本
    def preproc_text(self, text):
        # tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')  # 加载分词器
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)  # 加载分词器
        line = text
        row = line.strip("\n").split("\t")
        if len(row) != 2:
            raise RuntimeError("Data is illegal: " + line)
        labels = torch.LongTensor([int(row[0])])
        sen_code = tokenizer(row[1], return_tensors='pt', padding='max_length',
                             truncation=True, max_length=self.max_seq_len)  # max_length:truncation/padding
        tokens = torch.LongTensor(sen_code['input_ids'][0])
        token_type_ids = torch.LongTensor(sen_code['token_type_ids'][0])
        attention_mask = torch.LongTensor(sen_code['attention_mask'][0])
        return tokens, token_type_ids, attention_mask, labels

    # 进行预测
    def predict(self, text):
        # 将文本处理成bert需要的输入样式
        tokens, segment_ids, attn_masks, labels = self.preproc_text(text)
        with torch.no_grad():  # 不需要计算梯度
            # 输入x(token), attention_mask, token_type_ids
            loss, logits = self.model(tokens.reshape([1, -1]), attn_masks.reshape([1, -1]),
                                      segment_ids.reshape([1, -1]), labels)
            top_probs, top_index = logits.topk(1)
            # 预测的概率、最大的概率对应的index
            return logits.cpu().numpy()[0], top_index.cpu().numpy()[0]


def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


if __name__ == '__main__':
    saved_model_path = '/Users/sugarmei/PycharmProjects/untitled2/my_model.pkl' # 修改路径
    max_seq_len = 510
    p = Model_Predictor(max_seq_len=max_seq_len, path=saved_model_path)
    text = '0	挺没有意思的一本书，总觉得韩寒这人吧，老这样写小说真的挺没劲的。'
    # 预测标签： [0] 概率分布： [[9.9998164e-01 1.8343979e-05]] 对应文本： 0	挺没有意思的一本书，总觉得韩寒这人吧，老这样写小说真的挺没劲的。
    # text = '0	没意思'
    # text = '1	好疯狂，居然有点羡慕.......'
    # 预测标签： [1] 概率分布： [[2.167351e-05 9.999783e-01]] 对应文本： 1	好疯狂，居然有点羡慕.......
    # text = '0	太假了。喜欢牧流冰。'
    prob, pred = p.predict(text)
    print("预测标签：", pred, "概率分布：", softmax(prob, -1), "对应文本：", text)
