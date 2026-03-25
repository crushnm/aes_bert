"""

返回一个DataSet给DataLoader迭代mini-batch
"""
import torch
from transformers import BertTokenizer
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize

# 下载nltk必要的数据包
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

model_name = Path("bert-base-uncased").resolve()
# model_name = 'hfl/chinese-roberta-wwm-ext'  # roberta中文预训练模型
# model_name = 'google-bert/bert-base-uncased'
# model_name = 'google-bert/bert-base-uncased'
class Prepare_DataSet(torch.utils.data.Dataset):
    def __init__(self, max_len=512,
                 data_file='C:/Users/HuaFei/Desktop/task/aes_bert-bisru++-attention-作文评分asap/data/asap-aes/training_set_rel3.xlsx', # 路径记得修改
                 subset=4, tmp=1, shallow_feature_file=None):
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(model_name)  # 加载分词器
        self.subset = subset  # 当前训练那个子集合
        self.tmp = tmp # 修正标签范围
        self.shallow_features = None
        
        if data_file != None:
            # 读取全部数据
            self.raw_datas = pd.read_excel(data_file)
            print(len(self.raw_datas))
            print(self.raw_datas)
            # 根据self.subset筛选当前设置的子集合测试和训练样本
            # essay_set 所属的子集
            # domain1_score 评分
            # essay 具体作文内容
            self.subset_data = self.raw_datas.loc[self.raw_datas['essay_set'] == self.subset]
            
            self.essays = self.subset_data['essay']
            self.scores = self.subset_data['domain1_score']
            
            # 加载浅层特征（如果提供）
            if shallow_feature_file is not None:
                import pickle
                # 根据subset加载对应的浅层特征文件
                subset_feature_file = f"{shallow_feature_file.replace('.pkl', '')}_subset_{self.subset}.pkl"
                print(f"加载浅层特征: {subset_feature_file}")
                with open(subset_feature_file, 'rb') as f:
                    feature_data = pickle.load(f)
                    subset_features = feature_data['features']
                    self.shallow_features = torch.tensor(subset_features, dtype=torch.float32)
                    print(f"浅层特征形状: {self.shallow_features.shape}")

            # print("len(self.essays):", len(self.essays))

    def __len__(self):
        return len(self.subset_data)

    def __getitem__(self, item):
        # print('item:',item)
        # print("len(self.essays):", len(self.essays))
        essay = self.essays.iloc[item]
        score = self.scores.iloc[item]
        # print("essay:", essay)
        # print("score:", score)
        # 使用nltk的sent_tokenize进行句子切分
        sentences = sent_tokenize(essay)
        
        # 构建整体输入格式 '[CLS] ' + 句子 + ' [SEP] '
        doc_str = ''
        doc_str_tmp = ''
        for r in sentences:
            doc_str_tmp += '[CLS] ' + r + ' [SEP] '
            # 做判断 如果长度大于了512 则不再添加句子
            if len(doc_str_tmp) >= self.max_len:
                break  # 退出循环 doc_str仍然为未更新的字符串结果
            else:
                doc_str = doc_str_tmp  # 更新到doc_str
        # print('doc_str:',doc_str)
        # 获取id显示
        indexed_tokens = self.tokenizer.encode(doc_str.strip(), add_special_tokens=False)

        # print("indexed_tokens:", indexed_tokens)
        # print("len(indexed_tokens):", len(indexed_tokens))

        # 记录 句向量 的位置 即是[CLS]的位置 通过判断id是否为101
        indices = [i for i, x in enumerate(indexed_tokens) if x == 101]

        # print("indices:",indices)

        # 分句向量
        segments_ids = []
        segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
        segments_indices.append(len(indexed_tokens))
        for i in range(len(segments_indices) - 1):
            # len(segments_indices)-1=10  共10句话  区分两个句子两个句子，
            # 请将第一个句子中的每个单词加上“[SEP]”token赋值为0，第二个句子中的所有token赋值为1。
            semgent_len = segments_indices[i + 1] - segments_indices[i]
            if i % 2 == 0:
                segments_ids.extend([0] * semgent_len)
            else:
                segments_ids.extend([1] * semgent_len)

        # print("segments_ids:",segments_ids)
        # print("len(segments_ids):", len(segments_ids))

        sen_code = self.tokenizer(essay, return_tensors='pt', padding='max_length',
                                  truncation=True, max_length=self.max_len)  # max_length:truncation/padding
        tokens = torch.LongTensor(sen_code['input_ids'][0])
        token_type_ids = torch.LongTensor(sen_code['token_type_ids'][0])
        attention_mask = torch.LongTensor(sen_code['attention_mask'][0])

        # 进行长度补全 或者 截断
        indexed_tokens = pad_list(indexed_tokens, self.max_len, -1)
        # print("indexed_tokens:", indexed_tokens)

        segments_ids = pad_list(segments_ids, self.max_len, -1)

        #  一般使用torch.nn.utils.rnn.pad_sequence对序列进行填充，填充的长度为批次中最长序列的长度，再
        #  对序列进行填充之后，就需要用torch.nn.utils.rnn.pack_padded_sequence 和
        #  torch.nn.utils.rnn.pad_packed_sequence两个函数进行处理，处理的过程如下：
        # 填充后的输入序列先经过torch.nn.utils.rnn.pack_padded_sequence的处理，
        # 然后会得到一个PackedSequence类型的对象，可以直接给RNN进行处理。
        # RNN处理PackedSequence类型的数据后，会返回一个PackedSequence类型的输出。
        # 最后使用torch.nn.utils.rnn.pad_packed_sequence函数将经过RNN后的输出数据在重新进行填充，得到正常的每个batch等长的序列。

        indexed_tokens = torch.LongTensor(indexed_tokens)
        segments_ids = torch.LongTensor(segments_ids)
        
        # 对indices进行padding处理，使其长度一致
        # 使用max_len作为最大长度，-1作为padding值
        indices_padded = pad_list(indices, self.max_len, -1)
        indices = torch.LongTensor(indices_padded)

        # print("len(indexed_tokens):", len(indexed_tokens))
        # print("len(segments_ids):", len(segments_ids))
        # print("len(indices):", len(indices))
        # text
        # indices：句子向量位置 用于提取每个句子的语义表示 后输入到循环网络
        # tokens
        # segments_ids
        # label

        # 该函数用padding_value来填充一个可变长度的张量列表。将长度较短的序列填充为和最长序列相同的长度。
        # indexed_tokens = pad_sequence(indexed_tokens, batch_first=True, padding_value=0)
        # segments_ids = pad_sequence(segments_ids, batch_first=True, padding_value=0)
        # indices = pad_sequence(indices, batch_first=True, padding_value=0)

        # print({"text": essay, "indices": indices, "indexed_tokens": indexed_tokens, "segments_ids": segments_ids,
        #        "label": int(score)})
        
        # 提取句子级输入
        # 使用nltk的sent_tokenize进行句子切分
        sentences = sent_tokenize(essay)
        
        # 对每个句子添加[CLS]和[SEP]标记
        processed_sentences = []
        for sentence in sentences:
            processed_sentences.append('[CLS] ' + sentence + ' [SEP]')
        
        # 将所有处理后的句子连接起来
        essay_with_markers = ' '.join(processed_sentences)
        
        # 对处理后的essay进行tokenization
        sen_code = self.tokenizer(essay_with_markers, return_tensors='pt', padding='max_length',
                                  truncation=True, max_length=self.max_len)
        
        # 确保sentence_tokens与tokens维度一致
        sentence_tokens = sen_code['input_ids'][0]  # [max_len]
        sentence_attention_masks = sen_code['attention_mask'][0]  # [max_len]
        sentence_token_type_ids = sen_code['token_type_ids'][0]  # [max_len]
        
        result = {
            "text": essay, 
            "tokens": tokens, 
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": int(score) - self.tmp,
            "topic_id": self.subset - 1,  # ASAP子集1-8，转换为0-7的索引
            "sentence_tokens": sentence_tokens,
            "sentence_attention_masks": sentence_attention_masks,
            "sentence_token_type_ids": sentence_token_type_ids
        }
        
        # 添加浅层特征（如果有）
        if self.shallow_features is not None:
            result["shallow_features"] = self.shallow_features[item]
        result["indices"] = indices
        
        return result


def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad


# if __name__ == '__main__':
#     a = Prepare_DataSet()
#     for i in range(1772):
#         a.__getitem__(i)
