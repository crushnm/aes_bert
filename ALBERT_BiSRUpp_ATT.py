import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import BertModel, LongformerModel, LongformerConfig
from model.xLSTM import MultiLevelxLSTM
from model.shallow_features import ShallowFeatureFusion
from model.topic_similarity import TopicSimilarityModule
from model.ScConv import ScConv as ScConvModule
from pathlib import Path

model_name = Path("bert-base-uncased").resolve()
# model_name = 'google-bert/bert-base-uncased'
# model_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
# model_name = 'princeton-nlp/sup-simcse-roberta-large'

# if __name__ == '__main__':
#     model = ALBERT_BiSRUpp_ATT(sequence_length=512,num_classes = 11)
#     x = torch.randn((2, 8))
#     # 在0-7范围内，随机生成两个数，当做标签
#     y = torch.randint(0, 8, [2])
    
    

class ALBERT_BiSRUpp_ATT(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, proj_size=512, num_layers=1, dropout=0.3, attn_dropout=0.1,
                 num_heads=8, bidirectional=True, layer_norm=True, normalize_after=True, highway_bias=-2.0,
                 attention_every_n_layers=2, rescale=True, nn_rnn_compatible_return=False,
                 proj_input_to_hidden_first=True, weight_c_init=1.0, linear_high=256, num_classes=2, use_cuda=False,
                 sequence_length=512, use_longformer=True, phrase_window_size=3, xlstm_cell_type='slstm',
                 use_shallow_features=True, use_topic_similarity=True, num_topics=8,
                 # 消融实验参数
                 use_scconv_mta=True,  # 是否使用SCConv+MTA（多层语义嵌入与注意力）
                pretrained_model='longformer',  # 预训练模型选择: 'longformer', 'bert', 'roberta'
                # 消融实验参数
                use_mta_w=True,  # 是否使用单词级注意力
                use_mta_p=True,  # 是否使用短语级注意力
                use_mta_s=True,  # 是否使用句子级注意力
                use_fine_grained_topic=True,  # 是否使用细粒度主题特征
                use_shallow_semantic=True,  # 是否使用浅层语义特征
                use_xlstm_w=True,  # 是否使用单词级xLSTM
                use_xlstm_p=True,  # 是否使用短语级xLSTM
                use_xlstm_s=True):  # 是否使用句子级xLSTM
        super(ALBERT_BiSRUpp_ATT, self).__init__()
        # 预训练模型
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.attention_size = proj_size
        self.use_cuda = use_cuda
        self.use_longformer = use_longformer
        self.phrase_window_size = phrase_window_size
        self.use_shallow_features = use_shallow_features
        self.use_topic_similarity = use_topic_similarity
        self.num_topics = num_topics
        
        # 消融实验配置
        self.use_scconv_mta = use_scconv_mta  # 是否使用多层语义嵌入与注意力
        self.pretrained_model = pretrained_model.lower()
        
        # 新增消融实验配置
        self.use_mta_w = use_mta_w  # 是否使用单词级注意力
        self.use_mta_p = use_mta_p  # 是否使用短语级注意力
        self.use_mta_s = use_mta_s  # 是否使用句子级注意力
        self.use_fine_grained_topic = use_fine_grained_topic  # 是否使用细粒度主题特征
        self.use_shallow_semantic = use_shallow_semantic  # 是否使用浅层语义特征
        self.use_xlstm_w = use_xlstm_w  # 是否使用单词级xLSTM
        self.use_xlstm_p = use_xlstm_p  # 是否使用短语级xLSTM
        self.use_xlstm_s = use_xlstm_s  # 是否使用句子级xLSTM
        
        # 处理浅层语义特征的开关
        if not self.use_shallow_semantic:
            self.use_shallow_features = False
        
        # 根据消融实验配置选择预训练模型
        if self.pretrained_model == 'bert':
            print("Using BERT as pretrained model")
            self.pre_trained = BertModel.from_pretrained(model_name)
            self.use_longformer = False
        elif self.pretrained_model == 'roberta':
            print("Using RoBERTa as pretrained model")
            from transformers import RobertaModel
            try:
                self.pre_trained = RobertaModel.from_pretrained('roberta-base')
            except:
                # 如果加载失败，使用本地路径
                roberta_path = Path("roberta-base").resolve()
                self.pre_trained = RobertaModel.from_pretrained(roberta_path)
            self.use_longformer = False
        else:  # longformer (默认)
            print("Using Longformer as pretrained model")
            self.pre_trained = BertModel.from_pretrained(model_name)
            # use_longformer 保持原值
        
        # Longformer层 - 用于提取长距离依赖的多层级语义
        # 只有在使用SCConv+MTA且使用longformer时才初始化这些模块
        if self.use_longformer and self.use_scconv_mta:
            # 方案：使用预训练的 Longformer，然后在 BERT 输出上应用
            # 创建一个轻量级的 Longformer 配置
            try:
                # 尝试加载预训练的 Longformer
                self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
                print("Successfully loaded pretrained Longformer model")
            except:
                # 如果加载失败，创建一个新的 Longformer
                print("Creating new Longformer configuration")
                longformer_config = LongformerConfig(
                    vocab_size=30522,  # 与 BERT 相同
                    hidden_size=input_size,
                    num_hidden_layers=2,  # 使用较少的层数
                    num_attention_heads=num_heads,
                    intermediate_size=input_size * 4,
                    attention_window=[128, 128],  # 局部注意力窗口
                    attention_probs_dropout_prob=attn_dropout,
                    hidden_dropout_prob=dropout,
                    max_position_embeddings=4096  # 支持长文本
                )
                self.longformer = LongformerModel(longformer_config)
            
            # 添加一个投影层，将 BERT 输出映射到 Longformer 的嵌入空间
            self.bert_to_longformer = nn.Linear(input_size, input_size)
        
        # 浅层特征处理模块
        if self.use_shallow_features:
            self.shallow_norm = nn.BatchNorm1d(9)
            self.shallow_proj = nn.Sequential(
                nn.Linear(9, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 128)
            )
        
        # SCConv+MTA模块：多层语义嵌入与注意力
        if self.use_scconv_mta:
            # 短语级别提取 - 使用卷积捕获n-gram特征
            # 修复cuDNN问题：使用更稳定的配置
            self.phrase_conv = nn.Conv1d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=phrase_window_size,
                padding=phrase_window_size // 2,
                groups=1,
                bias=True
            )
            
            # 句子级别提取 - 使用平均池化
            self.sentence_pooling = nn.AdaptiveAvgPool1d(1)
            
            # 多层级特征融合
            self.word_proj = nn.Linear(input_size, input_size)
            self.phrase_proj = nn.Linear(input_size, input_size)
            self.sentence_proj = nn.Linear(input_size, input_size)
            
            # 门控融合机制
            self.fusion_gate = nn.Sequential(
                nn.Linear(input_size * 3, input_size),
                nn.Sigmoid()
            )
            
            self.fusion_output = nn.Linear(input_size * 3, input_size)
            self.fusion_norm = nn.LayerNorm(input_size)
        else:
            # 不使用SCConv+MTA时，只需要简单的投影层
            self.simple_proj = nn.Linear(input_size, input_size)
        
        # Multi-level xLSTM - 替代BiSRUpp
        # 使用xLSTM分别处理word_level, phrase_level, sentence_level
        self.xlstm = MultiLevelxLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            cell_type=xlstm_cell_type,  # 'slstm' or 'mlstm'
            bidirectional=bidirectional,
            dropout=dropout,
            layer_norm=layer_norm
        )
        # 注意力机制
        self.bi = 1
        if bidirectional:
            self.bi = 2
        # 是否使用use_cuda
        
        # 多层级注意力机制（MTA）参数
        if self.use_cuda:
            # 单词级注意力
            self.w_omega_word = nn.Parameter(torch.zeros(self.hidden_size * self.bi, self.attention_size).cuda())
            self.u_omega_word = nn.Parameter(torch.zeros(self.attention_size).cuda())
            # 短语级注意力
            self.w_omega_phrase = nn.Parameter(torch.zeros(self.hidden_size * self.bi, self.attention_size).cuda())
            self.u_omega_phrase = nn.Parameter(torch.zeros(self.attention_size).cuda())
            # 句子级注意力
            self.w_omega_sentence = nn.Parameter(torch.zeros(self.hidden_size * self.bi, self.attention_size).cuda())
            self.u_omega_sentence = nn.Parameter(torch.zeros(self.attention_size).cuda())
        else:
            # 单词级注意力
            w_omega_word = torch.zeros(self.hidden_size * self.bi, self.attention_size)
            nn.init.xavier_uniform_(w_omega_word)
            self.w_omega_word = nn.Parameter(w_omega_word)
            u_omega_word = torch.zeros(self.attention_size, 1)
            nn.init.xavier_uniform_(u_omega_word)
            self.u_omega_word = nn.Parameter(u_omega_word)
            
            # 短语级注意力
            w_omega_phrase = torch.zeros(self.hidden_size * self.bi, self.attention_size)
            nn.init.xavier_uniform_(w_omega_phrase)
            self.w_omega_phrase = nn.Parameter(w_omega_phrase)
            u_omega_phrase = torch.zeros(self.attention_size, 1)
            nn.init.xavier_uniform_(u_omega_phrase)
            self.u_omega_phrase = nn.Parameter(u_omega_phrase)
            
            # 句子级注意力
            w_omega_sentence = torch.zeros(self.hidden_size * self.bi, self.attention_size)
            nn.init.xavier_uniform_(w_omega_sentence)
            self.w_omega_sentence = nn.Parameter(w_omega_sentence)
            u_omega_sentence = torch.zeros(self.attention_size, 1)
            nn.init.xavier_uniform_(u_omega_sentence)
            self.u_omega_sentence = nn.Parameter(u_omega_sentence)
        
        # 主题相似度模块
        if self.use_topic_similarity:
            self.topic_similarity = TopicSimilarityModule(
                hidden_size=input_size,
                num_topics=num_topics,
                use_learnable_topics=True,
                similarity_type='cosine',
                dropout=dropout
            )
            # 主题相似度特征维度
            if self.use_scconv_mta:
                topic_feature_dim = 3  # word, phrase, sentence各一个
            else:
                topic_feature_dim = 1  # 只有一个整体特征
        else:
            topic_feature_dim = 0
        
        # 计算最终特征维度
        if self.use_scconv_mta:
            # 使用多层语义：word + phrase + sentence
            # 需要考虑 xlstm_cell_type 的影响
            if xlstm_cell_type == 'all':
                # 'all' 模式：(slstm + mlstm) * num_directions = hidden_size * 2 * num_directions
                xlstm_output_dim = self.hidden_size * self.bi * 2 * 3  # * 3 是因为有 word, phrase, sentence
            else:
                # 'slstm' 或 'mlstm' 模式：hidden_size * num_directions
                xlstm_output_dim = self.hidden_size * self.bi * 3
        else:
            # 不使用多层语义：只有单一输出
            if xlstm_cell_type == 'all':
                xlstm_output_dim = self.hidden_size * self.bi * 2
            else:
                xlstm_output_dim = self.hidden_size * self.bi
        
        shallow_feature_dim = 128 if self.use_shallow_features else 0
        # 考虑句子级嵌入特征维度
        sentence_embedding_dim = input_size if self.use_scconv_mta else 0
        final_feature_dim = xlstm_output_dim + shallow_feature_dim + topic_feature_dim
        
        self.Linear = nn.Linear(final_feature_dim, num_classes)
        self.ReLU = nn.ReLU()

        self.ce_loss_fct = nn.CrossEntropyLoss()  # Log_softmax() + NLLLoss() 组成

    def _update_mask(self, mask, bsz, xq, xk_size):
        """
        更新掩码以适应MTA操作
        """
        # 这里简化实现，实际应用中可能需要更复杂的掩码处理
        if mask is None:
            return None
        return mask
    
    def _mta_convolution(self, scores, mask, chunk_start_ids, kernel):
        """
        MTA卷积操作
        """
        # 简化实现，实际应用中可能需要更复杂的卷积逻辑
        bsz, n_heads, seq_len_q, seq_len_k = scores.shape
        
        # 调整形状以适应卷积操作
        # [batch_size * num_heads, 1, seq_len_q, seq_len_k]
        scores_reshaped = scores.view(-1, 1, seq_len_q, seq_len_k)
        
        # 应用卷积
        # 这里使用简化的卷积实现，实际应用中可能需要更复杂的处理
        kernel_size = kernel.shape[-2:]
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        
        # 将提供的kernel复制到卷积层
        if self.use_cuda:
            conv.weight.data = kernel.view(n_heads, 1, *kernel_size).mean(dim=0).unsqueeze(0).cuda()
            conv = conv.cuda()
        else:
            conv.weight.data = kernel.view(n_heads, 1, *kernel_size).mean(dim=0).unsqueeze(0)
        
        # 应用卷积
        scores_conv = conv(scores_reshaped)
        
        # 恢复形状
        # [batch_size, num_heads, seq_len_q, seq_len_k]
        scores_conv = scores_conv.view(bsz, n_heads, seq_len_q, seq_len_k)
        
        return scores_conv
    
    def attention_net(self, rnn_output, w_omega, u_omega, num_attention_heads=1, use_mta=False, mta_args=None):
        """
        多token注意力机制 (Multi-Token Attention)
        
        Args:
            rnn_output: [sequence_length, batch_size, hidden_size * self.bi]
            w_omega: [hidden_size * self.bi, attention_size]
            u_omega: [attention_size]
            num_attention_heads: 注意力头数量
            use_mta: 是否使用多token注意力
            mta_args: MTA相关参数
            
        Returns:
            attn_output: [batch_size, hidden_size * self.bi]
        """
        # 1、(sequence_length, batch_size, hidden_size * self.bi)
        sequence_length, batch_size, hidden_dim = rnn_output.shape
        
        # 调整形状以适应多头注意力
        # (batch_size, sequence_length, hidden_dim)
        rnn_output_reshaped = rnn_output.permute(1, 0, 2)
        # 确保hidden_dim可以被注意力头数量整除
        assert hidden_dim % num_attention_heads == 0, "hidden_dim must be divisible by num_attention_heads"
        head_dim = hidden_dim // num_attention_heads
        
        wq = nn.Linear(hidden_dim, num_attention_heads * head_dim, bias=False).cuda()
        wk = nn.Linear(hidden_dim, num_attention_heads * head_dim, bias=False).cuda()
        wv = nn.Linear(hidden_dim, num_attention_heads * head_dim, bias=False).cuda()
        wo = nn.Linear(num_attention_heads * head_dim, hidden_dim, bias=False).cuda()
        
        # 参数初始化
        nn.init.xavier_uniform_(wq.weight)
        nn.init.xavier_uniform_(wk.weight)
        nn.init.xavier_uniform_(wv.weight)
        nn.init.xavier_uniform_(wo.weight)
        
        # 计算查询、键、值
        q = wq(rnn_output_reshaped)  # [batch_size, sequence_length, num_heads * head_dim]
        k = wk(rnn_output_reshaped)  # [batch_size, sequence_length, num_heads * head_dim]
        v = wv(rnn_output_reshaped)  # [batch_size, sequence_length, num_heads * head_dim]
        
        # 调整形状以适应注意力计算
        # [batch_size, num_heads, sequence_length, head_dim]
        q = q.view(batch_size, sequence_length, num_attention_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, sequence_length, num_attention_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, sequence_length, num_attention_heads, head_dim).permute(0, 2, 1, 3)
        
        # MTA特定处理
        if use_mta and mta_args is not None:
            # 更新掩码
            mask = self._update_mask(mask=None, bsz=batch_size, xq=q, xk_size=k.size(-2))
        
        # 计算注意力分数
        # [batch_size, num_heads, sequence_length, sequence_length]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # MTA卷积处理
        if use_mta and mta_args is not None:
            # 前softmax MTA卷积
            if mta_args.get('query_kernel_size', None) is not None and mta_args.get('key_kernel_size', None) is not None:
                # 初始化MTA卷积核
                kernel_size = (mta_args['query_kernel_size'], mta_args['key_kernel_size'])
                mta_kernel = torch.nn.parameter.Parameter(
                    torch.empty(num_attention_heads, 1, *kernel_size).cuda()
                )
                nn.init.xavier_uniform_(mta_kernel)
                
                # 应用MTA卷积
                attn_scores = self._mta_convolution(
                    scores=attn_scores, 
                    mask=mask, 
                    chunk_start_ids=None, 
                    kernel=mta_kernel
                )
            
            # 前softmax线性头部处理
            if mta_args.get('pre_sm_linear_head', False):
                # 调整注意力分数形状
                # [batch_size, sequence_length, sequence_length, num_heads]
                attn_scores_reshaped = attn_scores.permute(0, 2, 3, 1)
                # 线性变换
                wpsm = nn.Linear(num_attention_heads, num_attention_heads, bias=False).cuda()
                nn.init.xavier_uniform_(wpsm.weight)
                attn_scores_reshaped = wpsm(attn_scores_reshaped)
                # 恢复形状
                attn_scores = attn_scores_reshaped.permute(0, 3, 1, 2)
        
        # 注意力得分归一化
        if use_mta and mask is not None:
            attn_scores = attn_scores + mask
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # 后softmax MTA处理
        if use_mta and mta_args is not None:
            # 后softmax MTA卷积
            if mta_args.get('after_sm_query_kernel_size', None) is not None and mta_args.get('after_sm_key_kernel_size', None) is not None:
                # 初始化MTA卷积核
                kernel_size = (mta_args['after_sm_query_kernel_size'], mta_args['after_sm_key_kernel_size'])
                if self.use_cuda:
                    mta_kernel_after_sm = torch.nn.parameter.Parameter(
                        torch.empty(num_attention_heads, 1, *kernel_size).cuda()
                    )
                else:
                    mta_kernel_after_sm = torch.nn.parameter.Parameter(
                        torch.empty(num_attention_heads, 1, *kernel_size)
                    )
                nn.init.xavier_uniform_(mta_kernel_after_sm)
                
                # 应用MTA卷积
                attn_probs = self._mta_convolution(
                    scores=attn_probs, 
                    mask=mask, 
                    chunk_start_ids=None, 
                    kernel=mta_kernel_after_sm
                )
                
                # 处理掩码
                if mask is not None:
                    attn_probs = torch.where(mask == float('-inf'), 0.0, attn_probs)
            
            # 后softmax线性头部处理
            if mta_args.get('post_sm_linear_head', False):
                # 调整注意力概率形状
                # [batch_size, sequence_length, sequence_length, num_heads]
                attn_probs_reshaped = attn_probs.permute(0, 2, 3, 1)
                # 线性变换
                if self.use_cuda:
                    wposm = nn.Linear(num_attention_heads, num_attention_heads, bias=False).cuda()
                else:
                    wposm = nn.Linear(num_attention_heads, num_attention_heads, bias=False)
                nn.init.xavier_uniform_(wposm.weight)
                attn_probs_reshaped = wposm(attn_probs_reshaped)
                # 恢复形状
                attn_probs = attn_probs_reshaped.permute(0, 3, 1, 2)
        
        # 计算加权和
        # [batch_size, num_heads, sequence_length, head_dim]
        attn_output = torch.matmul(attn_probs, v)
        
        # 调整形状并通过输出投影
        # [batch_size, sequence_length, num_heads * head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, num_attention_heads * head_dim)
        # [batch_size, sequence_length, hidden_dim]
        attn_output = wo(attn_output)
        
        # 对序列长度维度进行平均池化
        # [batch_size, hidden_dim]
        attn_output = torch.mean(attn_output, dim=1)
        
        return attn_output

    def extract_multilevel_embeddings(self, word_embeddings, indices=None):
        """
        提取多层级语义嵌入：单词级、短语级、句子级
        
        Args:
            word_embeddings: [batch_size, seq_len, hidden_size]
            indices: [batch_size, num_sentences] 或 [num_sentences]，记录每个句子[CLS] token的位置
        
        Returns:
            word_level: [batch_size, seq_len, hidden_size]
            phrase_level: [batch_size, seq_len, hidden_size]
            sentence_level: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = word_embeddings.shape
        
        # 1. 单词级别嵌入 (Word-level)
        word_level = self.word_proj(word_embeddings)  # [batch_size, seq_len, hidden_size]
        
        # 2. 短语级别嵌入 (Phrase-level) - 使用ScConv提取局部n-gram特征
        # 转换维度以适配ScConv: [batch_size, hidden_size, seq_len]
        word_embeddings_1d = word_embeddings.transpose(1, 2).contiguous()

        # 尝试使用ScConv
        # 确保scconv是ScConv实例
        self.scconv = ScConvModule(op_channel=hidden_size, group_kernel_size=3).cuda()

        # 调整输入维度以适配ScConv
        # ScConv期望输入: [batch_size, channels, height, width]
        # 将3D输入转换为4D: [batch_size, hidden_size, seq_len, 1]
        word_embeddings_2d = word_embeddings_1d.unsqueeze(-1)
        phrase_level = self.scconv(word_embeddings_2d)  # [batch_size, hidden_size, seq_len, 1]
        phrase_level = phrase_level.squeeze(-1)  # [batch_size, hidden_size, seq_len]
        phrase_level = self.ReLU(phrase_level)
        
        phrase_level = phrase_level.transpose(1, 2)  # [batch_size, seq_len, hidden_size]
        phrase_level = self.phrase_proj(phrase_level)
        
        return word_level, phrase_level,

    def forward(self, tokens, token_type_ids, attention_mask, labels, shallow_features=None, topic_ids=None, indices=None, sentence_tokens=None, sentence_attention_masks=None, sentence_token_type_ids=None):
        # 1. 预训练模型提取基础特征（单词级）
        if self.pretrained_model == 'roberta':
            # RoBERTa 不使用 token_type_ids
            bert_output = self.pre_trained(tokens, attention_mask, 
                                          output_hidden_states=True, return_dict=True)
            word_embeddings = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        elif self.pretrained_model == 'bert':
            # BERT 和 Longformer 使用 token_type_ids
            bert_output = self.pre_trained(tokens, attention_mask, token_type_ids, 
                                          output_hidden_states=True, return_dict=True)
            word_embeddings = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 2. 处理句子级输入（如果提供）
        sentence_embeddings = None
        if sentence_tokens is not None and self.use_longformer:
            # 处理每个样本的句子级输入 [batch_size, seq_len]
            batch_size = tokens.shape[0]
            sentence_features = []
            
            for i in range(batch_size):
                # 句子输入 [seq_len]
                sent = sentence_tokens[i:i+1]  # [1, seq_len]
                attn = sentence_attention_masks[i:i+1]  # [1, seq_len]
                token_type = sentence_token_type_ids[i:i+1]  # [1, seq_len]
                
                # 创建全局注意力掩码
                global_attention_mask = torch.zeros_like(sent, dtype=torch.long)
                global_attention_mask[:, 0] = 1  # [CLS] token 使用全局注意力
                
                # 使用 Longformer 处理句子
                longformer_output = self.longformer(
                    input_ids=sent,
                    attention_mask=attn,
                    token_type_ids=token_type,
                    global_attention_mask=global_attention_mask,
                    return_dict=True
                )
                # 提取 [CLS] 位置的特征作为句子表示
                sentence_feat = longformer_output.last_hidden_state[:, 0, :].unsqueeze(1)  # [1, 1, hidden_size]
                sentence_features.append(sentence_feat)
            
            # 将句子特征堆叠为批次
            sentence_level = torch.cat(sentence_features, dim=0)  # [batch_size, 1, hidden_size]
        
        # 3. 根据配置选择是否使用Longformer和多层语义嵌入
        if self.use_scconv_mta:
            # 使用完整的SCConv+MTA模块                # Longformer层 - 提取长距离依赖和多层级语义
            batch_size, seq_len = tokens.shape
            global_attention_mask = torch.zeros_like(tokens, dtype=torch.long)
            global_attention_mask[:, 0] = 1  # [CLS] token 使用全局注意力

            # 使用 Longformer 处理
            longformer_output = self.longformer(
                input_ids=tokens,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                global_attention_mask=global_attention_mask,
                return_dict=True
            )
            # 使用 Longformer 的输出
            word_embeddings = longformer_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
            word_level, phrase_level = self.extract_multilevel_embeddings(word_embeddings, indices)
            
            word_level = torch.transpose(word_level, 0, 1)
            phrase_level = torch.transpose(phrase_level, 0, 1)
            
            # 扩展sentence_level的seq_len维度以匹配其他特征
            # 将[1, batch_size, hidden_size]扩展为[seq_len, batch_size, hidden_size]
            sentence_level = torch.transpose(sentence_level, 0, 1)  # [1, batch_size, hidden_size]
            sentence_level = sentence_level.repeat(seq_len, 1, 1)  # [seq_len, batch_size, hidden_size]
            
            # 处理xLSTM的消融实验
            if self.use_xlstm_w and self.use_xlstm_p and self.use_xlstm_s:
                # 使用完整的xLSTM
                word_out, phrase_out, sentence_out = self.xlstm(word_level, phrase_level, sentence_level)
            else:
                # 根据配置选择xLSTM输入
                xlstm_input_w = word_level if self.use_xlstm_w else torch.zeros_like(word_level)
                xlstm_input_p = phrase_level if self.use_xlstm_p else torch.zeros_like(phrase_level)
                xlstm_input_s = sentence_level if self.use_xlstm_s else torch.zeros_like(sentence_level)
                
                word_out, phrase_out, sentence_out = self.xlstm(xlstm_input_w, xlstm_input_p, xlstm_input_s)
            
            # 多层级注意力机制（MTA）
            # 单词级注意力
            if self.use_mta_w:
                word_out = self.attention_net(word_out, self.w_omega_word, self.u_omega_word)
            else:
                word_out = torch.mean(word_out, dim=0)  # 使用平均池化替代
            
            # 短语级注意力
            if self.use_mta_p:
                phrase_out = self.attention_net(phrase_out, self.w_omega_phrase, self.u_omega_phrase)
            else:
                phrase_out = torch.mean(phrase_out, dim=0)  # 使用平均池化替代
            
            # 句子级注意力
            if self.use_mta_s:
                sentence_out = self.attention_net(sentence_out, self.w_omega_sentence, self.u_omega_sentence)
            else:
                sentence_out = torch.mean(sentence_out, dim=0)  # 使用平均池化替代
            
            if self.use_topic_similarity:
                # 获取[CLS]位置的句子级表示
                # 计算细粒度主题特征
                word_level_for_sim = torch.transpose(word_level, 0, 1)
                phrase_level_for_sim = torch.transpose(phrase_level, 0, 1)
                sentence_level_for_sim = torch.transpose(sentence_level, 0, 1)

                topic_similarity_features = self.topic_similarity(
                    word_embeddings=word_level_for_sim,
                    phrase_embeddings=phrase_level_for_sim,
                    sentence_embeddings=sentence_level_for_sim,
                    topic_ids=topic_ids
                )  # [batch_size, 3]
            
            # 7. 特征拼接（多层级）
            feature_list = [word_out, phrase_out, sentence_out]
            
            # 如果有句子级嵌入，添加到特征列表
            if sentence_embeddings is not None:
                feature_list.append(sentence_embeddings.squeeze(1))  # [batch_size, hidden_size]
            
        else:
            # W/o SCConv+MTA: 不使用多层语义嵌入与注意力
            # 只使用单一的特征提取路径
            
            # 简单投影
            output = self.simple_proj(word_embeddings)
            
            output = torch.transpose(output, 0, 1)  # [seq_len, batch_size, hidden_size]
            
            single_out = self.xlstm(output, output, output)[0]  # 只取第一个输出
            
            # 使用注意力机制聚合特征
            single_out = self.attention_net(single_out, self.w_omega_word, self.u_omega_word)  # [batch_size, hidden_size * bi]
            
            if self.use_topic_similarity:
                # 获取[CLS]位置的句子级表示
                output_transposed = torch.transpose(output, 0, 1)  # [batch_size, seq_len, hidden_size]
                sentence_cls = output_transposed[:, 0, :]  # [batch_size, hidden_size]
                
                # 只计算句子级别的主题相似度
                topic_similarity_features = self.topic_similarity(
                    word_embeddings=output_transposed,
                    phrase_embeddings=output_transposed,
                    sentence_embeddings=sentence_cls,
                    topic_ids=topic_ids
                )  # [batch_size, 3]
                # 只取句子级别的相似度
                topic_similarity_features = topic_similarity_features[:, 2:3]  # [batch_size, 1]
            
            feature_list = [single_out]
            
            # 如果有句子级嵌入，添加到特征列表
            if sentence_embeddings is not None:
                feature_list.append(sentence_embeddings.squeeze(1))  # [batch_size, hidden_size]
        
        if self.use_shallow_features and shallow_features is not None:
            shallow_norm = self.shallow_norm(shallow_features)
            shallow_proj = self.shallow_proj(shallow_norm)
            feature_list.append(shallow_proj)
        
        if self.use_topic_similarity:
            feature_list.append(topic_similarity_features)
        
        att_output = torch.cat(feature_list, dim=-1)
        
        logits = self.Linear(att_output)  # [batch_size, num_classes]
        
        loss = self.ce_loss_fct(logits, labels)
       
        return loss, logits
