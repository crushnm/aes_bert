"""
主题相似度计算模块
计算单词、短语、句子级别的语义表示与主题提示语的相似度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TopicSimilarityModule(nn.Module):
    """
    计算多层级语义表示与主题提示语的相似度
    Similarity_i = S_i · T / (||S_i|| * ||T||)
    
    其中：
    - S_i: 第i个单词/短语/句子的语义表示
    - T: 主题提示语的嵌入表示
    - Similarity_i: 第i个元素与主题的语义相关度
    """
    
    # 定义八个主题的英文文本
    TOPICS = {
        1: "The effects computers have on people",
        2: "Censorship in the libraries",
        3: "How the features of a setting affected a cyclist",
        4: "Why an extract from Winter Hibiscus was concluded in the way the author did",
        5: "The mood created by the author in an extract from Narciso Rodriguez",
        6: "The difficulties faced by the builders of the Empire State Building in allowing dirigibles to dock there",
        7: "Write a story about patience",
        8: "The benefits of laughter"
    }
    
    def __init__(self, hidden_size=768, num_topics=8, use_learnable_topics=True, 
                 similarity_type='cosine', dropout=0.1, topic_embedding_model='bert-base-uncased'):
        """
        Args:
            hidden_size: 语义表示的维度
            num_topics: 主题数量（对应ASAP数据集的8个子集）
            use_learnable_topics: 是否使用可学习的主题嵌入
            similarity_type: 相似度计算方式 ('cosine', 'dot', 'bilinear')
            dropout: dropout比率
            topic_embedding_model: 用于获取主题嵌入的预训练模型
        """
        super(TopicSimilarityModule, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_topics = num_topics
        self.similarity_type = similarity_type
        self.topic_embedding_model = topic_embedding_model
        
        # 主题嵌入表示
        if use_learnable_topics:
            # 可学习的主题嵌入（每个子集对应一个主题向量）
            self.topic_embeddings = nn.Parameter(torch.randn(num_topics, hidden_size))
            nn.init.xavier_uniform_(self.topic_embeddings)
        else:
            # 使用固定的主题嵌入（需要外部提供）
            self.register_buffer('topic_embeddings', torch.zeros(num_topics, hidden_size))
        
        # 初始化主题嵌入（如果需要）
        if not use_learnable_topics:
            self._initialize_topic_embeddings()
        
        # 相似度计算层
        if similarity_type == 'bilinear':
            # 双线性相似度: S_i^T W T
            self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)
        elif similarity_type == 'mlp':
            # MLP相似度
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
            )
        
        # 投影层（可选）
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def compute_similarity(self, semantic_repr, topic_embedding):
        """
        计算语义表示与主题嵌入的相似度
        
        Args:
            semantic_repr: [batch_size, seq_len, hidden_size] 或 [batch_size, hidden_size]
            topic_embedding: [hidden_size] 主题嵌入向量
            
        Returns:
            similarity: [batch_size, seq_len] 或 [batch_size] 相似度分数
        """
        if self.similarity_type == 'cosine':
            # 余弦相似度
            # 归一化
            semantic_norm = F.normalize(semantic_repr, p=2, dim=-1)
            topic_norm = F.normalize(topic_embedding, p=2, dim=-1)
            
            # 计算余弦相似度
            if semantic_repr.dim() == 3:
                # [batch_size, seq_len, hidden_size] · [hidden_size] -> [batch_size, seq_len]
                similarity = torch.matmul(semantic_norm, topic_norm)
            else:
                # [batch_size, hidden_size] · [hidden_size] -> [batch_size]
                similarity = torch.matmul(semantic_norm, topic_norm)
        
        elif self.similarity_type == 'dot':
            # 点积相似度
            if semantic_repr.dim() == 3:
                similarity = torch.matmul(semantic_repr, topic_embedding)
            else:
                similarity = torch.matmul(semantic_repr, topic_embedding)
        
        elif self.similarity_type == 'bilinear':
            # 双线性相似度
            batch_size = semantic_repr.size(0)
            if semantic_repr.dim() == 3:
                seq_len = semantic_repr.size(1)
                # 扩展topic_embedding
                topic_expanded = topic_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                similarity = self.bilinear(semantic_repr, topic_expanded).squeeze(-1)
            else:
                topic_expanded = topic_embedding.unsqueeze(0).expand(batch_size, -1)
                similarity = self.bilinear(semantic_repr, topic_expanded).squeeze(-1)
        
        elif self.similarity_type == 'mlp':
            # MLP相似度
            batch_size = semantic_repr.size(0)
            if semantic_repr.dim() == 3:
                seq_len = semantic_repr.size(1)
                topic_expanded = topic_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                concat = torch.cat([semantic_repr, topic_expanded], dim=-1)
                similarity = self.mlp(concat).squeeze(-1)
            else:
                topic_expanded = topic_embedding.unsqueeze(0).expand(batch_size, -1)
                concat = torch.cat([semantic_repr, topic_expanded], dim=-1)
                similarity = self.mlp(concat).squeeze(-1)
        
        return similarity
    
    def forward(self, word_embeddings, phrase_embeddings, sentence_embeddings, 
                topic_ids=None, return_individual=False):
        """
        计算多层级的主题相似度特征
        
        Args:
            word_embeddings: [batch_size, seq_len, hidden_size] 单词级语义表示
            phrase_embeddings: [batch_size, seq_len, hidden_size] 短语级语义表示
            sentence_embeddings: [batch_size, hidden_size] 句子级语义表示（通过[CLS]获取）
            topic_ids: [batch_size] 主题ID（对应ASAP子集1-8）
            return_individual: 是否返回各层级的相似度
            
        Returns:
            similarity_features: [batch_size, feature_dim] 拼接后的主题相关特征
            或者
            (similarity_w, similarity_p, similarity_s): 各层级的相似度
        """
        if sentence_embeddings is None:
            raise ValueError("sentence_embeddings 不能为空")
        
        batch_size = sentence_embeddings.size(0)
        
        # 获取主题嵌入
        if topic_ids is not None:
            # 确保主题ID在有效范围内（1-8），并转换为索引（0-7）
            topic_indices = torch.clamp(topic_ids - 1, 0, self.num_topics - 1)  # 转换为0-based索引
            # 根据topic_indices选择对应的主题嵌入
            topic_embedding = self.topic_embeddings[topic_indices]  # [batch_size, hidden_size]
        else:
            # 使用平均主题嵌入
            topic_embedding = self.topic_embeddings.mean(dim=0)  # [hidden_size]
            topic_embedding = topic_embedding.unsqueeze(0).expand(batch_size, -1)  # [batch_size, hidden_size]
        
        # 投影（可选）
        topic_proj = self.key_proj(topic_embedding)
        
        # 计算各层级的相似度
        similarity_results = []
        
        for i in range(batch_size):
            # 句子级相似度（始终计算）
            sentence_proj = self.query_proj(sentence_embeddings[i].unsqueeze(0)).squeeze(0)
            sim_s = self.compute_similarity(sentence_proj, topic_proj[i])
            sim_s_pooled = torch.mean(sim_s)# 标量
            
            if word_embeddings is not None and phrase_embeddings is not None:
                # 计算单词级和短语级相似度（细粒度）
                word_proj = self.query_proj(word_embeddings[i])
                sim_w = self.compute_similarity(word_proj, topic_proj[i])  # [seq_len]
                sim_w_pooled = torch.mean(sim_w)  # 标量
                
                phrase_proj = self.query_proj(phrase_embeddings[i])
                sim_p = self.compute_similarity(phrase_proj, topic_proj[i])  # [seq_len]
                sim_p_pooled = torch.mean(sim_p)  # 标量
                
                # 拼接三个层级的相似度
                similarity_vec = torch.stack([sim_w_pooled, sim_p_pooled, sim_s_pooled])  # [3]
            else:
                # 只计算句子级相似度（粗粒度）
                similarity_vec = sim_s.unsqueeze(0)  # [1]
            
            similarity_results.append(similarity_vec)
        
        # 拼接所有样本
        similarity_features = torch.stack(similarity_results, dim=0)  # [batch_size, feature_dim]
        
        if return_individual:
            # 返回各层级的详细相似度
            similarity_w = []
            similarity_p = []
            similarity_s = []
            
            for i in range(batch_size):
                if word_embeddings is not None:
                    word_proj = self.query_proj(word_embeddings[i])
                    sim_w = self.compute_similarity(word_proj, topic_proj[i])
                    similarity_w.append(sim_w)
                
                if phrase_embeddings is not None:
                    phrase_proj = self.query_proj(phrase_embeddings[i])
                    sim_p = self.compute_similarity(phrase_proj, topic_proj[i])
                    similarity_p.append(sim_p)
                
                sentence_proj = self.query_proj(sentence_embeddings[i].unsqueeze(0)).squeeze(0)
                sim_s = self.compute_similarity(sentence_proj, topic_proj[i])
                similarity_s.append(sim_s)
            
            if word_embeddings is not None:
                similarity_w = torch.stack(similarity_w, dim=0)  # [batch_size, seq_len]
            else:
                similarity_w = None
            
            if phrase_embeddings is not None:
                similarity_p = torch.stack(similarity_p, dim=0)  # [batch_size, seq_len]
            else:
                similarity_p = None
            
            similarity_s = torch.stack(similarity_s, dim=0)  # [batch_size]
            
            return similarity_features, (similarity_w, similarity_p, similarity_s)
        
        return similarity_features
    
    def _initialize_topic_embeddings(self):
        """
        使用预训练模型初始化主题嵌入
        """
        try:
            # 加载预训练模型和分词器
            tokenizer = AutoTokenizer.from_pretrained(self.topic_embedding_model)
            model = AutoModel.from_pretrained(self.topic_embedding_model)
            model.eval()
            
            # 为每个主题生成嵌入
            topic_embeddings = []
            for i in range(1, self.num_topics + 1):
                if i in self.TOPICS:
                    topic_text = self.TOPICS[i]
                    # 分词
                    inputs = tokenizer(topic_text, return_tensors='pt', padding=True, truncation=True)
                    # 生成嵌入
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # 使用 [CLS] token 的嵌入作为主题表示
                        topic_emb = outputs.last_hidden_state[:, 0, :].squeeze()
                        topic_embeddings.append(topic_emb)
                else:
                    # 如果没有对应主题，使用随机嵌入
                    topic_embeddings.append(torch.randn(self.hidden_size))
            
            # 拼接主题嵌入
            topic_embeddings = torch.stack(topic_embeddings, dim=0)
            # 归一化
            topic_embeddings = F.normalize(topic_embeddings, p=2, dim=-1)
            
            # 设置主题嵌入
            self.topic_embeddings.data = topic_embeddings
            print(f"主题嵌入初始化完成，形状: {topic_embeddings.shape}")
        except Exception as e:
            print(f"初始化主题嵌入时出错: {e}")
            print("使用随机初始化的主题嵌入")
    
    def set_topic_embeddings(self, topic_embeddings):
        """
        设置外部提供的主题嵌入
        
        Args:
            topic_embeddings: [num_topics, hidden_size] 主题嵌入矩阵
        """
        if isinstance(topic_embeddings, torch.Tensor):
            self.topic_embeddings.data = topic_embeddings
        else:
            self.topic_embeddings.data = torch.tensor(topic_embeddings, dtype=torch.float32)


class TopicAwareAttention(nn.Module):
    """
    主题感知的注意力机制
    结合主题相似度来调整注意力权重
    """
    
    def __init__(self, hidden_size, attention_size, dropout=0.1):
        super(TopicAwareAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # 注意力权重
        self.w_omega = nn.Parameter(torch.zeros(hidden_size, attention_size))
        self.u_omega = nn.Parameter(torch.zeros(attention_size, 1))
        
        # 主题调制
        self.topic_gate = nn.Sequential(
            nn.Linear(1, attention_size),
            nn.Sigmoid()
        )
        
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.xavier_uniform_(self.u_omega)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, topic_similarity):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            topic_similarity: [batch_size, seq_len] 主题相似度分数
            
        Returns:
            attention_output: [batch_size, hidden_size]
            attention_weights: [batch_size, seq_len]
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # 计算注意力分数
        # [batch_size, seq_len, hidden_size] @ [hidden_size, attention_size]
        attn_scores = torch.matmul(hidden_states, self.w_omega)  # [batch_size, seq_len, attention_size]
        attn_scores = torch.tanh(attn_scores)
        
        # 主题调制
        topic_sim_expanded = topic_similarity.unsqueeze(-1)  # [batch_size, seq_len, 1]
        topic_modulation = self.topic_gate(topic_sim_expanded)  # [batch_size, seq_len, attention_size]
        
        # 融合主题信息
        attn_scores = attn_scores * topic_modulation
        
        # 计算注意力权重
        attn_weights = torch.matmul(attn_scores, self.u_omega).squeeze(-1)  # [batch_size, seq_len]
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 加权求和
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        attention_output = torch.sum(hidden_states * attn_weights_expanded, dim=1)  # [batch_size, hidden_size]
        
        return attention_output, attn_weights


# 测试代码
if __name__ == '__main__':
    # 测试主题相似度模块
    batch_size = 4
    seq_len = 128
    hidden_size = 768
    num_topics = 8
    
    # 创建模块
    topic_sim_module = TopicSimilarityModule(
        hidden_size=hidden_size,
        num_topics=num_topics,
        similarity_type='cosine'
    )
    
    # 模拟输入
    word_emb = torch.randn(batch_size, seq_len, hidden_size)
    phrase_emb = torch.randn(batch_size, seq_len, hidden_size)
    sentence_emb = torch.randn(batch_size, hidden_size)
    topic_ids = torch.randint(0, num_topics, (batch_size,))
    
    # 计算相似度
    similarity_features = topic_sim_module(word_emb, phrase_emb, sentence_emb, topic_ids)
    
    print(f"相似度特征形状: {similarity_features.shape}")  # [batch_size, 3]
    print(f"相似度特征:\n{similarity_features}")
    
    # 测试主题感知注意力
    topic_attn = TopicAwareAttention(hidden_size, 512)
    topic_similarity = torch.rand(batch_size, seq_len)
    attn_output, attn_weights = topic_attn(word_emb, topic_similarity)
    
    print(f"\n注意力输出形状: {attn_output.shape}")  # [batch_size, hidden_size]
    print(f"注意力权重形状: {attn_weights.shape}")  # [batch_size, seq_len]
