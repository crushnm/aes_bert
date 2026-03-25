"""
浅层特征提取模块
用于从文本中提取语法、语义、复杂度等特征
"""
import re
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
import language_tool_python
from collections import Counter


class ShallowFeatureExtractor:
    """
    提取文本的浅层特征
    包括：语法错误、作文长度、语义复杂度、单词复杂度、句子复杂度、从句数量
    """
    
    def __init__(self, use_grammar_check=True):
        """
        Args:
            use_grammar_check: 是否使用语法检查工具（需要安装language_tool_python）
        """
        self.use_grammar_check = use_grammar_check
        self.tool = None
        
        if use_grammar_check:
            try:
                # 初始化语法检查工具
                self.tool = language_tool_python.LanguageTool('en-US')
                print("语法检查工具初始化成功")
            except Exception as e:
                print(f"语法检查工具初始化失败: {e}")
                print("将使用简化的语法错误检测")
                self.use_grammar_check = False
        
        # 从句标志词
        self.subordinating_conjunctions = {
            'after', 'although', 'as', 'because', 'before', 'even', 'if', 
            'once', 'since', 'than', 'that', 'though', 'unless', 'until', 
            'when', 'whenever', 'where', 'wherever', 'whether', 'while', 'who',
            'whom', 'whose', 'which', 'what'
        }
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        提取所有浅层特征
        
        Args:
            text: 输入文本
            
        Returns:
            features: 包含9个特征的字典
        """
        features = {}
        
        # 1. 语法错误数量
        grammar_errors = self._count_grammar_errors(text)
        features['grammar_errors'] = grammar_errors
        
        # 2. 作文长度特征
        word_count, char_count = self._get_length_features(text)
        features['word_count'] = word_count
        features['char_count'] = char_count
        
        # 3. 语义复杂程度（单词长度的均值和方差）
        word_len_mean, word_len_var = self._get_word_complexity(text)
        features['word_len_mean'] = word_len_mean
        features['word_len_var'] = word_len_var
        
        # 4. 单词复杂程度（字符长度>6的单词数量）
        long_word_count = self._count_long_words(text)
        features['long_word_count'] = long_word_count
        
        # 5. 句子复杂程度（句子长度的均值和方差）
        sent_len_mean, sent_len_var = self._get_sentence_complexity(text)
        features['sent_len_mean'] = sent_len_mean
        features['sent_len_var'] = sent_len_var
        
        # 6. 从句数量
        clause_count = self._count_clauses(text)
        features['clause_count'] = clause_count
        
        return features
    
    def _count_grammar_errors(self, text: str) -> int:
        """
        统计语法错误数量
        包括：拼写错误、冠词误用、单复数误用等
        """
        if self.use_grammar_check and self.tool is not None:
            try:
                matches = self.tool.check(text)
                return len(matches)
            except:
                pass
        
        # 简化版本：统计常见错误
        error_count = 0
        
        # 检查冠词误用（简化版）
        # 例如：a apple -> an apple
        words = text.lower().split()
        for i in range(len(words) - 1):
            if words[i] == 'a' and words[i+1] and words[i+1][0] in 'aeiou':
                error_count += 1
            elif words[i] == 'an' and words[i+1] and words[i+1][0] not in 'aeiou':
                error_count += 1
        
        return error_count
    
    def _get_length_features(self, text: str) -> tuple:
        """
        获取作文长度特征
        Returns:
            (word_count, char_count)
        """
        # 单词数量
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        # 字符数量（不包括空格和标点）
        char_count = sum(len(word) for word in words)
        
        return word_count, char_count
    
    def _get_word_complexity(self, text: str) -> tuple:
        """
        获取语义复杂程度（单词长度的均值和方差）
        Returns:
            (mean, variance)
        """
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0, 0.0
        
        word_lengths = [len(word) for word in words]
        mean = np.mean(word_lengths)
        var = np.var(word_lengths)
        
        return float(mean), float(var)
    
    def _count_long_words(self, text: str) -> float:
        """
        统计单词复杂程度（根据长单词数量进行评分）
        评分规则：
        - 1-2个长单词(>6个字符)：1分
        - 3-4个长单词：2分
        - 5-6个长单词：3分
        - 大于6个长单词：4分
        
        Returns:
            平均得分（如果文本中没有单词则返回0）
        """
        words = re.findall(r'\b\w+\b', text)
        scores = []
        for word in words:
            if len(word) >= 7:
                scores.append(4)
            elif len(word) >= 5:
                scores.append(3)
            elif len(word) >= 3:
                scores.append(2)
            elif len(word) >= 1:
                scores.append(1)
            else:
                scores.append(0)
        score = sum(scores)/len(scores)
        
        # 如果没有单词，返回0分，否则返回计算得到的分数
        return score
    
    def _get_sentence_complexity(self, text: str) -> tuple:
        """
        获取句子复杂程度（句子长度的均值和方差）
        Returns:
            (mean, variance)
        """
        # 按句号、问号、感叹号分割句子
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0, 0.0
        
        # 计算每个句子的单词数
        sentence_lengths = []
        for sent in sentences:
            words = re.findall(r'\b\w+\b', sent)
            sentence_lengths.append(len(words))
        
        mean = np.mean(sentence_lengths)
        var = np.var(sentence_lengths)
        
        return float(mean), float(var)
    
    def _count_clauses(self, text: str) -> int:
        """
        统计从句数量
        通过检测从属连词来估计从句数量
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        clause_count = 0
        for word in words:
            if word in self.subordinating_conjunctions:
                clause_count += 1
        
        # 也可以通过逗号数量来辅助判断
        # comma_count = text.count(',')
        
        return clause_count
    
    def extract_batch_features(self, texts: List[str]) -> torch.Tensor:
        """
        批量提取特征
        
        Args:
            texts: 文本列表
            
        Returns:
            features_tensor: [batch_size, 9] 的特征张量
        """
        batch_features = []
        
        for text in texts:
            features = self.extract_features(text)
            # 按固定顺序组织特征
            feature_vector = [
                features['grammar_errors'],
                features['word_count'],
                features['char_count'],
                features['word_len_mean'],
                features['word_len_var'],
                features['long_word_count'],
                features['sent_len_mean'],
                features['sent_len_var'],
                features['clause_count']
            ]
            batch_features.append(feature_vector)
        
        return torch.tensor(batch_features, dtype=torch.float32)


class ShallowFeatureFusion(nn.Module):
    """
    浅层特征融合模块
    将提取的9个浅层特征与深度特征融合
    """
    
    def __init__(self, shallow_feature_dim=9, deep_feature_dim=1000, 
                 fusion_dim=512, dropout=0.3):
        """
        Args:
            shallow_feature_dim: 浅层特征维度（默认9）
            deep_feature_dim: 深度特征维度（来自xLSTM+Attention）
            fusion_dim: 融合后的特征维度
            dropout: dropout比率
        """
        super(ShallowFeatureFusion, self).__init__()
        
        self.shallow_feature_dim = shallow_feature_dim
        self.deep_feature_dim = deep_feature_dim
        
        # 浅层特征归一化
        self.shallow_norm = nn.BatchNorm1d(shallow_feature_dim)
        
        # 浅层特征投影
        self.shallow_proj = nn.Sequential(
            nn.Linear(shallow_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128)
        )
        
        # 深度特征投影
        self.deep_proj = nn.Linear(deep_feature_dim, fusion_dim - 128)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim)
        )
        
        # 门控机制：动态调整浅层和深层特征的权重
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )
    
    def forward(self, shallow_features, deep_features):
        """
        Args:
            shallow_features: [batch_size, 9] 浅层特征
            deep_features: [batch_size, deep_feature_dim] 深度特征
            
        Returns:
            fused_features: [batch_size, fusion_dim] 融合后的特征
        """
        # 归一化浅层特征
        shallow_norm = self.shallow_norm(shallow_features)
        
        # 投影浅层特征
        shallow_proj = self.shallow_proj(shallow_norm)  # [batch_size, 128]
        
        # 投影深度特征
        deep_proj = self.deep_proj(deep_features)  # [batch_size, fusion_dim-128]
        
        # 拼接
        concat_features = torch.cat([shallow_proj, deep_proj], dim=1)  # [batch_size, fusion_dim]
        
        # 融合
        fused = self.fusion(concat_features)
        
        # 门控
        gate = self.gate(fused)
        fused_features = gate * fused + (1 - gate) * concat_features
        
        return fused_features


# 使用示例
if __name__ == '__main__':
    # 测试特征提取
    extractor = ShallowFeatureExtractor(use_grammar_check=False)
    
    sample_text = """
    This is a sample essay for testing. The essay contains multiple sentences 
    with different lengths. Some words are quite long, such as "extraordinary" 
    and "comprehensive". We also have subordinate clauses because we want to 
    test the clause detection. Although this is just a test, it should work well.
    """
    
    features = extractor.extract_features(sample_text)
    print("提取的特征:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # 测试批量提取
    texts = [sample_text, sample_text]
    batch_features = extractor.extract_batch_features(texts)
    print(f"\n批量特征张量形状: {batch_features.shape}")
    print(f"特征值:\n{batch_features}")
