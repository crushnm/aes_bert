"""
从ASAP数据集中提取浅层特征并保存
"""
import pandas as pd
import numpy as np
import torch
from model.shallow_features import ShallowFeatureExtractor
from tqdm import tqdm
import pickle


def extract_and_save_features(data_file, output_file, subset=None, use_grammar_check=False):
    """
    从Excel文件中提取浅层特征并保存
    
    Args:
        data_file: 输入的Excel文件路径
        output_file: 输出的特征文件路径（.pkl格式）
        subset: 子集编号（1-8），如果为None则提取所有数据
        use_grammar_check: 是否使用语法检查工具
    """
    print(f"正在读取数据文件: {data_file}")
    df = pd.read_excel(data_file)
    
    # 根据子集筛选数据
    if subset is not None:
        print(f"筛选子集: {subset}")
        df = df.loc[df['essay_set'] == subset]
        if len(df) == 0:
            print(f"警告: 子集 {subset} 没有数据")
            return None
    
    # 初始化特征提取器
    print("初始化特征提取器...")
    extractor = ShallowFeatureExtractor(use_grammar_check=use_grammar_check)
    
    # 提取特征
    print("开始提取浅层特征...")
    all_features = []
    all_feature_dicts = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取特征"):
        # 假设文本列名为 'essay' 或 'text'
        if 'essay' in df.columns:
            text = str(row['essay'])
        elif 'text' in df.columns:
            text = str(row['text'])
        else:
            # 尝试找到包含文本的列
            text_col = [col for col in df.columns if 'text' in col.lower() or 'essay' in col.lower()]
            if text_col:
                text = str(row[text_col[0]])
            else:
                raise ValueError("找不到文本列，请检查数据格式")
        
        # 提取特征
        features = extractor.extract_features(text)
        all_feature_dicts.append(features)
        
        # 转换为向量
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
        all_features.append(feature_vector)
    
    # 转换为numpy数组
    features_array = np.array(all_features, dtype=np.float32)
    
    # 保存特征
    print(f"保存特征到: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'features': features_array,
            'feature_dicts': all_feature_dicts,
            'feature_names': [
                'grammar_errors', 'word_count', 'char_count',
                'word_len_mean', 'word_len_var', 'long_word_count',
                'sent_len_mean', 'sent_len_var', 'clause_count'
            ],
            'subset': subset
        }, f)
    
    # 打印统计信息
    print("\n特征统计信息:")
    print(f"样本数量: {len(features_array)}")
    print(f"特征维度: {features_array.shape[1]}")
    print("\n各特征的均值和标准差:")
    feature_names = [
        'grammar_errors', 'word_count', 'char_count',
        'word_len_mean', 'word_len_var', 'long_word_count',
        'sent_len_mean', 'sent_len_var', 'clause_count'
    ]
    for i, name in enumerate(feature_names):
        mean = np.mean(features_array[:, i])
        std = np.std(features_array[:, i])
        print(f"  {name}: mean={mean:.2f}, std={std:.2f}")
    
    return features_array


def extract_features_by_subset(data_file, output_prefix, use_grammar_check=False):
    """
    按子集提取浅层特征并保存为多个文件
    
    Args:
        data_file: 输入的Excel文件路径
        output_prefix: 输出文件前缀
        use_grammar_check: 是否使用语法检查工具
    """
    print(f"按子集提取浅层特征: {data_file}")
    
    # 遍历所有子集（1-8）
    for subset in range(1, 9):
        print("\n" + "=" * 60)
        print(f"处理子集: {subset}")
        print("=" * 60)
        
        # 生成输出文件名
        output_file = f"{output_prefix}_subset_{subset}.pkl"
        
        # 提取并保存特征
        extract_and_save_features(data_file, output_file, subset=subset, use_grammar_check=use_grammar_check)
    
    print("\n所有子集的特征提取完成！")


def load_shallow_features(feature_file):
    """
    加载保存的浅层特征
    
    Args:
        feature_file: 特征文件路径
        
    Returns:
        features: numpy数组或torch张量
    """
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)
    return data['features']


if __name__ == '__main__':
    # 提取训练集特征（按子集）
    train_file = 'D:/PyCharm/py_project/aes_bert/aes_bert-bisru++-attention/train_asap.xlsx'
    train_output_prefix = 'train_shallow_features'
    
    print("=" * 60)
    print("按子集提取训练集浅层特征")
    print("=" * 60)
    extract_features_by_subset(train_file, train_output_prefix, use_grammar_check=False)
    
    # 提取测试集特征（按子集）
    test_file = 'D:/PyCharm/py_project/aes_bert/aes_bert-bisru++-attention/test_asap.xlsx'
    test_output_prefix = 'test_shallow_features'
    
    print("\n" + "=" * 60)
    print("按子集提取测试集浅层特征")
    print("=" * 60)
    extract_features_by_subset(test_file, test_output_prefix, use_grammar_check=False)
    
    print("\n特征提取完成！")
