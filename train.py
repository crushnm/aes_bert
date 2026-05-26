"""
"""
import sys
import io
# 设置标准输出为UTF-8编码，解决Linux上的Unicode编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import logging
import torch.utils.data as data
import tqdm
import numpy as np
import torch
# from radam import radam
# 设置随机数 使实现可复现
from evaluate import evaluation
import optimization
# from model.BERT_BiSRUpp_MACNN_ATT import BERT_B_MACNN_ATT
# from model.bert_CNN_BiSRUpp_ATT import bert_CNN_BiSRUpp_ATT
from model.ALBERT_BiSRUpp_ATT import ALBERT_BiSRUpp_ATT
# from model.ALBERT_BiSAGRU_AT import ALBETR_BiSARNN_AT
from Prepare_DataSet import Prepare_DataSet

seed = 6666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

logging.basicConfig(level=logging.INFO,  # 指定根日志记录器级别
                    format="%(asctime)s - %(levelname)s - %(message)s")  # 使用指定字符串格式


def evaluates_pr(labels, predictions):
    TP, TN, FP, FN = 0, 0, 0, 0
    for label, prediction in zip(labels, predictions):
        if label == 1 and prediction == 1:
            TP += 1
        elif label == 0 and prediction == 0:
            TN += 1
        elif label == 1 and prediction == 0:
            FP += 1
        elif label == 0 and prediction == 1:
            FN += 1
    precise = TP / (TP + FN + 0.0001)
    recall = TP / (TP + FP + 0.0001)
    return precise, recall


# 利用混淆矩阵进行多标签分类计算每一个类的F1
def evaluates_mul_pr(labels, predictions):
    hx = [[0] * 8 for _ in range(8)]  # 混淆矩阵
    for label, prediction in zip(labels, predictions):
        # 真实标签对应预测的标签 0,0
        hx[label][prediction] += 1  # 对应位置+1
    # 等待统计完成 对混淆矩阵进行pr的计算
    tran = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H'}
    for i in range(len(hx)):
        rowsum, colsum = sum(hx[i]) + 0.001, sum(hx[r][i] for r in range(len(hx))) + 0.001
        p = hx[i][i] / float(colsum)
        r = hx[i][i] / float(rowsum)
        f1 = 2 * p * r / (p + r)
        logging.info('部' + tran[i] + ':precision: ' + str(p) + ' recall: ' + str(r) + ' f1-score: ' + str(f1))


def att_score_visual(att_score, words, seq_len):
    # att_score:[batch_size, seq_len, 1]
    # words:[batch_size,]
    f = open('attention.html', 'w')
    f.write('<html style="margin:0;padding:0;"><body style="margin:0;padding:0;">\n')
    f.write('<div style="margin:25px;">\n')
    # 一句话->每个字->红色注意力得分
    # 红色越深 得分越高
    for i in range(len(att_score)):
        f.write('<p style="margin:10px;">\n')
        att = att_score[i]  # 第i个句子的注意力得分
        for j in range(len(words[i])):  # 第i个句子的第j个词
            alpha = "{:.2f}".format(att)  # 保留两位小数
            f.write('\t<span style="margin-left:3px;background-color:rgba(255,0,0,{0})">{1}</span>\n'
                    .format(alpha, words[j]))
        f.write('</p>\n')
    f.write('</div>\n')
    f.write('</body></html>')
    f.close()


def evaluates(master_gpu_id, model, val_data, batch_size=1,
              use_cuda=False, num_workers=2):
    model.eval()  # 测试模式
    test_data_loader = data.DataLoader(dataset=val_data,
                                       pin_memory=use_cuda,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       shuffle=False)
    total_loss = 0.0
    correct_sum = 0
    process_num = 0
    infos = []  # 存储预测的index 和 label
    num_batch = test_data_loader.__len__()
    logging.info("Evaluating Model".center(60, "="))  # 开始验证模型性能

    for step, batch in enumerate(tqdm.tqdm(test_data_loader, unit="batch", ncols=100, desc="Evaluating process: ")):
        tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]  # 获取token
        token_type_ids = batch["token_type_ids"].cuda(master_gpu_id) if use_cuda else batch[
            "token_type_ids"]  # 获取token_type_ids
        attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda else batch[
            "attention_mask"]  # 获取attention_mask
        labels = batch['labels'].cuda(master_gpu_id) if use_cuda else batch['labels']
        
        # 获取浅层特征（如果有）
        shallow_features = None
        if 'shallow_features' in batch:
            shallow_features = batch['shallow_features'].cuda(master_gpu_id) if use_cuda else batch['shallow_features']
        
        # 获取主题ID（如果有）
        topic_ids = None
        if 'topic_id' in batch:
            topic_ids = batch['topic_id'].cuda(master_gpu_id) if use_cuda else batch['topic_id']
        
        # 获取句子位置特征（如果有）
        indices = None
        if 'indices' in batch:
            indices = batch['indices'].cuda(master_gpu_id) if use_cuda else batch['indices']
        
        # 获取句子级输入（如果有）
        sentence_tokens = None
        sentence_attention_masks = None
        sentence_token_type_ids = None
        if 'sentence_tokens' in batch:
            sentence_tokens = batch['sentence_tokens'].cuda(master_gpu_id) if use_cuda else batch['sentence_tokens']
            sentence_attention_masks = batch['sentence_attention_masks'].cuda(master_gpu_id) if use_cuda else batch['sentence_attention_masks']
            sentence_token_type_ids = batch['sentence_token_type_ids'].cuda(master_gpu_id) if use_cuda else batch['sentence_token_type_ids']
        
        loss, logits = model(tokens, attention_mask, token_type_ids, labels, shallow_features, topic_ids, indices, sentence_tokens, sentence_attention_masks, sentence_token_type_ids)
        loss = loss.mean()
        loss_val = loss.item()  # 取值
        total_loss += loss_val  # 统计总的损失值

        # 返回batch个样本的最大值的[key,index]
        _, top_index = logits.topk(1)  # 取最大值 即预测概率最大的那一类
        # 统计batch个label相等的个数并求和
        correct_sum += (top_index.view(-1) == labels).sum().item()  # 进行预测是否正确的判断
        process_num += labels.shape[0]  # 统计总的样本数
        # 存储预测和真实标签
        for label, prediction in zip(labels, top_index.view(-1)):
            infos.append((prediction, label))
    logging.info('eval total avg loss:%s', format(total_loss / num_batch, "0.4f"))  # 验证结束 打印结果
    logging.info("Correct Prediction: " + str(correct_sum))
    logging.info("Accuracy Rate: " + format(correct_sum / process_num, "0.4f"))
    # 计算精确率和召回率
    labels = [info[0] for info in infos]
    predictions = [info[1] for info in infos]

    #result = evaluation(labels, predictions)
    #logging.info("验证结果：" + str(result))

    # precision, recall = evaluates_pr(labels, predictions)
    # evaluates_mul_pr(labels, predictions)


def test_model(master_gpu_id, model, test_data, batch_size=1,
               use_cuda=False, num_workers=2):
    """测试模型"""
    model.eval()  # 测试模式
    test_data_loader = data.DataLoader(dataset=test_data,
                                       pin_memory=use_cuda,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       shuffle=False)
    total_loss = 0.0
    correct_sum = 0
    process_num = 0
    true_labels = []  # 存储真实标签
    pred_labels = []  # 存储预测标签
    num_batch = test_data_loader.__len__()
    logging.info("Testing Model".center(60, "="))  # 开始测试模型性能

    with torch.no_grad():  # 测试时不需要计算梯度
        for step, batch in enumerate(tqdm.tqdm(test_data_loader, unit="batch", ncols=100, desc="Testing process: ")):
            tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]
            token_type_ids = batch["token_type_ids"].cuda(master_gpu_id) if use_cuda else batch["token_type_ids"]
            attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda else batch["attention_mask"]
            labels = batch['labels'].cuda(master_gpu_id) if use_cuda else batch['labels']
            
            # 获取浅层特征（如果有）
            shallow_features = None
            if 'shallow_features' in batch:
                shallow_features = batch['shallow_features'].cuda(master_gpu_id) if use_cuda else batch['shallow_features']
            
            # 获取主题ID（如果有）
            topic_ids = None
            if 'topic_id' in batch:
                topic_ids = batch['topic_id'].cuda(master_gpu_id) if use_cuda else batch['topic_id']

            # 获取句子位置特征（如果有）
            indices = None
            if 'indices' in batch:
                indices = batch['indices'].cuda(master_gpu_id) if use_cuda else batch['indices']

            # 获取句子级输入（如果有）
            sentence_tokens = None
            sentence_attention_masks = None
            sentence_token_type_ids = None
            if 'sentence_tokens' in batch:
                sentence_tokens = batch['sentence_tokens'].cuda(master_gpu_id) if use_cuda else batch['sentence_tokens']
                sentence_attention_masks = batch['sentence_attention_masks'].cuda(master_gpu_id) if use_cuda else batch['sentence_attention_masks']
                sentence_token_type_ids = batch['sentence_token_type_ids'].cuda(master_gpu_id) if use_cuda else batch['sentence_token_type_ids']

            loss, logits = model(tokens, attention_mask, token_type_ids, labels, shallow_features, topic_ids, indices, sentence_tokens, sentence_attention_masks, sentence_token_type_ids)
            loss = loss.mean()
            loss_val = loss.item()
            total_loss += loss_val

            # 获取预测结果
            _, top_index = logits.topk(1)
            correct_sum += (top_index.view(-1) == labels).sum().item()
            process_num += labels.shape[0]

            # 存储真实标签和预测标签
            true_labels.extend(labels.cpu().numpy().tolist())
            pred_labels.extend(top_index.view(-1).cpu().numpy().tolist())
    
    # 输出测试结果
    logging.info('test total avg loss:%s', format(total_loss / num_batch, "0.4f"))
    logging.info("Correct Prediction: " + str(correct_sum))
    logging.info("Accuracy Rate: " + format(correct_sum / process_num, "0.4f"))
    
    # 使用evaluation函数计算详细指标
    result = evaluation(true_labels, pred_labels)
    logging.info("测试结果：" + str(result))
    logging.info("测试指标说明：[0分差, 0.5分差, 1分差, 1.5分差, >=2分差, <=0.5分差率, <=1分差率, Pearson相关系数, QWK]")
    
    return result


def train_epoch(master_gpu_id, model, optimizer, data_loader,
                gradient_accumulation_steps, use_cuda):
    model.train()  # 训练模式
    data_loader.dataset.is_training = True
    total_loss = 0.0  # 总共的损失
    correct_sum = 0  # 正确预测总和
    process_sum = 0  # 处理数据的总和
    num_batch = data_loader.__len__()
    num_sample = data_loader.dataset.__len__()
    p_bar = tqdm.tqdm(data_loader, unit="batch", ncols=100)  # 进度条封装
    p_bar.set_description('train step loss')
    for step, batch in enumerate(p_bar):
        # # print({"text": essay, "indices": indices, "indexed_tokens": indexed_tokens, "segments_ids": segments_ids,
        #         #        "label": int(score)})
        
        
        
        tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]  # 获取token
        token_type_ids = batch["token_type_ids"].cuda(master_gpu_id) if use_cuda else batch[
            "token_type_ids"]  # 获取token_type_ids
        attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda else batch[
            "attention_mask"]  # 获取attention_mask
        labels = batch['labels'].cuda(master_gpu_id) if use_cuda else batch['labels']

        # 获取浅层特征（如果有）
        shallow_features = None
        if 'shallow_features' in batch:
            shallow_features = batch['shallow_features'].cuda(master_gpu_id) if use_cuda else batch['shallow_features']
        
        # 获取主题ID（如果有）
        topic_ids = None
        if 'topic_id' in batch:
            topic_ids = batch['topic_id'].cuda(master_gpu_id) if use_cuda else batch['topic_id']
        
        # 获取句子位置特征（如果有）
        indices = None
        if 'indices' in batch:
            indices = batch['indices'].cuda(master_gpu_id) if use_cuda else batch['indices']

        # 获取句子级输入（如果有）
        sentence_tokens = None
        sentence_attention_masks = None
        sentence_token_type_ids = None
        if 'sentence_tokens' in batch:
            sentence_tokens = batch['sentence_tokens'].cuda(master_gpu_id) if use_cuda else batch['sentence_tokens']
            sentence_attention_masks = batch['sentence_attention_masks'].cuda(master_gpu_id) if use_cuda else batch['sentence_attention_masks']
            sentence_token_type_ids = batch['sentence_token_type_ids'].cuda(master_gpu_id) if use_cuda else batch['sentence_token_type_ids']

        loss, logits = model(tokens, attention_mask, token_type_ids, labels, shallow_features, topic_ids, indices, sentence_tokens, sentence_attention_masks, sentence_token_type_ids)

        # 取平均
        loss = loss.mean()

        if gradient_accumulation_steps > 1:  # 梯度累积次数 其实意思就是去多次的平均值
            # 如果gradient_accumulation_steps == 1的话 代表不累计梯度 即不与前面的mini-batch相关
            loss /= gradient_accumulation_steps
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()  # 梯度下降
            model.zero_grad()  # 梯度归零
        loss_val = loss.item()
        total_loss += loss_val

        # 统计
        _, top_index = logits.topk(1)
        correct_sum += (top_index.view(-1) == labels).sum().item()
        process_sum += labels.shape[0]
        p_bar.set_description('train step loss ' + format(loss_val, "0.4f"))
    logging.info("Total Training Samples:%s ", num_sample)
    logging.info('train total avg loss:%s', total_loss / num_batch)
    logging.info("Correct Prediction: " + str(correct_sum))
    logging.info("Accuracy Rate: " + format(correct_sum / process_sum, "0.4f"))
    return total_loss / num_batch  # 返回平均每一个batch的损失值


def saved_model(model, path):
    pass


def trains(master_gpu_id, model, epochs, optimizer, train_data, val_data, test_data,
           batch_size, gradient_accumulation_steps=1,
           use_cuda=False, num_workers=4):
    logging.info("Start Training".center(60, "="))
    train_data_loader = data.DataLoader(dataset=train_data,
                                        pin_memory=use_cuda,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True)  # 是否打乱
    # for epoch in range(1, epochs + 1):
    #     logging.info("Training Epoch: " + str(epoch))
    #     avg_loss = train_epoch(master_gpu_id, model, optimizer, train_data_loader,
    #                            gradient_accumulation_steps, use_cuda)
    #     logging.info("Average Loss: " + format(avg_loss, "0.4f"))
    #     evaluates(master_gpu_id, model, val_data, batch_size, use_cuda, num_workers)  # 验证模型
    # # 保存整个模型
    # torch.save(model, "my_model_woXLSTM.pkl")
    #
    # # 训练完成后进行测试
    # logging.info("Training Completed, Starting Test Phase".center(60, "="))
    test_model(master_gpu_id, model, test_data, batch_size, use_cuda, num_workers)
import warnings
warnings.filterwarnings('ignore')

def main():
    max_len = 512
    subset = 1
    num_classes = {
        1: 11,  # 2-12
        2: 6,  # 1-6
        3: 4,  # 0-3
        4: 4,  # 0-3
        5: 5,  # 0-4
        6: 5,  # 0-4
        7: 31,  # 0-30
        8: 61,  # 0-60
    }
    
    tmps = {
        1: 2,  # 2-12
        2: 1,  # 1-6
        3: 0,  # 0-3
        4: 0,  # 0-3
        5: 0,  # 0-4
        6: 0,  # 0-4
        7: 0,  # 0-30
        8: 0,  # 0-60
    }
    print(num_classes[subset])

    # 初始化模型
    # xlstm_cell_type: 'slstm' (scalar LSTM) 或 'mlstm' (matrix LSTM)
    # use_shallow_features: 是否使用浅层特征
    # use_topic_similarity: 是否使用主题相似度特征
    model = ALBERT_BiSRUpp_ATT(
        sequence_length=max_len,
        num_classes=num_classes[subset],
        xlstm_cell_type='slstm',  # 可以改为 'mlstm' 使用矩阵LSTM
        use_shallow_features=True,  # 启用浅层特征融合
        use_topic_similarity=True,  # 启用主题相似度计算
        num_topics=8  # ASAP数据集有8个子集
    )
    # 输出模型参数量
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))

    # 输出模型信息
    # logging.info(model)
    logging.info("Initialize Model Done".center(60, "="))

    # 初始化数据集 训练集和测试集
    # 改这里就行
    train_path = 'train_asap.xlsx'
    train_shallow_features = 'train_shallow_features.pkl'  # 浅层特征文件
    full_train_data = Prepare_DataSet(
        max_len=max_len, 
        data_file=train_path,
        subset=subset,
        tmp=tmps[subset],
        shallow_feature_file=train_shallow_features if model.use_shallow_features else None
    )
    logging.info("Load Full Training Dataset Done, Total line: %s", full_train_data.__len__())
    
    # 从训练集中分出1/4作为验证集
    total_train_size = full_train_data.__len__()
    val_size = total_train_size // 4
    train_size = total_train_size - val_size
    
    train_data, val_data = torch.utils.data.random_split(
        full_train_data, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)  # 使用固定种子保证可复现
    )
    logging.info("Split Dataset - Training: %s, Validation: %s", train_size, val_size)

    test_path = 'test_asap.xlsx'
    test_shallow_features = 'test_shallow_features.pkl'  # 浅层特征文件
    test_data = Prepare_DataSet(
        max_len=max_len, 
        data_file=test_path,
        subset=subset,
        tmp=tmps[subset],
        shallow_feature_file=test_shallow_features if model.use_shallow_features else None
    )
    logging.info("Load Test Dataset Done, Total test line: %s", test_data.__len__())

    # 初始化优化器
    # 超参数设置
    training_data_len = train_data.__len__()
    epochs = 20
    batch_size = 2
    gradient_accumulation_steps = 1
    init_lr = 1e-5
    warm_up_proportion = 0.1
    optimizer = optimization.init_bert_adam_optimizer(model, training_data_len, epochs, batch_size,
                                                      gradient_accumulation_steps, init_lr, warm_up_proportion)

    # 设置GPU
    gpu_ids = '0'
    # gpu_ids = '0,1'
    use_cuda = gpu_ids != '-1'
    if len(gpu_ids) == 1 and use_cuda:  # 一个gpu
        master_gpu_id = int(gpu_ids)  # 主master
        model = model.cuda(int(gpu_ids)) if use_cuda else model
    elif use_cuda:  # 多个gpu
        gpu_ids = [int(each) for each in gpu_ids.split(",")]  # ','分割的int数字
        master_gpu_id = gpu_ids[0]
        model = model.cuda(gpu_ids[0])  # Moves all model parameters and buffers to the GPU
        logging.info("Start multi-gpu dataparallel training/evaluating...")
        # 前提是在device_ids[0]中保存有parameters and buffers 即 model.cuda(gpu_ids[0])
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)  # 自动拷贝参数和缓存到所有的GPUs上
    else:  # 不使用gpu
        master_gpu_id = None
    # 开始训练
    trains(master_gpu_id=master_gpu_id, model=model, epochs=epochs,
           optimizer=optimizer, train_data=train_data, val_data=val_data, test_data=test_data,
           batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
           use_cuda=use_cuda, num_workers=1)


if __name__ == '__main__':
    main()
