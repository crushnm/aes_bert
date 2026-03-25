"""
消融实验训练脚本
支持三种消融实验配置：
1. W/o SCConv+MTA: 去除多层语义嵌入与注意力
2. W/ BERT: 使用BERT替代Longformer
3. W/ RoBERTa: 使用RoBERTa替代Longformer
"""
import logging
import torch.utils.data as data
import tqdm
import numpy as np
import torch
from evaluate import evaluation
import optimization
from model.ALBERT_BiSRUpp_ATT import ALBERT_BiSRUpp_ATT
from Prepare_DataSet import Prepare_DataSet
import argparse

# 设置随机数
seed = 6666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def train_epoch(master_gpu_id, model, optimizer, data_loader,
                gradient_accumulation_steps, use_cuda):
    """训练一个epoch"""
    model.train()
    data_loader.dataset.is_training = True
    total_loss = 0.0
    correct_sum = 0
    process_sum = 0
    num_batch = data_loader.__len__()
    num_sample = data_loader.dataset.__len__()
    p_bar = tqdm.tqdm(data_loader, unit="batch", ncols=100)
    p_bar.set_description('train step loss')
    
    for step, batch in enumerate(p_bar):
        tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]
        token_type_ids = batch["token_type_ids"].cuda(master_gpu_id) if use_cuda else batch["token_type_ids"]
        attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda else batch["attention_mask"]
        labels = batch['labels'].cuda(master_gpu_id) if use_cuda else batch['labels']

        shallow_features = None
        if 'shallow_features' in batch:
            shallow_features = batch['shallow_features'].cuda(master_gpu_id) if use_cuda else batch['shallow_features']
        
        topic_ids = None
        if 'topic_id' in batch:
            topic_ids = batch['topic_id'].cuda(master_gpu_id) if use_cuda else batch['topic_id']

        loss, logits = model(tokens, attention_mask, token_type_ids, labels, shallow_features, topic_ids)
        loss = loss.mean()

        if gradient_accumulation_steps > 1:
            loss /= gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        
        loss_val = loss.item()
        total_loss += loss_val

        _, top_index = logits.topk(1)
        correct_sum += (top_index.view(-1) == labels).sum().item()
        process_sum += labels.shape[0]
        p_bar.set_description('train step loss ' + format(loss_val, "0.4f"))
    
    logging.info("Total Training Samples:%s ", num_sample)
    logging.info('train total avg loss:%s', total_loss / num_batch)
    logging.info("Correct Prediction: " + str(correct_sum))
    logging.info("Accuracy Rate: " + format(correct_sum / process_sum, "0.4f"))
    return total_loss / num_batch


def evaluates(master_gpu_id, model, val_data, batch_size=1,
              use_cuda=False, num_workers=2):
    """验证模型"""
    model.eval()
    test_data_loader = data.DataLoader(dataset=val_data,
                                       pin_memory=use_cuda,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       shuffle=False)
    total_loss = 0.0
    correct_sum = 0
    process_num = 0
    num_batch = test_data_loader.__len__()
    logging.info("Evaluating Model".center(60, "="))

    with torch.no_grad():
        for step, batch in enumerate(tqdm.tqdm(test_data_loader, unit="batch", ncols=100, desc="Evaluating process: ")):
            tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]
            token_type_ids = batch["token_type_ids"].cuda(master_gpu_id) if use_cuda else batch["token_type_ids"]
            attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda else batch["attention_mask"]
            labels = batch['labels'].cuda(master_gpu_id) if use_cuda else batch['labels']
            
            shallow_features = None
            if 'shallow_features' in batch:
                shallow_features = batch['shallow_features'].cuda(master_gpu_id) if use_cuda else batch['shallow_features']
            
            topic_ids = None
            if 'topic_id' in batch:
                topic_ids = batch['topic_id'].cuda(master_gpu_id) if use_cuda else batch['topic_id']
            
            loss, logits = model(tokens, attention_mask, token_type_ids, labels, shallow_features, topic_ids)
            loss = loss.mean()
            total_loss += loss.item()

            _, top_index = logits.topk(1)
            correct_sum += (top_index.view(-1) == labels).sum().item()
            process_num += labels.shape[0]
    
    logging.info('eval total avg loss:%s', format(total_loss / num_batch, "0.4f"))
    logging.info("Correct Prediction: " + str(correct_sum))
    logging.info("Accuracy Rate: " + format(correct_sum / process_num, "0.4f"))


def trains(master_gpu_id, model, epochs, optimizer, train_data, val_data,
           batch_size, gradient_accumulation_steps=1,
           use_cuda=False, num_workers=4, save_path='model.pkl'):
    """训练模型"""
    logging.info("Start Training".center(60, "="))
    train_data_loader = data.DataLoader(dataset=train_data,
                                        pin_memory=use_cuda,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True)
    for epoch in range(1, epochs + 1):
        logging.info("Training Epoch: " + str(epoch))
        avg_loss = train_epoch(master_gpu_id, model, optimizer, train_data_loader,
                               gradient_accumulation_steps, use_cuda)
        logging.info("Average Loss: " + format(avg_loss, "0.4f"))
        evaluates(master_gpu_id, model, val_data, batch_size, use_cuda, num_workers)
    
    # 保存模型
    torch.save(model, save_path)
    logging.info(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='消融实验训练脚本')
    parser.add_argument('--ablation', type=str, default='full',
                        choices=['full', 'wo_scconv_mta', 'bert', 'roberta'],
                        help='消融实验类型: full(完整模型), wo_scconv_mta(去除SCConv+MTA), bert(使用BERT), roberta(使用RoBERTa)')
    parser.add_argument('--subset', type=int, default=1, help='ASAP子集ID (1-8)')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    
    args = parser.parse_args()
    
    # 配置参数
    max_len = 512
    subset = args.subset
    
    num_classes = {
        1: 11, 2: 6, 3: 4, 4: 4, 5: 5, 6: 5, 7: 31, 8: 61,
    }
    
    tmps = {
        1: 2, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
    }
    
    # 根据消融实验类型配置模型
    if args.ablation == 'full':
        # 完整模型
        logging.info("训练完整模型 (Longformer + SCConv+MTA)")
        model = ALBERT_BiSRUpp_ATT(
            sequence_length=max_len,
            num_classes=num_classes[subset],
            xlstm_cell_type='slstm',
            use_shallow_features=True,
            use_topic_similarity=True,
            num_topics=8,
            use_scconv_mta=True,
            pretrained_model='longformer'
        )
        save_path = f'model_full_subset{subset}.pkl'
        
    elif args.ablation == 'wo_scconv_mta':
        # W/o SCConv+MTA
        logging.info("训练消融模型: W/o SCConv+MTA (去除多层语义嵌入与注意力)")
        model = ALBERT_BiSRUpp_ATT(
            sequence_length=max_len,
            num_classes=num_classes[subset],
            xlstm_cell_type='slstm',
            use_shallow_features=True,
            use_topic_similarity=True,
            num_topics=8,
            use_scconv_mta=False,  # 关键：去除SCConv+MTA
            pretrained_model='longformer'
        )
        save_path = f'model_wo_scconv_mta_subset{subset}.pkl'
        
    elif args.ablation == 'bert':
        # W/ BERT
        logging.info("训练消融模型: W/ BERT (使用BERT替代Longformer)")
        model = ALBERT_BiSRUpp_ATT(
            sequence_length=max_len,
            num_classes=num_classes[subset],
            xlstm_cell_type='slstm',
            use_shallow_features=True,
            use_topic_similarity=True,
            num_topics=8,
            use_scconv_mta=True,
            pretrained_model='bert'  # 关键：使用BERT
        )
        save_path = f'model_bert_subset{subset}.pkl'
        
    elif args.ablation == 'roberta':
        # W/ RoBERTa
        logging.info("训练消融模型: W/ RoBERTa (使用RoBERTa替代Longformer)")
        model = ALBERT_BiSRUpp_ATT(
            sequence_length=max_len,
            num_classes=num_classes[subset],
            xlstm_cell_type='slstm',
            use_shallow_features=True,
            use_topic_similarity=True,
            num_topics=8,
            use_scconv_mta=True,
            pretrained_model='roberta'  # 关键：使用RoBERTa
        )
        save_path = f'model_roberta_subset{subset}.pkl'
    
    logging.info("Initialize Model Done".center(60, "="))

    # 加载数据
    train_path = 'train_asap.xlsx'
    train_shallow_features = 'train_shallow_features.pkl'
    full_train_data = Prepare_DataSet(
        max_len=max_len,
        data_file=train_path,
        subset=subset,
        tmp=tmps[subset],
        shallow_feature_file=train_shallow_features if model.use_shallow_features else None
    )
    logging.info("Load Full Training Dataset Done, Total line: %s", full_train_data.__len__())
    
    # 分割训练集和验证集
    total_train_size = full_train_data.__len__()
    val_size = total_train_size // 4
    train_size = total_train_size - val_size
    
    train_data, val_data = torch.utils.data.random_split(
        full_train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    logging.info("Split Dataset - Training: %s, Validation: %s", train_size, val_size)

    # 初始化优化器
    training_data_len = train_data.__len__()
    epochs = args.epochs
    batch_size = args.batch_size
    gradient_accumulation_steps = 1
    init_lr = 1e-5
    warm_up_proportion = 0.1
    optimizer = optimization.init_bert_adam_optimizer(model, training_data_len, epochs, batch_size,
                                                      gradient_accumulation_steps, init_lr, warm_up_proportion)

    # 设置GPU
    gpu_ids = args.gpu
    use_cuda = gpu_ids != '-1'
    if len(gpu_ids) == 1 and use_cuda:
        master_gpu_id = int(gpu_ids)
        model = model.cuda(int(gpu_ids)) if use_cuda else model
    elif use_cuda:
        gpu_ids = [int(each) for each in gpu_ids.split(",")]
        master_gpu_id = gpu_ids[0]
        model = model.cuda(gpu_ids[0])
        logging.info("Start multi-gpu dataparallel training/evaluating...")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        master_gpu_id = None
    
    # 开始训练
    trains(master_gpu_id=master_gpu_id, model=model, epochs=epochs,
           optimizer=optimizer, train_data=train_data, val_data=val_data,
           batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
           use_cuda=use_cuda, num_workers=1, save_path=save_path)


if __name__ == '__main__':
    main()
