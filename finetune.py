import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import os
import random
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seq_emb_gen import EnhancedAccountTransformer as SeqModel,MIXER_TO_IDX,MIXER_ADDRESSES,TRANSACTION_TYPE_TO_IDX,TRANSACTION_TYPES
import dgl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime
import json
def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class JointEmbeddingModel(nn.Module):
    def __init__(self, graph_model, seq_model, projection_dim=128, 
                 freeze_graph_layers=0, freeze_seq_layers=0):
        super(JointEmbeddingModel, self).__init__()
        self.graph_model = graph_model
        self.seq_model = seq_model
        # 只解冻最后一层
        for i, conv in enumerate(graph_model.encoder.convs):
            requires_grad = (i == len(graph_model.encoder.convs) - 1)  
            for param in conv.parameters():
                param.requires_grad = requires_grad

        for i, layer in enumerate(seq_model.transformer_encoder.layers):
            requires_grad = (i == len(seq_model.transformer_encoder.layers) - 1)
            for param in layer.parameters():
                param.requires_grad = requires_grad   

        # #冻结encoder层
        # for param in self.graph_model.encoder.parameters():
        #     param.requires_grad = False
        # for param in self.seq_model.transformer_encoder.parameters():
        #     param.requires_grad = False
        # 获取输入特征维度
        self.graph_dim = 32  # 假设图模型输出维度为32
        self.seq_dim = 64    # 假设序列模型输出维度为64

        # 投影头
        # self.projector = nn.Sequential(
        #     nn.Linear(self.graph_dim + self.seq_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, projection_dim)
        # )
        self.projector = nn.Sequential(
            nn.Linear(self.graph_dim + self.seq_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),  # 增加dropout率
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, projection_dim)
        )
    def forward(self, graph_x, seq_x):
        try:
            # 获取图数据
            x = graph_x.ndata['features']
            edge_index = torch.stack(graph_x.edges())
            edge_attr = graph_x.edata.get('features', None)
            
            # 获取batch信息
            batch_num_nodes = graph_x.batch_num_nodes()
            batch = []
            for i, num_nodes in enumerate(batch_num_nodes):
                batch.extend([i] * num_nodes.item())
            batch = torch.tensor(batch, dtype=torch.long, device=x.device)
            
            # 获取图嵌入 - 先获取64维的图表示，然后通过projection得到32维
            graph_embeddings = self.graph_model(x, edge_index, batch, edge_attr)  # [batch_size, 64]
            graph_features = self.graph_model.projection(graph_embeddings)  # [batch_size, 32]
            
            # 获取序列特征
            seq_output = self.seq_model(seq_x)  # [batch_size, 64]
            seq_features = seq_output['global_repr']
            
            # 检查维度
            assert graph_features.size(0) == seq_features.size(0), \
                f"Batch size mismatch: graph={graph_features.size(0)}, seq={seq_features.size(0)}"
            assert graph_features.size(1) == self.graph_dim, \
                f"Graph feature dimension mismatch: {graph_features.size(1)} != {self.graph_dim}"
            assert seq_features.size(1) == self.seq_dim, \
                f"Sequence feature dimension mismatch: {seq_features.size(1)} != {self.seq_dim}"
            graph_features = F.normalize(graph_features, p=2, dim=1)
            seq_features = F.normalize(seq_features, p=2, dim=1)
    
            # 拼接特征 [batch_size, 96]
            joint_features = torch.cat([graph_features, seq_features], dim=1)
            
            # 投影 [batch_size, projection_dim]
            projected = self.projector(joint_features)
            projected = F.normalize(projected, p=2, dim=1)
            
            return projected
            
        except Exception as e:
            print(f"Forward pass error: {e}")
            print(f"Graph info: nodes={graph_x.number_of_nodes()}, edges={graph_x.number_of_edges()}")
            print(f"Batch num nodes: {graph_x.batch_num_nodes()}")
            if 'graph_features' in locals():
                print(f"graph_features shape: {graph_features.shape}")
            if 'seq_features' in locals():
                print(f"seq_features shape: {seq_features.shape}")
            if 'joint_features' in locals():
                print(f"joint_features shape: {joint_features.shape}")
            raise

def load_data():
    """加载训练数据"""
    # 加载预处理的子图数据
    processed_dir = '/data2/wzy/experiment/Dataset/Graph/processed_2ego/'
    
    # 加载交易序列数据
    with open('/data2/wzy/experiment/Dataset/Seq/node_sequences.pkl', 'rb') as f:
        transactions_dict = pickle.load(f)
    
    # 直接从pair_edges.csv读取节点对及其标签
    edges_path = '/data2/wzy/experiment/Dataset/D/train_pair_edges.csv'
    df_edges = pd.read_csv(edges_path)
    
    # 打乱数据
    df_edges = df_edges.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_edges, processed_dir, transactions_dict

class NodePairDataset(Dataset):
    """节点对数据集，保持与预训练时相同的输入格式"""
    def __init__(self, df_edges, processed_dir, transactions_dict, max_seq_length=30):
        self.df_edges = df_edges
        self.processed_dir = processed_dir
        self.transactions_dict = transactions_dict
        self.max_seq_length = max_seq_length
        
        # 过滤有效的节点对（同时具有子图数据和交易序列）
        valid_pairs = []
        valid_labels = []
        # import pdb;pdb.set_trace()
        
        for _, row in df_edges.iterrows():
            node1, node2 = row['node1'], row['node2']
            label = row['label']
            
            # 检查子图数据是否存在
            graph1_path = os.path.join(processed_dir, f"{node1}_2hop_topk20.pt")
            graph2_path = os.path.join(processed_dir, f"{node2}_2hop_topk20.pt")
            
            # 检查交易序列是否存在
            has_seq1 = node1 in transactions_dict
            has_seq2 = node2 in transactions_dict

            if (os.path.exists(graph1_path) and os.path.exists(graph2_path) and 
                has_seq1 and has_seq2):
                valid_pairs.append((node1, node2))
                valid_labels.append(label)
        
        self.node_pairs = valid_pairs
        self.labels = valid_labels
        print(f"有效节点对数量: {len(self.node_pairs)}")
        
    def __len__(self):
        return len(self.node_pairs)
        
    def __getitem__(self, idx):
        node1, node2 = self.node_pairs[idx]
        label = self.labels[idx]
        
        # 加载子图数据
        graph_data1 = torch.load(os.path.join(self.processed_dir, f"{node1}_2hop_topk20.pt"))
        graph_data2 = torch.load(os.path.join(self.processed_dir, f"{node2}_2hop_topk20.pt"))
        
        # 处理交易序列数据
        seq1 = self.transactions_dict[node1]
        seq2 = self.transactions_dict[node2]
        
        # 准备序列特征（与EthereumTransactionDataset保持一致）
        def prepare_seq_features(transactions):
            features = []
            for tx in transactions[:self.max_seq_length]:
                to_node, timestamp, value, method, gas_price, tx_count = tx
                
                # 获取混合器索引和交易类型索引
                try:
                    mixer_idx = MIXER_TO_IDX.get(to_node, len(MIXER_ADDRESSES))
                except:
                    mixer_idx = len(MIXER_ADDRESSES)
                    
                try:
                    tx_type_idx = TRANSACTION_TYPE_TO_IDX.get(method, len(TRANSACTION_TYPES))
                except:
                    tx_type_idx = len(TRANSACTION_TYPES)
                
                # 归一化特征
                norm_timestamp = timestamp / (10**10)
                
                features.append([mixer_idx, norm_timestamp, tx_type_idx, gas_price, tx_count])
            
            # 填充序列
            padding_length = self.max_seq_length - len(features)
            if padding_length > 0:
                n_mixers = len(MIXER_ADDRESSES)
                n_tx_types = len(TRANSACTION_TYPES)
                padding = [[n_mixers, 0, n_tx_types, 0, 0]] * padding_length
                features = features + padding
            
            return torch.tensor(features, dtype=torch.float32)
        
        seq_data1 = prepare_seq_features(seq1)
        seq_data2 = prepare_seq_features(seq2)
        
        return {
            'graph_data1': graph_data1,
            'graph_data2': graph_data2,
            'seq_data1': seq_data1,
            'seq_data2': seq_data2,
            'label': torch.tensor(label, dtype=torch.long)
        }

# def contrastive_loss(features1, features2, labels, margin=5):
#     batch_size = features1.shape[0]
    
#     # 计算欧氏距离的平方
#     euclidean_distance = F.pairwise_distance(features1, features2, p=2)
#     euclidean_distance_sq = euclidean_distance ** 2
    
#     # 对比损失公式: 
#     # L = (1-Y) * (1/2) * D² + Y * (1/2) * max(0, margin - D)²
#     # 其中Y=1表示相关节点对，Y=0表示不相关节点对
    
#     # 正样本损失：最小化距离
#     positive_loss = labels * euclidean_distance_sq
    
#     # 负样本损失：将距离推到至少为margin
#     # max(0, margin - distance)²
#     negative_loss = (1 - labels) * torch.clamp(margin - euclidean_distance, min=0) ** 2
    
#     # 总损失
#     loss = torch.mean(0.5 * (positive_loss + negative_loss))
    
#     return loss
def contrastive_loss(features1, features2, labels, margin=5.0):  # 降低margin值
    batch_size = features1.shape[0]
    euclidean_distance = F.pairwise_distance(features1, features2, p=2)
    euclidean_distance_sq = euclidean_distance ** 2
    
    # 添加温度参数
    temperature = 0.07
    positive_loss = labels * euclidean_distance_sq
    negative_loss = (1 - labels) * torch.clamp(margin - euclidean_distance, min=0) ** 2
    
    # 添加权重平衡
    pos_weight = 1  
    loss = torch.mean(0.5 * (pos_weight * positive_loss + negative_loss))
    
    return loss
import logging
import os
from datetime import datetime

def train_joint_model(joint_model, train_loader, val_loader, epochs=50, lr=0.0001, device='cuda'):
    """训练联合模型并计算评估指标"""
    # 设置日志记录
    log_dir = "/data2/wzy/experiment/magic/subgraph_level/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, joint_model.parameters()),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    best_f1 = 0
    best_model_state = None
    best_metrics = None
    
    # 设置指标阈值
    min_f1 = 0.85  # 最小F1分数要求
    min_precision = 0.85  # 最小精确率要求
    min_recall = 0.85  # 最小召回率要求
    
    # thresholds = np.linspace(0.3, 0.9, 9)
    thresholds = [0.9]
    # thresholds = [0.7]
    for epoch in range(epochs):
        joint_model.train()
        total_loss = 0
        batch_count = 0
        
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            if batch is None:
                continue
                
            graph_data1 = batch['graph_data1'].to(device)
            graph_data2 = batch['graph_data2'].to(device)
            seq_data1 = batch['seq_data1'].to(device)
            seq_data2 = batch['seq_data2'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            features1 = joint_model(graph_data1, seq_data1)
            features2 = joint_model(graph_data2, seq_data2)
            
            loss = contrastive_loss(features1, features2, labels, margin=2.0)
            
            with torch.no_grad():
                cos_sim = F.cosine_similarity(features1, features2)
                predictions = (cos_sim + 1) / 2
                train_preds.extend(predictions.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(joint_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / max(batch_count, 1)
        
        train_metrics = calculate_metrics_with_best_threshold(
            np.array(train_preds), 
            np.array(train_labels),
            thresholds
        )
        
        # 验证
        joint_model.eval()
        val_loss = 0
        val_count = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                    
                graph_data1 = batch['graph_data1'].to(device)
                graph_data2 = batch['graph_data2'].to(device)
                seq_data1 = batch['seq_data1'].to(device)
                seq_data2 = batch['seq_data2'].to(device)
                labels = batch['labels'].to(device)
                
                features1 = joint_model(graph_data1, seq_data1)
                features2 = joint_model(graph_data2, seq_data2)
                
                loss = contrastive_loss(features1, features2, labels, margin=2.0)
                
                cos_sim = F.cosine_similarity(features1, features2)
                predictions = (cos_sim + 1) / 2
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                val_loss += loss.item()
                val_count += 1
        
        avg_val_loss = val_loss / max(val_count, 1)
        scheduler.step(avg_val_loss)
        
        val_metrics = calculate_metrics_with_best_threshold(
            np.array(val_preds), 
            np.array(val_labels),
            thresholds
        )
        
        # 修改保存最佳模型的策略
        current_f1 = val_metrics['f1']
        current_precision = val_metrics['precision']
        current_recall = val_metrics['recall']
        
        # 检查是否满足所有指标要求
        if (current_f1 >= min_f1 and 
            current_precision >= min_precision and 
            current_recall >= min_recall):
            
            # 如果满足基本要求，进一步检查是否比之前的最佳模型更好
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_state = {k: v.cpu().clone() for k, v in joint_model.state_dict().items()}
                best_metrics = val_metrics
                logging.info(f"\n保存新的最佳模型，当前指标:")
                logging.info(f"  F1 Score: {current_f1:.4f}")
                logging.info(f"  Precision: {current_precision:.4f}")
                logging.info(f"  Recall: {current_recall:.4f}")
                logging.info(f"  AUC: {val_metrics['auc']:.4f}")
                logging.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                logging.info(f"  FPR: {val_metrics['fpr']:.4f}")
                logging.info(f"  Best Threshold: {val_metrics['threshold']:.4f}")
        else:
            # 记录当前指标，但不保存模型
            logging.info(f"\n当前指标未达到要求:")
            logging.info(f"  F1 Score: {current_f1:.4f} (需要 >= {min_f1})")
            logging.info(f"  Precision: {current_precision:.4f} (需要 >= {min_precision})")
            logging.info(f"  Recall: {current_recall:.4f} (需要 >= {min_recall})")
    
    # 如果没有找到满足所有要求的模型，使用F1分数最高的模型
    if best_model_state is None:
        logging.info("\n未找到同时满足所有指标要求的模型，使用F1分数最高的模型")
        best_model_state = {k: v.cpu().clone() for k, v in joint_model.state_dict().items()}
        best_metrics = val_metrics
    
    # 加载最佳模型状态
    if best_model_state is not None:
        joint_model.load_state_dict(best_model_state)
    
    return joint_model, best_metrics
def calculate_metrics_with_best_threshold(predictions, labels, thresholds):
    best_f1 = 0
    best_metrics = None
    
    for threshold in thresholds:
        binary_preds = (predictions > threshold).astype(int)
        
        # 计算混淆矩阵
        cm = confusion_matrix(labels, binary_preds)
        tn, fp, fn, tp = cm.ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            'auc': roc_auc_score(labels, predictions),
            'accuracy': accuracy_score(labels, binary_preds),
            'f1': f1_score(labels, binary_preds),
            'precision': precision_score(labels, binary_preds),
            'recall': recall_score(labels, binary_preds),
            'threshold': threshold,
            'fpr': fpr,
            'confusion_matrix': cm
        }
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_metrics = metrics
    
    return best_metrics
def print_metrics(metrics):
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  FPR: {metrics['fpr']:.4f}")
    print(f"  Best Threshold: {metrics['threshold']:.4f}")
    print("  Confusion Matrix:")
    print(metrics['confusion_matrix'])

def collate_fn(batch):
    """
    处理包含图数据和序列数据的批次
    """
    # 过滤掉为 None 的项
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # 分别收集图数据、序列数据和标签
    graphs1 = []
    graphs2 = []
    seq_data1 = []
    seq_data2 = []
    labels = []

    for item in batch:
        # 确保图有边
        g1 = item['graph_data1']
        g2 = item['graph_data2']
        if isinstance(g1, dict) and 'graph' in g1:
            g1 = g1['graph']
        if isinstance(g2, dict) and 'graph' in g2:
            g2 = g2['graph']
            
        if g1 is not None and g1.number_of_edges() > 0 and g2 is not None and g2.number_of_edges() > 0:
            graphs1.append(g1)
            graphs2.append(g2)
            seq_data1.append(item['seq_data1'])
            seq_data2.append(item['seq_data2'])
            labels.append(item['label'])

    if not graphs1:  # 如果没有有效的图数据
        return None

    # 将序列数据和标签转换为张量
    seq_data1 = torch.stack(seq_data1)
    seq_data2 = torch.stack(seq_data2)
    labels = torch.stack(labels)

    # 批处理图数据
    batched_graphs1 = dgl.batch(graphs1)
    batched_graphs2 = dgl.batch(graphs2)

    return {
        'graph_data1': batched_graphs1,
        'graph_data2': batched_graphs2,
        'seq_data1': seq_data1,
        'seq_data2': seq_data2,
        'labels': labels
    }
def evaluate_on_test(joint_model, processed_dir, transactions_dict, max_seq_length, device, result_path=None):
    """
    用训练好的joint_model对test_pair_edges.csv进行评估
    """
    import torch.nn.functional as F
    import numpy as np

    # 1. 加载测试集
    test_edges = pd.read_csv('/data2/wzy/experiment/Dataset/D/test_pair_edges.csv')
    test_dataset = NodePairDataset(
        df_edges=test_edges,
        processed_dir=processed_dir,
        transactions_dict=transactions_dict,
        max_seq_length=max_seq_length
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    # 2. 评估
    joint_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            graph_data1 = batch['graph_data1'].to(device)
            graph_data2 = batch['graph_data2'].to(device)
            seq_data1 = batch['seq_data1'].to(device)
            seq_data2 = batch['seq_data2'].to(device)
            labels = batch['labels'].to(device)

            features1 = joint_model(graph_data1, seq_data1)
            features2 = joint_model(graph_data2, seq_data2)
            cos_sim = F.cosine_similarity(features1, features2)
            predictions = (cos_sim + 1) / 2
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 3. 计算评估指标
    # thresholds = np.linspace(0.3, 0.8, 9)
    thresholds = [0.9]
    metrics = calculate_metrics_with_best_threshold(all_preds, all_labels, thresholds)
    print("\n测试集评估指标：")
        # 保存评估指标到文件

    print_metrics(metrics)
    result_dict = {
        "AUC": float(metrics["auc"]),
        "Accuracy": float(metrics["accuracy"]),
        "F1 Score": float(metrics["f1"]),
        "Precision": float(metrics["precision"]),
        "Recall": float(metrics["recall"]),
        "FPR": float(metrics["fpr"]),
        "FNR": float(metrics["confusion_matrix"][1,0] / (metrics["confusion_matrix"][1,0] + metrics["confusion_matrix"][1,1]) if (metrics["confusion_matrix"][1,0] + metrics["confusion_matrix"][1,1]) > 0 else 0),
        "Confusion Matrix": metrics["confusion_matrix"].tolist()
    }

    if result_path is not None:
        with open(result_path, 'w') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)
        print(f"评估指标已保存至: {result_path}")

    return result_dict
    # return metrics
def train_main():
    # 设置设备和随机种子
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print(f"Using device: {device}")

    # 加载模型结构
    from subgraph_level_emb import Model, Encoder
    from seq_emb_gen import EnhancedAccountTransformer as SeqModel
    from seq_emb_gen import MAX_SEQ_LENGTH

    # 初始化图模型
    graph_encoder = Encoder(
        input_dim=16, hidden_dim=64, num_layers=2,
        conv_type='graphsage', dropout=0.2, sage_aggregator='mean'
    )
    graph_model = Model(encoder=graph_encoder, projection_dim=32, tau=0.2)

    # 加载图模型权重
    print("Loading graph model weights...")
    checkpoint = torch.load("/data2/wzy/experiment/model/node_repr_model_subgraph_level_2ego_graphsage.pth", map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        graph_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        graph_model.load_state_dict(checkpoint)

    # 初始化序列模型
    seq_model = SeqModel(
        d_model=64, nhead=4, num_layers=3, dim_feedforward=256,
        dropout=0.1, max_seq_length=MAX_SEQ_LENGTH
    )
    # 加载序列模型权重
    print("Loading sequence model weights...")
    seq_checkpoint = torch.load("/data2/wzy/experiment/model/seq_model/ablation_model_0.7_0.3.pth", map_location=device)
    if isinstance(seq_checkpoint, dict) and 'model_state_dict' in seq_checkpoint:
        seq_model.load_state_dict(seq_checkpoint['model_state_dict'])
    else:
        seq_model.load_state_dict(seq_checkpoint)

    # 创建联合模型
    joint_model = JointEmbeddingModel(
        graph_model=graph_model,
        seq_model=seq_model,
        projection_dim=128,
        freeze_graph_layers=1,
        freeze_seq_layers=2
    ).to(device)

    # 加载数据
    df_edges, processed_dir, transactions_dict = load_data()
    print(f"Total number of node pairs: {len(df_edges)}")
    print(f"Positive samples: {len(df_edges[df_edges['label'] == 1])}")
    print(f"Negative samples: {len(df_edges[df_edges['label'] == 0])}")

    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        df_edges, test_size=0.2, random_state=42, stratify=df_edges['label']
    )

    # 创建数据集和加载器
    train_dataset = NodePairDataset(train_df, processed_dir, transactions_dict, MAX_SEQ_LENGTH)
    val_dataset = NodePairDataset(val_df, processed_dir, transactions_dict, MAX_SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    print("Starting training...")
    trained_model, best_metrics = train_joint_model(
        joint_model=joint_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=5e-4,
        device=device
    )

    # 保存模型
    save_path = "/data2/wzy/experiment/model/subgraph_level_model/joint_model_D_0.9.pth"
    print(f"Saving model to {save_path}")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': {
            'projection_dim': 128,
            'max_seq_length': MAX_SEQ_LENGTH,
            'graph_hidden_dim': 64,
            'seq_hidden_dim': 64
        },
        'best_metrics': best_metrics,
    }, save_path)
    print("\n最终最佳评估指标:")
    print_metrics(best_metrics)
    print("\nTraining completed!")

def eval_main():
    # 设置设备和随机种子
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print(f"Using device: {device}")

    # 加载模型结构
    from subgraph_level_emb import Model, Encoder
    from seq_emb_gen import EnhancedAccountTransformer as SeqModel
    from seq_emb_gen import MAX_SEQ_LENGTH

    # 初始化图模型
    graph_encoder = Encoder(
        input_dim=16, hidden_dim=64, num_layers=2,
        conv_type='graphsage', dropout=0.2, sage_aggregator='mean'
    )
    graph_model = Model(encoder=graph_encoder, projection_dim=32, tau=0.2)

    # 初始化序列模型
    seq_model = SeqModel(
        d_model=64, nhead=4, num_layers=3, dim_feedforward=256,
        dropout=0.1, max_seq_length=MAX_SEQ_LENGTH
    )

    # 创建联合模型
    joint_model = JointEmbeddingModel(
        graph_model=graph_model,
        seq_model=seq_model,
        projection_dim=128,
        freeze_graph_layers=1,
        freeze_seq_layers=2
    ).to(device)

    # 加载模型权重
    print("Loading joint model weights...")
    checkpoint = torch.load("/data2/wzy/experiment/model/subgraph_level_model/joint_model_D_0.9.pth", map_location=device)
    joint_model.load_state_dict(checkpoint['model_state_dict'])

    # 加载子图和交易序列数据
    processed_dir = '/data2/wzy/experiment/Dataset/Graph/processed_2ego/'
    with open('/data2/wzy/experiment/Dataset/Seq/node_sequences.pkl', 'rb') as f:
        transactions_dict = pickle.load(f)


    # 评估
    evaluate_on_test(
        joint_model=joint_model,
        processed_dir=processed_dir,
        transactions_dict=transactions_dict,
        max_seq_length=MAX_SEQ_LENGTH,
        device=device,
        result_path="/data2/wzy/experiment/outputs/evaluation_results/demixnet/test_eval_result_D_thr_0.9.json"
    )

def predict_node_pairs(nodes, joint_model, processed_dir, transactions_dict, max_seq_length=30, device='cuda', threshold=0.85):
    """
    对给定节点列表中的所有可能节点对进行关系预测
    """
    import itertools
    import torch.nn.functional as F
    import dgl
    from torch.utils.data import DataLoader
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)    
    # 创建节点对数据集
    node_pairs = list(itertools.combinations(nodes, 2))
    pair_data = []
    for node1, node2 in node_pairs:
        pair_data.append({
            'node1': node1,
            'node2': node2,
            'label': 0  # 临时标签，预测时不会使用
        })
    
    # 创建数据集
    test_dataset = NodePairDataset(
        df_edges=pd.DataFrame(pair_data),
        processed_dir=processed_dir,
        transactions_dict=transactions_dict,
        max_seq_length=max_seq_length
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 每次处理一个节点对
        shuffle=False,
        num_workers=0,  # 简单预测不需要多进程
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    # 预测
    joint_model.eval()
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if batch is None:
                continue
                
            graph_data1 = batch['graph_data1'].to(device)
            graph_data2 = batch['graph_data2'].to(device)
            seq_data1 = batch['seq_data1'].to(device)
            seq_data2 = batch['seq_data2'].to(device)
            
            # 获取预测结果
            features1 = joint_model(graph_data1, seq_data1)
            features2 = joint_model(graph_data2, seq_data2)
            cos_sim = F.cosine_similarity(features1, features2)
            prediction = (cos_sim + 1) / 2
            
            # 记录结果
            result = {
                'node1': node_pairs[i][0],
                'node2': node_pairs[i][1],
                'predicted_label': int(prediction.item() > threshold),
                'confidence': prediction.item()
            }
            results.append(result)
            
            # 打印进度
            print(f"\n节点对 {i+1}/{len(node_pairs)}:")
            print(f"节点1: {result['node1']}")
            print(f"节点2: {result['node2']}")
            print(f"预测标签: {result['predicted_label']} (0:不相关, 1:相关)")
            print(f"置信度: {result['confidence']:.4f}")
    
    return results

def predict_main(node_list):
    """
    主预测函数
    """
    # 设置设备和随机种子
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print(f"Using device: {device}")

    # 加载模型结构
    from subgraph_level_emb import Model, Encoder
    from seq_emb_gen import EnhancedAccountTransformer as SeqModel
    from seq_emb_gen import MAX_SEQ_LENGTH

    # 初始化图模型
    graph_encoder = Encoder(
        input_dim=16, hidden_dim=64, num_layers=2,
        conv_type='graphsage', dropout=0.2, sage_aggregator='mean'
    )
    graph_model = Model(encoder=graph_encoder, projection_dim=32, tau=0.2)

    # 初始化序列模型
    seq_model = SeqModel(
        d_model=64, nhead=4, num_layers=3, dim_feedforward=256,
        dropout=0.1, max_seq_length=MAX_SEQ_LENGTH
    )

    # 创建联合模型
    joint_model = JointEmbeddingModel(
        graph_model=graph_model,
        seq_model=seq_model,
        projection_dim=128,
        freeze_graph_layers=1,
        freeze_seq_layers=2
    ).to(device)

    # 加载模型权重
    print("Loading joint model weights...")
    checkpoint = torch.load("/data2/wzy/experiment/model/subgraph_level_model/joint_model.pth", map_location=device)
    joint_model.load_state_dict(checkpoint['model_state_dict'])
    set_seed(42)

    # 加载子图和交易序列数据
    processed_dir = '/data2/wzy/experiment/Dataset/Graph/processed_2ego/'
    with open('/data2/wzy/experiment/Dataset/Seq/node_sequences.pkl', 'rb') as f:
        transactions_dict = pickle.load(f)

    # 进行预测
    results = predict_node_pairs(
        nodes=node_list,
        joint_model=joint_model,
        processed_dir=processed_dir,
        transactions_dict=transactions_dict,
        max_seq_length=MAX_SEQ_LENGTH,
        device=device
    )
    
    return results

if __name__ == "__main__":
    # 只运行训练或只运行评估，二选一
    # train_main()

    # eval_main()
    # nodes = [
    # "0x12b6EDb437b532D775319DA877473a4D93e8457f".lower(),
    # "0xe9f7cd0c410257850324d86536bd165cf4306e80".lower(),
    # ]
    addresses = [
        "0x0864b5ef4d8086cd0062306f39adea5da5bd2603",
        # "0x1031ffaf5d00c6bc1ee0978eb7ec196b1d164129",
        "0x05492cbc8fb228103744ecca0df62473b2858810",
        "0x0784051d5136a5ccb47ddb3a15243890f5268482",
        # "0xad6a4ace6dcc21c93ca9dbc8a21c7d3a726c1fb1",
        # "0x03e89f2e1ebcea5d94c1b530f638cea3950c2e2b",
    ]
    # nodes = [
    #     # "0x35869B5fACa988519681Bb16CbB9E193Ed574aA3".lower(),
    #     # "0x65760288f19cFf476B80A36a61F9DEDAb16Bab49".lower(),
    #     # "0x72ccbb2002254bd8a0485d13a3a2faa9b0f992c6".lower(),
    #     "0xA89DB320721f517c36b1256314aD4575F12342e8".lower(),
    #     "0xae40379049d22D7CB780D3a908ddED0c30be46EA".lower()
    # ]


    # 进行预测
    results = predict_main(addresses)
