import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import random
import os
def set_seed(seed=42):
    """设置所有随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"随机种子已设置为: {seed}")

# 定义混合服务地址常量
MIXER_ADDRESSES = [
    '0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc',
    '0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936',
    '0x910cbd523d972eb0a6f4cae4618ad62622b39dbf',
    '0xa160cdab225685da1d56aa342ad8841c3b53f291'
]

MAX_SEQ_LENGTH = 30

# 设置GPU相关参数
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
torch.backends.cudnn.deterministic = False  # 关闭确定性模式以提高性能

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 创建地址到索引的映射
MIXER_TO_IDX = {addr: idx for idx, addr in enumerate(MIXER_ADDRESSES)}

# 交易类型编码
TRANSACTION_TYPES = ['Deposit', 'Withdraw']
TRANSACTION_TYPE_TO_IDX = {tx_type: idx for idx, tx_type in enumerate(TRANSACTION_TYPES)}

class EthereumTransactionDataset(Dataset):
    def __init__(self, transactions_dict, max_seq_length=MAX_SEQ_LENGTH ):
        """
        transactions_dict: 字典，键为账户地址，值为交易序列列表
        max_seq_length: 只保留最近 max_seq_length 个交易，超过则截断，不足则填充
        """
        self.transactions = []
        self.account_addresses = []
        
        # **动态创建索引映射**
        self.mixer_to_idx = {}
        self.tx_type_to_idx = {}

        # 遍历数据，建立映射
        for tx_sequence in transactions_dict.values():
            for tx in tx_sequence:
                to_node, _, _, method, _, _ = tx
                if to_node not in self.mixer_to_idx:
                    self.mixer_to_idx[to_node] = len(self.mixer_to_idx)  # 分配索引
                if method not in self.tx_type_to_idx:
                    self.tx_type_to_idx[method] = len(self.tx_type_to_idx)  # 分配索引
        
        # **处理交易数据**
        for address, tx_sequence in transactions_dict.items():
            # **只保留最近的 max_seq_length 条交易**
            tx_sequence = tx_sequence[-max_seq_length:]
            
            # 交易转换为特征矩阵
            features = []
            for tx in tx_sequence:
                to_node, timestamp, value, method, gas_price, tx_count = tx
                
                # 获取或默认索引
                mixer_idx = self.mixer_to_idx.get(to_node, -1)
                tx_type_idx = self.tx_type_to_idx.get(method, -1)
                
                # 归一化时间戳和 gas 价格
                norm_timestamp = timestamp / (10**10)  # 适当调整
                norm_gas_price = gas_price  # 适当调整
                
                features.append([mixer_idx, norm_timestamp, tx_type_idx, norm_gas_price, tx_count])
            
            padding_length = max_seq_length - len(features)
            if padding_length > 0:
                n_mixers = len(self.mixer_to_idx)
                n_tx_types = len(self.tx_type_to_idx)
                padding = [[n_mixers, 0, n_tx_types, 0, 0]] * padding_length
                features = features + padding  # 后面填充
            
            # 转换为 NumPy，确保一致形状
            self.transactions.append(np.array(features, dtype=np.float32))
            self.account_addresses.append(address)
    
    def __len__(self):
        return len(self.transactions)
    
    def __getitem__(self, idx):
        return torch.tensor(self.transactions[idx]), self.account_addresses[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=MAX_SEQ_LENGTH ):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransactionEmbedding(nn.Module):
    def __init__(self, n_mixers=4, n_tx_types=2, d_model=64):
        super(TransactionEmbedding, self).__init__()
        
        # 修复：改用正确的padding_idx值而非-1
        self.mixer_embedding = nn.Embedding(n_mixers + 1, d_model, padding_idx=n_mixers)
        self.tx_type_embedding = nn.Embedding(n_tx_types + 1, d_model, padding_idx=n_tx_types)
        
        # 连续特征的线性映射
        self.timestamp_proj = nn.Linear(1, d_model // 4)
        self.gas_price_proj = nn.Linear(1, d_model // 4)
        self.popularity_proj = nn.Linear(1, d_model // 4)
        
        # 合并层
        self.combine_layer = nn.Linear(d_model + d_model + (d_model // 4) * 3, d_model)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, 5]
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 分离特征
        mixer_idx = x[:, :, 0].long()
        timestamp = x[:, :, 1].unsqueeze(-1)
        tx_type_idx = x[:, :, 2].long()
        gas_price = x[:, :, 3].unsqueeze(-1)
        popularity = x[:, :, 4].unsqueeze(-1)
        
        # 重要修复：处理超出范围的索引
        n_mixers = self.mixer_embedding.weight.size(0) - 1
        n_tx_types = self.tx_type_embedding.weight.size(0) - 1
        
        # 将负值或超出范围的索引替换为padding_idx
        mixer_idx = torch.clamp(mixer_idx, 0, n_mixers)
        mixer_idx = torch.where(mixer_idx < 0, torch.tensor(n_mixers, device=mixer_idx.device), mixer_idx)
        
        tx_type_idx = torch.clamp(tx_type_idx, 0, n_tx_types)
        tx_type_idx = torch.where(tx_type_idx < 0, torch.tensor(n_tx_types, device=tx_type_idx.device), tx_type_idx)
        
        # 嵌入离散特征
        mixer_emb = self.mixer_embedding(mixer_idx)
        tx_type_emb = self.tx_type_embedding(tx_type_idx)
        
        # 映射连续特征
        timestamp_emb = self.timestamp_proj(timestamp)
        gas_price_emb = self.gas_price_proj(gas_price)
        popularity_emb = self.popularity_proj(popularity)
        
        # 合并所有特征
        combined = torch.cat([
            mixer_emb, tx_type_emb, timestamp_emb, gas_price_emb, popularity_emb
        ], dim=-1)
        
        # 映射到统一维度
        return self.combine_layer(combined)
    
class EnhancedAccountTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, max_seq_length=MAX_SEQ_LENGTH):
        super(EnhancedAccountTransformer, self).__init__()
        
        # 基础特征嵌入，与原来相同
        self.transaction_embedding = TransactionEmbedding(
            n_mixers=len(MIXER_ADDRESSES),
            n_tx_types=len(TRANSACTION_TYPES),
            d_model=d_model
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 双向Transformer编码器 - 用于账户表示
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers,enable_nested_tensor=False)
        
        # 单向Transformer解码器 - 用于序列建模
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 自回归预测层
        self.mixer_predictor = nn.Linear(d_model, len(MIXER_ADDRESSES))
        
        # 全局账户表示层
        self.global_representation = nn.Linear(d_model, d_model)
        
        # 掩码生成
        self.register_buffer('seq_mask', torch.ones(max_seq_length, max_seq_length).triu_(diagonal=1).bool())
        
    def forward(self, x):
        # 提取混合服务索引
        mixer_indices = x[:, :, 0].long()
        batch_size, seq_len = mixer_indices.size()
        
        # 检查序列长度是否与模型预期一致
        expected_seq_len = self.seq_mask.shape[0]
        if seq_len != expected_seq_len:
            print(f"警告: 序列长度不匹配! 输入长度={seq_len}, 模型期望长度={expected_seq_len}")
            # 调整输入序列长度
            if seq_len < expected_seq_len:
                # 输入太短，需要填充
                padding_data = torch.zeros(batch_size, expected_seq_len - seq_len, x.size(2), device=x.device)
                x = torch.cat([x, padding_data], dim=1)
                # 重新提取混合服务索引
                mixer_indices = x[:, :, 0].long()
                seq_len = expected_seq_len
            else:
                # 输入太长，需要截断
                x = x[:, :expected_seq_len, :]
                mixer_indices = mixer_indices[:, :expected_seq_len]
                seq_len = expected_seq_len
        
        # 生成填充掩码
        n_mixers = self.transaction_embedding.mixer_embedding.weight.size(0) - 1
        padding_mask = (mixer_indices == n_mixers)
        
        # 步骤1: 嵌入交易特征
        embedded = self.transaction_embedding(x)
        embedded = self.positional_encoding(embedded)
        # print(f"Before TransformerEncoder, embedded.shape: {embedded.shape}")  
        # 步骤2: 双向Transformer编码 - 获取账户表示
        memory = self.transformer_encoder(
            embedded,
            src_key_padding_mask=padding_mask
        )
        # print(f"padding_mask shape: {padding_mask.shape}")
        # print(f"padding_mask sum (per batch): {padding_mask.sum(dim=1)}")
        # print(f"After TransformerEncoder, memory.shape: {memory.shape}") 
        
        # 步骤3: 计算全局账户表示
        # 使用注意力掩码获取有效位置的平均表示
        valid_mask = ~padding_mask
        expanded_mask = valid_mask.unsqueeze(-1).expand(-1, -1, memory.size(-1))

        # import pdb; pdb.set_trace()
        # print(f"valid_mask shape: {valid_mask.shape}")  # 预期: (batch_size, seq_len)
        # print(f"expanded_mask shape: {expanded_mask.shape}")  # 预期: (batch_size, seq_len, d_model)
        # print(f"memory shape: {memory.shape}")
        # valid_mask = ~padding_mask
        # expanded_mask = valid_mask[:, :memory.shape[1]].unsqueeze(-1).expand(-1, -1, memory.shape[-1])

        sum_repr = torch.sum(memory * expanded_mask, dim=1)
        sum_mask = torch.sum(expanded_mask, dim=1)
        global_repr = sum_repr / (sum_mask + 1e-10)
        global_repr = self.global_representation(global_repr)
        
        # 步骤4: 单向Transformer解码 - 自回归序列建模
        # 使用因果掩码确保每个位置只能看到之前的位置
        tgt_mask = self.seq_mask[:seq_len, :seq_len]
        
        # 使用整个序列作为输入，但每个位置只能看到之前的位置
        decoded = self.transformer_decoder(
            embedded,  # 查询
            memory,    # 键和值
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask
        )

        
        # 步骤5: 预测每个位置的下一个混合服务
        mixer_logits = self.mixer_predictor(decoded)
        
        return {
            'global_repr': global_repr,          # 用于混合服务偏好预测
            'seq_representations': decoded,      # 序列中每个位置的表示
            'mixer_logits': mixer_logits,        # 每个位置对下一个混合服务的预测
            'padding_mask': padding_mask         # 填充掩码
        }

def compute_enhanced_losses(model_outputs, batch_x):
    """
    计算改进后的多任务损失
    """
    # 提取模型输出
    global_repr = model_outputs['global_repr']
    mixer_logits = model_outputs['mixer_logits']
    padding_mask = model_outputs['padding_mask']
    
    # 提取目标值
    mixer_indices = batch_x[:, :, 0].long()
    batch_size, seq_len = mixer_indices.size()
    n_mixers = model.transaction_embedding.mixer_embedding.weight.size(0) - 1
    
    # 1. 自回归序列预测损失
    # 偏移目标：预测下一个位置
    target_indices = torch.zeros_like(mixer_indices)
    target_indices[:, :-1] = mixer_indices[:, 1:]  # 将目标偏移一位
    target_indices = torch.clamp(target_indices, 0, n_mixers-1) 
    
    # 创建有效位置掩码：只考虑非填充且有下一个位置的情况
    valid_positions = ~padding_mask.clone()
    valid_positions[:, -1] = False  # 最后一个位置没有下一个位置，设为无效
    
    
    # 去掉填充位置，将数据整形为2D
    flat_logits = mixer_logits[valid_positions]
    flat_targets = target_indices[valid_positions]
    assert flat_targets.max() < n_mixers, f"目标标签越界：最大值 {flat_targets.max()}"
    
    # 计算交叉熵损失
    if flat_logits.size(0) > 0:
        seq_loss = F.cross_entropy(flat_logits, flat_targets)
    else:
        seq_loss = torch.tensor(0.0, device=global_repr.device)
    
    # 2. 混合服务偏好预测损失
    # 使用全局表示预测混合服务的整体偏好
    weight_matrix = model.transaction_embedding.mixer_embedding.weight[:n_mixers]
    predicted_prefs = torch.matmul(global_repr, weight_matrix.transpose(0, 1))
    predicted_prefs = F.softmax(predicted_prefs, dim=1)
    
    # 计算真实偏好
    mixer_onehot = F.one_hot(torch.clamp(mixer_indices, 0, n_mixers-1), n_mixers).float()
    valid_mask = (~padding_mask).float().unsqueeze(-1)
    true_counts = (mixer_onehot * valid_mask).sum(dim=1)
    true_prefs = true_counts / (true_counts.sum(dim=1, keepdim=True) + 1e-10)
    
    # 计算KL散度损失
    epsilon = 1e-6
    smoothed_preds = predicted_prefs * (1 - epsilon) + epsilon / n_mixers
    pref_loss = F.kl_div(torch.log(smoothed_preds), true_prefs, reduction='batchmean')
    
    # 3. 下一步预测准确率 (仅用于监控，不参与优化)
    with torch.no_grad():
        predictions = mixer_logits.argmax(dim=-1)
        correct = (predictions == target_indices) & valid_positions
        accuracy = correct.sum().float() / (valid_positions.sum() + 1e-10)
    
    return {
        'seq_loss': seq_loss,
        'pref_loss': pref_loss,
        'accuracy': accuracy,
        'total_loss': 0.7 * seq_loss + 0.3 * pref_loss
    }
def worker_init_fn(worker_id):
    """确保DataLoader的worker进程也使用确定性随机性"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def train_enhanced_model(model, transactions_dict, num_epochs=50, batch_size=32, lr=0.0001):
    """
    使用改进的序列建模方法训练模型
    """
    # 数据准备部分与原来相同
    dataset = EthereumTransactionDataset(transactions_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
    
    # 优化器和学习率调度
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    model = model.to(device)
    model.train()
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_seq_loss = 0
        total_pref_loss = 0
        total_accuracy = 0
        
        # 动态调整序列损失和偏好损失的权重
        lambda_seq = min(0.8, 0.6 + epoch * 0.01)  # 随着训练进展增加序列损失权重
        lambda_pref = 1.0 - lambda_seq
        
        for batch_idx, (batch_x, batch_addresses) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch_x)
            
            # 计算损失
            losses = compute_enhanced_losses(outputs, batch_x)
            seq_loss = losses['seq_loss']
            pref_loss = losses['pref_loss']
            accuracy = losses['accuracy']
            
            # 组合损失
            loss = lambda_seq * seq_loss + lambda_pref * pref_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录统计信息
            total_seq_loss += seq_loss.item()
            total_pref_loss += pref_loss.item()
            total_accuracy += accuracy.item()
            
            # # 每50个批次打印进度
            # if batch_idx % 50 == 0:
            #     print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, "
            #           f"Seq Loss: {seq_loss.item():.4f}, Pref Loss: {pref_loss.item():.4f}, "
            #           f"Accuracy: {accuracy.item():.4f}")
        
        # 计算平均损失
        avg_seq_loss = total_seq_loss / len(dataloader)
        avg_pref_loss = total_pref_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        
        # 更新学习率
        avg_loss = lambda_seq * avg_seq_loss + lambda_pref * avg_pref_loss
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed. "
            f"Seq Loss: {avg_seq_loss:.4f}, Pref Loss: {avg_pref_loss:.4f}, "
            f"Accuracy: {avg_accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"保存新的最佳模型，损失: {best_loss:.4f}")
            torch.save(model.state_dict(), "best_account_model.pth")
    
    return model

def get_account_embedding(model, account_transactions, max_seq_length=MAX_SEQ_LENGTH ):
    model.eval()
    
    # 准备数据
    features = []
    # import pdb; pdb.set_trace()
    for tx in account_transactions[:max_seq_length]:
        # 修正：保持与训练数据一致的解包顺序
        to_node, timestamp, value, method, gas_price, tx_count = tx
        
        # 处理可能的键错误
        try:
            mixer_idx = MIXER_TO_IDX[to_node]
        except KeyError:
            print(f"警告: 未知混合服务地址 {to_node}，使用默认值0")
            mixer_idx = 0
            
        try:
            tx_type_idx = TRANSACTION_TYPE_TO_IDX[method]
        except KeyError:
            print(f"警告: 未知交易类型 {method}，使用默认值0")
            tx_type_idx = 0
        
        norm_timestamp = timestamp / (10**10)
        norm_gas_price = gas_price
        
        features.append([mixer_idx, norm_timestamp, tx_type_idx, norm_gas_price, tx_count])
    
    # 填充序列
    padding_length = max_seq_length - len(features)
    if padding_length > 0:
        # 使用正确的填充值
        n_mixers = len(MIXER_ADDRESSES)
        n_tx_types = len(TRANSACTION_TYPES)
        padding = [[n_mixers, 0, n_tx_types, 0, 0]] * padding_length
        features = features + padding  # 后面填充，与训练保持一致
    
    # 转换为张量并添加批量维度
    input_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        # 打印序列长度和模型期望长度
    # print(f"输入张量形状: {input_tensor.shape}")
    # print(f"模型seq_mask形状: {model.seq_mask.shape}")
    
    # 不使用混合精度处理
    with torch.no_grad():
        # print(input_tensor.shape)
        output_dict = model(input_tensor)
        
        # 从输出字典中提取全局表示
        global_repr = output_dict['global_repr']
    
    return global_repr.squeeze(0).cpu().numpy()


# 示例用法
if __name__ == "__main__":
    set_seed(42)
    # 创建模型实例
    model = EnhancedAccountTransformer(
        d_model=64,
        nhead=4,
        num_layers=3,
        max_seq_length=MAX_SEQ_LENGTH 
    )
    
    # # 读取数据
    # print("正在加载数据...")
    with open('/data2/wzy/experiment/Dataset/Seq/node_sequences.pkl', 'rb') as f:
        transactions_dict = pickle.load(f)
    import pandas as pd
    csv_path = '/data2/wzy/experiment/Dataset/D/D1.csv'
    df = pd.read_csv(csv_path,header=None)
    from_addresses = df[0].unique().tolist()
    filtered_transactions = {
        k: v for k, v in transactions_dict.items() 
        if k in from_addresses
    }

    # 训练模型
    print("开始训练模型...")
    # trained_model = train_enhanced_model(model, filtered_transactions, num_epochs=100,lr=0.001)
    # torch.save(trained_model.state_dict(), "/data2/wzy/experiment/model/best_account_model_D1.pth")
    model_state_dict = torch.load("/data2/wzy/experiment/model/best_account_model_D1.pth")
    model.load_state_dict(model_state_dict)
    model = model.to(device)  # 将模型移到GPU
    trained_model = model

    # 为所有地址生成嵌入向量
    print("正在生成嵌入向量...")
    # eth_file_path = "/data2/wzy/experiment/Dataset/Graph/ETH.csv"
    # eth_df = pd.read_csv(eth_file_path)
    # from_addresses = eth_df['From'].unique().tolist()
    # import pandas as pd
    # csv_path = '/data2/wzy/experiment/Dataset/D/D1.csv'
    # df = pd.read_csv(csv_path,header=None)
    # from_addresses = df[0].unique().tolist()

    address_embeddings = {}
        
    # 优化批处理生成嵌入
    batch_size = 32  # 增加批量大小以提高GPU利用率
    for i in range(0, len(from_addresses), batch_size):
        batch_addresses = from_addresses[i:i+batch_size]
        batch_tx_sequences = []
        
        # 获取批次的交易序列
        for address in batch_addresses:
            tx_sequence = transactions_dict.get(address, [])
            batch_tx_sequences.append(tx_sequence)
        
        # 批量处理并生成嵌入向量
        for j, address in enumerate(batch_addresses):
            address_embeddings[address] = get_account_embedding(trained_model, batch_tx_sequences[j])

    
    # 保存嵌入向量
    print("正在保存嵌入向量...")
    with open('/data2/wzy/experiment/Dataset/Seq/node_embeddings_D1.pkl', 'wb') as f:
        pickle.dump(address_embeddings, f)
    print('嵌入向量已保存完成')