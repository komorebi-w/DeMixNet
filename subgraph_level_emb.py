import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import glob
import pandas as pd
import argparse
import random
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import networkx as nx
import argparse
import os.path as osp
import random
from time import perf_counter as t

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv
import logging
import matplotlib.pyplot as plt
from model import Encoder, Model, drop_feature
# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # 启用CUDA动态共享内存分配

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
def get_dataset_name(dataset_path):
    # 只取文件名（不带后缀）
    return os.path.splitext(os.path.basename(dataset_path))[0]

def setup_logging_and_dirs(dataset_path):
    dataset_name = get_dataset_name(dataset_path)
    log_dir = os.path.join('logs', dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    # 配置logging
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )
    return log_dir
class LightSubgraphDataset(Dataset):
    """使用预处理的轻量级子图数据集"""
    def __init__(self, processed_dir, file_list=None):
        self.processed_dir = processed_dir
        
        if file_list is None:
            self.file_list = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
        else:
            self.file_list = [f"{name}_2hop_topk20.pt" for name in file_list]
            
        print(f"Dataset created with {len(self.file_list)} pre-processed subgraphs.")
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        pt_file = self.file_list[idx]
        try:
            # 直接加载预处理好的数据
            data = torch.load(os.path.join(self.processed_dir, pt_file))
            return data
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            return None

def dropout_edge_with_attr(edge_index, edge_attr=None, p=0.2, force_undirected=False):

    edge_index, edge_mask = dropout_edge(edge_index, p=p, force_undirected=force_undirected)
    
    if edge_attr is not None:
        edge_attr = edge_attr[edge_mask]
        
    return edge_index, edge_attr
def process_batch_data(view_data, device):
    """处理批次数据并移到GPU"""
    batch_x = [] 
    batch_edge_index = []
    batch_edge_attr = []
    batch_indices = []
    
    offset = 0
    for i, (x, edge_index, edge_attr) in enumerate(view_data):
        num_nodes = x.size(0)
        batch_x.append(x)
        batch_edge_index.append(edge_index + offset)
        if edge_attr is not None:
            batch_edge_attr.append(edge_attr)
        batch_indices.extend([i] * num_nodes)
        offset += num_nodes
    
    x = torch.cat(batch_x, dim=0).to(device)
    edge_index = torch.cat(batch_edge_index, dim=1).to(device)
    batch = torch.tensor(batch_indices, dtype=torch.long, device=device)
    edge_attr = torch.cat(batch_edge_attr, dim=0).to(device) if batch_edge_attr else None
    
    return x, edge_index, batch, edge_attr

def train_model(model, dataloader, epochs=100, lr=0.001, device='cuda', patience=10, log_dir=None):
    model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 初始化最佳损失和模型状态
    best_loss = float('inf')
    best_model_state = None
    
    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    patience_counter = 0
    loss_list = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        # 记录每个epoch的开始时间
        epoch_start = t()
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated(device)
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_start = t()
            try:
                if batch is None:
                    continue
                
                # 处理数据
                graphs = batch['graphs']
                view1_data = []
                view2_data = []
                
                for g in graphs:
                    if g.number_of_nodes() < 2:
                        continue
                        
                    # 将图移到CPU处理数据增强
                    g = g.to('cpu')
                    x = g.ndata['features']
                    edge_index = torch.stack(g.edges())
                    edge_attr = g.edata.get('features', None)


                    # 生成两个视图
                    # import pdb;pdb.set_trace()
                    x1 = drop_feature(x, 0.2)
                    x2 = drop_feature(x, 0.2)
                    edge_index1, edge_attr1 = dropout_edge_with_attr(edge_index, edge_attr, p=0.2)
                    edge_index2, edge_attr2 = dropout_edge_with_attr(edge_index, edge_attr, p=0.2)
                    
                    view1_data.append((x1, edge_index1, edge_attr1))
                    view2_data.append((x2, edge_index2, edge_attr2))
                
                if not view1_data:
                    continue
                
                # 批处理数据
                batch_size = len(view1_data)
                device_data1 = process_batch_data(view1_data, device)
                device_data2 = process_batch_data(view2_data, device)
                
                # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    z1 = model(*device_data1)
                    z2 = model(*device_data2)
                    loss = model.loss(z1, z2)
                
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                batch_count += 1
                
                # 主动清理不需要的变量
                del device_data1, device_data2, z1, z2, loss
                del view1_data, view2_data
                torch.cuda.empty_cache()
                
                # 监控每个batch的处理时间和内存使用
                if batch_count % 10 == 0:
                    batch_time = t() - batch_start
                    current_memory = torch.cuda.memory_allocated(device)
                    memory_diff = current_memory - start_memory
                    print(f"\nBatch {batch_count} - Time: {batch_time:.2f}s, Memory: {memory_diff/1024**2:.1f}MB")
                
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                torch.cuda.empty_cache()
                continue
            
        avg_loss = total_loss / max(batch_count, 1)
        loss_list.append(avg_loss)
        epoch_time = t() - epoch_start  # 提前赋值，保证后面一定有
        current_memory = torch.cuda.memory_allocated(device)
        memory_diff = current_memory - start_memory
        logging.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s, Memory: {memory_diff/1024**2:.1f}MB, LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(avg_loss)
        
        # 早停逻辑
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        if log_dir is not None:
            plt.figure()
            plt.plot(loss_list, label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
            plt.close()
    return model

def generate_embeddings(model, dataloader, device, output_path='subgraph_embeddings.pkl'):
    """生成子图级别的嵌入"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model.eval()
    all_embeddings = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            if batch is None:
                continue
                
            try:
                graphs = batch['graphs']
                file_names = batch['file_names']
                
                for g, file_name in zip(graphs, file_names):
                    if g is None or g.number_of_nodes() == 0:
                        continue
                    
                    try:
                        g = g.to(device)
                        # 获取子图嵌入
                        embedding = model.get_embedding(g)
                        # 保存子图嵌入
                        all_embeddings[file_name] = embedding.cpu().numpy()
                        
                        g = g.to('cpu')
                        del embedding
                        
                    except Exception as e:
                        print(f"处理子图 {file_name} 时出错: {e}")
                        continue
                    
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"处理批次时出错: {e}")
                continue
    
    # 保存所有嵌入
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(all_embeddings, f)
        print(f"所有子图嵌入已保存到 {output_path}")
        print(f"共生成了 {len(all_embeddings)} 个子图的嵌入表示")
    except Exception as e:
        print(f"保存结果时出错: {e}")
        temp_path = output_path + '.backup'
        with open(temp_path, 'wb') as f:
            pickle.dump(all_embeddings, f)
        print(f"结果已保存到备份文件: {temp_path}")


def collate_fn(batch):
    # 过滤掉为 None 的项
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # 分别收集图、文件名、节点映射
    graphs = []
    file_names = []
    node_mappings = []

    for item in batch:
        g = item.get('graph')
        if g is not None and g.number_of_edges() > 0:  # 确保图有边
            graphs.append(g)
            file_names.append(item.get('file_name').split('_')[0])
            node_mappings.append(item.get('node_mapping'))

    if not graphs:
        return None

    return {
        'graphs': graphs,  # 图对象列表
        'file_names': file_names,
        'node_mappings': node_mappings
    }
def init_cuda(gpu_id):
    """初始化CUDA设备"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # 设置CUDA设备
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    # 清理CUDA缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # 设置CUDA参数
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Using CUDA device {device} with {torch.cuda.get_device_name(device)}")
    print(f"Initial CUDA memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    
    return device
def main(args):
    if args.dataset and args.dataset.endswith('.csv'):
        log_dir = setup_logging_and_dirs(args.dataset)
        dataset_name = get_dataset_name(args.dataset)
    else:
        log_dir = setup_logging_and_dirs('default')
        dataset_name = 'default'
    # 初始化CUDA
    device = init_cuda(args.gpu)
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # 设置路径
    processed_dir = '/data2/wzy/experiment/Dataset/Graph/processed_2ego/'  # 设置预处理数据的默认路径
    output_dir = args.output_dir if args.output_dir else f'/data2/wzy/experiment/outputs/node_embeddings_subgraph_level_2ego/{dataset_name}'
    model_save_path = args.model_path if args.model_path else f'/data2/wzy/experiment/model/node_repr_model_subgraph_level_2ego_{dataset_name}.pth'
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果指定了特定数据集
    if args.dataset:
        if args.dataset.endswith('.csv'):
            # 读取csv文件中的node_id
            with open(args.dataset, 'r') as f:
                file_list = [line.strip() for line in f if line.strip()]
        else:
            file_list = [f"{args.dataset}.pt"]
    else:
        file_list = None
    # 创建数据集
    dataset = LightSubgraphDataset(processed_dir, file_list)
    print(f"找到 {len(dataset)} 个预处理子图")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=True if torch.cuda.is_available() else False
    )
    # 定义编码器和模型
    input_dim = 16  # 输入特征维度
    hidden_dim = args.hidden_dim
    output_dim = args.embedding_dim
    
    # 模型加载或创建
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        
        # 创建模型架构
        encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=args.num_layers,
            conv_type=args.conv_type,
            dropout=args.dropout,
            sage_aggregator=args.sage_aggregator if hasattr(args, 'sage_aggregator') else 'mean'
        )
        model = Model(encoder=encoder, projection_dim=output_dim, tau=args.tau).to(device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    else:
        # 创建新模型
        encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=args.num_layers,
            conv_type=args.conv_type,
            dropout=args.dropout
        )
        model = Model(encoder=encoder, projection_dim=output_dim, tau=args.tau).to(device)
    
    # 为了性能优化，可以使用torch.compile (PyTorch 2.0+)
    if hasattr(torch, 'compile') and not args.no_compile:
        try:
            model = torch.compile(model)
            print("使用torch.compile加速模型")
        except:
            print("torch.compile不可用，使用标准模型")
    
    # 训练模式
    if not args.eval_only:
        model = train_model(
            model=model,
            dataloader=dataloader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            patience=args.patience,
            log_dir=log_dir
        )
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'seed': args.seed,
            'args': vars(args),
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
    
    # 生成嵌入
    generate_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        output_path=os.path.join(output_dir, 'all_node_embeddings.pkl')
    )
    
    print("处理完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("图对比学习训练参数")
    # 数据参数
    parser.add_argument("--dataset", type=str, default=None, help="要处理的特定数据集")
    parser.add_argument("--output_dir", type=str, default=None, help="保存嵌入的目录")
    
    # 性能优化选项
    parser.add_argument("--no_compile", action="store_true", help="不使用torch.compile加速")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default=None, help="保存模型的路径")
    parser.add_argument("--load_model", type=str, default=None, help="加载现有模型的路径")
    parser.add_argument("--hidden_dim", default=64, type=int, help="隐藏层维度")
    parser.add_argument("--embedding_dim", default=32, type=int, help="输出嵌入维度")
    parser.add_argument("--num_layers", default=2, type=int, help="GCN层数")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout比例")
    parser.add_argument("--tau", default=0.1, type=float, help="对比学习温度参数")
    
    # 模型类型参数
 
    parser.add_argument("--conv_type", default='graphsage', type=str, 
                    choices=['gcn', 'gat', 'edge_feature', 'graphsage'], 
                    help="卷积层类型")
    parser.add_argument("--sage_aggregator", default='mean', type=str,
                    choices=['mean', 'max', 'sum'], 
                    help="GraphSAGE聚合器类型")
    
    # 训练参数
    parser.add_argument("--batch_size", default=128, type=int, help="批次大小")
    parser.add_argument("--gpu", default=2, type=int, help="使用的GPU ID")
    parser.add_argument("--epochs", default=100, type=int, help="训练轮数")
    parser.add_argument("--lr", default=5e-4, type=float, help="学习率")
    parser.add_argument("--patience", default=10, type=int, help="早停耐心值")
    parser.add_argument("--workers", default=4, type=int, help="数据加载器工作进程数")
    parser.add_argument("--seed", default=42, type=int, help="随机种子，用于可重现性")
    parser.add_argument("--eval_only", action="store_true", help="仅评估，不训练")
    
    args = parser.parse_args()
    main(args)


# nohup python3 subgraph_level_emb.py     --conv_type graphsage  --model_path /data2/wzy/experiment/model/node_repr_model_subgraph_level_2ego_graphsage.pth     --gpu 1     --batch_size 64     --output_dir /data2/wzy/experiment/outputs/node_embeddings_subgraph_level_2ego_graphsage > bat64_2ego_graphsage.out 2>&1 &