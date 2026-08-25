import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv,SAGEConv
from torch_scatter import scatter_add

def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


class EdgeFeatureConv(nn.Module):
    """
    考虑边特征的图卷积层
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeFeatureConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 节点特征变换
        self.node_transform = nn.Linear(in_channels, out_channels)
        
        # 边特征变换
        self.edge_transform = nn.Linear(3, out_channels)  # 假设边特征维度为3
        
        # 基础图卷积
        self.conv = GraphConv(in_channels, out_channels)
        
        # 边特征注意力机制
        self.edge_attention = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_features]
            
        Returns:
            更新后的节点特征 [num_nodes, out_channels]
        """
        # 基础图卷积
        conv_out = self.conv(x, edge_index)
        
        # 如果没有边特征，返回基础卷积结果
        if edge_attr is None:
            return conv_out
            
        # 处理边特征
        edge_feat = self.edge_transform(edge_attr)
        
        # 对每条边，获取源节点和目标节点
        src, dst = edge_index
        
        # 获取源节点和目标节点的特征
        src_feat = x[src]
        dst_feat = x[dst]
        
        # 计算边特征对节点特征的注意力权重
        cat_feat = torch.cat([src_feat, dst_feat], dim=1)
        attention = self.edge_attention(cat_feat)
        
        # 聚合边特征信息到目标节点
        edge_msg = edge_feat * attention
        
        # 使用scatter_add累加每个目标节点接收的所有边消息
        out = conv_out.clone()
        out = scatter_add(edge_msg, dst, dim=0, out=out)
            
        return out
class AttentivePooling(nn.Module):
    """
    子图级别的注意力池化层
    """
    def __init__(self, hidden_dim):
        super(AttentivePooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, batch):
        """
        Args:
            x: 节点特征矩阵 [num_nodes, hidden_dim]
            batch: 指示每个节点属于哪个子图的张量 [num_nodes]
        Returns:
            子图表示 [num_graphs, hidden_dim]
        """
        # 计算每个节点的注意力分数
        attention_scores = self.attention(x)  # [num_nodes, 1]
        
        num_graphs = batch.max().item() + 1
        graph_embeddings = torch.zeros(num_graphs, x.size(1), device=x.device)
        
        for i in range(num_graphs):
            mask = (batch == i)
            x_i = x[mask]  # [num_nodes_in_graph, hidden_dim]
            attn_i = attention_scores[mask]  # [num_nodes_in_graph, 1]
            attn_i = torch.softmax(attn_i, dim=0)  # 只对当前子图做softmax
            # 加权求和得到子图表示
            graph_embeddings[i] = torch.sum(x_i * attn_i, dim=0)
            
        return graph_embeddings

class Encoder(nn.Module):
    """
    包含边特征的图编码器
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, 
                 conv_type='edge_feature', dropout=0.2, activation='prelu',
                 sage_aggregator='mean'):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList()
        
        # 保存GraphSAGE特定参数
        self.sage_aggregator = sage_aggregator
        
        # 根据指定类型选择卷积层
        if conv_type == 'gcn':
            conv_layer = lambda in_dim, out_dim: GCNConv(in_dim, out_dim)
        elif conv_type == 'gat':
            conv_layer = lambda in_dim, out_dim: GATConv(in_dim, out_dim)
        elif conv_type == 'graphsage':
            conv_layer = lambda in_dim, out_dim: SAGEConv(in_dim, out_dim, 
                                                        aggr=self.sage_aggregator)
        elif conv_type == 'edge_feature':
            conv_layer = lambda in_dim, out_dim: EdgeFeatureConv(in_dim, out_dim)
        else:
            raise ValueError(f"不支持的卷积类型: {conv_type}")
        
        # 第一个卷积层
        self.convs.append(conv_layer(input_dim, hidden_dim))
        
        # 额外的卷积层
        for _ in range(num_layers - 1):
            self.convs.append(conv_layer(hidden_dim, hidden_dim))
        
        # 选择激活函数
        if activation == 'prelu':
            self.acts = nn.ModuleList([nn.PReLU() for _ in range(num_layers)])
        elif activation == 'relu':
            self.acts = nn.ModuleList([nn.ReLU() for _ in range(num_layers)])
        elif activation == 'leakyrelu':
            self.acts = nn.ModuleList([nn.LeakyReLU(0.2) for _ in range(num_layers)])
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        self.dropout = dropout
        self.conv_type = conv_type
    def forward(self, x, edge_index, edge_attr=None):
        """
        通过编码器的前向传播
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_attr: 边特征（可选）
            
        Returns:
            节点嵌入
        """
        # for i, (conv, act) in enumerate(zip(self.convs, self.acts)):
        #     print(f"Layer {i} input x[:5]:", x[:5])
        #     if self.conv_type == 'edge_feature' and edge_attr is not None:
        #         x = conv(x, edge_index, edge_attr)
        #     else:
        #         x = conv(x, edge_index)
        #     print(f"Layer {i} output x[:5]:", x[:5])
        #     x = act(x)
        #     if i < len(self.convs) - 1:
        #         x = F.dropout(x, p=self.dropout, training=self.training)
        for i, (conv, act) in enumerate(zip(self.convs, self.acts)):
            if self.conv_type == 'edge_feature' and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
                
            x = act(x)
            if i < len(self.convs) - 1:  # 最后一层不使用dropout
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
class Model(nn.Module):
    def __init__(self, encoder, projection_dim, tau=0.5):
        super(Model, self).__init__()
        self.encoder = encoder
        self.input_dim = encoder.convs[0].in_channels
        self.hidden_dim = encoder.convs[0].out_channels
        
        # 使用AttentivePooling替代原来的graph_pool
        self.graph_pool = AttentivePooling(self.hidden_dim)

        # 投影头的两层 MLP
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, projection_dim)

        self.tau = tau
        
    def to(self, device):
        super().to(device)
        self.encoder = self.encoder.to(device)
        self.graph_pool = self.graph_pool.to(device)
        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)
        return self
    
    def forward(self, x, edge_index, batch, edge_attr=None):
        try:
            device = x.device
            edge_index = edge_index.to(device)
            batch = batch.to(device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)
            
            assert x.dim() == 2, f"Expected 2D input tensor, got {x.dim()}D"
            assert edge_index.dim() == 2, f"Expected 2D edge_index tensor, got {edge_index.dim()}D"
            assert edge_index.size(0) == 2, f"Edge index must have shape [2, num_edges], got {edge_index.shape}"
            
            # 1. 首先通过编码器获取节点嵌入
            node_embeddings = self.encoder(x, edge_index, edge_attr)
            
            # 2. 使用AttentivePooling进行图级池化
            graph_embeddings = self.graph_pool(node_embeddings, batch)
            
            return graph_embeddings
            
        except Exception as e:
            print(f"Forward pass error: {e}")
            print(f"Shapes: x={x.shape}, edge_index={edge_index.shape}, batch={batch.shape}")
            if edge_attr is not None:
                print(f"edge_attr={edge_attr.shape}")
            raise

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        """
        通过投影头将图表示映射到对比学习空间
        """
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        计算两组图表示之间的相似度
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        计算子图级别的对比损失
        """
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        """
        计算总的对比损失
        """
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        loss = (l1 + l2) * 0.5
        return loss.mean() if mean else loss.sum()

    def get_embedding(self, g):
        """
        获取单个图的嵌入表示
        """
        x = g.ndata['features']
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0)
        batch = torch.zeros(g.number_of_nodes(), dtype=torch.long, device=x.device)
        edge_attr = None
        if 'features' in g.edata:
            edge_attr = g.edata['features']
        
        with torch.no_grad():
            # 获取节点嵌入
            node_embeddings = self.encoder(x, edge_index, edge_attr)
            # 使用AttentivePooling进行图级池化
            graph_embeddings = self.graph_pool(node_embeddings, batch)
            # 通过投影层获得最终维度的嵌入
            z = self.projection(graph_embeddings)
        
        return z[0]
