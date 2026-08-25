# Subgraph-level final computation

本目录整理了 `experiment/magic/subgraph_level` 的最终计算代码，包含图子图对比学习、序列特征、联合微调、节点对扩展和可视化脚本。

## 文件说明

- `subgraph_level_emb.py`: 子图编码器训练与节点嵌入生成入口。
- `model.py`: 图编码器、注意力池化和对比学习模型。
- `seq_emb_gen.py`: 交易序列编码模型及其特征字典。
- `finetune.py`: 图嵌入与交易序列嵌入的联合训练/评估。
- `extend_pairs.py`: 节点对预测和随机节点对扩展。
- `visul.py`: 表征距离统计与可视化。

运行前请确认脚本中的数据集、模型 checkpoint 和输出路径指向当前环境中的有效文件。

示例：

```bash
python subgraph_level_emb.py --conv_type graphsage --gpu 1 --batch_size 64
```
