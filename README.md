# Fashion Matching with Deep Metric Learning

## 📌 项目概述
本项目实现了一个基于深度度量学习的时尚匹配系统，能够判断两件衣服是否属于同一款式（相同类别）。系统使用三元组网络架构和三元组损失函数，通过ResNet50骨干网络提取特征，在嵌入空间中学习服装的相似性表示。

## 📂 数据集结构
数据集需要遵循以下格式：
- **图像文件命名格式**：`a_b.jpg` (或.jpeg/.png)
  - `a`：服装类别ID (同一款式服装共享相同ID)
  - `b`：任意标识符 (如视角、颜色变体等)
## 🧠 核心算法

### 1. 模型架构
```python
class FashionMatcher(nn.Module):
  def __init__(self, embedding_size=128):
      super().__init__()
      backbone = models.resnet50(pretrained=True)
      self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
      self.embedding = nn.Sequential(
          nn.Flatten(),
          nn.Linear(2048, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(0.3),
          nn.Linear(512, embedding_size)
      )
```
## 运行命令
```bash
python suanfa_3.py \
  --train_data ./fashion_data/train \  # 训练集路径
  --test_data ./fashion_data/test \    # 测试集路径
  --batch 64 \                        # 批大小
  --epochs 20 \                       # 训练轮数
  --model_path best_model.pth \        # 模型保存路径
  --threshold 0.85                    # 相似度阈值
```
核心算法
1. 模型架构
骨干网络: ResNet50 (ImageNet预训练)

嵌入层:

全连接层 (2048 → 512)

BatchNorm + ReLU + Dropout(0.3)

输出层 (512 → 128维嵌入向量)

2. 三元组损失函数
python
class OnlineTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        mask_positive = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_negative = ~mask_positive
        
        hardest_positive = (pairwise_dist * mask_positive).max(dim=1)[0]
        hardest_negative = (pairwise_dist + 1e6 * mask_positive).min(dim=1)[0]
        
        losses = torch.relu(hardest_positive - hardest_negative + self.margin)
        return losses.mean()
使用最困难样本挖掘(hardest mining)策略

边界值(margin)默认为0.5

3. 数据划分策略
分层抽样: 按类别分层划分训练/验证集

每类80%用于训练，20%用于验证

每类至少保留1个样本作为验证

运行指南
环境要求
text
Python >= 3.6
PyTorch >= 1.7
torchvision
scikit-learn
tqdm
Pillow
seaborn
训练模型
bash
python suanfa_3.py \
  --train_data [训练集路径] \
  --test_data [测试集路径] \
  --batch [批大小] \
  --epochs [训练轮数] \
  --model_path [模型保存路径] \
  --threshold [相似度阈值]
示例：

bash
python suanfa_3.py \
  --train_data ./fashion_data/train \
  --test_data ./fashion_data/test \
  --batch 64 \
  --epochs 20 \
  --model_path best_model.pth \
  --threshold 0.85
仅评估模型
bash
python suanfa_3.py \
  --eval \
  --test_data [测试集路径] \
  --model_path [模型路径] \
  --threshold [相似度阈值]
参数说明
参数	默认值	说明
--train_data	无	训练数据集路径 (必需)
--test_data	无	测试数据集路径 (必需)
--batch	64	训练批量大小
--epochs	10	训练轮数
--model_path	'fashion_matcher.pth'	模型保存/加载路径
--eval	False	仅评估模式
--threshold	0.85	相似度判定阈值
输出结果
评估完成后将生成：

evaluation_report.txt - 包含完整评估指标

分类报告 (精确率/召回率/F1分数)

ROC曲线和AUC值

混淆矩阵

性能指标
准确率: 相同/不同类别判断准确率

推理速度: 单对图像匹配时间 (毫秒)

ROC AUC: 模型整体区分能力指标

分类报告: 详细分类性能指标

注意事项
数据集应包含足够类别 (建议>100类)

每类至少2张图像 (建议5-10张)

图像尺寸建议256×256以上

相似度阈值可根据实际需求调整


