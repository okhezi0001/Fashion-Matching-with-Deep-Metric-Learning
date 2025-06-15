import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
import copy
import warnings
import argparse
# 忽略警告
warnings.filterwarnings('ignore')


# ----------------------
# 1. 数据预处理与加载
# ----------------------
class DeepFashionDataset(Dataset):
    def __init__(self, root_dir, transform=None, min_samples=2):
        """
        root_dir: 数据集路径
        transform: 图像预处理
        min_samples: 最小样本数要求（过滤样本不足的类别）
        """
        self.root_dir = root_dir
        all_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # 从文件名提取标签 (a_b.jpg中的a部分)
        self.labels = {}
        self.label_to_indices = {}
        for file in all_files:
            label = file.split('_')[0]  # 使用a作为标签
            self.labels[file] = label

            # 构建标签到文件索引的映射
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(file)

        # 过滤样本数不足的类别
        self.valid_labels = [label for label, files in self.label_to_indices.items()
                             if len(files) >= min_samples]

        # 创建有效文件列表
        self.image_files = []
        for label in self.valid_labels:
            self.image_files.extend(self.label_to_indices[label])

        # 创建数字标签映射
        self.unique_labels = sorted(set(self.valid_labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}

        # 统计数据集信息
        self.num_classes = len(self.unique_labels)
        self.num_images = len(self.image_files)

        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 重新构建标签映射（只包含有效类别）
        self.labels = {}
        for file in self.image_files:
            label = file.split('_')[0]
            self.labels[file] = label

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 处理单个索引
        file_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, file_name)
        img = Image.open(img_path).convert('RGB')

        label_str = self.labels[file_name]
        label_idx = self.label_to_idx[label_str]

        if self.transform:
            img = self.transform(img)

        return img, label_idx, file_name

    def get_dataset_info(self):
        """返回数据集统计信息"""
        class_counts = {label: len(files) for label, files in self.label_to_indices.items()
                        if label in self.valid_labels}
        return {
            "num_classes": self.num_classes,
            "num_images": self.num_images,
            "class_distribution": class_counts
        }

    def get_indices_for_label(self, label):
        """获取指定标签的所有索引"""
        return [i for i, file in enumerate(self.image_files) if self.labels[file] == label]


# ----------------------
# 2. 三元组数据生成器
# ----------------------
class TripletSampler:
    """在线生成困难三元组 - 基于文件名标签"""

    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.label_to_files = {
            label: files for label, files in dataset.label_to_indices.items()
            if label in dataset.valid_labels
        }

        # 确保有足够的类别生成批次
        if len(self.label_to_files) < batch_size // 2:
            raise ValueError(f"Not enough valid classes ({len(self.label_to_files)}) for batch size {batch_size}")

    def __iter__(self):
        # 为每个batch选择锚点、正样本、负样本
        for _ in range(len(self.dataset) // self.batch_size):
            # 随机选择batch_size/2个类别（每个类别选2张图）
            selected_labels = random.sample(
                list(self.label_to_files.keys()),
                k=self.batch_size // 2
            )

            anchors, positives, negatives = [], [], []
            for label in selected_labels:
                # 从当前标签对应的文件中随机选两个
                files = self.label_to_files[label]
                anchor_file, positive_file = random.sample(files, 2)
                anchors.append(anchor_file)
                positives.append(positive_file)

                # 选择不同标签的文件作为负样本
                negative_label = random.choice([
                    l for l in self.label_to_files.keys() if l != label
                ])
                negative_file = random.choice(self.label_to_files[negative_label])
                negatives.append(negative_file)

            # 获取文件在数据集中的索引
            anchor_idxs = [self.dataset.image_files.index(f) for f in anchors]
            positive_idxs = [self.dataset.image_files.index(f) for f in positives]
            negative_idxs = [self.dataset.image_files.index(f) for f in negatives]

            # 将所有索引合并为一个列表
            all_idxs = anchor_idxs + positive_idxs + negative_idxs
            yield all_idxs

    def __len__(self):
        return len(self.dataset) // self.batch_size


# ----------------------
# 3. 模型架构
# ----------------------
class FashionMatcher(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        # 使用预训练的ResNet50作为骨干
        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]  # 移除原始全连接层
        )
        # 自定义嵌入层
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.embedding(features)


# ----------------------
# 4. 三元组损失函数
# ----------------------
class OnlineTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # 计算所有样本间的距离矩阵
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        # 创建标签掩码
        mask_positive = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_negative = ~mask_positive

        # 计算最困难正样本和负样本
        hardest_positive = (pairwise_dist * mask_positive).max(dim=1)[0]
        hardest_negative = (pairwise_dist + 1e6 * mask_positive).min(dim=1)[0]

        # 计算三元组损失
        losses = torch.relu(hardest_positive - hardest_negative + self.margin)
        return losses.mean()


# ----------------------
# 5. 特征提取辅助函数
# ----------------------
def extract_features(model, dataset, device, batch_size=64):
    """提取整个数据集的特征向量"""
    model.eval()
    features_dict = {}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    with torch.no_grad():
        for images, labels, filenames in tqdm(loader, desc='Extracting features'):
            images = images.to(device, non_blocking=True)
            features = model(images).cpu().numpy()

            # L2归一化
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

            for i, filename in enumerate(filenames):
                features_dict[filename] = {
                    'feature': features[i],
                    'label': dataset.labels[filename]
                }

    return features_dict


# ----------------------
# 6. 评估辅助函数
# ----------------------
def evaluate_accuracy(features_dict, dataset, num_pairs=1000, threshold=0.85):
    """评估模型在数据集上的准确率"""
    # 生成测试对
    test_pairs = []
    all_files = list(features_dict.keys())

    # 生成正样本对（相同衣服）
    same_count = 0
    while same_count < num_pairs // 2:
        # 随机选择一个类别
        label = random.choice(dataset.valid_labels)
        files = dataset.label_to_indices[label]

        # 确保该类有至少两张图片
        if len(files) < 2:
            continue

        # 随机选择两张不同的图片
        img1, img2 = random.sample(files, 2)
        test_pairs.append((img1, img2, 1))  # 1表示相同衣服
        same_count += 1

    # 生成负样本对（不同衣服）
    diff_count = 0
    while diff_count < num_pairs // 2:
        # 随机选择两个不同的类别
        label1, label2 = random.sample(dataset.valid_labels, 2)
        files1 = dataset.label_to_indices[label1]
        files2 = dataset.label_to_indices[label2]

        # 从每个类别随机选择一张图片
        img1 = random.choice(files1)
        img2 = random.choice(files2)
        test_pairs.append((img1, img2, 0))  # 0表示不同衣服
        diff_count += 1

    # 评估测试对
    y_true = []
    y_pred = []

    for img1, img2, label in test_pairs:
        feat1 = features_dict[img1]['feature']
        feat2 = features_dict[img2]['feature']

        # 计算相似度（点积）
        similarity = np.dot(feat1, feat2)

        # 记录结果
        y_true.append(label)
        y_pred.append(1 if similarity > threshold else 0)

    # 计算准确率
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    return accuracy


# ----------------------
# 7. 训练流程
# ----------------------
def train_model(train_dir, model_save_path='fashion_matcher.pth', epochs=20, batch_size=64):
    # 配置参数
    EMBEDDING_SIZE = 128
    LR = 1e-4
    VAL_SPLIT = 0.2  # 20%作为验证集
    MIN_SAMPLES_PER_CLASS = 2  # 每个类别最小样本数
    VAL_NUM_PAIRS = 1000  # 验证使用的配对数量
    VAL_THRESHOLD = 0.85  # 验证使用的阈值

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载完整数据集（自动过滤样本不足的类别）
    full_dataset = DeepFashionDataset(train_dir, min_samples=MIN_SAMPLES_PER_CLASS)
    full_info = full_dataset.get_dataset_info()
    print(f"Full Dataset Info (after filtering):")
    print(f"- Classes: {full_info['num_classes']}")
    print(f"- Images: {full_info['num_images']}")
    print(f"- Min class count: {min(full_info['class_distribution'].values())}")
    print(f"- Max class count: {max(full_info['class_distribution'].values())}")

    # 划分训练集和验证集（分层抽样）
    # 为每个类别收集索引
    class_indices = {}
    for label in full_dataset.valid_labels:
        class_indices[label] = full_dataset.get_indices_for_label(label)

    train_indices = []
    val_indices = []

    # 对每个类别分别划分
    for label, indices in class_indices.items():
        n = len(indices)
        n_val = max(1, int(n * VAL_SPLIT))  # 验证集至少1个样本
        n_train = n - n_val

        # 随机打乱并分配
        random.shuffle(indices)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])

    print(f"\nTraining Set: {len(train_indices)} images")
    print(f"Validation Set: {len(val_indices)} images")

    # 创建训练集和验证集
    train_dataset = DeepFashionDataset(train_dir, min_samples=MIN_SAMPLES_PER_CLASS)
    val_dataset = DeepFashionDataset(train_dir, min_samples=MIN_SAMPLES_PER_CLASS)

    # 创建数据加载器
    train_sampler = TripletSampler(train_dataset, batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # 模型初始化
    model = FashionMatcher(EMBEDDING_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = OnlineTripletLoss(margin=0.5)

    # 训练循环
    best_val_accuracy = 0.0
    best_model_state = None
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

        for batch_idx, (images, labels, filenames) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)

            # 前向传播
            embeddings = model(images)

            # 计算损失
            loss = criterion(embeddings, labels)

            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

        # 计算平均损失
        avg_loss = epoch_loss / len(progress_bar)
        train_losses.append(avg_loss)

        # 验证评估 - 使用与测试相同的评估标准
        print("Validating...")
        # 提取验证集特征
        val_features = extract_features(model, val_dataset, device, batch_size=64)
        # 评估验证准确率
        val_accuracy = evaluate_accuracy(
            val_features,
            val_dataset,
            num_pairs=VAL_NUM_PAIRS,
            threshold=VAL_THRESHOLD
        )
        val_accuracies.append(val_accuracy)

        # 更新学习率
        scheduler.step(val_accuracy)

        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {val_accuracy:.4f}')

        # 保存最佳模型（基于验证准确率）
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_save_path)
            print(f"Saved best model with val accuracy: {val_accuracy:.4f}")

    # 最终保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"Final best model saved to {model_save_path} with val accuracy: {best_val_accuracy:.4f}")
    else:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")




    return model


# ----------------------
# 8. 测试评估
# ----------------------
def evaluate_model(model, test_dir, threshold=0.85, num_pairs=1000, save_plots=True):
    """在测试集上评估模型性能"""
    device = next(model.parameters()).device
    print(f"\nEvaluating model on test set...")

    # 加载测试数据集（自动过滤样本不足的类别）
    test_dataset = DeepFashionDataset(test_dir, min_samples=2)
    test_info = test_dataset.get_dataset_info()
    print(f"Test Dataset Info (after filtering):")
    print(f"- Classes: {test_info['num_classes']}")
    print(f"- Images: {test_info['num_images']}")

    # 预计算所有测试图像特征
    print("Precomputing test features...")
    features_dict = extract_features(model, test_dataset, device)

    # 评估测试准确率
    test_accuracy = evaluate_accuracy(
        features_dict,
        test_dataset,
        num_pairs=num_pairs,
        threshold=threshold
    )

    # 生成测试对用于详细评估
    test_pairs = []
    all_files = list(features_dict.keys())

    # 生成正样本对（相同衣服）
    same_count = 0
    while same_count < num_pairs // 2:
        label = random.choice(test_dataset.valid_labels)
        files = test_dataset.label_to_indices[label]
        if len(files) < 2:
            continue
        img1, img2 = random.sample(files, 2)
        test_pairs.append((img1, img2, 1))
        same_count += 1

    # 生成负样本对（不同衣服）
    diff_count = 0
    while diff_count < num_pairs // 2:
        label1, label2 = random.sample(test_dataset.valid_labels, 2)
        files1 = test_dataset.label_to_indices[label1]
        files2 = test_dataset.label_to_indices[label2]
        img1 = random.choice(files1)
        img2 = random.choice(files2)
        test_pairs.append((img1, img2, 0))
        diff_count += 1

    # 详细评估
    y_true = []
    y_pred = []
    similarities = []
    inference_times = []

    for img1, img2, label in test_pairs:
        feat1 = features_dict[img1]['feature']
        feat2 = features_dict[img2]['feature']

        start_time = time.time()
        similarity = np.dot(feat1, feat2)
        end_time = time.time()

        y_true.append(label)
        similarities.append(similarity)
        y_pred.append(1 if similarity > threshold else 0)
        inference_times.append(end_time - start_time)

    # 计算评估指标
    avg_inference_time = np.mean(inference_times) * 1000  # 毫秒

    # 分类报告和混淆矩阵
    print("\n" + "=" * 50)
    print(f"Evaluation Results (Threshold={threshold:.2f})")
    print("=" * 50)
    print(f"Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Average Inference Time per Pair: {avg_inference_time:.4f} ms")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Different', 'Same']))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)


    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)



    # 保存评估结果
    with open('evaluation_report.txt', 'w') as f:
        f.write("=" * 50 + "\n")
        f.write(f"Fashion Matching Model Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Set Size: {test_info['num_images']} images\n")
        f.write(f"Number of Classes: {test_info['num_classes']}\n")
        f.write(f"Test Pairs: {num_pairs} (50% same, 50% different)\n")
        f.write(f"Threshold: {threshold}\n\n")
        f.write(f"Accuracy: {test_accuracy * 100:.2f}%\n")
        f.write(f"Average Inference Time per Pair: {avg_inference_time:.4f} ms\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=['Different', 'Same']))
        f.write(f"\nROC AUC: {roc_auc:.4f}")

    print("\nEvaluation complete. Results saved to evaluation_report.txt")

    return test_accuracy, roc_auc


# ----------------------
# 9. 主程序
# ----------------------
if __name__ == "__main__":
    # 配置路径
    if __name__ == "__main__":
        # 创建命令行参数解析器
        parser = argparse.ArgumentParser(description='Fashion Matching Model Training and Evaluation')
        parser.add_argument('--train_data', type=str, help='Path to training dataset directory')
        parser.add_argument('--test_data', type=str, required=True, help='Path to testing dataset directory')
        parser.add_argument('--batch', type=int, default=64, help='Batch size for training and evaluation')
        parser.add_argument('--model_path', type=str, default='fashion_matcher.pth', help='Path to save/load model')
        parser.add_argument('--eval', action='store_true', help='Only evaluate model without training')
        parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (if not eval mode)')
        parser.add_argument('--threshold', type=float, default=0.85, help='Similarity threshold for evaluation')

        args = parser.parse_args()

        # 设备设置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # 评估模式：直接加载模型进行评估
        if args.eval:
            print("Running in evaluation-only mode...")
            model = FashionMatcher().to(device)

            # 检查模型文件是否存在
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model file not found: {args.model_path}")

            print(f"Loading model from {args.model_path}...")
            model.load_state_dict(torch.load(args.model_path, map_location=device))

            # 评估模型
            print(f"Evaluating model on test set: {args.test_data}")
            evaluate_model(
                model,
                args.test_data,
                threshold=args.threshold,
                num_pairs=10000
            )

        # 训练+评估模式：先训练再评估
        else:
            if not args.train_data:
                raise ValueError("Training data path is required in train mode")

            print("Starting training process...")
            print(f"Training data: {args.train_data}")
            print(f"Test data: {args.test_data}")
            print(f"Batch size: {args.batch}")
            print(f"Epochs: {args.epochs}")
            print(f"Model will be saved to: {args.model_path}")

            # 训练模型
            model = train_model(
                train_dir=args.train_data,
                model_save_path=args.model_path,
                epochs=args.epochs,
                batch_size=args.batch
            )

            # 评估模型
            print("Evaluating model on test set...")
            evaluate_model(
                model,
                args.test_data,
                threshold=args.threshold,
                num_pairs=10000
            )

        print("All operations completed successfully!")