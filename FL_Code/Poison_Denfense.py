import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score

# =======================
# 1. 设置随机种子和设备
# =======================

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =======================
# 2. 数据加载和预处理
# =======================

def process_csv_to_vector(file_path, max_length=2000):
    """
    将CSV文件转换为固定长度的数据向量，通过截断或填充来达到max_length。

    Args:
        file_path (str): CSV文件的路径
        max_length (int): 每个样本的最大长度（默认为2000）

    Returns:
        np.ndarray: 处理后的数据，形状为 (max_length, num_columns)
    """
    df = pd.read_csv(file_path)
    data = df.to_numpy()
    if data.shape[0] > max_length:
        data = data[:max_length, :]
    elif data.shape[0] < max_length:
        padding = np.full((max_length - data.shape[0], data.shape[1]), -2)
        data = np.vstack((data, padding))
    return data  # 保持为2D

def load_data(data_folder, max_length=2000):
    """
    从指定文件夹加载数据，将每个子文件夹视为一个类别。

    Args:
        data_folder (str): 数据文件夹的路径
        max_length (int): 每个样本的最大长度（默认为2000）

    Returns:
        tuple: (数据, 标签)
    """
    all_data = []
    all_labels = []

    # 遍历子文件夹，将每个子文件夹作为一个类别
    for label, subfolder in enumerate(sorted(os.listdir(data_folder))):
        subfolder_path = os.path.join(data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(subfolder_path, file)
                    vector = process_csv_to_vector(file_path, max_length)
                    all_data.append(vector)
                    all_labels.append(label)

    data = np.array(all_data, dtype=np.float32)  # 形状: (样本数, 2000, num_columns)
    labels = np.array(all_labels, dtype=np.int32)

    return data, labels

# **请根据您的环境更新数据文件夹路径**
data_folder = "C:\\Users\\wwyl5\\anaconda3\\envs\\llm_pcap\\virus_new3class"  # 替换为您的实际路径

# 加载数据
print("Loading data...")
data, labels = load_data(data_folder)
print(f"Data shape: {data.shape}")  # 例如: (样本数, 2000, num_columns)

# 分割训练集和测试集（80:20）
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# 标准化缩放
X_train_mean = X_train.mean()
X_train_std = X_train.std()

X_train_scaled = (X_train - X_train_mean) / X_train_std
X_test_scaled = (X_test - X_train_mean) / X_train_std

# 检查是否存在NaN或无穷大值
if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
    print("Training data contains NaN or infinite values.")
    # 根据需要处理或移除有问题的数据点

# 填充宽度维度，使其能被8整除
num_columns = X_train_scaled.shape[2]
if num_columns % 8 != 0:
    pad_size = 8 - (num_columns % 8)
    print(f"Padding width dimension by {pad_size} to make it divisible by 8.")
    X_train_scaled = np.pad(
        X_train_scaled,
        pad_width=((0, 0), (0, 0), (0, pad_size)),
        mode='constant',
        constant_values=0
    )
    X_test_scaled = np.pad(
        X_test_scaled,
        pad_width=((0, 0), (0, 0), (0, pad_size)),
        mode='constant',
        constant_values=0
    )
    num_columns += pad_size
    print(f"New number of columns after padding: {num_columns}")
else:
    print("No padding needed for width dimension.")

# 添加序列维度 (N, T, D)
# LSTM expects input shape: (batch_size, sequence_length, input_size)
X_train = X_train_scaled  # 形状: (样本数, 2000, num_columns)
X_test = X_test_scaled    # 形状: (样本数, 2000, num_columns)
print(f"Reshaped training set: {X_train.shape}")
print(f"Reshaped testing set: {X_test.shape}")

# 在填充后复制原始测试数据（保持原始数据不被修改）
X_test_original_np = X_test.copy()  # 形状: (样本数, 2000, num_columns)

# 转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# =======================
# 4. 定义多层基于LSTM的自编码器模型
# =======================

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(LSTMAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 编码器
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 解码器
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        # 编码
        _, (hidden, cell) = self.encoder(x)

        # 准备解码器的输入，全零张量
        decoder_input = torch.zeros(batch_size, seq_len, self.hidden_size).to(x.device)

        # 解码
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

        # 将解码器输出映射回输入尺寸
        decoded = self.output_layer(decoder_output)
        return decoded

# =======================
# 5. 创建数据加载器并关联类标签
# =======================

# 创建 TensorDatasets
train_dataset = TensorDataset(X_train_tensor, torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(X_test_tensor, torch.tensor(y_test, dtype=torch.long))

# 创建 DataLoaders
batch_size = 16  # 根据您的内存限制调整
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =======================
# 6. 初始化模型、损失函数、优化器和学习率调度器
# =======================

input_size = num_columns
hidden_size = 16 # 可以根据需要调整
num_layers = 3

autoencoder = LSTMAutoencoder(input_size, hidden_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# =======================
# 7. 训练自编码器或加载已有模型
# =======================

model_path = 'best_lstm_autoencoder.pth'

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    autoencoder.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")
else:
    print("No existing model found. Starting training...")
    num_epochs = 25
    patience = 10
    best_val_loss = np.inf
    trigger_times = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        autoencoder.train()
        train_loss = 0
        for inputs, labels_batch in train_loader:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels_batch in test_loader:
                inputs = inputs.to(device)
                labels_batch = labels_batch.to(device)

                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        avg_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(autoencoder.state_dict(), model_path)
            print(f"Model saved to {model_path}.")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    # 绘制训练和验证损失
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()
    plt.show()

# =======================
# 8. 可视化重建结果
# =======================

autoencoder.eval()
with torch.no_grad():
    sample_inputs, sample_labels = next(iter(test_loader))
    sample_inputs = sample_inputs.to(device)
    reconstructed = autoencoder(sample_inputs)

# 将张量移动到CPU并转换为numpy数组以便绘图
sample_inputs = sample_inputs.cpu().numpy()
reconstructed = reconstructed.cpu().numpy()

# 仅绘制一个原始和重建的样本
n = 1  # 要可视化的样本数量
for i in range(n):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(sample_inputs[i].T, color='blue')
    plt.title('Original')
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    plt.subplot(2, 1, 2)
    plt.plot(reconstructed[i].T, color='red')
    plt.title('Reconstructed')
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    plt.tight_layout()
    plt.show()

# =======================
# 9. 应用数据扰动到测试数据（随机竄改）
# =======================

def poison_test_data(X_test, y_test, poison_fraction=0.5):
    """
    将测试数据的一半进行污染，随机从另一类别复制某一笔数据的前600行替换原始数据。

    Args:
        X_test (np.ndarray): 测试数据，形状为 (samples, 2000, num_columns)
        y_test (np.ndarray): 测试标签，形状为 (samples,)
        poison_fraction (float): 污染比例，默认为0.5（50%）

    Returns:
        tuple: (被污染的数据, 更新后的标签, 异常标签)
    """
    print(f"Poisoning started...")
    num_samples = X_test.shape[0]
    num_poison = int(poison_fraction * num_samples)

    # 确保每次运行结果一致
    np.random.seed(42)

    # 随机选择要污染的样本索引
    poisoned_indices = np.random.choice(num_samples, num_poison, replace=False)
    print(f"Number of samples to poison: {num_poison}")

    # 创建异常标签
    anomaly_labels = np.zeros(num_samples)
    anomaly_labels[poisoned_indices] = 1

    # 从不同类别选择数据进行污染
    unique_classes = np.unique(y_test)
    if len(unique_classes) < 2:
        print("Not enough classes to perform data poisoning.")
        return X_test, y_test.copy(), anomaly_labels

    # 创建一个映射：类别 -> 样本索引
    class_to_indices = {cls: np.where(y_test == cls)[0] for cls in unique_classes}

    X_test_poisoned = X_test.copy()  # 保留原始测试数据

    for idx in poisoned_indices:
        original_class = y_test[idx]
        # 选择一个不同的类别
        possible_classes = unique_classes[unique_classes != original_class]
        if len(possible_classes) == 0:
            print(f"No other classes available to poison sample {idx}.")
            continue
        infecting_class = np.random.choice(possible_classes)
        infecting_indices = class_to_indices[infecting_class]
        if len(infecting_indices) == 0:
            print(f"No samples found for infecting class {infecting_class}. Skipping sample {idx}.")
            continue
        # 随机选择一个感染来源样本
        infecting_sample_idx = np.random.choice(infecting_indices)
        infecting_sample = X_test[infecting_sample_idx]
        # 替换前600行
        X_test_poisoned[idx, 1000:2000, :] = infecting_sample[:1000, :]

    print(f"Poisoning completed.")
    return X_test_poisoned, y_test.copy(), anomaly_labels



# 应用数据扰动到测试数据
print("Applying data perturbation to test data...")
X_test_perturbed_np, y_test_perturbed_np, anomaly_labels = poison_test_data(
    X_test_original_np, y_test
)

print(f"Perturbed test data shape: {X_test_perturbed_np.shape}")
print(f"Original test data shape: {X_test_original_np.shape}")
print(f"Anomaly labels shape: {anomaly_labels.shape}")

# 确保样本数和形状一致
assert X_test_perturbed_np.shape == X_test_original_np.shape, \
    "Mismatch in shape between perturbed and original test data."

# 转换为张量
X_test_perturbed_tensor = torch.tensor(X_test_perturbed_np, dtype=torch.float32).to(device)

# =======================
# 10. 计算重建误差（异常分数）
# =======================

autoencoder.eval()
with torch.no_grad():
    # 使用被扰动的测试数据作为模型输入
    reconstructions = autoencoder(X_test_perturbed_tensor)
    print(f"Reconstructions shape: {reconstructions.shape}")

    # 获取每个样本所属的类别的原型
    test_labels_tensor = torch.tensor(y_test_perturbed_np, dtype=torch.long).to(device)

    # 计算重建误差（MSE）作为异常分数
    reconstruction_errors = torch.mean(
        (reconstructions - X_test_perturbed_tensor) ** 2, dim=[1, 2]
    ).cpu().numpy()

print(f"Reconstruction errors shape: {reconstruction_errors.shape}")

# =======================
# 11. 设置异常检测阈值（基于最佳F1分数）
# =======================

# 使用重建误差和真实标签计算最佳阈值以最大化F1分数
precision, recall, thresholds_pr = precision_recall_curve(anomaly_labels, reconstruction_errors)

# 计算每个阈值的F1分数
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # 避免除以零

# 找到F1分数最大的阈值
best_index = np.argmax(f1_scores)
best_threshold = thresholds_pr[best_index]
best_f1 = f1_scores[best_index]

print(f"Best threshold based on F1 score: {best_threshold}")
print(f"Best F1 score: {best_f1:.4f}")

# =======================
# 12. 检测异常
# =======================

# 使用最佳阈值进行异常检测
predictions = (reconstruction_errors > best_threshold).astype(int)
print(f"Predictions shape: {predictions.shape}")

# =======================
# 13. 评估检测性能
# =======================

# 确保样本数一致
assert anomaly_labels.shape[0] == predictions.shape[0], \
    "Mismatch in sample size between anomaly labels and predictions."

print("Confusion Matrix:")
print(confusion_matrix(anomaly_labels, predictions))

print("\nClassification Report:")
print(classification_report(anomaly_labels, predictions))

# 计算ROC AUC
roc_auc = roc_auc_score(anomaly_labels, reconstruction_errors)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# 绘制ROC曲线
fpr, tpr, thresholds_roc = roc_curve(anomaly_labels, reconstruction_errors)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# =======================
# 14. 绘制重建误差直方图
# =======================

# 分离正常样本和异常样本的重建误差
normal_errors = reconstruction_errors[anomaly_labels == 0]
anomaly_errors = reconstruction_errors[anomaly_labels == 1]

plt.figure(figsize=(10, 6))
plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal', color='blue', edgecolor='black')
plt.hist(anomaly_errors, bins=50, alpha=0.6, label='Anomaly', color='red', edgecolor='black')
plt.axvline(best_threshold, color='green', linestyle='dashed', linewidth=2, label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Number of Samples')
plt.title('Distribution of Reconstruction Errors')
plt.legend()
plt.show()
