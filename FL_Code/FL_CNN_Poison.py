!pip install tensorflow_federated==0.80.0
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow_federated.python.aggregators import DifferentiallyPrivateFactory


def data_poison(data, labels):
    import random
    # 获取唯一的类别列表
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        print("类别数量不足，无法进行数据投毒。")
        return data, labels

    # 随机选择两个不同的类别
    infected_class = np.random.choice(unique_classes)
    remaining_classes = unique_classes[unique_classes != infected_class]
    infecting_class = np.random.choice(remaining_classes)

    print(f"被感染类（将被修改）：{infected_class}")
    print(f"感染类（感染源）：{infecting_class}")

    # 获取被感染类和感染类的样本索引
    infected_indices = np.where(labels == infected_class)[0]
    infecting_indices = np.where(labels == infecting_class)[0]

    # 对被感染类中的每个样本进行修改
    for idx in infected_indices:
        # 随机选择一个感染类的样本
        infecting_sample_idx = np.random.choice(infecting_indices)
        infecting_sample = data[infecting_sample_idx]

        # 替换前500行
        data[idx][:500, :] = infecting_sample[:500, :]

    return data, labels

# 数据预处理函数
def process_csv_to_vector(file_path, max_length=2000):
    df = pd.read_csv(file_path)
    data = df.to_numpy()
    if data.shape[0] > max_length:
        data = data[:max_length, :]
    elif data.shape[0] < max_length:
        padding = np.full((max_length - data.shape[0], data.shape[1]), -2)
        data = np.vstack((data, padding))
    return data  # 保持二维结构，不扁平化

def load_data(data_folder, max_length=2000):
    all_data = []
    all_labels = []

    # 遍历每个子文件夹（每个文件夹代表一个标签）
    for label, subfolder in enumerate(os.listdir(data_folder)):
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

# 数据文件夹路径（请根据实际路径修改）
data_folder = "/content/drive/MyDrive/virus_new3class"

# 加载数据
data, labels = load_data(data_folder)
data, labels = data_poison(data, labels)
print(f"数据形状: {data.shape}")  # 例如: (样本数, 2000, num_columns)

# 切分数据集为训练集和测试集（80:20 比例）
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 标准化数据
num_samples, num_time_steps, num_columns = data.shape
scaler = StandardScaler()

# 重塑为二维以进行标准化
X_train_reshaped = X_train.reshape(-1, num_columns)
X_test_reshaped = X_test.reshape(-1, num_columns)

# 标准化
scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

print("数据标准化完成")

# 定义输入形状和类别数
input_shape = (2000, X_train.shape[2], 1)  # (高度, 宽度, 通道数)
num_classes = len(np.unique(y_train))
print(f"输入形状: {input_shape}")
print(f"类别数: {num_classes}")

# 重塑训练集和测试集的形状以适应 Conv2D
X_train = X_train_scaled.reshape(X_train.shape[0], 2000, X_train.shape[2], 1)
X_test = X_test_scaled.reshape(X_test.shape[0], 2000, X_test.shape[2], 1)
print(f"重塑后的训练集形状: {X_train.shape}")
print(f"重塑后的测试集形状: {X_test.shape}")

# 分配数据到不同的客户端
num_clients = 10  # 增加客户端数量以提高联邦学习效果
client_size = len(X_train) // num_clients

client_datasets = []
for client_id in range(num_clients):
    start_index = client_id * client_size
    end_index = start_index + client_size if client_id != num_clients - 1 else len(X_train)

    client_x = X_train[start_index:end_index]
    client_y = y_train[start_index:end_index]

    # 创建 tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((client_x, client_y))
    dataset = dataset.shuffle(buffer_size=256).batch(32)
    client_datasets.append(dataset)

print(f"客户端数量: {len(client_datasets)}")

# 获取输入规范
example_dataset = client_datasets[0]
input_spec = example_dataset.element_spec

# 定义 CNN 模型（不编译）
def create_cnn_model(input_shape, num_classes):
    model = Sequential()

    # 第一个卷积层和池化层
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 第二个卷积层和池化层
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 展平层
    model.add(Flatten())

    # 全连接层
    model.add(Dense(128, activation='relu'))

    # 输出层
    model.add(Dense(num_classes, activation='softmax'))

    return model

# 定义模型函数
def model_fn():
    keras_model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)
    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# 定义优化器
def client_optimizer_fn():
    return tf.keras.optimizers.Adam(learning_rate=0.2)  # 使用 Adam 优化器

def server_optimizer_fn():
    return tf.keras.optimizers.Adam(learning_rate=1)



# 构建联邦平均过程
iterative_process = tff.learning.algorithms.build_unweighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=client_optimizer_fn,
    server_optimizer_fn=server_optimizer_fn
)

# 初始化过程
state = iterative_process.initialize()

# 定义评估函数
def evaluate(state, test_data):
    keras_model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)
    tff.learning.assign_weights_to_keras_model(keras_model, state.model)
    loss, accuracy = keras_model.evaluate(test_data[0], test_data[1], verbose=0)
    return loss, accuracy

# 训练轮数
NUM_ROUNDS = 25  # 根据需要调整轮数

# 构建联邦评估过程
evaluation = tff.learning.algorithms.build_fed_eval(
    model_fn=model_fn
)

# 在训练过程中添加评估步骤
for round_num in range(1, NUM_ROUNDS + 1):
    state, metrics = iterative_process.next(state, client_datasets)
    train_accuracy = metrics['client_work']['train']['sparse_categorical_accuracy']
    train_loss = metrics['client_work']['train']['loss']
    print(f'第 {round_num} 轮, 训练准确率: {train_accuracy:.4f}, 训练损失: {train_loss:.4f}')

print("训练完成")
