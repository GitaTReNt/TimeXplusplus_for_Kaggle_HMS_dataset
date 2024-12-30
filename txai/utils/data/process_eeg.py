# preprocess_eegs.py

import os
import torch
from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
from tqdm import tqdm  # 进度条库

# 定义选定的 EEG 通道，排除 EKG
FEATS = ['Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']
FEAT2IDX = {x: y for y, x in zip(FEATS, range(len(FEATS)))}


def get_lowpass_coeffs(cutoff_freq=20, sampling_rate=200, order=4):
    """
    预计算Butterworth低通滤波器的系数。

    参数：
    - cutoff_freq (float): 截止频率（Hz）
    - sampling_rate (float): 采样率（Hz）
    - order (int): 滤波器阶数

    返回：
    - b, a: 滤波器系数
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


# 预计算滤波器系数
b, a = get_lowpass_coeffs()


def butter_lowpass_filter(data, b, a):
    """
    应用Butterworth低通滤波器到数据。

    参数：
    - data (np.ndarray): 输入信号，形状为 (T, C)
    - b, a: 滤波器系数

    返回：
    - filtered_data (np.ndarray): 滤波后的信号，形状为 (T, C)
    """
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def preprocess_transform(X, t, b, a, downsample_factor=10):
    """
    对EEG数据应用低通滤波、下采样并规范化时间向量。

    参数：
    - X (torch.Tensor): 原始EEG数据，形状为 (T, C)
    - t (torch.Tensor): 时间向量，形状为 (T,)
    - b, a: 滤波器系数
    - downsample_factor (int): 下采样因子

    返回：
    - tuple: (滤波后的X, 下采样后的t)
    """
    # 将X转换为NumPy数组
    X_np = X.cpu().numpy()

    # 应用低通滤波器
    X_filtered = butter_lowpass_filter(X_np, b, a)

    # 下采样
    X_filtered = X_filtered[::downsample_factor, :]  # 形状 (T_down, C)
    t_down = t[::downsample_factor]

    # 规范化时间向量 t 到 0-50 秒范围
    t_min = t_down.min()
    t_max = t_down.max()
    if t_max - t_min == 0:
        t_norm = torch.zeros_like(t_down)
    else:
        t_norm = 50.0 * (t_down - t_min) / (t_max - t_min)

    # 创建数组副本以消除负 stride
    X_filtered = torch.tensor(X_filtered.copy(), dtype=torch.float32).cpu()
    t_norm = t_norm.cpu()

    return X_filtered, t_norm


def preprocess_and_save_eegs(all_eegs, train_df, path='/root/autodl-tmp/time/datasets/hmsprocessed', downsample_factor=10):
    """
    预处理所有 EEG 数据并保存为 .pt 文件。

    参数：
    - all_eegs (dict): EEG 数据字典，键为 eeg_id，值为 (T, C) 的 NumPy 数组
    - train_df (pd.DataFrame): 训练集 DataFrame，包含 eeg_id 和目标标签
    - path (str): 保存预处理后数据的路径
    - downsample_factor (int): 下采样因子
    """
    os.makedirs(path, exist_ok=True)

    # 将 train_df 中的 eeg_id 转换为字符串集合
    eeg_ids_in_train = set(train_df['eeg_id'].astype(str).tolist())

    # 处理每个 EEG
    for i, (eeg_id, data) in enumerate(tqdm(all_eegs.items(), desc="Preprocessing EEGs")):
        if i % 100 == 0 and i != 0:
            print(f'Processed {i} / {len(all_eegs)} EEGs')

        # 将 NumPy 数组转换为 Tensor
        X = torch.tensor(data, dtype=torch.float32)  # (T, C)

        # 确保 eeg_id 为字符串
        eeg_id_str = str(eeg_id)

        # 打印前10个 eeg_id 以进行调试
        if i < 10:
            print(f"Processing eeg_id: {eeg_id_str}")

        # 检查 eeg_id 是否存在于 train.csv
        if eeg_id_str not in eeg_ids_in_train:
            print(f"[WARNING] No matching row found for eeg_id: {eeg_id_str}")
            continue

        # 获取对应的目标标签
        row = train_df[train_df['eeg_id'].astype(str) == eeg_id_str]
        if row.empty:
            print(f"[WARNING] No matching row found for eeg_id: {eeg_id_str}")
            continue
        y = row.iloc[0][['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].values.astype(
            np.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # 应用预处理
        t = torch.tensor(np.arange(len(X)), dtype=torch.float32)
        X_filtered, t_norm = preprocess_transform(X, t, b, a, downsample_factor)

        # 保存预处理后的数据
        torch.save({
            'X': X_filtered,  # (T_down, C)
            't': t_norm,  # (T_down,)
            'y': y  # (6,)
        }, os.path.join(path, f'{eeg_id_str}.pt'))

    print('All EEGs have been preprocessed and saved.')


def load_all_eegs(data_path):
    """
    加载所有 EEG 数据到字典。

    参数：
    - data_path (str): EEG 数据存放的目录路径

    返回：
    - all_eegs (dict): 键为 eeg_id，值为 (T, C) 的 NumPy 数组
    """
    all_eegs = {}
    eeg_files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
    for eeg_file in tqdm(eeg_files, desc="Loading EEGs"):
        eeg_id = os.path.splitext(eeg_file)[0]
        df_eeg = pd.read_parquet(os.path.join(data_path, eeg_file), columns=FEATS)
        data = df_eeg.values.astype('float32')  # (T, C)
        all_eegs[eeg_id] = data
    return all_eegs


if __name__ == "__main__":
    # 定义数据路径
    train_csv_path = '/root/autodl-tmp/time/datasets/hms/train.csv'  # 请根据实际路径修改
    eegs_dir = '/root/autodl-tmp/time/datasets/hms/train_eegs/'  # 请根据实际路径修改

    # 检查 train.csv 是否存在
    if not os.path.isfile(train_csv_path):
        print(f"[ERROR] train.csv not found at {train_csv_path}")
        exit(1)

    # 加载训练集 CSV
    train_df = pd.read_csv(train_csv_path)

    # 检查 EEG 数据目录是否存在
    if not os.path.isdir(eegs_dir):
        print(f"[ERROR] EEGs directory not found at {eegs_dir}")
        exit(1)

    # 加载所有 EEG 数据
    all_eegs = load_all_eegs(eegs_dir)

    # 预处理并保存 EEG 数据
    preprocess_and_save_eegs(all_eegs, train_df, path='/root/autodl-tmp/time/datasets/hmsprocessed/', downsample_factor=10)
