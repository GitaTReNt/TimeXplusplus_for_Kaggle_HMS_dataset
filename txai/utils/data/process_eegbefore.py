import os
import pandas as pd

import torch
from torch.utils.data import Dataset

def process_EEG(base_path='E:/kaggle/hms', save_path='E:/kaggle/processed_eegs'):
    """
    逐个加载并保存处理后的 EEG 数据。

    参数:
        base_path (str): 数据集的根目录。
        save_path (str): 处理后数据的保存路径。

    返回:
        None
    """
    train_csv_path = os.path.join(base_path, 'train.csv')
    eeg_folder = os.path.join(base_path, 'train_eegs')

    # 检查 train.csv 是否存在
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"训练数据文件不存在: {train_csv_path}")

    # 加载 train.csv
    train_metadata = pd.read_csv(train_csv_path)

    # 投票字段作为分类标签
    votes = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    train_metadata['label'] = train_metadata[votes].idxmax(axis=1)
    label_mapping = {vote: idx for idx, vote in enumerate(votes)}
    train_metadata['label'] = train_metadata['label'].map(label_mapping)

    # 筛选存在的文件
    train_metadata = train_metadata[train_metadata['eeg_id'].apply(
        lambda x: os.path.exists(os.path.join(eeg_folder, f"{x}.parquet")))]
    print(f"过滤后剩余数据: {train_metadata.shape}")

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 遍历每条数据并保存
    for _, row in train_metadata.iterrows():
        eeg_id = row['eeg_id']
        eeg_file = os.path.join(eeg_folder, f"{eeg_id}.parquet")
        save_file = os.path.join(save_path, f"{eeg_id}.pt")

        # 如果已处理，跳过
        if os.path.exists(save_file):
            continue

        try:
            # 读取 .parquet 文件
            eeg_signal = pd.read_parquet(eeg_file)

            # 提取时间序列片段
            start_idx = int(row['eeg_label_offset_seconds'] * 200)
            end_idx = start_idx + 50 * 200

            if start_idx >= len(eeg_signal) or end_idx > len(eeg_signal):
                continue

            eeg_sample = eeg_signal.iloc[start_idx:end_idx].values

            if eeg_sample.shape[0] != 10_000:  # 50 * 200
                continue

            # 转换为张量并保存
            eeg_tensor = torch.tensor(eeg_sample, dtype=torch.float32)
            label_tensor = torch.tensor(row['label'], dtype=torch.long)
            torch.save((eeg_tensor, label_tensor), save_file)

        except Exception as e:
            print(f"处理文件 {eeg_file} 时出错: {e}")

    print(f"所有数据已保存到 {save_path}")

if __name__ == "__main__":
    process_EEG(base_path="E:/kaggle/hms", save_path="E:/kaggle/processed_eegs")
