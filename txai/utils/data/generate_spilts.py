import os
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
import logging
import gc
from tqdm import tqdm
import numpy as np
from scipy.signal import butter, lfilter  # 使用 lfilter
import argparse

# 配置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义选定的 EEG 通道，排除 EKG
FEATS = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']
FEAT2IDX = {x:y for y, x in zip(FEATS, range(len(FEATS)))}

def get_lowpass_coeffs(cutoff_freq=20, sampling_rate=200, order=4):
    """
    预计算Butterworth低通滤波器的系数。
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# 预计算滤波器系数
b, a = get_lowpass_coeffs()

def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    """
    应用Butterworth低通滤波器到数据。
    使用 lfilter 代替 filtfilt，适合因果滤波。
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

def preprocess_transform(segment, t_axis, b, a, downsample_factor=10, fixed_length=1000, snippet_id=None):
    """
    对EEG数据应用低通滤波、下采样并规范化时间向量。
    """
    # 计算每个通道的方差
    variances = np.var(segment, axis=0)
    constant_channels = np.where(variances < 1e-6)[0]  # 使用小阈值检测近似常数通道

    if len(constant_channels) > 0:
        logging.warning(f"Snippet {snippet_id}: Constant channels detected: {constant_channels}. Adding small noise.")
        for ch in constant_channels:
            # 增加较大的随机噪声以防止滤波器引入 NaNs
            segment[:, ch] += np.random.normal(0, 1e-3, size=segment.shape[0])

    # 应用低通滤波器
    try:
        filtered_segment = butter_lowpass_filter(segment, cutoff_freq=20, sampling_rate=200, order=4)
    except Exception as e:
        logging.error(f"Snippet {snippet_id}: Error during filtering: {e}")
        raise e

    # 检查滤波后是否引入了 NaNs
    if np.isnan(filtered_segment).any():
        logging.warning(f"Snippet {snippet_id}: NaNs found after filtering. Replacing NaNs with zeros.")
        filtered_segment = np.nan_to_num(filtered_segment, nan=0.0)

    # 下采样
    filtered_segment = filtered_segment[::downsample_factor, :]  # [1000,8]
    t_down = t_axis[::downsample_factor]

    # 规范化时间向量 t 到 0-50 秒范围
    t_min = t_down.min()
    t_max = t_down.max()
    if t_max - t_min == 0:
        t_norm = np.zeros_like(t_down)
    else:
        t_norm = 50.0 * (t_down - t_min) / (t_max - t_min)

    # 最终检查
    if np.isnan(filtered_segment).any():
        logging.error(f"Snippet {snippet_id}: Filtered segment still contains NaNs after replacement!")
        raise AssertionError("Filtered segment contains NaNs!")
    if np.isnan(t_norm).any():
        logging.error(f"Snippet {snippet_id}: Normalized time axis contains NaNs!")
        raise AssertionError("Normalized time axis contains NaNs!")

    return filtered_segment, t_norm

class SnippetDataset(Dataset):
    """
    用于训练集的懒加载，每次在 __getitem__ 时读取对应 .pt 文件，返回 (X, t, y)。
    """
    def __init__(self, snippet_paths):
        self.snippet_paths = snippet_paths

    def __len__(self):
        return len(self.snippet_paths)

    def __getitem__(self, idx):
        snippet_dict = torch.load(self.snippet_paths[idx])
        X = snippet_dict["X"]  # shape (1000, 8)
        t = snippet_dict["t"]  # shape (1000,)
        y = snippet_dict["y"]  # scalar
        return (X, t, y)

def extract_eeg_segment_with_times(
        file_path,
        offset_seconds,
        duration_seconds=50,
        sample_rate=200,
        columns=None
):
    """
    从 Parquet 文件中提取指定偏移点后的 EEG 片段 (duration_seconds秒)，
    并构建对应的时间轴 t (长度 = duration_seconds * sample_rate)。
    """
    start_idx = int(offset_seconds * sample_rate)
    end_idx = start_idx + int(duration_seconds * sample_rate)

    # 读取指定列
    df_eeg = pd.read_parquet(file_path, columns=columns)
    rows = len(df_eeg)

    # 确保索引范围
    start_idx = max(0, start_idx)
    end_idx = min(rows, end_idx)

    # 截取 EEG 片段
    eeg = df_eeg.iloc[start_idx:end_idx].to_numpy(dtype=np.float32)  # shape ~ (10000, #channels)

    # 处理 NaNs
    for j in range(eeg.shape[1]):
        x = eeg[:, j]
        if np.isnan(x).any():
            m = np.nanmean(x)
            if np.isnan(m):
                x[:] = 0  # 如果全是 NaN，设为零
            else:
                x = np.nan_to_num(x, nan=m)
            eeg[:, j] = x

    # 构建时间轴
    length = eeg.shape[0]
    t_axis = np.linspace(offset_seconds, offset_seconds + duration_seconds, num=length, endpoint=False, dtype=np.float32)

    return eeg, t_axis

import matplotlib.pyplot as plt
def generate_5fold_splits_chunked(
        data_path,
        output_path,
        n_splits=5
):
    """
    1. 从 train.csv 读取元数据，每行 -> 提取 50 秒 EEG + 时间轴 t + label(取投票最大) -> 保存到单独 .pt
    2. 使用 StratifiedKFold 做5折，把 snippet 路径 + label 分成 (train_idx, test_idx)，
       再对 test_idx 做一次二分，得到 val_idx + test_idx2。
    3. 对训练集：用 SnippetDataset + DataLoader 实现懒加载并存入 final dict；
       对验证集、测试集：一次性加载 (X, t, y) 并拼为 (N, T, C)/(N, T)/(N, )。
    4. 最后在 output_path 下保存 split_hms_{fold}.pt。
    """
    # 读取 CSV
    train_csv_path = os.path.join(data_path, "train.csv")
    if not os.path.isfile(train_csv_path):
        logging.error(f"train.csv not found at {train_csv_path}")
        return
    train_metadata = pd.read_csv(train_csv_path)
    #检查train.csv里的例子
    print(train_metadata.shape)
    #看看csv里面前10行数据
    print(train_metadata.head(10))
    train_eegs_path = os.path.join(data_path, "train_eegs")

    # 获取实际存在的 eeg_ids
    available_eeg_ids = set(os.listdir(train_eegs_path))
    available_eeg_ids = {eid.split('.parquet')[0] for eid in available_eeg_ids if eid.endswith('.parquet')}
    #输出前十个看一下
    print(list(available_eeg_ids)[:10])
    # 筛选出 train_metadata 的eeg_id列存在于数据文件夹中的 eeg_ids
    train_metadata['eeg_id_str'] = train_metadata['eeg_id'].astype(str)
    train_metadata = train_metadata[train_metadata['eeg_id_str'].isin(available_eeg_ids)]
    #看看train_metadata里的例子
    print(train_metadata.shape)


    if train_metadata.empty:
        logging.error("No matching eeg_ids found between train.csv and train_eegs folder.")
        return
    visualize_down=False

    # 定义 EEG 通道
    columns_to_load = FEATS  # 已定义的8个通道

    # 创建临时目录：存放每个样本 snippet_{i}.pt
    temp_snippets_dir = os.path.join(output_path, "temp_snippets")
    os.makedirs(temp_snippets_dir, exist_ok=True)

    snippet_paths = []  # 记录 snippet 文件路径
    snippet_labels = []  # 记录每个 snippet 的"整条"标签(投票最大)

    # 遍历 train_metadata，每行生成一个 snippet .pt
    for i, row in tqdm(train_metadata.iterrows(), total=len(train_metadata), desc="Extracting snippets"):
        eeg_id = row["eeg_id"]
        offset_sec = row["eeg_label_offset_seconds"]
        file_path = os.path.join(train_eegs_path, f"{eeg_id}.parquet")

        if not os.path.isfile(file_path):
            logging.warning(f"Missing EEG file: {file_path}")
            continue

        # 提取 50 秒 EEG & 时间轴
        segment, t_axis = extract_eeg_segment_with_times(
            file_path,
            offset_seconds=offset_sec,
            duration_seconds=50,
            sample_rate=200,
            columns=columns_to_load
        )

        # 预处理：滤波 + 下采样
        try:
            filtered_segment, t_norm = preprocess_transform(
                segment,
                t_axis,
                b,
                a,
                downsample_factor=10,
                fixed_length=1000,
                snippet_id=i
            )
        except AssertionError as e:
            logging.error(f"Snippet {i}: {e}. Skipping this snippet.")
            continue

        # 计算该样本的 label：从 6列投票中选最大
        vote_cols = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
        votes = row[vote_cols].to_numpy()
        label_idx = int(np.argmax(votes))  # 0~5

        # 可视化第一个 snippet 的 EEG 信号
        if not visualize_down:
            plt.figure(figsize=(24, 10))
            haha=200
            for c in range(filtered_segment.shape[1]):
                plt.plot(filtered_segment[:, c] + c*haha, label=f'Channel {c + 1}')
            plt.title(f'EEG Signal for Class {label_idx}')
            plt.legend()
            plt.savefig('eeg_signal_sample.png')
            plt.close()

            visualize_down = True  # 确保仅绘制一次

        # 把 (X, t, y) 存到 snippet_i.pt
        snippet_dict = {
            "X": torch.tensor(filtered_segment.copy(), dtype=torch.float32),  # shape (1000, 8)
            "t": torch.tensor(t_norm.copy(), dtype=torch.float32),  # shape (1000,)
            "y": torch.tensor(label_idx, dtype=torch.long)  # scalar
        }
        snippet_file = os.path.join(temp_snippets_dir, f"snippet_{i}.pt")
        torch.save(snippet_dict, snippet_file)

        snippet_paths.append(snippet_file)
        snippet_labels.append(label_idx)

        if (i + 1) % 1000 == 0:
            logging.info(f"... extracted {i + 1}/{len(train_metadata)} snippets")

    snippet_paths = np.array(snippet_paths)
    snippet_labels = np.array(snippet_labels, dtype=np.int64)
    logging.info(f"==> Done extracting {len(snippet_paths)} snippet files.\n")

    # 5折交叉验证
    logging.info("==> Starting 5-fold split...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_id = 1

    for train_idx, test_idx in skf.split(snippet_paths, snippet_labels):
        # test_idx -> 拆分为 val_idx + test_idx2
        X_train_paths, X_test_paths = snippet_paths[train_idx], snippet_paths[test_idx]
        y_train, y_test = snippet_labels[train_idx], snippet_labels[test_idx]

        X_val_paths, X_test_paths2, y_val, y_test2 = train_test_split(
            X_test_paths, y_test, test_size=0.5, random_state=fold_id
        )

        logging.info(f"[Fold {fold_id}] #Train={len(X_train_paths)}, #Val={len(X_val_paths)}, #Test={len(X_test_paths2)}")

        # 训练集：使用懒加载(自定义Dataset+DataLoader)
        train_dataset = SnippetDataset(X_train_paths.tolist())
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,  # 根据需要调整
            shuffle=True,
            num_workers=4,  # 根据需要调整
            pin_memory=True  # 如果使用 GPU
        )

        # 验证集：一次性加载并拼成 (val_X, val_t, val_y)
        val_X_list, val_t_list, val_y_list = [], [], []
        for vp in X_val_paths:
            snippet_dict = torch.load(vp, map_location='cpu')
            val_X_list.append(snippet_dict["X"])
            val_t_list.append(snippet_dict["t"])
            val_y_list.append(snippet_dict["y"])

        val_X = torch.stack(val_X_list, dim=0)  # shape (N_val, 1000, 8)
        val_t = torch.stack(val_t_list, dim=0)  # shape (N_val, 1000)
        val_y = torch.stack(val_y_list, dim=0)  # shape (N_val,)

        # 测试集：一次性加载并拼成 (test_X, test_t, test_y)
        test_X_list, test_t_list, test_y_list = [], [], []
        for tp in X_test_paths2:
            snippet_dict = torch.load(tp, map_location='cpu')
            test_X_list.append(snippet_dict["X"])
            test_t_list.append(snippet_dict["t"])
            test_y_list.append(snippet_dict["y"])

        test_X = torch.stack(test_X_list, dim=0)  # shape (N_test, 1000, 8)
        test_t = torch.stack(test_t_list, dim=0)  # shape (N_test, 1000)
        test_y = torch.stack(test_y_list, dim=0)  # shape (N_test,)

        # 封装成字典并保存
        fold_dataset = {
            "train_loader": train_loader,  # PyTorch DataLoader
            "val": (val_X, val_t, val_y),  # (N_val, 1000, 8)/(N_val, 1000)/(N_val,)
            "test": (test_X, test_t, test_y)  # (N_test, 1000, 8)/(N_test, 1000)/(N_test,)
        }

        os.makedirs(output_path, exist_ok=True)
        fold_file = os.path.join(output_path, f"split_hms_{fold_id}.pt")

        torch.save(fold_dataset, fold_file)
        logging.info(f"[Fold {fold_id}] => saved to: {fold_file}\n")

        # 释放内存
        del val_X, val_t, val_y, test_X, test_t, test_y, val_X_list, val_t_list, val_y_list, test_X_list, test_t_list, test_y_list, snippet_dict
        gc.collect()

        fold_id += 1
        logging.info("Next fold")

    logging.info("==> All folds saved. Done!\n")

def main():
    parser = argparse.ArgumentParser(description="Generate 5-fold splits for EEG data.")
    parser.add_argument('--data_path', default="/root/autodl-tmp/time/datasets/hms" ,type=str, required=False,
                        help='Path to the data directory containing train.csv and train_eegs/')
    parser.add_argument('--output_path', default="/root/autodl-tmp/time/datasets/hmstrain" ,type=str, required=False, help='Path to save the split files')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    generate_5fold_splits_chunked(
        data_path=args.data_path,
        output_path=args.output_path,
        n_splits=args.n_splits
    )

if __name__ == "__main__":
    main()
