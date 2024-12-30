import pandas as pd
import os
# 读取 train.csv
train_csv_path = '/root/autodl-tmp/time/datasets/hms/train.csv'  # 请替换为实际路径
train_eegs_path = '/root/autodl-tmp/time/datasets/hms/train_eegs'  # 请替换为实际路径
train_metadata = pd.read_csv(train_csv_path)
available_eeg_ids = set(os.listdir(train_eegs_path))
# 查看前几行
print(train_metadata['eeg_id'].head())
print("eeg_id in train.csv 数据类型：", train_metadata['eeg_id'].dtype)

eegs_dir = '/root/autodl-tmp/time/datasets/hms/train_eegs/'  # 请根据实际路径修改
eeg_files = [f for f in os.listdir(eegs_dir) if f.endswith('.parquet')]
eeg_ids_in_data = set([os.path.splitext(f)[0] for f in eeg_files])

print(f"EEG 数据文件夹中的 eeg_id 数量：{len(eeg_ids_in_data)}")
print(f"示例 EEG 文件名：{list(eeg_ids_in_data)[:10]}")


# import pandas as pd
#
# # 加载 train.csv
# train_csv_path = '/root/autodl-tmp/time/datasets/hms/train.csv'
# train_df = pd.read_csv(train_csv_path)
#
# # 查看 eeg_id 的数据类型
# print("eeg_id 数据类型：", train_df['eeg_id'].dtype)
#
# # 查看前10个 eeg_id
# print("前10个 eeg_id：", train_df['eeg_id'].head(10).tolist())
#
# import os
#
# # EEG 数据文件夹路径
# eegs_dir = '/root/autodl-tmp/time/datasets/hms/train_eegs/'  # 请根据实际路径修改
# eeg_files = [f for f in os.listdir(eegs_dir) if f.endswith('.parquet')]
# eeg_ids_in_data = set([os.path.splitext(f)[0] for f in eeg_files])
#
# print(f"EEG 数据文件夹中的 eeg_id 数量：{len(eeg_ids_in_data)}")
# print(f"示例 EEG 文件名：{list(eeg_ids_in_data)[:10]}")
#
# import pandas as pd
# import os
#
# # 加载 train.csv
# train_csv_path = '/root/autodl-tmp/time/datasets/hms/train.csv'
# train_df = pd.read_csv(train_csv_path)
#
# # 获取 train.csv 中的所有 eeg_id，转换为字符串
# eeg_ids_in_train = set(train_df['eeg_id'].astype(str).tolist())
#
# # EEG 数据文件夹路径
# eegs_dir = '/root/autodl-tmp/time/datasets/hms/train_eegs/'  # 请根据实际路径修改
# eeg_files = [f for f in os.listdir(eegs_dir) if f.endswith('.parquet')]
# eeg_ids_in_data = set([os.path.splitext(f)[0] for f in eeg_files])
#
# # 找出不匹配的 eeg_id
# missing_in_train = eeg_ids_in_train - eeg_ids_in_data
# missing_in_data = eeg_ids_in_data - eeg_ids_in_train
#
# print(f"在 train.csv 中但不在数据目录中的 eeg_id 数量：{len(missing_in_train)}")
# print(f"示例：{list(missing_in_train)[:10]}")
#
# print(f"在数据目录中但不在 train.csv 中的 eeg_id 数量：{len(missing_in_data)}")
# print(f"示例：{list(missing_in_data)[:10]}")
#
# import pandas as pd
#
# # 示例：加载一个 EEG 文件
# eeg_file = '/root/autodl-tmp/time/datasets/hms/train_eegs/3965080689.parquet'  # 替换为实际文件名
# df_eeg = pd.read_parquet(eeg_file)
# print("nihao:",df_eeg.shape)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # # generate_splits.py
# #
# # import os
# # import numpy as np
# # import pandas as pd
# # import torch
# # from sklearn.model_selection import StratifiedKFold, train_test_split
# # import gc  # 添加垃圾回收模块
# # from tqdm import tqdm
# #
# #
# # def generate_5fold_splits_chunked(data_path, output_path, n_splits=5, folds_to_generate=None):
# #     """
# #     生成 5 折分割，并保存每一折的数据。
# #
# #     参数：
# #     - data_path (str): 原始数据路径，包含 'train.csv' 和 'processed_eegs/' 目录
# #     - output_path (str): 分割后数据的保存路径
# #     - n_splits (int): 折数
# #     - folds_to_generate (list or range): 要生成的折编号
# #     """
# #     train_csv_path = os.path.join(data_path, "train.csv")
# #     if not os.path.isfile(train_csv_path):
# #         print(f"[ERROR] train.csv not found at {train_csv_path}")
# #         exit(1)
# #     train_metadata = pd.read_csv(train_csv_path)
# #     processed_eegs_path = os.path.join(data_path, "hmsprocessed")  # 预处理后的数据路径
# #
# #     snippet_paths = []  # 记录 snippet 文件路径
# #     snippet_labels = []  # 记录每个 snippet 的"整条"标签(投票最大)
# #
# #     vote_cols = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
# #
# #     for i, row in tqdm(train_metadata.iterrows(), total=len(train_metadata),
# #                        desc="Collecting snippet paths and labels"):
# #         eeg_id = str(row['eeg_id'])
# #         snippet_file = os.path.join(processed_eegs_path, f"{eeg_id}.pt")
# #         if not os.path.isfile(snippet_file):
# #             print(f"[WARNING] Missing snippet file: {snippet_file}")
# #             continue
# #
# #         votes = row[vote_cols].to_numpy()
# #         label_idx = int(np.argmax(votes))  # 0~5
# #
# #         snippet_paths.append(snippet_file)
# #         snippet_labels.append(label_idx)
# #
# #     snippet_paths = np.array(snippet_paths)
# #     snippet_labels = np.array(snippet_labels, dtype=np.int64)
# #     print(f"==> Found {len(snippet_paths)} snippet files.\n")
# #
# #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
# #     fold_id = 1
# #     for train_idx, test_idx in skf.split(snippet_paths, snippet_labels):
# #         if folds_to_generate and fold_id not in folds_to_generate:
# #             fold_id += 1
# #             continue
# #         X_train_paths, X_test_paths = snippet_paths[train_idx], snippet_paths[test_idx]
# #         y_train, y_test = snippet_labels[train_idx], snippet_labels[test_idx]
# #
# #         # 再 split 一半 => val/test
# #         X_val_paths, X_test_paths2, y_val, y_test2 = train_test_split(
# #             X_test_paths, y_test, test_size=0.5, random_state=fold_id
# #         )
# #
# #         # 验证集 => 一次性加载
# #         val_X_list, val_t_list, val_y_list = [], [], []
# #         for p in X_val_paths:
# #             snippet_dict = torch.load(p, map_location='cpu')  # 使用 CPU 加载
# #             val_X_list.append(snippet_dict["X"])
# #             val_t_list.append(snippet_dict["t"])
# #             val_y_list.append(snippet_dict["y"])
# #         val_X = torch.stack(val_X_list, dim=0)
# #         val_t = torch.stack(val_t_list, dim=0)
# #         val_y = torch.stack(val_y_list, dim=0)
# #
# #         # 测试集 => 一次性加载
# #         test_X_list, test_t_list, test_y_list = [], [], []
# #         for p in X_test_paths2:
# #             snippet_dict = torch.load(p, map_location='cpu')  # 使用 CPU 加载
# #             test_X_list.append(snippet_dict["X"])
# #             test_t_list.append(snippet_dict["t"])
# #             test_y_list.append(snippet_dict["y"])
# #         test_X = torch.stack(test_X_list, dim=0)
# #         test_t = torch.stack(test_t_list, dim=0)
# #         test_y = torch.stack(test_y_list, dim=0)
# #
# #         dataset_info = {
# #             # 这里只保存 train_paths，而非 train_loader
# #             "train_paths": X_train_paths.tolist(),
# #             # val/test
# #             "val_X": val_X, "val_t": val_t, "val_y": val_y,
# #             "test_X": test_X, "test_t": test_t, "test_y": test_y
# #         }
# #         os.makedirs(output_path, exist_ok=True)
# #         fold_file = os.path.join(output_path, f"split_hms_{fold_id}.pt")
# #         torch.save(dataset_info, fold_file)
# #         print(f"[Fold {fold_id}] => #Train={len(X_train_paths)}, #Val={val_X.shape[0]}, #Test={test_X.shape[0]}")
# #         print(f"Saved to: {fold_file}\n")
# #
# #         # 释放内存
# #         del val_X, val_t, val_y, test_X, test_t, test_y, val_X_list, val_t_list, val_y_list, test_X_list, test_t_list, test_y_list, snippet_dict
# #         gc.collect()
# #
# #         fold_id += 1
# #         print("next fold")
# #
# #     print("==> All folds saved. Done!\n")
# #
# #
# # if __name__ == "__main__":
# #     import sys
# #
# #     data_path = "/root/autodl-tmp/time/datasets/hms"  # 原始数据路径，包含 'train.csv' 和 'processed_eegs/'
# #     output_path = "/root/autodl-tmp/time/datasets/hmstrain"  # 分割后数据的保存路径
# #
# #     # 获取要生成的折叠编号
# #     if len(sys.argv) > 1:
# #         fold_number = int(sys.argv[1])
# #         generate_5fold_splits_chunked(data_path, output_path, n_splits=5, folds_to_generate=[fold_number])
# #     else:
# #         generate_5fold_splits_chunked(data_path, output_path, n_splits=5, folds_to_generate=range(1, 6))
# #
# #     # 简单检查第1折
# #     split_1_file = os.path.join(output_path, "split_hms_1.pt")
# #     if os.path.isfile(split_1_file):
# #         fold_data = torch.load(split_1_file)
# #         print("Loaded fold 1 keys:", fold_data.keys())
# #
# #         # 取出 train_paths, val, test
# #         print("train_paths sample:", fold_data["train_paths"][:5])
# #         print("val shape:", fold_data["val_X"].shape)
# #         print("test shape:", fold_data["test_X"].shape)
# #     else:
# #         print(f"[WARNING] Fold 1 file not found at {split_1_file}")
#
#
# # generate_splits.py
#
# # import os
# # import numpy as np
# # import pandas as pd
# # import torch
# # from sklearn.model_selection import StratifiedKFold, train_test_split
# # import gc  # 添加垃圾回收模块
# # from tqdm import tqdm
# #
# #
# # def generate_5fold_splits(data_path, output_path, n_splits=5, folds_to_generate=None):
# #     """
# #     生成 5 折分割，并保存每一折的数据。
# #
# #     参数：
# #     - data_path (str): 原始数据路径，包含 'train.csv' 和 'processed_eegs/' 目录
# #     - output_path (str): 分割后数据的保存路径
# #     - n_splits (int): 折数
# #     - folds_to_generate (list or range, optional): 要生成的折编号
# #     """
# #     train_csv_path = os.path.join(data_path, "train.csv")
# #     if not os.path.isfile(train_csv_path):
# #         print(f"[ERROR] train.csv not found at {train_csv_path}")
# #         exit(1)
# #     train_metadata = pd.read_csv(train_csv_path)
# #     processed_eegs_path = os.path.join(data_path, "processed_eegs")  # 预处理后的数据路径
# #
# #     snippet_paths = []  # 记录 snippet 文件路径
# #     snippet_labels = []  # 记录每个 snippet 的"整条"标签(投票最大)
# #
# #     vote_cols = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
# #
# #     for i, row in tqdm(train_metadata.iterrows(), total=len(train_metadata),
# #                        desc="Collecting snippet paths and labels"):
# #         eeg_id = str(row['eeg_id'])
# #         snippet_file = os.path.join(processed_eegs_path, f"{eeg_id}.pt")
# #         if not os.path.isfile(snippet_file):
# #             print(f"[WARNING] Missing snippet file: {snippet_file}")
# #             continue
# #
# #         votes = row[vote_cols].to_numpy()
# #         label_idx = int(np.argmax(votes))  # 0~5
# #
# #         snippet_paths.append(snippet_file)
# #         snippet_labels.append(label_idx)
# #
# #     snippet_paths = np.array(snippet_paths)
# #     snippet_labels = np.array(snippet_labels, dtype=np.int64)
# #     print(f"==> Found {len(snippet_paths)} snippet files.\n")
# #
# #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
# #     fold_id = 1
# #     for train_idx, test_idx in skf.split(snippet_paths, snippet_labels):
# #         if folds_to_generate and fold_id not in folds_to_generate:
# #             fold_id += 1
# #             continue
# #         X_train_paths, X_test_paths = snippet_paths[train_idx], snippet_paths[test_idx]
# #         y_train, y_test = snippet_labels[train_idx], snippet_labels[test_idx]
# #
# #         # 再 split 一半 => val/test
# #         X_val_paths, X_test_paths2, y_val, y_test2 = train_test_split(
# #             X_test_paths, y_test, test_size=0.5, random_state=fold_id
# #         )
# #
# #         # 验证集 => 一次性加载
# #         val_X_list, val_t_list, val_y_list = [], [], []
# #         for p in X_val_paths:
# #             snippet_dict = torch.load(p, map_location='cpu')  # 使用 CPU 加载
# #             val_X_list.append(snippet_dict["X"])
# #             val_t_list.append(snippet_dict["t"])
# #             val_y_list.append(snippet_dict["y"])
# #         val_X = torch.stack(val_X_list, dim=0)
# #         val_t = torch.stack(val_t_list, dim=0)
# #         val_y = torch.stack(val_y_list, dim=0)
# #
# #         # 测试集 => 一次性加载
# #         test_X_list, test_t_list, test_y_list = [], [], []
# #         for p in X_test_paths2:
# #             snippet_dict = torch.load(p, map_location='cpu')  # 使用 CPU 加载
# #             test_X_list.append(snippet_dict["X"])
# #             test_t_list.append(snippet_dict["t"])
# #             test_y_list.append(snippet_dict["y"])
# #         test_X = torch.stack(test_X_list, dim=0)
# #         test_t = torch.stack(test_t_list, dim=0)
# #         test_y = torch.stack(test_y_list, dim=0)
# #
# #         dataset_info = {
# #             # 这里只保存 train_paths，而非 train_loader
# #             "train_paths": X_train_paths.tolist(),
# #             # val/test
# #             "val_X": val_X, "val_t": val_t, "val_y": val_y,
# #             "test_X": test_X, "test_t": test_t, "test_y": test_y
# #         }
# #         os.makedirs(output_path, exist_ok=True)
# #         fold_file = os.path.join(output_path, f"split_hms_{fold_id}.pt")
# #         torch.save(dataset_info, fold_file)
# #         print(f"[Fold {fold_id}] => #Train={len(X_train_paths)}, #Val={val_X.shape[0]}, #Test={test_X.shape[0]}")
# #         print(f"Saved to: {fold_file}\n")
# #
# #         # 释放内存
# #         del val_X, val_t, val_y, test_X, test_t, test_y, val_X_list, val_t_list, val_y_list, test_X_list, test_t_list, test_y_list, snippet_dict
# #         gc.collect()
# #
# #         fold_id += 1
# #         print("next fold")
# #
# #     print("==> All folds saved. Done!\n")
# #
# #
# # if __name__ == "__main__":
# #     import sys
# #
# #     data_path = "/root/autodl-tmp/time/datasets/hms"  # 原始数据路径，包含 'train.csv' 和 'processed_eegs/'
# #     output_path = "/root/autodl-tmp/time/datasets/hmstrain"  # 分割后数据的保存路径
# #
# #     # 获取要生成的折叠编号
# #     if len(sys.argv) > 1:
# #         try:
# #             fold_number = int(sys.argv[1])
# #             generate_5fold_splits(data_path, output_path, n_splits=5, folds_to_generate=[fold_number])
# #         except ValueError:
# #             print(f"[ERROR] Invalid fold number provided: {sys.argv[1]}")
# #             exit(1)
# #     else:
# #         generate_5fold_splits(data_path, output_path, n_splits=5, folds_to_generate=range(1, 6))
# #
# #     # 简单检查第1折
# #     split_1_file = os.path.join(output_path, "split_hms_1.pt")
# #     if os.path.isfile(split_1_file):
# #         fold_data = torch.load(split_1_file)
# #         print("Loaded fold 1 keys:", fold_data.keys())
# #
# #         # 取出 train_paths, val, test
# #         print("train_paths sample:", fold_data["train_paths"][:5])
# #         print("val shape:", fold_data["val_X"].shape)
# #         print("test shape:", fold_data["test_X"].shape)
# #     else:
# #         print(f"[WARNING] Fold 1 file not found at {split_1_file}")
# #
# #
#
#
# # generate_splits.py
# import os
# import torch
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from torch.utils.data import DataLoader, Dataset
# import logging
# import gc
# from tqdm import tqdm
# import numpy as np
# from scipy.signal import butter, filtfilt
# import argparse
#
# # 配置 logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
# # 定义选定的 EEG 通道，排除 EKG
# FEATS = ['Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']
# FEAT2IDX = {x: y for y, x in zip(FEATS, range(len(FEATS)))}
#
#
# def get_lowpass_coeffs(cutoff_freq=20, sampling_rate=200, order=4):
#     """
#     预计算Butterworth低通滤波器的系数。
#     """
#     nyquist = 0.5 * sampling_rate
#     normal_cutoff = cutoff_freq / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return b, a
#
#
# # 预计算滤波器系数
# b, a = get_lowpass_coeffs()
#
#
# def butter_lowpass_filter(data, b, a):
#     """
#     应用Butterworth低通滤波器到数据。
#     """
#     filtered_data = filtfilt(b, a, data, axis=0)
#     return filtered_data
#
#
# def preprocess_transform(segment, t_axis, b, a, downsample_factor=10, fixed_length=1000):
#     """
#     对EEG数据应用低通滤波、下采样并规范化时间向量。
#     """
#     # 应用低通滤波器
#     filtered_segment = butter_lowpass_filter(segment, b, a)
#
#     # 下采样
#     filtered_segment = filtered_segment[::downsample_factor, :]  # [1000,8]
#     t_down = t_axis[::downsample_factor]
#
#     # 规范化时间向量 t 到 0-50 秒范围
#     t_min = t_down.min()
#     t_max = t_down.max()
#     if t_max - t_min == 0:
#         t_norm = np.zeros_like(t_down)
#     else:
#         t_norm = 50.0 * (t_down - t_min) / (t_max - t_min)
#
#     # 确保数据无异常
#     assert not np.isnan(filtered_segment).any(), "Filtered segment contains NaNs!"
#     assert not np.isnan(t_norm).any(), "Normalized time axis contains NaNs!"
#
#     return filtered_segment, t_norm
#
#
# class SnippetDataset(Dataset):
#     """
#     用于训练集的懒加载，每次在 __getitem__ 时读取对应 .pt 文件，返回 (X, t, y)。
#     """
#
#     def __init__(self, snippet_paths):
#         self.snippet_paths = snippet_paths
#
#     def __len__(self):
#         return len(self.snippet_paths)
#
#     def __getitem__(self, idx):
#         snippet_dict = torch.load(self.snippet_paths[idx])
#         X = snippet_dict["X"]  # shape (T, C)
#         t = snippet_dict["t"]  # shape (T,)
#         y = snippet_dict["y"]  # int或长张量
#         return (X, t, y)
#
#
# def extract_eeg_segment_with_times(
#         file_path,
#         offset_seconds,
#         duration_seconds=50,
#         sample_rate=200,
#         columns=None
# ):
#     """
#     从 Parquet 文件中提取指定偏移点后的 EEG 片段 (duration_seconds秒)，
#     并构建对应的时间轴 t (长度 = duration_seconds * sample_rate)。
#     """
#     start_idx = int(offset_seconds * sample_rate)
#     print(f"start_idx: {start_idx}")
#     end_idx = start_idx + int(duration_seconds * sample_rate)
#     print(f"end_idx: {end_idx}")
#
#     # 读取指定列
#     df_eeg = pd.read_parquet(file_path, columns=columns)
#     rows = len(df_eeg)
#     print(f"rows: {rows}")
#     # 确保索引范围
#     # start_idx = max(0, start_idx)
#     # end_idx = min(rows, end_idx)
#
#     # 截取 EEG 片段
#     eeg = df_eeg.iloc[start_idx:end_idx].to_numpy(dtype=np.float32)  # shape ~ (10000, #channels)
#     print(f"eeg.shape: {eeg.shape}")
#     for j in range(eeg.shape[1]):
#         x = eeg[:, j]
#         if np.isnan(x).any():
#             m = np.nanmean(x)
#             if np.isnan(m):
#                 print(f"NaN found in column {j}")
#                 exit(1)
#                 x[:] = 0  # 如果全是 NaN，设为零
#             else:
#                 x = np.nan_to_num(x, nan=m)
#                 print(f"NaN found in column {j}, replaced with mean {m}")
#             eeg[:, j] = x
#
#     # 构建时间轴：线性从 offset_seconds 到 offset_seconds + duration_seconds
#     length = eeg.shape[0]
#     t_axis = np.linspace(offset_seconds, offset_seconds + duration_seconds, num=length, endpoint=False,
#                          dtype=np.float32)
#
#     return eeg, t_axis
#
#
# def generate_5fold_splits_chunked(
#         data_path,
#         output_path,
#         n_splits=5
# ):
#     """
#     1. 从 train.csv 读取元数据，每行 -> 提取 50 秒 EEG + 时间轴 t + label(取投票最大) -> 保存到单独 .pt
#     2. 使用 StratifiedKFold 做5折，把 snippet 路径 + label 分成 (train_idx, test_idx)，
#        再对 test_idx 做一次二分，得到 val_idx + test_idx2。
#     3. 对训练集：用 SnippetDataset + DataLoader 实现懒加载并存入 final dict；
#        对验证集、测试集：一次性加载 (X, t, y) 并拼为 (N, T, C)/(N, T)/(N, )。
#     4. 最后在 output_path 下保存 split_hms_{fold}.pt。
#     """
#     # 读取 CSV
#     train_csv_path = os.path.join(data_path, "train.csv")
#     if not os.path.isfile(train_csv_path):
#         logging.error(f"train.csv not found at {train_csv_path}")
#         return
#     train_metadata = pd.read_csv(train_csv_path)
#     train_eegs_path = os.path.join(data_path, "train_eegs")
#
#     # 定义 EEG 通道
#     columns_to_load = FEATS  # 已定义的8个通道
#
#     # 创建临时目录：存放每个样本 snippet_{i}.pt
#     temp_snippets_dir = "/root/autodl-tmp/time/datasets/hmsprocessed"
#     os.makedirs(temp_snippets_dir, exist_ok=True)
#
#     snippet_paths = []  # 记录 snippet 文件路径
#     snippet_labels = []  # 记录每个 snippet 的"整条"标签(投票最大)
#
#     # 遍历 train_metadata，每行生成一个 snippet .pt
#     for i, row in tqdm(train_metadata.iterrows(), total=len(train_metadata), desc="Extracting snippets"):
#         eeg_id = row["eeg_id"]
#         offset_sec = row["eeg_label_offset_seconds"]
#         file_path = os.path.join(train_eegs_path, f"{eeg_id}.parquet")
#
#         if not os.path.isfile(file_path):
#             logging.warning(f"Missing EEG file: {file_path}")
#             continue
#
#         # 提取 50 秒 EEG & 时间轴
#         segment, t_axis = extract_eeg_segment_with_times(
#             file_path,
#             offset_seconds=offset_sec,
#             duration_seconds=50,
#             sample_rate=200,
#             columns=columns_to_load
#         )
#
#         # 预处理：滤波 + 下采样
#         filtered_segment, t_norm = preprocess_transform(
#             segment,
#             t_axis,
#             b,
#             a,
#             downsample_factor=10,
#             fixed_length=1000
#         )
#
#         # 计算该样本的 label：从 6列投票中选最大
#         vote_cols = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
#         votes = row[vote_cols].to_numpy()
#         label_idx = int(np.argmax(votes))  # 0~5
#         filtered_segment = np.ascontiguousarray(filtered_segment.copy())
#
#         print(f"filtered_segment.shape: {filtered_segment.shape}, strides: {filtered_segment.strides}")
#         print(f"t_norm.shape: {t_norm.shape}, strides: {t_norm.strides}")
#
#         # 把 (X, t, y) 存到 snippet_i.pt
#         snippet_dict = {
#             "X": torch.tensor(np.ascontiguousarray(filtered_segment.copy()), dtype=torch.float32),  # shape (1000, 8)
#             "t": torch.tensor(t_norm, dtype=torch.float32),  # shape (1000,)
#             "y": torch.tensor(label_idx, dtype=torch.long)  # scalar
#         }
#         snippet_file = os.path.join(temp_snippets_dir, f"snippet_{i}.pt")
#         torch.save(snippet_dict, snippet_file)
#
#         snippet_paths.append(snippet_file)
#         snippet_labels.append(label_idx)
#
#         if (i + 1) % 1000 == 0:
#             logging.info(f"... extracted {i + 1}/{len(train_metadata)} snippets")
#
#     snippet_paths = np.array(snippet_paths)
#     snippet_labels = np.array(snippet_labels, dtype=np.int64)
#     logging.info(f"==> Done extracting {len(snippet_paths)} snippet files.\n")
#
#     # 5折交叉验证
#     logging.info("==> Starting 5-fold split...")
#
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     fold_id = 1
#
#     for train_idx, test_idx in skf.split(snippet_paths, snippet_labels):
#         # test_idx -> 拆分为 val_idx + test_idx2
#         X_train_paths, X_test_paths = snippet_paths[train_idx], snippet_paths[test_idx]
#         y_train, y_test = snippet_labels[train_idx], snippet_labels[test_idx]
#
#         X_val_paths, X_test_paths2, y_val, y_test2 = train_test_split(
#             X_test_paths, y_test, test_size=0.5, random_state=fold_id
#         )
#
#         logging.info(
#             f"[Fold {fold_id}] #Train={len(X_train_paths)}, #Val={len(X_val_paths)}, #Test={len(X_test_paths2)}")
#
#         # 训练集：使用懒加载(自定义Dataset+DataLoader)
#         train_dataset = SnippetDataset(X_train_paths.tolist())
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=16,  # 根据需要调整
#             shuffle=True,
#             num_workers=4  # 根据需要调整
#         )
#
#         # 验证集：一次性加载并拼成 (val_X, val_t, val_y)
#         val_X_list, val_t_list, val_y_list = [], [], []
#         for vp in X_val_paths:
#             snippet_dict = torch.load(vp, map_location='cpu')
#             val_X_list.append(snippet_dict["X"])
#             val_t_list.append(snippet_dict["t"])
#             val_y_list.append(snippet_dict["y"])
#
#         val_X = torch.stack(val_X_list, dim=0)  # shape (N_val, 1000, 8)
#         val_t = torch.stack(val_t_list, dim=0)  # shape (N_val, 1000)
#         val_y = torch.stack(val_y_list, dim=0)  # shape (N_val,)
#
#         # 测试集：一次性加载并拼成 (test_X, test_t, test_y)
#         test_X_list, test_t_list, test_y_list = [], [], []
#         for tp in X_test_paths2:
#             snippet_dict = torch.load(tp, map_location='cpu')
#             test_X_list.append(snippet_dict["X"])
#             test_t_list.append(snippet_dict["t"])
#             test_y_list.append(snippet_dict["y"])
#
#         test_X = torch.stack(test_X_list, dim=0)  # shape (N_test, 1000, 8)
#         test_t = torch.stack(test_t_list, dim=0)  # shape (N_test, 1000)
#         test_y = torch.stack(test_y_list, dim=0)  # shape (N_test,)
#
#         # 封装成字典并保存
#         fold_dataset = {
#             "train_loader": train_loader,  # PyTorch DataLoader
#             "val": (val_X, val_t, val_y),  # (N_val, 1000, 8)/(N_val, 1000)/(N_val,)
#             "test": (test_X, test_t, test_y)  # (N_test, 1000, 8)/(N_test, 1000)/(N_test,)
#         }
#
#         os.makedirs(output_path, exist_ok=True)
#         fold_file = os.path.join(output_path, f"split_hms_{fold_id}.pt")
#
#         torch.save(fold_dataset, fold_file)
#         logging.info(f"[Fold {fold_id}] => saved to: {fold_file}\n")
#
#         # 释放内存
#         del val_X, val_t, val_y, test_X, test_t, test_y, val_X_list, val_t_list, val_y_list, test_X_list, test_t_list, test_y_list, snippet_dict
#         gc.collect()
#
#         fold_id += 1
#         logging.info("Next fold")
#
#     logging.info("==> All folds saved. Done!\n")
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Generate 5-fold splits for EEG data.")
#     parser.add_argument('--data_path', default="/root/autodl-tmp/time/datasets/hms" ,type=str, required=False,
#                         help='Path to the data directory containing train.csv and train_eegs/')
#     parser.add_argument('--output_path', default="/root/autodl-tmp/time/datasets/hmstrain" ,type=str, required=False, help='Path to save the split files')
#     parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation')
#     args = parser.parse_args()
#
#     generate_5fold_splits_chunked(
#         data_path=args.data_path,
#         output_path=args.output_path,
#         n_splits=args.n_splits
#     )
#
#
# if __name__ == "__main__":
#     main()
