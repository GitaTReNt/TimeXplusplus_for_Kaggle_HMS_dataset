import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
import pyarrow.parquet as pq


###########################################
# 1) 自定义函数：提取 EEG 片段 (X) + 时间轴 (t)
###########################################
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

    # 打开 Parquet 并读取
    parquet_file = pq.ParquetFile(file_path)
    num_rows = parquet_file.metadata.num_rows

    # 确保索引范围
    start_idx = max(0, start_idx)
    end_idx = min(num_rows, end_idx)

    # 这里假设只有1个 row group，如果文件本身很大也可能需要进一步分块
    table = parquet_file.read_row_group(0, columns=columns)
    df = table.to_pandas()

    segment = df.iloc[start_idx:end_idx].to_numpy(dtype=np.float32)  # shape ~ (10000, #channels)

    # 构建时间轴：线性从 offset_seconds 到 offset_seconds + duration_seconds
    # 索引数 = segment.shape[0]
    length = segment.shape[0]
    t_axis = np.linspace(offset_seconds, offset_seconds + duration_seconds, num=length, endpoint=False,
                         dtype=np.float32)

    return segment, t_axis


###########################################
# 2) 自定义 Dataset：按需加载每个 snippet 文件
###########################################
class SnippetDataset(Dataset):
    """
    用于训练集的懒加载，每次在 __getitem__ 时读取对应 .pt 文件，返回 (X, t, y)。
    """

    def __init__(self, snippet_paths):
        """
        snippet_paths: 一个字符串列表，每个元素是 snippet_{i}.pt 文件的路径
        """
        self.snippet_paths = snippet_paths

    def __len__(self):
        return len(self.snippet_paths)

    def __getitem__(self, idx):
        # 加载对应 snippet 文件
        snippet_dict = torch.load(self.snippet_paths[idx])
        # snippet_dict 应包含 "X", "t", "y"
        X = snippet_dict["X"]  # shape (T, C)
        t = snippet_dict["t"]  # shape (T,)
        y = snippet_dict["y"]  # int或长张量
        return (X, t, y)


###########################################
# 3) 核心函数：生成 5 折数据，分批存储避免爆内存
###########################################
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
    # 3.1 读取 CSV
    train_csv_path = os.path.join(data_path, "train.csv")
    train_metadata = pd.read_csv(train_csv_path)
    train_eegs_path = os.path.join(data_path, "train_eegs")

    # 指定想加载的通道列
    columns_to_load = [
        'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
        'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8',
        'T4', 'T6', 'O2', 'EKG'
    ]

    # 3.2 创建临时目录：存放每个样本 snippet_{i}.pt
    temp_snippets_dir = os.path.join(output_path, "temp_snippets")
    os.makedirs(temp_snippets_dir, exist_ok=True)

    snippet_paths = []  # 记录 snippet 文件路径
    snippet_labels = []  # 记录每个 snippet 的"整条"标签(投票最大)

    # 3.3 遍历 train_metadata，每行生成一个 snippet .pt
    for i, row in train_metadata.iterrows():
        eeg_id = row["eeg_id"]
        offset_sec = row["eeg_label_offset_seconds"]
        file_path = os.path.join(train_eegs_path, f"{eeg_id}.parquet")

        if not os.path.isfile(file_path):
            print(f"[WARNING] Missing EEG file: {file_path}")
            continue

        # 提取 50 秒 EEG & 时间轴
        segment, t_axis = extract_eeg_segment_with_times(
            file_path,
            offset_seconds=offset_sec,
            duration_seconds=50,
            sample_rate=200,
            columns=columns_to_load
        )

        # 计算该样本的 label：从 6列投票中选最大
        vote_cols = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
        votes = row[vote_cols].to_numpy()
        label_idx = int(np.argmax(votes))  # 0~5

        # 把 (X, t, y) 存到 snippet_i.pt
        snippet_dict = {
            "X": torch.tensor(segment, dtype=torch.float32),  # shape (T, C)
            "t": torch.tensor(t_axis, dtype=torch.float32),  # shape (T,)
            "y": torch.tensor(label_idx, dtype=torch.long)  # scalar
        }
        snippet_file = os.path.join(temp_snippets_dir, f"snippet_{i}.pt")
        torch.save(snippet_dict, snippet_file)

        snippet_paths.append(snippet_file)
        snippet_labels.append(label_idx)

        if (i + 1) % 1000 == 0:
            print(f"... extracted {i + 1}/{len(train_metadata)} snippets")

    snippet_paths = np.array(snippet_paths)
    snippet_labels = np.array(snippet_labels, dtype=np.int64)
    print(f"==> Done extracting {len(snippet_paths)} snippet files.\n")

    # 3.4 5折交叉验证
    print("==> Starting 5-fold split...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_id = 1

    for train_idx, test_idx in skf.split(snippet_paths, snippet_labels):
        # test_idx -> 拆分为 val_idx + test_idx2
        X_train_paths, X_test_paths = snippet_paths[train_idx], snippet_paths[test_idx]
        y_train, y_test = snippet_labels[train_idx], snippet_labels[test_idx]

        X_val_paths, X_test_paths2, y_val, y_test2 = train_test_split(
            X_test_paths, y_test, test_size=0.5, random_state=fold_id
        )

        print(f"[Fold {fold_id}] #Train={len(X_train_paths)}, #Val={len(X_val_paths)}, #Test={len(X_test_paths2)}")

        # ========== 3.4.1 训练集：使用懒加载(自定义Dataset+DataLoader) ========== #
        train_dataset = SnippetDataset(X_train_paths.tolist())
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,  # 你可根据需要改
            shuffle=True,
            num_workers=0  # 可根据需要改
        )

        # ========== 3.4.2 验证集：一次性加载并拼成 (val_X, val_t, val_y) ========== #
        val_X_list, val_t_list, val_y_list = [], [], []
        for vp, label_ in zip(X_val_paths, y_val):
            snippet_dict = torch.load(vp)
            # snippet_dict["y"] 其实也应该 == label_，这里保持一致
            val_X_list.append(snippet_dict["X"])
            val_t_list.append(snippet_dict["t"])
            val_y_list.append(snippet_dict["y"])

        val_X = torch.stack(val_X_list, dim=0)  # shape (N_val, T, C)
        val_t = torch.stack(val_t_list, dim=0)  # shape (N_val, T)
        val_y = torch.stack(val_y_list, dim=0)  # shape (N_val,)
        del val_X_list, val_t_list, val_y_list

        # ========== 3.4.3 测试集：一次性加载并拼成 (test_X, test_t, test_y) ========== #
        test_X_list, test_t_list, test_y_list = [], [], []
        for tp, label_ in zip(X_test_paths2, y_test2):
            snippet_dict = torch.load(tp)
            test_X_list.append(snippet_dict["X"])
            test_t_list.append(snippet_dict["t"])
            test_y_list.append(snippet_dict["y"])

        test_X = torch.stack(test_X_list, dim=0)
        test_t = torch.stack(test_t_list, dim=0)
        test_y = torch.stack(test_y_list, dim=0)
        del test_X_list, test_t_list, test_y_list

        # ========== 3.4.4 封装成字典并保存 ========== #
        fold_dataset = {
            "train_loader": train_loader,  # PyTorch DataLoader
            "val": (val_X, val_t, val_y),  # (N_val, T, C/T, /)
            "test": (test_X, test_t, test_y)  # (N_test, T, C/T, /)
        }

        os.makedirs(output_path, exist_ok=True)
        fold_file = os.path.join(output_path, f"split_hms_{fold_id}.pt")

        torch.save(fold_dataset, fold_file)
        print(f"[Fold {fold_id}] => saved to: {fold_file}\n")

        fold_id += 1

    print("==> All folds saved. Done!\n")


###########################################
# 4) 脚本入口：示例调用
###########################################
if __name__ == "__main__":
    # 你需要根据自己的数据路径调整：
    data_path = "../../../datasets/hms"  # 包含 train.csv, train_eegs/ ...
    output_path = "../../../datasets/hmstrain"

    generate_5fold_splits_chunked(data_path, output_path, n_splits=5)

    # 简单检查第1折
    split_1_file = os.path.join(output_path, "split_hms_1.pt")
    fold_data = torch.load(split_1_file)
    print("Loaded fold 1 keys:", fold_data.keys())

    # 取出 train_loader, val, test
    train_loader = fold_data["train_loader"]
    val_X, val_t, val_y = fold_data["val"]
    test_X, test_t, test_y = fold_data["test"]
    print("val_X shape:", val_X.shape, "| val_t shape:", val_t.shape, "| val_y shape:", val_y.shape)
    print("test_X shape:", test_X.shape, "| test_t shape:", test_t.shape, "| test_y shape:", test_y.shape)
