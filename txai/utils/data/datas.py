# import os
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedKFold, train_test_split
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class SnippetDataset(Dataset):
#     """
#     用于训练集的懒加载，每次在 __getitem__ 时读取对应 .pt 文件，返回 (X, t, y)。
#     """
#     def __init__(self, snippet_paths, transform=None):
#         self.snippet_paths = snippet_paths  # List[str], e.g. ["temp_snippets/snippet_0.pt", ...]
#         self.transform = transform  # transform 参数，用于数据变换
#     def __len__(self):
#         return len(self.snippet_paths)
#
#     def __getitem__(self, idx):
#         snippet_dict = torch.load(self.snippet_paths[idx])
#         X = snippet_dict["X"].to(device)  # (T, C)
#         t = snippet_dict["t"].to(device)  # (T,)
#         y = snippet_dict["y"].to(device)  # scalar
#         if self.transform:
#             X, t = self.transform(X, t)
#         return X, t, y
#
# def generate_5fold_splits_chunked(data_path, output_path, n_splits=5):
#     """
#     仅保存 “train_paths” 而不直接保存 train_loader。
#     验证/测试也只存 (X, t, y) 的张量，避免载入时出现 Dataset 反序列化问题。
#     """
#     train_csv_path = os.path.join(data_path, "train.csv")
#     train_metadata = pd.read_csv(train_csv_path)
#
#     temp_snippets_dir = os.path.join(output_path, "temp_snippets")
#     snippet_paths = []
#     snippet_labels = []
#
#     vote_cols = ["seizure_vote","lpd_vote","gpd_vote","lrda_vote","grda_vote","other_vote"]
#
#     for i, row in train_metadata.iterrows():
#         snippet_file = os.path.join(temp_snippets_dir, f"snippet_{i}.pt")
#         if not os.path.isfile(snippet_file):
#             print(f"[WARNING] Missing snippet file: {snippet_file}")
#             continue
#
#         votes = row[vote_cols].to_numpy()
#         label_idx = int(np.argmax(votes))  # 0~5
#
#         snippet_paths.append(snippet_file)
#         snippet_labels.append(label_idx)
#
#     snippet_paths = np.array(snippet_paths)
#     snippet_labels = np.array(snippet_labels, dtype=np.int64)
#     print(f"==> Found {len(snippet_paths)} snippet files.\n")
#
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     fold_id = 1
#     for train_idx, test_idx in skf.split(snippet_paths, snippet_labels):
#         X_train_paths, X_test_paths = snippet_paths[train_idx], snippet_paths[test_idx]
#         y_train, y_test = snippet_labels[train_idx], snippet_labels[test_idx]
#
#         # 再 split 一半 => val/test
#         X_val_paths, X_test_paths2, y_val, y_test2 = train_test_split(
#             X_test_paths, y_test, test_size=0.5, random_state=fold_id
#         )
#
#         # 验证集 => 一次性加载
#         val_X_list, val_t_list, val_y_list = [], [], []
#         for p in X_val_paths:
#             snippet_dict = torch.load(p)
#             val_X_list.append(snippet_dict["X"])
#             val_t_list.append(snippet_dict["t"])
#             val_y_list.append(snippet_dict["y"])
#         val_X = torch.stack(val_X_list, dim=0)
#         val_t = torch.stack(val_t_list, dim=0)
#         val_y = torch.stack(val_y_list, dim=0)
#
#         # 测试集 => 一次性加载
#         test_X_list, test_t_list, test_y_list = [], [], []
#         for p in X_test_paths2:
#             snippet_dict = torch.load(p)
#             test_X_list.append(snippet_dict["X"])
#             test_t_list.append(snippet_dict["t"])
#             test_y_list.append(snippet_dict["y"])
#         test_X = torch.stack(test_X_list, dim=0)
#         test_t = torch.stack(test_t_list, dim=0)
#         test_y = torch.stack(test_y_list, dim=0)
#
#         dataset_info = {
#             # 这里只保存 train_paths，而非 train_loader
#             "train_paths": X_train_paths.tolist(),
#             # val/test
#             "val_X": val_X, "val_t": val_t, "val_y": val_y,
#             "test_X": test_X, "test_t": test_t, "test_y": test_y
#         }
#         os.makedirs(output_path, exist_ok=True)
#         fold_file = os.path.join(output_path, f"split_hms_{fold_id}.pt")
#         torch.save(dataset_info, fold_file)
#         print(f"[Fold {fold_id}] => #Train={len(X_train_paths)}, #Val={val_X.shape[0]}, #Test={test_X.shape[0]}")
#         print(f"Saved to: {fold_file}\n")
#
#         fold_id += 1
#
#     print("==> All folds saved. Done!\n")
#
# if __name__ == "__main__":
#     data_path = "../../../datasets/hms"
#     output_path = "../../../datasets/hmstrain"
#
#     generate_5fold_splits_chunked(data_path, output_path, n_splits=5)
#
#     # 简单检查
#     split_1_file = os.path.join(output_path, "split_hms_1.pt")
#     data_1 = torch.load(split_1_file)
#     print("split_1_file keys:", data_1.keys())
#     # train_paths, val_X, val_t, val_y, test_X, test_t, test_y
#     print("train_paths sample:", data_1["train_paths"][:5])
#     print("val shape:", data_1["val_X"].shape)
#     print("test shape:", data_1["test_X"].shape)


# datas.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import gc  # 添加垃圾回收模块

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SnippetDataset(Dataset):
    """
    用于训练集的懒加载，每次在 __getitem__ 时读取对应 .pt 文件，返回 (X, t, y)。
    确保仅包含所需的 EEG 通道，并已应用低通滤波器。
    """
    def __init__(self, snippet_paths, transform=None, selected_channels=None):
        self.snippet_paths = snippet_paths  # List[str], e.g. ["temp_snippets/snippet_0.pt", ...]
        self.transform = transform  # transform 参数，用于数据变换
        self.selected_channels = selected_channels  # List[str], 要选择的通道

    def __len__(self):
        return len(self.snippet_paths)

    def __getitem__(self, idx):
        snippet_dict = torch.load(self.snippet_paths[idx])  # 使用 CPU 加载
        X = snippet_dict["X"]  # (T, C)
        t = snippet_dict["t"] # (T,)
        y = snippet_dict["y"]  # scalar

        # 如果指定了要选择的通道，则进行选择
        if self.selected_channels is not None:
            # 假设 `snippet_dict["X"]` 的列顺序与 `columns_to_load` 一致
            columns_to_load = [
                'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
                'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8',
                'T4', 'T6', 'O2', 'EKG'
            ]
            selected_indices = [columns_to_load.index(ch) for ch in self.selected_channels]
            X = X[:, selected_indices]  # 选择所需的通道

        if self.transform:
            X, t = self.transform(X, t)

        return X, t, y

def generate_5fold_splits_chunked(data_path, output_path, n_splits=5, folds_to_generate=None):
    """
    仅保存 “train_paths” 而不直接保存 train_loader。
    验证/测试也只存 (X, t, y) 的张量，避免载入时出现 Dataset 反序列化问题。
    """
    train_csv_path = os.path.join(data_path, "train.csv")
    train_metadata = pd.read_csv(train_csv_path)
    train_eegs_path = os.path.join(data_path, "train_eegs")

    # 指定要加载的所有通道（包含 EKG）
    columns_to_load = [
        'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
        'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8',
        'T4', 'T6', 'O2', 'EKG'
    ]

    # 创建临时目录：存放每个样本 snippet_{i}.pt
    temp_snippets_dir = os.path.join(output_path, "temp_snippets")
    os.makedirs(temp_snippets_dir, exist_ok=True)

    snippet_paths = []  # 记录 snippet 文件路径
    snippet_labels = []  # 记录每个 snippet 的"整条"标签(投票最大)

    vote_cols = ["seizure_vote","lpd_vote","gpd_vote","lrda_vote","grda_vote","other_vote"]

    for i, row in train_metadata.iterrows():
        snippet_file = os.path.join(temp_snippets_dir, f"snippet_{i}.pt")
        if not os.path.isfile(snippet_file):
            print(f"[WARNING] Missing snippet file: {snippet_file}")
            continue

        votes = row[vote_cols].to_numpy()
        label_idx = int(np.argmax(votes))  # 0~5

        snippet_paths.append(snippet_file)
        snippet_labels.append(label_idx)

    snippet_paths = np.array(snippet_paths)
    snippet_labels = np.array(snippet_labels, dtype=np.int64)
    print(f"==> Found {len(snippet_paths)} snippet files.\n")

    print("==> Starting 5-fold split...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_id = 1

    for train_idx, test_idx in skf.split(snippet_paths, snippet_labels):
        if fold_id not in folds_to_generate:
            fold_id += 1
            continue
        X_train_paths, X_test_paths = snippet_paths[train_idx], snippet_paths[test_idx]
        y_train, y_test = snippet_labels[train_idx], snippet_labels[test_idx]

        # 再 split 一半 => val/test
        X_val_paths, X_test_paths2, y_val, y_test2 = train_test_split(
            X_test_paths, y_test, test_size=0.5, random_state=fold_id
        )

        # 验证集 => 一次性加载
        val_X_list, val_t_list, val_y_list = [], [], []
        for p in X_val_paths:
            snippet_dict = torch.load(p, map_location='cpu')  # 使用 CPU 加载
            val_X_list.append(snippet_dict["X"])
            val_t_list.append(snippet_dict["t"])
            val_y_list.append(snippet_dict["y"])
        val_X = torch.stack(val_X_list, dim=0)
        val_t = torch.stack(val_t_list, dim=0)
        val_y = torch.stack(val_y_list, dim=0)

        # 测试集 => 一次性加载
        test_X_list, test_t_list, test_y_list = [], [], []
        for p in X_test_paths2:
            snippet_dict = torch.load(p, map_location='cpu')  # 使用 CPU 加载
            test_X_list.append(snippet_dict["X"])
            test_t_list.append(snippet_dict["t"])
            test_y_list.append(snippet_dict["y"])
        test_X = torch.stack(test_X_list, dim=0)
        test_t = torch.stack(test_t_list, dim=0)
        test_y = torch.stack(test_y_list, dim=0)

        dataset_info = {
            # 这里只保存 train_paths，而非 train_loader
            "train_paths": X_train_paths.tolist(),
            # val/test
            "val_X": val_X, "val_t": val_t, "val_y": val_y,
            "test_X": test_X, "test_t": test_t, "test_y": test_y
        }
        os.makedirs(output_path, exist_ok=True)
        fold_file = os.path.join(output_path, f"split_hms_{fold_id}.pt")
        torch.save(dataset_info, fold_file)
        print(f"[Fold {fold_id}] => #Train={len(X_train_paths)}, #Val={val_X.shape[0]}, #Test={test_X.shape[0]}")
        print(f"Saved to: {fold_file}\n")

        # 释放内存
        del val_X, val_t, val_y, test_X, test_t, test_y, val_X_list, val_t_list, val_y_list, test_X_list, test_t_list, test_y_list, snippet_dict
        gc.collect()

        fold_id += 1
        print("next fold")

    print("==> All folds saved. Done!\n")

if __name__ == "__main__":
    import sys

    data_path = "../../../datasets/hms"
    output_path = "../../../datasets/hmstrain"

    # 获取要生成的折叠编号
    if len(sys.argv) > 1:
        fold_number = int(sys.argv[1])
        generate_5fold_splits_chunked(data_path, output_path, n_splits=5, folds_to_generate=[fold_number])
    else:
        generate_5fold_splits_chunked(data_path, output_path, n_splits=5, folds_to_generate=range(1,6))

    # # 简单检查第1折
    # split_1_file = os.path.join(output_path, "split_hms_1.pt")
    # fold_data = torch.load(split_1_file)
    # print("Loaded fold 1 keys:", fold_data.keys())
    #
    # # 取出 train_paths, val, test
    # print("train_paths sample:", fold_data["train_paths"][:5])
    # print("val shape:", fold_data["val_X"].shape)
    # print("test shape:", fold_data["test_X"].shape)

