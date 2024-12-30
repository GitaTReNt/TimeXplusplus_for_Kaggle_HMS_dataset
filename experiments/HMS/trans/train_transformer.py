# # import os
# # import torch
# # from torch.utils.data import DataLoader
# # #忽视warning
# # import warnings
# # warnings.filterwarnings("ignore")
# #
# #
# # from txai.trainers.train_transformer import train
# # from txai.models.encoders.transformer_simple import TransformerMVTS
# # from txai.utils.predictors import eval_mvts_transformer
# #
# # # 从“文件一”或某个公共模块导入 SnippetDataset
# # from txai.utils.data.datas import SnippetDataset
# #
# #
# #
# #
# #
# #
# # def main():
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     print(f"Using device: {device}")
# #     data_path = "../../../datasets/hmstrain"  # 这里放了 split_hms_{i}.pt
# #
# #     n_folds = 5
# #     for i in range(1, n_folds + 1):
# #         split_file = os.path.join(data_path, f"split_hms_{i}.pt")
# #         if not os.path.isfile(split_file):
# #             print(f"[Warning] {split_file} not found.")
# #             exit(0)
# #
# #         dataset_info = torch.load(split_file)  # {"train_paths", "val_X", "val_t", "val_y", "test_X", ...}
# #         train_paths = dataset_info["train_paths"]
# #         val_X, val_t, val_y = dataset_info["val_X"], dataset_info["val_t"], dataset_info["val_y"]
# #         test_X, test_t, test_y = dataset_info["test_X"], dataset_info["test_t"], dataset_info["test_y"]
# #
# #         def normalize_time(X, t):
# #             """
# #             规范化时间向量 t，使其范围从0到50秒。
# #
# #             Params:
# #                 X (torch.Tensor): 输入特征，形状为 (T, C)
# #                 t (torch.Tensor): 时间向量，形状为 (T,)
# #
# #             Returns:
# #                 tuple: 规范化后的 (X, t_norm)
# #             """
# #             t_min = t.min()
# #             t_max = t.max()
# #             # 防止除以零
# #             if t_max - t_min == 0:
# #                 t_norm = torch.zeros_like(t)
# #             else:
# #                 t_norm = 50.0 * (t - t_min) / (t_max - t_min)
# #             return X, t_norm
# #
# #         # 构建训练集 DataLoader
# #         train_dataset = SnippetDataset(train_paths, transform=normalize_time)
# #         # 在数据集构建后，检查一个样本
# #         sample_X, sample_t, sample_y = train_dataset[0]
# #         print(f"Sample X shape: {sample_X.shape}")  # 应为 (T, C)
# #         print(f"Sample t shape: {sample_t.shape}")  # 应为 (T,)
# #         print(f"Sample t values: {sample_t[0],sample_t[-1]}")  # 查看前10个时间步的值
# #
# #         train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
# #         print(f"Train DataLoader: {len(train_loader)} batches")
# #         #检查dataloader的ylabel的分布
# #         for j, (X, t, y) in enumerate(train_loader):
# #             print(f"Batch {i} - y: {y.tolist()}")
# #             if j == 3:
# #                 break
# #
# #
# #             #再看看ylabel的分布
# #
# #
# #
# #         print(f"\n========== Fold {i} ==========")
# #         print(f"#Train snippet = {len(train_dataset)}, #Val={val_X.shape[0]}, #Test={test_X.shape[0]}")
# #
# #         from collections import Counter
# #         # print(f"Fold {i} - Train labels distribution: {Counter(train_y.tolist())}")
# #         print(f"Fold {i} - Val labels distribution: {Counter(val_y.tolist())}")
# #         print(f"Fold {i} - Test labels distribution: {Counter(test_y.tolist())}")
# #
# #         def normalize_time_tensor(tensor):
# #             """
# #             规范化张量中的每个样本的时间向量 t 到 0-50 秒范围。
# #
# #             Params:
# #                 tensor (torch.Tensor): 形状 (N, T)
# #
# #             Returns:
# #                 torch.Tensor: 规范化后的时间向量，形状 (N, T)
# #             """
# #             t_min = tensor.min(dim=1, keepdim=True)[0]
# #             t_max = tensor.max(dim=1, keepdim=True)[0]
# #             duration = t_max - t_min
# #             duration[duration == 0] = 1  # 防止除以零
# #             t_norm = 50.0 * (tensor - t_min) / duration
# #             return t_norm
# #
# #         val_t = normalize_time_tensor(val_t)
# #         test_t = normalize_time_tensor(test_t)
# #         #check val_t and test_t
# #         print(f"Val t shape: {val_t.shape}")
# #         print(f"Val t values: {val_t[0]}")
# #         print(f"Val t values: {val_t[-1]}")
# #         print(f"Test t shape: {test_t.shape}")
# #         print(f"Test t values: {test_t[0]}")
# #         print(f"Test t values: {test_t[-1]}")
# #
# #         # 定义模型 (假设6类)
# #         model = TransformerMVTS(
# #             d_inp=val_X.shape[-1],
# #             max_len=val_X.shape[1],   # val_X: (N_val, T, C) => shape[1] = T
# #             n_classes=6,
# #             nlayers=1,
# #             trans_dim_feedforward=16,
# #             trans_dropout=0.1,
# #             d_pe=16,
# #             norm_embedding=False
# #         ).to(device)
# #
# #         optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
# #
# #         save_path = f"models/hms_transformer_fold={i}.pt"
# #         model, best_loss, best_auc = train(
# #             model=model,
# #             train_loader=train_loader,
# #             val_tuple=(val_X, val_t, val_y),
# #             n_classes=6,
# #             num_epochs=1,
# #             save_path=save_path,
# #             optimizer=optimizer,
# #             show_sizes=False,
# #             validate_by_step=16,
# #             use_scheduler=False
# #         )
# #
# #
# #
# #         # 测试
# #         test_f1 = eval_mvts_transformer((test_X, test_t, test_y), model, batch_size=16)
# #         print(f"[Fold {i}] Test F1 = {test_f1:.4f}")
# #
# # if __name__ == "__main__":
# #     main()
#
#
# # trans/train_transformer.py
#
# #
#
#
# # trans/train_transformer.py
#
# import os
# import torch
# from torch.utils.data import DataLoader
# import warnings
# import torch.multiprocessing as mp
#
# warnings.filterwarnings("ignore")
#
# # 你已有的库
# from txai.trainers.train_transformer import train
# from txai.models.encoders.transformer_simple import TransformerMVTS
# from txai.utils.predictors import eval_mvts_transformer
#
# # 从“文件一”或某个公共模块导入 SnippetDataset
# from txai.utils.data.datas import SnippetDataset
#
# from scipy.signal import butter, filtfilt
# import numpy as np
#
# # 定义选定的 EEG 通道，排除 EKG
# FEATS = ['Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']
# FEAT2IDX = {x: y for y, x in zip(FEATS, range(len(FEATS)))}
#
#
# def get_lowpass_coeffs(cutoff_freq=20, sampling_rate=200, order=4):
#     """
#     预计算Butterworth低通滤波器的系数。
#
#     参数：
#     - cutoff_freq (float): 截止频率（Hz）
#     - sampling_rate (float): 采样率（Hz）
#     - order (int): 滤波器阶数
#
#     返回：
#     - b, a: 滤波器系数
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
#     应用预计算的Butterworth低通滤波器到数据。
#
#     参数：
#     - data (np.ndarray): 输入信号，形状为 (T, C)
#     - b, a: 滤波器系数
#
#     返回：
#     - filtered_data (np.ndarray): 滤波后的信号，形状为 (T, C)
#     """
#     filtered_data = filtfilt(b, a, data, axis=0)
#     return filtered_data
#
#
# def preprocess_transform(X, t):
#     """
#     对EEG数据应用低通滤波器并规范化时间向量。
#
#     参数：
#     - X (torch.Tensor): 原始EEG数据，形状为 (T, C)
#     - t (torch.Tensor): 时间向量，形状为 (T,)
#
#     返回：
#     - tuple: (滤波后的X, 规范化后的t)
#     """
#     # 将X转换为NumPy数组
#     X_np = X.cpu().numpy()
#
#     # 应用低通滤波器
#     X_filtered = butter_lowpass_filter(X_np, b, a)
#
#     # 确保数组是连续的，避免负步幅
#     X_filtered = np.ascontiguousarray(X_filtered).copy()
#
#     # 转换回Torch张量
#     X_filtered = torch.tensor(X_filtered, dtype=torch.float32).to(X.device)
#
#     # 规范化时间向量 t 到 0-50 秒范围
#     t_min = t.min()
#     t_max = t.max()
#     if t_max - t_min == 0:
#         t_norm = torch.zeros_like(t)
#     else:
#         t_norm = 50.0 * (t - t_min) / (t_max - t_min)
#
#     return X_filtered, t_norm
#
#
# def normalize_time_tensor(tensor):
#     """
#     规范化张量中的每个样本的时间向量 t 到 0-50 秒范围。
#
#     参数：
#     - tensor (torch.Tensor): 形状 (N, T)
#
#     返回：
#     - torch.Tensor: 规范化后的时间向量，形状 (N, T)
#     """
#     t_min = tensor.min(dim=1, keepdim=True)[0]
#     t_max = tensor.max(dim=1, keepdim=True)[0]
#     duration = t_max - t_min
#     duration[duration == 0] = 1  # 防止除以零
#     t_norm = 50.0 * (tensor - t_min) / duration
#     return t_norm
#
#
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     data_path = "../../../datasets/hmstrain"  # 这里放了 split_hms_{i}.pt
#
#     n_folds = 5
#     for i in range(1, n_folds + 1):
#         split_file = os.path.join(data_path, f"split_hms_{i}.pt")
#         if not os.path.isfile(split_file):
#             print(f"[Warning] {split_file} not found.")
#             exit(0)
#
#         dataset_info = torch.load(split_file)  # {"train_paths", "val_X", "val_t", "val_y", "test_X", ...}
#         train_paths = dataset_info["train_paths"]
#         val_X, val_t, val_y = dataset_info["val_X"], dataset_info["val_t"], dataset_info["val_y"]
#         test_X, test_t, test_y = dataset_info["test_X"], dataset_info["test_t"], dataset_info["test_y"]
#
#         # 构建训练集 DataLoader，应用预处理转换，并选择所需通道
#         train_dataset = SnippetDataset(
#             train_paths,
#             transform=preprocess_transform,
#             selected_channels=FEATS  # 仅选择所需的8个通道
#         )
#
#         # 在数据集构建后，检查一个样本
#         sample_X, sample_t, sample_y = train_dataset[0]
#         print(f"Sample X shape: {sample_X.shape}")  # 应为 (T, C)
#         print(f"Sample t shape: {sample_t.shape}")  # 应为 (T,)
#         print(f"Sample t values: {sample_t[0], sample_t[-1]}")  # 查看前10个时间步的值
#
#         # 使用多个工作线程以提高数据加载性能
#         train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
#         print(f"Train DataLoader: {len(train_loader)} batches")
#
#         # 检查 DataLoader 的标签分布
#         for j, (X, t, y) in enumerate(train_loader):
#             print(f"Batch {j + 1} - y: {y.tolist()}")
#             if j == 3:
#                 break
#
#         print(f"\n========== Fold {i} ==========")
#         print(f"#Train snippet = {len(train_dataset)}, #Val={val_X.shape[0]}, #Test={test_X.shape[0]}")
#
#         from collections import Counter
#         print(f"Fold {i} - Val labels distribution: {Counter(val_y.tolist())}")
#         print(f"Fold {i} - Test labels distribution: {Counter(test_y.tolist())}")
#
#         # 规范化验证集和测试集的时间向量
#         val_t = normalize_time_tensor(val_t)
#         test_t = normalize_time_tensor(test_t)
#         # 检查 val_t 和 test_t
#         print(f"Val t shape: {val_t.shape}")
#         print(f"Val t values: {val_t[0]}")
#         print(f"Val t values: {val_t[-1]}")
#         print(f"Test t shape: {test_t.shape}")
#         print(f"Test t values: {test_t[0]}")
#         print(f"Test t values: {test_t[-1]}")
#
#         # 定义模型 (假设6类)
#         model = TransformerMVTS(
#             d_inp=len(FEATS),  # 确保与 FEATS 数量一致（8）
#             max_len=val_X.shape[1],  # val_X: (N_val, T, C) => shape[1] = T
#             n_classes=6,
#             nlayers=1,
#             trans_dim_feedforward=16,
#             trans_dropout=0.1,
#             d_pe=16,
#             norm_embedding=False
#         ).to(device)
#
#         optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
#
#         save_path = f"models/hms_transformer_fold={i}.pt"
#         model, best_loss, best_auc = train(
#             model=model,
#             train_loader=train_loader,
#             val_tuple=(val_X, val_t, val_y),
#             n_classes=6,
#             num_epochs=1,  # 可以增加 epoch 数量以提升性能
#             save_path=save_path,
#             optimizer=optimizer,
#             show_sizes=False,
#             validate_by_step=16,
#             use_scheduler=False
#         )
#
#         # 测试
#         test_f1 = eval_mvts_transformer((test_X, test_t, test_y), model, batch_size=16)
#         print(f"[Fold {i}] Test F1 = {test_f1:.4f}")
#
# if __name__ == "__main__":
#     #mp.set_start_method('spawn')
#     main()
#
# trans/train_transformer.py

import os
import torch
from torch.utils.data import DataLoader
import warnings
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

# 你已有的库
from txai.trainers.train_transformer import train
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.predictors import eval_mvts_transformer
from txai.utils.data.generate_spilts import SnippetDataset
# 不再需要从 txai.utils.data.datas 导入 SnippetDataset
# 因为 train_loader 已经包含了所需的数据

import numpy as np

# 定义选定的 EEG 通道，排除 EKG
FEATS = ['Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']
FEAT2IDX = {x: y for y, x in zip(FEATS, range(len(FEATS)))}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data_path = "../../../datasets/hmstrain"  # 这里放了 split_hms_{i}.pt

    n_folds = 5
    for i in range(1, n_folds + 1):
        split_file = os.path.join(data_path, f"split_hms_{i}.pt")
        if not os.path.isfile(split_file):
            print(f"[Warning] {split_file} not found.")
            exit(0)

        dataset_info = torch.load(split_file)  # {"train_loader", "val", "test"}
        train_loader = dataset_info["train_loader"]  # PyTorch DataLoader
        val_X, val_t, val_y = dataset_info["val"]  # (N_val, 1000, 8)/(N_val, 1000)/(N_val,)
        test_X, test_t, test_y = dataset_info["test"]  # (N_test, 1000, 8)/(N_test, 1000)/(N_test,)

        # 在数据集构建后，检查一个样本（从 train_loader）
        for sample in train_loader:
            sample_X, sample_t, sample_y = sample
            print(f"Sample X shape: {sample_X.shape}")  # 应为 (batch_size, 1000, 8)
            print(f"Sample t shape: {sample_t.shape}")  # 应为 (batch_size, 1000)
            print(f"Sample y shape: {sample_y.shape}")  # 应为 (batch_size,)
            break  # 仅检查第一个 batch

        # 查看 DataLoader 的标签分布（可选）
        label_counter = {}
        for batch_X, batch_t, batch_y in train_loader:
            for label in batch_y.tolist():
                label_counter[label] = label_counter.get(label, 0) + 1
        print(f"Train labels distribution: {label_counter}")

        print(f"\n========== Fold {i} ==========")
        print(f"#Train batches = {len(train_loader)}, #Val={val_X.shape[0]}, #Test={test_X.shape[0]}")

        from collections import Counter
        print(f"Fold {i} - Val labels distribution: {Counter(val_y.tolist())}")
        print(f"Fold {i} - Test labels distribution: {Counter(test_y.tolist())}")

        # 规范化验证集和测试集的时间向量已经在预处理阶段完成，因此无需再次规范化
        # 如果确认需要，可以检查时间向量是否在预期范围内，但无需修改
        print(f"Val t shape: {val_t.shape}")
        # print(f"Val t values: {val_t[0]}")
        # print(f"Val t values: {val_t[-1]}")
        print(f"Test t shape: {test_t.shape}")
        # print(f"Test t values: {test_t[0]}")
        # print(f"Test t values: {test_t[-1]}")

        # 定义模型 (假设6类)
        model = TransformerMVTS(
            d_inp=len(FEATS),  # 确保与 FEATS 数量一致（8）
            max_len=val_X.shape[1],  # val_X: (N_val, T, C) => shape[1] = T
            n_classes=6,
            nlayers=1,
            trans_dim_feedforward=16,
            trans_dropout=0.1,
            d_pe=16,
            norm_embedding=False
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        save_path = f"models/hms_transformer_fold={i}.pt"
        model, best_loss, best_auc = train(
            model=model,
            train_loader=train_loader,
            val_tuple=(val_X, val_t, val_y),
            n_classes=6,
            num_epochs=10,  # 可以增加 epoch 数量以提升性能
            save_path=save_path,
            optimizer=optimizer,
            show_sizes=False,
            validate_by_step=32,
            use_scheduler=False
        )

        # 测试
        test_f1 = eval_mvts_transformer((test_X, test_t, test_y), model, batch_size=16)
        print(f"[Fold {i}] Test F1 = {test_f1:.4f}")

        # 保存模型及其他信息（可选）
        # torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    main()
