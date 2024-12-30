import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

############################################################
# 假设你库里已有下列接口，无需重写：
#
# from txai.trainers.train_transformer import train
# from txai.models.encoders.transformer_simple import TransformerMVTS
# from txai.utils.predictors import eval_mvts_transformer
# ----------------------------------------------------------
# 但这里为了让脚本“自洽”，示范性写上 import。
# 实际使用时请确保导入路径无误。
############################################################
from txai.trainers.train_transformer import train
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.predictors import eval_mvts_transformer
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
import os
import torch
from torch.utils.data import DataLoader

# 你已有的库
from txai.trainers.train_transformer import train
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.predictors import eval_mvts_transformer

# 从“文件一”或某个公共模块导入 SnippetDataset
from txai.utils.data.datas import SnippetDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data_path = "../../../datasets/hmstrain"  # 这里放了 split_hms_{i}.pt

    n_folds = 5
    for i in range(1, n_folds + 1):
        split_file = os.path.join(data_path, f"split_hms_{i}.pt")
        if not os.path.isfile(split_file):
            print(f"[Warning] {split_file} not found.")
            continue

        dataset_info = torch.load(split_file)  # {"train_paths", "val_X", "val_t", "val_y", "test_X", ...}
        train_paths = dataset_info["train_paths"]
        val_X, val_t, val_y = dataset_info["val_X"], dataset_info["val_t"], dataset_info["val_y"]
        test_X, test_t, test_y = dataset_info["test_X"], dataset_info["test_t"], dataset_info["test_y"]

        # 构建训练集 DataLoader
        train_dataset = SnippetDataset(train_paths)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

        print(f"\n========== Fold {i} ==========")
        print(f"#Train snippet = {len(train_dataset)}, #Val={val_X.shape[0]}, #Test={test_X.shape[0]}")

        # 定义模型 (假设6类)
        model = TransformerMVTS(
            d_inp=val_X.shape[-1],
            max_len=val_X.shape[1],   # val_X: (N_val, T, C) => shape[1] = T
            n_classes=6,
            nlayers=1,
            trans_dim_feedforward=16,
            trans_dropout=0.1,
            d_pe=16,
            norm_embedding=False
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)

        save_path = f"models/hms_transformer_fold={i}.pt"
        model, best_loss, best_auc = train(
            model=model,
            train_loader=train_loader,
            val_tuple=(val_X, val_t, val_y),
            n_classes=6,
            num_epochs=100,
            save_path=save_path,
            optimizer=optimizer,
            show_sizes=False,
            validate_by_step=32,
            use_scheduler=False
        )
        print(f"[Fold {i}] => best_loss={best_loss:.4f}, best_auc={best_auc:.4f}")

        # 测试
        test_f1 = eval_mvts_transformer((test_X, test_t, test_y), model, batch_size=16)
        print(f"[Fold {i}] Test F1 = {test_f1:.4f}")

if __name__ == "__main__":
    main()


# ############################################################
# # 1) 自定义一个容器(可选)用于存储 (X, time, y)
# ############################################################
# class EpiChunk:
#     """
#     仅作演示，用来封装 (X, time, y).
#     X shape: (T, N, C) or (T, N)...
#     time shape: (T, N)
#     y shape: (N,)
#     """
#     def __init__(self, X, time, y):
#         self.X = X
#         self.time = time
#         self.y = y
#
#
# ############################################################
# # 2) 自定义 EpiDataset, 让 DataLoader 可迭代
# ############################################################
# class EpiDataset(Dataset):
#     """
#     把 (X, time, y) 格式包装成可被 DataLoader 取 batch 的 Dataset。
#     默认假设 X.shape=(T, N, C), time.shape=(T, N), y.shape=(N,).
#     每次 __getitem__(i) 会取第 i 个“样本”的列 => X[:, i, :], time[:, i], y[i].
#     """
#
#     def __init__(self, X, time, y):
#         """
#         X: (T, N, C)
#         time: (T, N)
#         y: (N,)
#         """
#         self.X = X
#         self.time = time
#         self.y = y
#
#         # 简单检查一下维度
#         assert self.X.shape[1] == self.time.shape[1] == self.y.shape[0], \
#             f"Inconsistent shapes: X={X.shape}, time={time.shape}, y={y.shape}"
#
#     def __len__(self):
#         # 样本数 = N
#         return self.y.shape[0]
#
#     def __getitem__(self, idx):
#         # 返回第 idx 个样本 => (x_i, t_i, y_i)
#         # x_i: (T, C), t_i: (T,), y_i: scalar
#         x_i = self.X[:, idx, :]
#         t_i = self.time[:, idx]
#         y_i = self.y[idx]
#         return x_i, t_i, y_i
#
#
# ############################################################
# # 3) 自己写 process_Epilepsy(...)
# #    - 读取并返回 (train_chunk, val_chunk, test_chunk)
# ############################################################
# def process_Epilepsy(split_no=1, device=None, base_path=None):
#     """
#     在此函数中，你需要：
#       1) 读取真实 Epilepsy 数据(比如 all_epilepsy.pt 或 .npy/.csv)
#       2) 根据 split_no = i 加载对应的训练/验证/测试索引
#       3) 切分成 train_X, val_X, test_X (形状 (T, N, C)),
#          以及 time, y
#       4) 移动到 device (可选)
#       5) 封装成 EpiChunk 并返回 (train_chunk, val_chunk, test_chunk).
#
#     下方仅示范: "随机生成 X/time/y + 简单七三分"。
#     你要改成真实的加载 & 切分逻辑。
#     """
#
#     # 举例: (T=100, N=500, C=20)
#
#     all_X = torch.load("your_EEG_data.pt")               # (T, N, C)
#     all_time = torch.arange(T).unsqueeze(1).repeat(1, N)  # (T, N)
#     all_y = torch.randint(low=0, high=2, size=(N,))       # 二分类 => 0/1
#
#     # 简单shuffle + 70/15/15切分
#     idxs = list(range(N))
#     random.shuffle(idxs)
#     train_size = int(0.7 * N)
#     val_size = int(0.15 * N)
#     idx_train = idxs[:train_size]
#     idx_val = idxs[train_size: train_size+val_size]
#     idx_test = idxs[train_size+val_size:]
#
#     # 拆分
#     train_X = all_X[:, idx_train, :]
#     train_time = all_time[:, idx_train]
#     train_y = all_y[idx_train]
#
#     val_X = all_X[:, idx_val, :]
#     val_time = all_time[:, idx_val]
#     val_y = all_y[idx_val]
#
#     test_X = all_X[:, idx_test, :]
#     test_time = all_time[:, idx_test]
#     test_y = all_y[idx_test]
#
#     if device is not None:
#         train_X = train_X.to(device)
#         train_time = train_time.to(device)
#         train_y = train_y.to(device)
#
#         val_X = val_X.to(device)
#         val_time = val_time.to(device)
#         val_y = val_y.to(device)
#
#         test_X = test_X.to(device)
#         test_time = test_time.to(device)
#         test_y = test_y.to(device)
#
#     # 封装
#     train_chunk = EpiChunk(train_X, train_time, train_y)
#     val_chunk = EpiChunk(val_X, val_time, val_y)
#     test_chunk = EpiChunk(test_X, test_time, test_y)
#
#     return train_chunk, val_chunk, test_chunk
#
#
# ############################################################
# # 4) 主脚本: 5 折循环 => 调用 train(...) => 测试评估
# ############################################################
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 你可以使用 Poly1CrossEntropyLoss, 也可以不显式声明
#     # clf_criterion = Poly1CrossEntropyLoss(num_classes=2, epsilon=1.0, weight=None, reduction='mean')
#
#     for i in range(1, 6):
#         # 清显存
#         torch.cuda.empty_cache()
#
#         # 1) 根据 split_no=i 加载 train/val/test
#         #    (下面随机示范, 你要改成真正 Epilepsy 数据的切分.)
#         train_chunk, val_chunk, test_chunk = process_Epilepsy(
#             split_no=i,
#             device=device,
#             base_path='/TimeX/datasets/hmstrain'  # 改成你真实路径
#         )
#
#         # 2) 构建训练集 DataLoader
#         train_dataset = EpiDataset(train_chunk.X, train_chunk.time, train_chunk.y)
#         train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#
#         print(f"\n========== Fold {i} ==========")
#         print("Train X shape:", train_chunk.X.shape)
#         print("Train y shape:", train_chunk.y.shape)
#
#         # 3) 验证 & 测试集打包成 (X, t, y)
#         val_tuple = (val_chunk.X, val_chunk.time, val_chunk.y)
#         test_tuple = (test_chunk.X, test_chunk.time, test_chunk.y)
#
#         # 4) 定义 TransformerMVTS
#         #    - d_inp = 通道数 (val_tuple[0].shape[-1])
#         #    - max_len = 时间长度 (val_tuple[0].shape[0])
#         #    - n_classes = 2 或 6, 看你任务
#         model = TransformerMVTS(
#             d_inp=val_tuple[0].shape[-1],
#             max_len=val_tuple[0].shape[0],
#             n_classes=2,  # 如果是 6 类( seizure,lpd,gpd,lrda,grda,other ) => n_classes=6
#             nlayers=1,
#             trans_dim_feedforward=16,
#             trans_dropout=0.1,
#             d_pe=16,
#             norm_embedding=False
#         ).to(device)
#
#         # 5) 优化器
#         optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
#
#         # 6) 调用库里的 train(...)
#         #    它一般返回 (model, best_loss, best_metric)
#         save_path = f"models/transformer_split={i}.pt"
#         model, best_loss, best_metric = train(
#             model=model,
#             train_loader=train_loader,
#             val_tuple=val_tuple,
#             n_classes=2,        # 跟上面要一致
#             num_epochs=300,     # 训练轮数
#             save_path=save_path,
#             optimizer=optimizer,
#             show_sizes=False,
#             validate_by_step=16,
#             use_scheduler=False
#         )
#
#         #print(f"[Fold {i}] => best_loss={best_loss:.4f}, best_metric={best_metric:.4f}")
#
#         # 7) 可选: 再保存一份CPU版本
#         model_sdict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
#         torch.save(model_sdict_cpu, f"models/transformer_split={i}_cpu.pt")
#
#         # 8) 测试集评估
#         #    eval_mvts_transformer(...) 也是库里现成
#         f1 = eval_mvts_transformer(test_tuple, model, batch_size=16)
#         print(f"[Fold {i}] Test F1 = {f1:.4f}")
#
#
# if __name__ == "__main__":
#     main()
