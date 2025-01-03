import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from txai.utils.data.datasets import DatasetwInds1
from txai.utils.data.datasets import DatasetwInds
from txai.utils.data.generate_spilts import SnippetDataset


class HMSDataset(torch.utils.data.Dataset):
    def __init__(self, X, t, y):
        """
        初始化 HMS 数据集。

        Args:
            X (Tensor): 特征数据，形状为 (T, N, C)。
            t (Tensor): 时间数据，形状为 (T, N)。
            y (Tensor): 标签数据，形状为 (N,)。
        """
        self.X = X
        self.t = t
        self.y = y

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        return self.X[:, idx, :], self.t[:, idx], self.y[idx]

def process_HMS(split_no=1, device=torch.device('cpu'), base_path='../../datasets/hmstrain2/'):
    """
    处理 HMS 数据集。

    Args:
        split_no (int): 折叠编号。
        device (torch.device): 设备（CPU 或 GPU）。
        base_path (str): 数据集的基础路径。

    Returns:
        train_loader (DataLoader): 训练数据加载器。
        val (tuple): 验证集数据 (X, t, y)。
        test (tuple): 测试集数据 (X, t, y)。
        gt_exps (Tensor, optional): Ground Truth 解释（如果存在）。
    """
    split_path = Path(base_path) / f'fold_1/split_hms_{split_no}.pt'
    datainfo = torch.load(split_path)

    train_loader = datainfo['train_loader']
    val_X, val_t, val_y = datainfo["val"]  # (N_val, 1000, 8)/(N_val, 1000)/(N_val,)
    test_X, test_t, test_y = datainfo["test"]  # (N_test, 1000, 8)/(N_test, 1000)/(N_test,)

    # 收集所有训练数据
    all_X = []
    all_t = []
    all_y = []
    for batch in tqdm(train_loader, desc=f"Processing Fold {split_no} Batches"):
        X, t, y = batch
        all_X.append(X)
        all_t.append(t)
        all_y.append(y)

    # 合并所有批次的数据
    all_X = torch.cat(all_X, dim=0)  # 形状: (N_train, 1000, 8)
    all_t = torch.cat(all_t, dim=0)  # 形状: (N_train, 1000)
    all_y = torch.cat(all_y, dim=0)  # 形状: (N_train,)

    # 转置 X 和 t 以匹配模型预期的形状 (T, N, C) 和 (T, N)
    all_X = all_X.transpose(0, 1)  # 形状: (1000, N_train, 8)
    all_t = all_t.transpose(0, 1)  # 形状: (1000, N_train)

    # 创建 Dataset 和 DataLoader
    train_dataset = HMSDataset(all_X, all_t, all_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 处理验证集和测试集
    val = (val_X.transpose(0, 1), val_t.transpose(0, 1), val_y)
    test = (test_X.transpose(0, 1), test_t.transpose(0, 1), test_y)

    # 如果有 Ground Truth 解释，可以在这里加载
    # 假设 `gt_exps` 存储在 datainfo 中
    #sgt_exps = datainfo.get("gt_exps", None)

    return train_loader, val, test