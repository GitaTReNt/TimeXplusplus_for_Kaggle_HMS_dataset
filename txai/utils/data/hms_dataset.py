from torch.utils.data import Dataset
import torch
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EEGDataset(Dataset):
    def __init__(self, data_path, seq_len=1000, downsample_factor=1):
        """
        初始化数据集，加载所有 .pt 文件路径
        :param data_path: 数据存储路径
        :param seq_len: 每个样本的目标时间序列长度
        :param downsample_factor: 下采样因子
        """
        self.data_files = [
            os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.pt')
        ]
        self.seq_len = seq_len
        self.downsample_factor = downsample_factor

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        加载单个样本并返回 (eeg_data, time_steps, label)
        :return: eeg_data: EEG信号 (seq_len, features)
                 time_steps: 时间步长 (seq_len,)
                 label: 样本的标签
        """
        data_path = self.data_files[idx]
        eeg_data, label = torch.load(data_path)  # 从 .pt 文件中加载数据和标签

        # 下采样 EEG 数据
        eeg_data = eeg_data[::self.downsample_factor]

        # 生成时间步 time_steps
        time_steps = torch.arange(eeg_data.shape[0]) * self.downsample_factor

        # 截断或填充时间序列数据到指定长度 seq_len
        if eeg_data.shape[0] > self.seq_len:
            # 截断
            eeg_data = eeg_data[:self.seq_len]
            time_steps = time_steps[:self.seq_len]
        else:
            # 填充
            padding_len = self.seq_len - eeg_data.shape[0]
            eeg_padding = torch.zeros((padding_len, eeg_data.shape[1]))  # 对特征维度补零
            time_padding = torch.zeros(padding_len)  # 对时间步补零
            eeg_data = torch.cat((eeg_data, eeg_padding), dim=0)
            time_steps = torch.cat((time_steps, time_padding), dim=0)

        return eeg_data, time_steps, label



def load_processed_eegs(save_path='E:/kaggle/processed_eegs'):
    """
    加载保存的 EEG 数据文件。

    参数:
        save_path (str): 保存处理后的 EEG 数据的路径。

    返回:
        List[torch.Tensor]: 所有 EEG 数据张量。
        List[torch.Tensor]: 所有标签张量。
    """
    eeg_data = []
    labels = []

    for file_name in os.listdir(save_path):
        if file_name.endswith('.pt'):
            # 加载 .pt 文件
            file_path = os.path.join(save_path, file_name)
            eeg_tensor, label_tensor = torch.load(file_path)
            eeg_data.append(eeg_tensor)
            labels.append(label_tensor)

    print(f"加载了 {len(eeg_data)} 个样本。")
    return eeg_data, labels

#
# from torch.utils.data import Dataset
# import torch
# import os
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class EEGDataset(Dataset):
#     def __init__(self, data_path):
#         """
#         初始化数据集，加载所有 .pt 文件路径
#         """
#         self.data_files = [
#             os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.pt')
#         ]
#
#     def __len__(self):
#         return len(self.data_files)
#
#     def __getitem__(self, idx):
#         """
#         加载单个样本并返回 (eeg_data, time_steps, label)
#         """
#         data_path = self.data_files[idx]
#         eeg_data, label = torch.load(data_path)  # 从 .pt 文件中加载数据和标签
#
#         # 生成时间步 T，假设时间步的长度与 eeg_data 的第 0 维相同
#         time_steps = torch.arange(eeg_data.shape[0])
#         #print("test:",time_steps)
#         return eeg_data, time_steps, label
#
#
# def load_processed_eegs(save_path='E:/kaggle/processed_eegs'):
#     """
#     加载保存的 EEG 数据文件。
#
#     参数:
#         save_path (str): 保存处理后的 EEG 数据的路径。
#
#     返回:
#         List[torch.Tensor]: 所有 EEG 数据张量。
#         List[torch.Tensor]: 所有标签张量。
#     """
#     eeg_data = []
#     labels = []
#
#     for file_name in os.listdir(save_path):
#         if file_name.endswith('.pt'):
#             # 加载 .pt 文件
#             file_path = os.path.join(save_path, file_name)
#             eeg_tensor, label_tensor = torch.load(file_path)
#             eeg_data.append(eeg_tensor)
#             labels.append(label_tensor)
#
#     print(f"加载了 {len(eeg_data)} 个样本。")
#     return eeg_data, labels


