from txai.utils.data.hms_dataset import EEGDataset
import torch
def process_EEG_data(data_path, device):
    # 加载数据集
    dataset = EEGDataset(data_path)

    # 划分训练集和验证集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    # 转换为张量格式
    val_X = torch.stack([data[0] for data in val_dataset]).to(device)  # EEG数据
    # for data in val_dataset:
    #     print(data[1])  # 打印 data[1] 的内容
    #     print(type(data[1]))  # 打印 data[1] 的类型
    val_t = torch.stack([data[1] for data in val_dataset]).to(device)  # 标签
    val_y = torch.stack([data[2] for data in val_dataset]).to(device)  # 标签

    # 同理创建测试集
    test_X = torch.stack([data[0] for data in test_dataset]).to(device)  # EEG数据
    # for data in val_dataset:
    #     print(data[1])  # 打印 data[1] 的内容
    #     print(type(data[1]))  # 打印 data[1] 的类型
    test_t = torch.stack([data[1] for data in test_dataset]).to(device)  # 标签
    test_y = torch.stack([data[2] for data in test_dataset]).to(device)  # 标签
    return {
        "train_loader": train_loader,
        "val": (val_X, val_t, val_y),  # `None` 是 times 的占位符
        "test": (test_X, test_t, test_y),
    }
