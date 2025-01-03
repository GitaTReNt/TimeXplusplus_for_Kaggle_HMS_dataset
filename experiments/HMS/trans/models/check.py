import torch
from txai.utils.data.generate_spilts import SnippetDataset
import warnings
warnings.filterwarnings("ignore")

split_file = "../../../../datasets/hmstrain/split_hms_1.pt"
data = torch.load(split_file)
print(data.keys())  # 检查文件内容
print(type(data['train_loader']))
print(type(data['val']), type(data['test']))
#train shape:
# print("data['train_loader'][0].shape, data['train_loader'][1].shape, data['train_loader'][2].shape", data['train_loader'][0].shape, data['train_loader'][1].shape, data['train_loader'][2].shape)
print("data.val[0].shape, data['val'][1].shape, data['val'][2].shape",data['val'][0].shape, data['val'][1].shape, data['val'][2].shape)
print("data['test'][0].shape, data['test'][1].shape, data['test'][2].shape", data['test'][0].shape, data['test'][1].shape, data['test'][2].shape)

import torch
from txai.utils.data.generate_spilts import SnippetDataset

# 1. 定义分片文件路径列表（可以手动指定几个分片文件路径用于测试）
snippet_paths = [
    "../../../../datasets/hmsprocessed/snippet_97.pt",  # 替换为实际路径
    "../../../../datasets/hmsprocessed/snippet_99.pt"
]

# 2. 验证分片路径
print("Number of snippets:", len(snippet_paths))
print("Snippet paths:", snippet_paths)

# 3. 加载一个分片文件并检查内容
print("Checking content of first snippet...")
snippet_data = torch.load(snippet_paths[0])  # 加载第一个分片
print("Keys in snippet file:", snippet_data.keys())
print("Shape of X:", snippet_data["X"].shape)
print("Shape of t:", snippet_data["t"].shape)
print("Shape of y:", snippet_data["y"].shape)


from torch.utils.data import DataLoader

# 验证 train_loader 是否正常工作
print("Checking train_loader...")
train_loader = torch.load(split_file)["train_loader"]  # 替换为实际文件路径

# 打印第一个批次
for batch in train_loader:
    X_batch, t_batch, y_batch = batch
    print("Batch X shape:", X_batch.shape)  # (batch_size, 1000, 8)
    print("Batch t shape:", t_batch.shape)  # (batch_size, 1000)
    print("Batch y shape:", y_batch.shape)  # (batch_size,)
    print("t_value0:", {t_batch[0]})
    print("Batch y values:", y_batch)
    print("X_batch[0]:",X_batch[0][1:10],X_batch[0].shape)
    break
