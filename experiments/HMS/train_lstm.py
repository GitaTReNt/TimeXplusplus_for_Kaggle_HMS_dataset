from txai.trainers.train_transformer import train
from txai.utils.predictors import eval_mvts_transformer
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.models.encoders.simple import LSTM
from txai.utils.data.process_EEG_data import process_EEG_data
import torch

# 设置设备



# 设置设备





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 配置损失函数
clf_criterion = Poly1CrossEntropyLoss(
    num_classes=6,  # 分类数
    epsilon=1.0,
    weight=None,
    reduction='mean',
)

# 数据路径
data_path = "E:/kaggle/processed_eegs"  # 替换为您数据的路径

# 加载数据集
D = process_EEG_data(data_path, device)

# 获取 DataLoader 和验证集
train_loader = D['train_loader']
val, test = D['val'], D['test']
print(f"Test X shape: {test[0].shape}, Test times shape: {test[1].shape}, Test y shape: {test[2].shape}")

# print("Validation tuple content:")
# print("X:", val[0])
# print("times:", val[1])
# print("y:", val[2])

# 初始化模型
model = LSTM(
    d_inp=20,  # 输入特征数量
    n_classes=6,  # 分类数量
)
model.to(device)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 定义保存路径
spath = "models/EEG_LSTM_model.pt"

# 训练模型
model, loss, auc = train(
    model,
    train_loader,
    val_tuple=val,
    n_classes=6,
    num_epochs=100,
    save_path=spath,
    optimizer=optimizer,
    show_sizes=False,
    use_scheduler=False,
    validate_by_step=None,
)

# 评估模型
from txai.utils.predictors import eval_mvts_transformer

f1 = eval_mvts_transformer(test, model)
print(f"Test F1: {f1:.4f}")

# 保存模型
model_sdict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
torch.save(model_sdict_cpu, spath)
