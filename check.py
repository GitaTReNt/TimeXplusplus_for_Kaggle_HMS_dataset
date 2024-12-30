import os
import pandas as pd

# 设置数据路径
data_dir = "E:/kaggle/hms/"

# 查看训练和测试文件
train_csv_path = os.path.join(data_dir, "train.csv")
test_csv_path = os.path.join(data_dir, "test.csv")

# 加载 CSV 文件
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# 打印文件信息
print("Train CSV Columns:", train_df.columns)
print("Train CSV Sample:\n", train_df.head())
print("Test CSV Columns:", test_df.columns)
print("Test CSV Sample:\n", test_df.head())
