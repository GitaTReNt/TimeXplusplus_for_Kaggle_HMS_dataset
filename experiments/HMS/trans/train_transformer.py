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
ALL_FEATS = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5',
            'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4',
            'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']

# 2. 排除EKG通道，保留19个通道
FEATS = [feat for feat in ALL_FEATS if feat != 'EKG']
FEAT2IDX = {x: y for y, x in zip(FEATS, range(len(FEATS)))}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data_path = "../../../datasets/hmstrain2/fold_1"  # 这里放了 split_hms_{i}.pt

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
        # model, best_loss, best_auc = train(
        #     model=model,
        #     train_loader=train_loader,
        #     val_tuple=(val_X, val_t, val_y),
        #     n_classes=6,
        #     num_epochs=500,  # 可以增加 epoch 数量以提升性能
        #     save_path=save_path,
        #     optimizer=optimizer,
        #     show_sizes=False,
        #     validate_by_step=16,
        #     use_scheduler=False
        # )
        model.load_state_dict(torch.load(save_path))
        # 测试
        test_f1 = eval_mvts_transformer((test_X, test_t, test_y), model, batch_size=16)
        print(f"[Fold {i}] Test F1 = {test_f1:.4f}")

        # 保存模型及其他信息（可选）
        # torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    main()
