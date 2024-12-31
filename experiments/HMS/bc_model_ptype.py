import torch
import argparse, os, time
import numpy as np
from sklearn import metrics
from tint.utils import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
#ignore warning
#bc_model_ptype.py
warnings.filterwarnings("ignore")

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data.generate_spilts import SnippetDataset
from txai.utils.predictors.eval import eval_mv4
from txai.utils.data.datasets import DatasetwInds, DatasetwInds1
from txai.utils.predictors.loss_cl import *
from txai.utils.predictors.select_models import simloss_on_val_wboth

is_timex = True

if is_timex:
    from txai.models.bc_model4 import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv4_consistency import train_mv6_consistency
else:
    from txai.models.bc_model import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv6_consistency import train_mv6_consistency

import warnings

warnings.filterwarnings("ignore", category=UserWarning)




import torch
from torch.utils.data import Subset
import random

import torch


def create_subset(X, t, y, fraction=0.5, seed=42):
    """
    从给定的数据集中随机选择指定比例的数据。

    Args:
        X (Tensor): 特征数据，形状为 (T, N, C)。
        t (Tensor): 时间数据，形状为 (T, N)。
        y (Tensor): 标签数据，形状为 (N,)。
        fraction (float): 要保留的数据比例（0 < fraction <= 1）。
        seed (int): 随机种子以确保可重复性。

    Returns:
        X_subset (Tensor): 特征子集，形状为 (T, subset_N, C)。
        t_subset (Tensor): 时间子集，形状为 (T, subset_N)。
        y_subset (Tensor): 标签子集，形状为 (subset_N,)。
    """
    assert 0 < fraction <= 1, "fraction 必须在 (0, 1] 之间。"
    torch.manual_seed(seed)
    N = X.shape[1]
    subset_size = int(N * fraction)
    indices = torch.randperm(N)[:subset_size]  # 随机选择指定比例的数据

    X_subset = X[:, indices, :]  # (T, subset_N, C)
    t_subset = t[:, indices]  # (T, subset_N)
    y_subset = y[indices]  # (subset_N,)

    return X_subset, t_subset, y_subset


def naming_convention(args):
    if args.eq_ge:
        name = "bc_eqge_split={}.pt"
    elif args.eq_pret:
        name = "bc_eqpret_split={}.pt"
    elif args.ge_rand_init:
        name = "bc_gerand_split={}.pt"
    elif args.no_ste:
        name = "bc_noste_split={}.pt"
    elif args.simclr:
        name = "bc_simclr_split={}.pt"
    elif args.no_la:
        name = "bc_nola_split={}.pt"
    elif args.no_con:
        name = "bc_nocon_split={}.pt"
    elif args.runtime_exp:
        name = None
    else:
        name = 'bc_full_split={}.pt'
    if not is_timex:
        name = 'our_' + name
    if args.lam != 1.0:
        name = name[:-3] + '_lam={}'.format(args.lam) + '.pt'

    return name


def main(args):
    tencoder_path = "/root/autodl-tmp/time/experiments/HMS/trans/models/hms_transformer_fold={}.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf_criterion = Poly1CrossEntropyLoss(
        num_classes=6,
        epsilon=1.0,
        weight=None,
        reduction='mean'
    )

    sim_criterion_label = LabelConsistencyLoss()
    sim_criterion_cons = EmbedConsistencyLoss()

    if args.no_la:
        sim_criterion = sim_criterion_cons
    elif args.no_con:
        sim_criterion = sim_criterion_label
    else:  # Regular
        sim_criterion = [sim_criterion_cons, sim_criterion_label]
        selection_criterion = simloss_on_val_wboth(sim_criterion, lam=1.0)

    targs = transformer_default_args

    for i in range(1, 6):

        datainfo = torch.load('../../datasets/hmstrain/split_hms_{}.pt'.format(i))
        train_loader = datainfo['train_loader']
        val_X, val_t, val_y = datainfo["val"]  # (N_val, 1000, 8)/(N_val, 1000)/(N_val,)
        test_X, test_t, test_y = datainfo["test"]  # (N_test, 1000, 8)/(N_test, 1000)/(N_test,)

        # Extract training data
        # 收集所有训练数据
        all_X = []
        all_t = []
        all_y = []
        #tqdm loading
        for batch in tqdm(train_loader, desc=f"Processing Fold {i} Batches"):
            X, t, y = batch
            all_X.append(X)
            all_t.append(t)
            all_y.append(y)

        # 合并所有批次的数据
        all_X = torch.cat(all_X, dim=0)  # 形状: (N_train, 1000, 8)
        all_t = torch.cat(all_t, dim=0)  # 形状: (N_train, 1000)
        all_y = torch.cat(all_y, dim=0)  # 形状: (N_train,)
        all_ids = torch.arange(all_X.shape[0])
        print("all_ids.shape:",all_ids.shape)
        print("all_ids:",all_ids[0])
        # 转置 X 和 t 以匹配模型预期的形状 (seq_len, batch_size, features)

#, all_ids
        # 创建 TensorDataset 包含 ids
        train_dataset = DatasetwInds1(all_X, all_t, all_y)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        all_X = all_X.transpose(0, 1)  # 形状: (1000, N_train, 8)
        # all_t = all_t.transpose(0, 1)  # 形状: (1000, N_train)
        # all_X = all_X.permute(1, 0, 2)  # [D=8, N_train, T=1000]
        all_t = all_t.permute(1, 0)  # [D=1, N_train, T=1000]，假设 D=1
        trainB = (all_X, all_t, all_y)
        val_X = val_X.transpose(0, 1)
        val_t = val_t.permute(1, 0)
        test_X = test_X.transpose(0, 1)
        test_t = test_t.permute(1, 0)
        print("valshape",val_X.shape)
        print("val_t.shape:",val_t.shape)
        print("val_y.shape:",val_y.shape)
        val_X_half, val_t_half, val_y_half = create_subset(val_X, val_t, val_y,fraction=0.1, seed=42)
        test_X_half, test_t_half, test_y_half = create_subset(test_X, test_t, test_y, fraction=0.1, seed=42)

        val = (val_X_half, val_t_half, val_y_half)
        print("val.shape:",val[0].shape)
        test = (test_X_half, test_t_half, test_y_half)
        #test = (test_X, test_t, test_y)
        print("tb0.shape:",trainB[0].shape)
        mu = trainB[0].mean(dim=1)
        print("mu.shape:",mu.shape)
        std = trainB[0].std(unbiased=True, dim=1)
        print("std.shape:",std.shape)

        abl_params = AblationParameters(
            equal_g_gt=args.eq_ge,
            g_pret_equals_g=args.eq_pret,
            label_based_on_mask=True,
            ptype_assimilation=True,
            side_assimilation=True,
            use_ste=(not args.no_ste),
        )

        loss_weight_dict = {
            'gsat': 1.0,
            'connect': 2.0
        }

        targs['trans_dim_feedforward'] = 16
        targs['trans_dropout'] = 0.1
        targs['norm_embedding'] = False
        print("val[0].shape[-1]:",val[0].shape[-1])# 8
        model = TimeXModel(
            d_inp=val[0].shape[-1],
            max_len=val[0].shape[0],
            n_classes=6,
            n_prototypes=50,
            gsat_r=args.r,
            transformer_args=targs,
            ablation_parameters=abl_params,
            loss_weight_dict=loss_weight_dict,
            masktoken_stats=(mu, std)
        )
        #看看加载模型的路径
        print("loaded from:",tencoder_path.format(i))
        model.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
        model.to(device)

        if is_timex:
            model.init_prototypes(train=trainB)

            if not args.ge_rand_init:
                model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

        for param in model.encoder_main.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)

        model_suffix = naming_convention(args)
        if model_suffix is not None:
            spath = os.path.join('models', model_suffix)
            spath = spath.format(i)
            print('saving at', spath)
        else:
            spath = None

        start_time = time.time()

        best_model = train_mv6_consistency(
            model,
            optimizer=optimizer,
            train_loader=train_loader,
            clf_criterion=clf_criterion,
            sim_criterion=sim_criterion,
            beta_exp=2.0,
            beta_sim=1.0,
            val_tuple=val,
            num_epochs=2,
            save_path=spath,
            train_tuple=trainB,
            early_stopping=True,
            selection_criterion=selection_criterion,
            label_matching=True,
            embedding_matching=True,
            use_scheduler=True
        )

        end_time = time.time()

        print('Time {}'.format(end_time - start_time))
        if args.runtime_exp:
            exit()

        sdict, config = torch.load(spath)

        model.load_state_dict(sdict)

        f1, _ = eval_mv4(test, model, masked=True)
        print('Test F1: {:.4f}'.format(f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ablations = parser.add_mutually_exclusive_group()
    ablations.add_argument('--eq_ge', action='store_true', help='G = G_E')
    ablations.add_argument('--eq_pret', action='store_true', help='G_pret = G')
    ablations.add_argument('--ge_rand_init', action='store_true', help="Randomly initialized G_E, i.e. don't copy")
    ablations.add_argument('--no_ste', action='store_true', help='Does not use STE')
    ablations.add_argument('--simclr', action='store_true', help='Uses SimCLR loss instead of consistency loss')
    ablations.add_argument('--no_la', action='store_true', help='No label alignment - just consistency loss')
    ablations.add_argument('--no_con', action='store_true', help='No consistency loss - just label')
    ablations.add_argument("--runtime_exp", action='store_true')

    parser.add_argument('--r', type=float, default=0.5, help='r for GSAT loss')
    parser.add_argument('--lam', type=float, default=1.0, help='lambda between label alignment and consistency loss')

    args = parser.parse_args()

    main(args)
