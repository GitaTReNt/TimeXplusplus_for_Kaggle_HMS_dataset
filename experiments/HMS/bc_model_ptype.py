#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bc_model_ptype.py for HMS dataset with TimeX++ (TimeXModel).

Usage Example:
  python bc_model_ptype.py --no_la --lam=0.5

This script will:
 - For each fold i in [1..5], load your black-box transformer model (transformer_split={i}.pt)
 - Load your pre-split data (split_hms_{i}.pt) which has train data (paths or X/t/y), val, test
 - Build TimeXModel for explanation, freeze black-box, train via train_mv6_consistency, evaluate
"""

import sys
import os
import argparse
import warnings
import torch
import numpy as np
from sklearn import metrics

# 1) 如果 TimeX++ 源码在相对路径外, 你可以手动加:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

warnings.filterwarnings("ignore", category=UserWarning)

# ============ 2) 导入 TimeX++ / ablation 需要的对象 ============
# “is_timex=True” 则使用 bc_model4 & train_mv4_consistency
is_timex = True
if is_timex:
    from txai.models.bc_model4 import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv4_consistency import train_mv6_consistency
else:
    from txai.models.bc_model import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv6_consistency import train_mv6_consistency

# 各种损失与选择逻辑
from txai.utils.predictors.loss import Poly1CrossEntropyLoss, LabelConsistencyLoss, EmbedConsistencyLoss, SimCLRLoss
from txai.utils.predictors.loss_smoother_stats import *
from txai.utils.predictors.loss_cl import *
from txai.utils.predictors.select_models import (simloss_on_val_cononly,
                                                 simloss_on_val_laonly,
                                                 simloss_on_val_wboth,
                                                 cosine_sim_for_simclr)

# 用于评估
from txai.utils.predictors.eval import eval_mv4

# ============ 3) 脚本本身的一些辅助函数 (命名后缀等) ============
def naming_convention(args):
    """
    根据 ablation 参数自动生成输出文件名后缀, e.g. bc_eqge_split={}.pt
    """
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
    else:
        name = 'bc_full_split={}.pt'

    if not is_timex:
        name = 'our_' + name

    if args.lam != 1.0:
        name = name[:-3] + '_lam={}.pt'.format(args.lam)

    return name

# ============ 4) 训练主逻辑 ============
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4.1 black-box transformer 路径, 假设你已有 “transformer_split={i}.pt”:
    blackbox_path = "./models/transformer_split={}.pt"

    # 4.2 5 折数据存放路径 “split_hms_{i}.pt”
    # 这里放你的 hms 数据 (split_1..5).pt
    data_path = "../../datasets/hmstrain"

    # 4.3 定义分类损失(6类 for HMS: seizure/lpd/gpd/lrda/grda/other)
    clf_criterion = Poly1CrossEntropyLoss(
        num_classes=6,
        epsilon=1.0,
        weight=None,
        reduction='mean'
    )

    # 4.4 构建 consistency / label alignment
    # 根据是否 simclr, no_la, no_con
    sim_criterion_label = LabelConsistencyLoss()
    if args.simclr:
        sim_criterion_cons = SimCLRLoss()
        sc_expand_args = {'simclr_training': True, 'num_negatives_simclr': 32}
    else:
        sim_criterion_cons = EmbedConsistencyLoss()
        sc_expand_args = {'simclr_training': False, 'num_negatives_simclr': 32}

    if args.no_la and not args.no_con:
        sim_criterion = sim_criterion_cons
        selection_criterion = simloss_on_val_cononly(sim_criterion)
        label_matching = False
        embedding_matching = True
    elif args.no_con and not args.no_la:
        sim_criterion = sim_criterion_label
        selection_criterion = simloss_on_val_laonly(sim_criterion)
        label_matching = True
        embedding_matching = False
    elif args.no_con and args.no_la:
        sim_criterion = None
        selection_criterion = None
        label_matching = False
        embedding_matching = False
    else:
        sim_criterion = [sim_criterion_cons, sim_criterion_label]
        if args.simclr:
            selection_criterion = simloss_on_val_wboth([cosine_sim_for_simclr, sim_criterion_label], lam=1.0)
        else:
            selection_criterion = simloss_on_val_wboth(sim_criterion, lam=1.0)
        label_matching = True
        embedding_matching = True

    # 4.5 TimeX++ Transformer config
    targs = transformer_default_args
    # 你可改： dropout/ feedforward
    targs['trans_dim_feedforward'] = 16
    targs['trans_dropout'] = 0.1
    targs['norm_embedding'] = False
    # targs['nlayers'] = 1  # 是否只用1层

    # 4.6 循环 5 折
    for i in range(1, 6):
        # 读取 “split_hms_{i}.pt” => 只存 "train_paths" or "train_loader"?
        # 建议：只存 train_paths, val_X/t/y, test_X/t/y
        fold_file = os.path.join(data_path, f"split_hms_{i}.pt")
        if not os.path.isfile(fold_file):
            print(f"[Warning] {fold_file} not found, skip.")
            continue

        print(f"\n========== Fold {i} ==========")
        # 4.6.1 加载 fold 数据 (train_paths, val_X, val_t, val_y, test_X, test_t, test_y)
        fold_data = torch.load(fold_file)
        # “只保存数据”场景 =>  fold_data["train_paths"] + fold_data["val_X"]...
        train_paths = fold_data["train_paths"]  # snippet_{i}.pt list
        val_X, val_t, val_y = fold_data["val_X"], fold_data["val_t"], fold_data["val_y"]
        test_X, test_t, test_y = fold_data["test_X"], fold_data["test_t"], fold_data["test_y"]

        # 构建 DataLoader => SnippetDataset
        from generate_5fold_splits_chunked import SnippetDataset  # or wherever your snippet dataset is
        train_dataset = SnippetDataset(train_paths)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        print(f"#train snippet = {len(train_dataset)}, #val={val_X.shape[0]}, #test={test_X.shape[0]}")

        # 4.6.2 构建 TimeXModel
        from txai.models.bc_model4 import TimeXModel, AblationParameters
        abl_params = AblationParameters(
            equal_g_gt = args.eq_ge,
            g_pret_equals_g = args.eq_pret,
            label_based_on_mask = True,
            ptype_assimilation = True,
            side_assimilation = True,
            use_ste = (not args.no_ste),
        )
        loss_weight_dict = {
            'gsat': 1.0,
            'connect': 2.0
        }

        # shape check => val_X: (N_val, T, C)
        model = TimeXModel(
            d_inp=val_X.shape[-1],
            max_len=val_X.shape[1],    # if (N_val, T, C)
            n_classes=6,              # HMS => 6
            n_prototypes=50,
            gsat_r=0.5,
            transformer_args=targs,
            ablation_parameters=abl_params,
            loss_weight_dict=loss_weight_dict,
            masktoken_stats=None  # e.g. (mu, std) if you want
        )

        # 4.6.3 加载黑盒 (transformer)
        blackbox_weights = torch.load(blackbox_path.format(i))  # => state_dict
        model.encoder_main.load_state_dict(blackbox_weights)
        model.to(device)

        # init prototypes => optional
        if is_timex:
            # 需构造 “(X, times, y)” => 这里因为 snippet是懒加载, 你可能一次性加载
            # or you skip
            pass

        # freeze blackbox
        for param in model.encoder_main.parameters():
            param.requires_grad = False

        # 4.6.4 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)

        # 4.6.5 训练 => naming
        model_suffix = naming_convention(args)
        if model_suffix is not None:
            spath = os.path.join("models", model_suffix)
            spath = spath.format(i)
            print("[Fold {}] saving at: {}".format(i, spath))
        else:
            spath = None

        # 调用 train_mv6_consistency
        from txai.trainers.train_mv6_consistency import train_mv6_consistency

        best_model = train_mv6_consistency(
            model,
            optimizer=optimizer,
            train_loader=train_loader,
            clf_criterion=clf_criterion,
            sim_criterion=sim_criterion,
            beta_exp=2.0,
            beta_sim=1.0,
            val_tuple=(val_X, val_t, val_y),
            num_epochs=50,
            save_path=spath,
            train_tuple=None,  # if not needed
            early_stopping=True,
            selection_criterion=selection_criterion,
            label_matching=label_matching,
            embedding_matching=embedding_matching,
            use_scheduler=True,
            clip_norm=False,   # if you want gradient clip
            **sc_expand_args
        )

        # 4.6.6 测试
        if spath is not None and os.path.isfile(spath):
            sdict, config = torch.load(spath)
            model.load_state_dict(sdict)

        from txai.utils.predictors.eval import eval_mv4
        f1, _, results_dict = eval_mv4((test_X, test_t, test_y), model, masked=True)
        print("Test F1: {:.4f}".format(f1))

        # 这里若需要 further analysis or saliency metrics, e.g. “generated_exps”:
        if "generated_exps" in results_dict and "gt_exps" in results_dict:
            gen_exps = results_dict["generated_exps"]
            gt_exps  = results_dict["gt_exps"]
            # 你可做 metrics, e.g. AUC

        print("Fold {} done.\n".format(i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ablations = parser.add_mutually_exclusive_group()
    ablations.add_argument('--eq_ge', action='store_true', help='G = G_E')
    ablations.add_argument('--eq_pret', action='store_true', help='G_pret = G')
    ablations.add_argument('--ge_rand_init', action='store_true', help='Random init G_E, no copying from blackbox')
    ablations.add_argument('--no_ste', action='store_true', help='Does not use STE in mask process')
    ablations.add_argument('--simclr', action='store_true', help='Use SimCLR consistency')
    ablations.add_argument('--no_la', action='store_true', help='No label alignment')
    ablations.add_argument('--no_con', action='store_true', help='No consistency loss')
    parser.add_argument('--runtime_exp', action='store_true', help='Exit after training (no eval)')

    parser.add_argument('--r', type=float, default=0.5, help='r for GSAT loss weighting')
    parser.add_argument('--lam', type=float, default=1.0, help='lambda weighting for label & consistency')

    args = parser.parse_args()
    main(args)
