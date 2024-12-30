import sys, os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error

sys.path.append(os.path.dirname(__file__))
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.models.run_model_utils import batch_forwards_TransformerMVTS
from txai.models.encoders.simple import CNN, LSTM
from torch.cuda.amp import autocast, GradScaler

default_scheduler_args = {
    'mode': 'max',
    'factor': 0.1,
    'patience': 10,
    'threshold': 0.00001,
    'threshold_mode': 'rel',
    'cooldown': 0,
    'min_lr': 1e-8,
    'eps': 1e-08,
    'verbose': True
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def one_hot(y_):
    if not (type(y_) is np.ndarray):
        y_ = y_.detach().clone().cpu().numpy()  # Assume it's a tensor
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def train(
        model,
        train_loader,
        val_tuple,
        n_classes,
        num_epochs,
        class_weights=None,
        optimizer=None,
        standardize=False,
        save_path=None,
        validate_by_step=None,
        criterion=None,
        scheduler_args=default_scheduler_args,
        show_sizes=False,
        regression=False,
        use_scheduler=True,
        counterfactual_training=False,
        max_mask_size=None,
        replace_method=None,
        print_freq=10,
        clip_grad=None,
        detect_irreg=False,
):
    scaler = GradScaler()

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    if criterion is None:
        if regression:
            criterion = torch.nn.MSELoss()
        else:
            criterion = Poly1CrossEntropyLoss(
                num_classes=n_classes,
                epsilon=1.0,
                weight=class_weights,
                reduction='mean'
            )

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

    if save_path is None:
        save_path = 'tmp.pt'

    train_loss, val_auc = [], []
    max_val_auc, best_epoch = 0, 0

    for epoch in range(num_epochs):
        # Train:
        model.train()
        epoch_train_loss = 0

        # Wrap the training loader with tqdm for progress tracking
        for batch_idx, (X, times, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)):
            X, times, y = X.to(device), times.to(device), y.to(device)

            # 在数据加载后，绘制第一个训练样本的EEG信号
            # X, t, y = next(iter(train_loader))
            # plt.figure(figsize=(12, 6))
            # for c in range(X.shape[2]):
            #     plt.plot(X[0, :, c].cpu().numpy(), label=f'Channel {c + 1}')
            # plt.title(f'EEG Signal for Class {y[0].item()}')
            # plt.legend()
            # # 保存图像到文件
            # plt.savefig('eeg_signal_sample.png')
            # plt.close()  # 关闭图形以释放内存
            # print("trainX=", X.shape)
            # print("traintimes=", times.shape)
            # print("trainy=", y.shape)
            # 打印模型输出和标签
            # if batch_idx == 0 and epoch == 0:
            #     out = model(X, times, captum_input=True, show_sizes=show_sizes)
            #     print(f"Model output (first 5 samples): {out[:5]}")
            #     print(f"Labels (first 5 samples): {y[:5]}")
            #     print(f"Labels dtype: {y.dtype}, Labels shape: {y.shape}")
            #     unique_labels = torch.unique(y)
            #     print(f"Unique labels in the first batch: {unique_labels}")




            out = model(X, times, captum_input=True, show_sizes=show_sizes)
            # print("out=", out.shape)
            optimizer.zero_grad()
            loss = criterion(out, y)
            scaler.scale(loss).backward()

            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            scaler.step(optimizer)
            scaler.update()

            if counterfactual_training:
                batch_size, T, d = X.shape[0], X.shape[1], X.shape[2]
                x_inds = torch.randperm(batch_size)[:(batch_size // 2)]
                xsamp = X[x_inds, :, :]
                masks = torch.ones_like(xsamp).float().to(xsamp.device)

                mms = max_mask_size if max_mask_size is not None else T * d
                if mms < 1 and mms > 0:
                    mms = X.shape[0] * X.shape[1] * mms
                mask_nums = torch.randint(0, high=int(mms), size=((batch_size // 2),))

                for i in range(masks.shape[0]):
                    cart = torch.cartesian_prod(torch.arange(T), torch.arange(d))[:mask_nums[i]]
                    masks[i, cart[:, 0], cart[:, 1]] = 0
                xmasked = replace_method(xsamp, masks)

                out = model(xmasked, times[x_inds, :], captum_input=True, show_sizes=show_sizes)

                optimizer.zero_grad()
                loss2 = criterion(out, y[x_inds])
                loss2.backward()
                optimizer.step()

                loss += loss2

            train_loss.append(loss.item())
            epoch_train_loss += loss.item()
            # break
        # Validation:
        model.eval()
        with torch.no_grad():
            X, times, y = val_tuple

            X, times, y = X.to(device), times.to(device), y.to(device)
            # print("valX=", X.shape)
            # print("valtimes=", times.shape)
            # print("valy=", y.shape)

            # print(X.shape, times.shape, y.shape)    #torch.Size([200, 100, 4]) torch.Size([200, 100]) torch.Size([100])
            #print("valbystep=", validate_by_step)
            if validate_by_step is not None:
                if isinstance(model, CNN) or isinstance(model, LSTM):
                    pred = torch.cat(
                        [model(xb, tb) for xb, tb in
                         zip(torch.split(X, validate_by_step, dim=1), torch.split(times, validate_by_step, dim=1))],
                        dim=0
                    )
                else:
                    # print("X=", X.shape)
                    # print("times=", times.shape)
                    pred, _ = batch_forwards_TransformerMVTS(model, X, times, batch_size=validate_by_step)
            else:
                # if detect_irreg:
                #     pred = model(X, times, show_sizes = show_sizes, src_mask = src_mask)
                # else:
                pred = model(X, times,captum_input=True, show_sizes=show_sizes)
            # print("pred=", pred.shape)
            # print("y=", y.shape)
            val_loss = criterion(pred, y)

            # Calculate AUROC:
            # auc = roc_auc_score(one_hot(y), pred.detach().cpu().numpy(), average = 'weighted')
            if regression:
                auc = -1.0 * mean_absolute_error(y.cpu().numpy(), pred.cpu().numpy())
            else:
                auc = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro', )

            if use_scheduler:
                scheduler.step(auc)  # Step the scheduler

            val_auc.append(auc)

            if auc > max_val_auc:
                max_val_auc = auc
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)
                # best_sd = model.state_dict()

        if (epoch + 1) % print_freq == 0:  # Print progress:
            # print('y', y)
            # print('pred', pred)
            met = 'MAE' if regression else 'F1'
            print('Epoch {}, Train Loss = {:.4f}, Val {} = {:.4f}'.format(epoch + 1, train_loss[-1], met, auc))

        # model.eval()
        # with torch.no_grad():
        #     X, times, y = val_tuple
        #     X, times, y = X.to(device), times.to(device), y.to(device)
        #     # 在 train 函数中添加调试代码
        #
        #     if validate_by_step is not None:
        #         if isinstance(model, CNN) or isinstance(model, LSTM):
        #             pred = torch.cat(
        #                 [model(xb, tb) for xb, tb in zip(torch.split(X, validate_by_step, dim=1), torch.split(times, validate_by_step, dim=1))],
        #                 dim=0
        #             )
        #         else:
        #             pred, _ = batch_forwards_TransformerMVTS(model, X, times, batch_size=validate_by_step)
        #     else:
        #         pred = model(X, times, show_sizes=show_sizes)
        #
        #     val_loss = criterion(pred, y)
        #
        #     if regression:
        #         auc = -1.0 * mean_absolute_error(y.cpu().numpy(), pred.cpu().numpy())
        #     else:
        #         auc = f1_score(y.cpu().numpy(), pred.argmax(dim=1).detach().cpu().numpy(), average='macro')
        #
        #     if use_scheduler:
        #         scheduler.step(auc)
        #
        #     val_auc.append(auc)
        #
        #     if auc > max_val_auc:
        #         max_val_auc = auc
        #         best_epoch = epoch
        #         torch.save(model.state_dict(), save_path)
        #
        # if (epoch + 1) % print_freq == 0:
        #     met = 'MAE' if regression else 'F1'
        #     print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss = {epoch_train_loss:.4f}, Val {met} = {auc:.4f}")

    # Load best model
    model.load_state_dict(torch.load(save_path))

    if save_path == 'tmp.pt':
        os.remove('tmp.pt')  # Remove temporarily stored file

    print(f'Best AUC achieved at Epoch {best_epoch}, AUC = {max_val_auc:.4f}')

    return model, train_loss, val_auc
