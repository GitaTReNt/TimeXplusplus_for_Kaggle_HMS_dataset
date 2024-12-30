import torch

def concat_all_dicts(dlist, org_v = False, ours = False):
    # Marries together all dictionaries
    # Will change based on output from model

    mother_dict = {k:[] for k in dlist[0].keys()}

    is_tensor_list = []

    for d in dlist:
        for k in d.keys():
            if k == 'smooth_src' and org_v:
                mother_dict[k].append(torch.stack(d[k], dim = -1))
            else:   
                mother_dict[k].append(d[k])

    mother_dict['pred'] = torch.cat(mother_dict['pred'], dim = 0).cpu()
    mother_dict['pred_mask'] = torch.cat(mother_dict['pred_mask'], dim = 0).cpu()
    mother_dict['mask_logits'] = torch.cat(mother_dict['mask_logits'], dim = 0).cpu()
    if org_v:
        mother_dict['concept_scores'] = torch.cat(mother_dict['concept_scores'], dim = 0).cpu()
    mother_dict['ste_mask'] = torch.cat(mother_dict['ste_mask'], dim = 0).cpu()
    # [[(), ()], ... 24]
    mother_dict['smooth_src'] = torch.cat(mother_dict['smooth_src'], dim = 1).cpu() # Will be (T, B, d, ne)

    L = len(mother_dict['all_z'])
    if ours:
        mother_dict['all_z'] = (
            torch.cat([mother_dict['all_z'][i][0] for i in range(L)], dim = 0).cpu(), 
            torch.cat([mother_dict['all_z'][i][1] for i in range(L)], dim = 0).cpu()
        )

        mother_dict['z_mask_list'] = torch.cat(mother_dict['z_mask_list'], dim = 0).cpu()

    return mother_dict

def batch_forwards(model, X, times, batch_size = 64, org_v = False, ours=False):
    '''
    Runs the model in batches for large datasets. Used to get lots of embeddings, outputs, etc.
        - Need to use this bc there's a specialized dictionary notation for output of the forward method (see concat_all_dicts)
    '''

    iters = torch.arange(0, X.shape[1], step = batch_size)
    out_list = []

    for i in range(len(iters)):
        if i == (len(iters) - 1):
            batch_X = X[:,iters[i]:,:]
            batch_times = times[:,iters[i]:]
        else:
            batch_X = X[:,iters[i]:iters[i+1],:]
            batch_times = times[:,iters[i]:iters[i+1]]

        with torch.no_grad():
            out = model(batch_X, batch_times, captum_input = False)

        out_list.append(out)

    out_full = concat_all_dicts(out_list, org_v = org_v, ours=ours)

    return out_full


def batch_forwards_TransformerMVTS(model, X, times, batch_size=16):
    """
    将 (X, times) 按第0维 (样本维) 分批, 并在分批预测后拼接结果.

    参数:
    --------
    model: 训练好的 TransformerMVTS 模型 (或类似)
    X: shape (N, T, C) => (样本数, 时间步, 通道)
    times: shape (N, T)  => (样本数, 时间步)
    batch_size: 每批样本数

    返回:
    --------
    outtotal: shape (N, num_classes)
    ztotal:   shape (N, hidden_dim)  (若你模型 out_z 对应embedding)
    """
    device = next(model.parameters()).device  # 拿到model所在device
    # 确保 X, times 在同一device
    X = X.to(device)
    times = times.to(device)

    n = X.shape[0]  # 样本数
    out_list = []
    z_list = []

    # 以 batch_size 为步长, slice 第 0 维
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        # batch_X shape: (B, T, C)
        batch_X = X[start:end]
        batch_times = times[start:end]  # (B, T)

        with torch.no_grad():
            # 前向推断
            # 注意: out shape => (B, num_classes), z => (B, hidden_dim) or something
            out, z, _ = model(batch_X, batch_times,
                              captum_input=True,
                              get_agg_embed=True)

        # print(f"batch_X shape: {batch_X.shape}")
        # print(f"batch_times shape: {batch_times.shape}")
        # print(f"out: {out.shape}")
        # print(f"z: {z.shape}")

        out_list.append(out)
        z_list.append(z)

    # 拼接, dim=0 => (N, num_classes) / (N, hidden_dim)
    outtotal = torch.cat(out_list, dim=0)
    ztotal = torch.cat(z_list, dim=0)

    # print("outtotal:", outtotal.shape)  # => (N, 6) if 6 classes
    # print("ztotal:", ztotal.shape)  # => (N, some_hidden_dim)

    return outtotal, ztotal

# def batch_forwards_TransformerMVTS(model, X, times, batch_size = 16):
#
#     iters = torch.arange(0, X.shape[1], step = batch_size)
#     out_list = []
#     z_list = []
#
#     for i in range(len(iters)):
#         if i == (len(iters) - 1):
#             batch_X = X[:,iters[i]:,:]
#             print("batchxshape:",batch_X.shape)
#             batch_times = times[:,iters[i]:]
#             print("batchtimeshape:",batch_times.shape)
#         else:
#             batch_X = X[:,iters[i]:iters[i+1],:]
#             print("batchxshape222:", batch_X.shape)
#             batch_times = times[:,iters[i]:iters[i+1]]
#             print("batchtimeshape222:", batch_times.shape)
#         with torch.no_grad():
#             out, z, _ = model(batch_X, batch_times, captum_input = False, get_agg_embed = True)
#         print("z:",z.shape)
#         print("out:",out.shape)
#         out_list.append(out)##########
#         z_list.append(z)
#     ztotal = torch.cat(z_list, dim = 0)
#     outtotal = torch.cat(out_list, dim = 0)
#     print("ztotal:",ztotal.shape)
#     print("outtotal:",outtotal.shape)
#     return outtotal, ztotal

