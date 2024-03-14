import random
import numpy as np
import scipy.stats as stats
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context, ops
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore import Tensor


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

def setup_seed(seed):
    ops.random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h

def get_embedding(model, input, device, type=False):
    batch_size = 64
    input = Tensor(input, dtype=ms.float32).to(device)
    if input.shape[0] > batch_size:
        embed = []
        i = 0
        while i <= input.shape[0] - 1:
            embed.append(model(input[i:i + batch_size].to(device), return_feat=True).detach().cpu())
            i += batch_size
        embed = ops.Concat()(embed, 0)
    else:
        embed = model(input.to(device), return_feat=True).detach().cpu()
    assert embed.shape[0] == input.shape[0]
    if type:
        return embed
    return embed.asnumpy()

def Proto(support, support_ys, query, opt=None):
    """Protonet classifier"""
    metric_list = []
    nc = support.shape[-1]
    support = ops.Normalize(p=2, axis=1)(support) if opt == 'l2' else support
    support = mnp.reshape(support, (-1, 1, 5, 8, nc))
    support = mnp.mean(support, axis=3)
    support = support[0][0]
    for idx1, item in enumerate(support):
        for idx2, item2 in enumerate(support):
            if idx1 == idx2:
                metric_list.append(-10000)
                continue
            dist = -((item2 - item) ** 2).sum(-1)
            metric_list.append(dist)
    matrix_array = mnp.asarray(metric_list)
    matrix_array = mnp.reshape(matrix_array, (5, 5))
    sim = mnp.argmax(matrix_array, axis=-1)
    return sim

def init_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 1e-3
    for param_group in optimizer.get_lr():
        param_group.assign(lr)

def adjust_learning_rate(optimizer, lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr *= (0.1 ** (epoch // 30))
    for param_group in optimizer.get_lr():
        param_group.assign(lr)

def cal_entropy(out):
    out = nn.Softmax()(out)
    entropies = []
    for item in out:
        out_log = mnp.log(item)
        entropy = (item * out_log).sum().detach().cpu()
        entropies.append(mnp.neg(entropy) / 5)
    return mnp.array(entropies)

def get_wrong(unlabel_targets, pseudo, index, type):
    gts = unlabel_targets[index]
    num = len(index)
    count = 0
    wrong_idx = []
    wrong_list = []
    for idx, item in enumerate(pseudo):
        gt = gts[idx]
        if type == 'nl':
            if gt == item:
                count += 1
                wrong_idx.append(idx)
                wrong_list.append(item)
        else:
            if gt != item:
                count += 1
                wrong_idx.append(idx)
                wrong_list.append(item)
    wrong_rate = count / (num + 1e-6)
    return wrong_rate, count, num, wrong_idx

def get_metrics(model, inputs, targets):
    out = model(inputs)
    out = nn.Sigmoid()(out)
    preds = ops.Argmax(axis=1)(out).detach().cpu().numpy()
    acc = (preds == targets).mean()
    return preds, acc

def get_preds(out):
    out = nn.Softmax()(out)
    preds = ops.Argmin(axis=1)(out).detach().cpu().numpy()
    confs = ops.Min(axis=1)(out).values.detach().cpu().numpy()
    return preds, confs

def get_preds_second(out):
    out = nn.Softmax()(out)
    preds = ops.Argmin(axis=1)(out).detach().cpu().numpy()
    for idx, item in enumerate(preds):
        out[idx][item] = 1
    preds = ops.Argmin(axis=1)(out).detach().cpu().numpy()
    confs = ops.Min(axis=1)(out).values.detach().cpu().numpy()
    return preds, confs

def get_preds_third(out):
    out = nn.Softmax()(out)
    preds = ops.Argmin(axis=1)(out).detach().cpu().numpy()
    for idx, item in enumerate(preds):
        out[idx][item] = 1
    preds = ops.Argmin(axis=1)(out).detach().cpu().numpy()
    for idx, item in enumerate(preds):
        out[idx][item] = 1
    preds = ops.Argmin(axis=1)(out).detach().cpu().numpy()
    confs = ops.Min(axis=1)(out).values.detach().cpu().numpy()
    return preds, confs

def remove_files(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)

def get_preds_fourth(out):
    out = nn.Softmax()(out)
    preds = ops.Argmin(axis=1)(out).detach().cpu().numpy()
    for idx, item in enumerate(preds):
        out[idx][item] = 1
    preds = ops.Argmin(axis=1)(out).detach().cpu().numpy()
    for idx, item in enumerate(preds):
        out[idx][item] = 1
    preds = ops.Argmin(axis=1)(out).detach().cpu().numpy()
    for idx, item in enumerate(preds):
        out[idx][item] = 1
    preds = ops.Argmin(axis=1)(out).detach().cpu().numpy()
    confs = ops.Min(axis=1)(out).values.detach().cpu().numpy()
    return preds, confs

def get_indexes_on_uncertainty(out, uncertainty_thres, confidence_thres=0.98, reverse=False):
    indexes, _ = get_indexes(out, confidence_thres, reverse)
    unlabel_entropy = cal_entropy(out)

    uncertain_indexes = np.where(unlabel_entropy <= uncertainty_thres)[0]
    final_indexes = [x for x in uncertain_indexes if x in indexes]
    return final_indexes

def get_indexes(out, threshold=0.98, reverse=False):
    preds = ops.Argmax(axis=1)(out).detach().cpu().numpy()
    out = nn.Softmax()(out)
    confs = ops.Max(axis=1)(out).values.detach().cpu().numpy()
    if reverse:
        indexes = np.where(confs <= threshold)[0]
    else:
        indexes = np.where(confs > threshold)[0]
    return indexes.tolist(), Tensor(preds, mstype.float32)

def sample_with_entropy(entropies, thres=0.2):
    sample_list = []
    for idx, entropy in enumerate(entropies):
        if entropy <= thres:
            sample_list.append(idx)
    return sample_list

def get_cl_by_sim(train_embeddings, unlabel_embeddings):
    out = get_proto(train_embeddings, unlabel_embeddings)
    preds = ops.Argmax(axis=1)(out).detach().cpu().numpy()
    for idx, item in enumerate(preds):
        out[idx][item] = 0
    preds = ops.Argmax(axis=1)(out).detach().cpu().numpy()
    confs = ops.Min(axis=1)(out).values.detach().cpu().numpy()
    return preds, confs

def get_proto(train_embedding, unlabel_embedding):
    support_set = train_embedding.view(5, -1, 512)
    centroid = ops.Mean(axis=1)(support_set).unsqueeze(0)
    query_set = unlabel_embedding.unsqueeze(1)
    neg_l2distance = ops.Sum((centroid - query_set) ** 2, -1).neg().view(-1, 5)
    sim = nn.Softmax(axis=1)(neg_l2_distance)
    return sim

def clean_cl(cl_dict, nl_pred):
    keys = cl_dict.keys()
    for idx, item in enumerate(nl_pred):
        if str(item) in keys:
            nl_pred[idx] = cl_dict[str(item)]
    return nl_pred