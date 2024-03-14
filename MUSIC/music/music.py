import math
import os
from copy import deepcopy
from sklearn import metrics
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import save_checkpoint

from tqdm import tqdm
from sklearn.preprocessing import normalize
from .utils import *
from .loss import non_k_softmax_loss, w_loss

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class linear_model(nn.Cell):
    def __init__(self, output_dim=5, input_dim=512):
        super(linear_model, self).__init__()
        self.linear = nn.Dense(input_dim, output_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Dense(500, output_dim)

    def construct(self, x):
        out = self.linear(x)
        return out

class MUSIC(object):
    def __init__(self, nlthreshold=0.4, plthreshold=0.9):
        self.device = "GPU" if context.get_context("device_target") == "GPU" else "CPU"
        self.nlthreshold = nlthreshold
        self.plthreshold = plthreshold

    def Negtive_Learning(self, clf, unlabel_embeddings, test_embeddings, test_targets, threshold=0.2):
        optimizer = nn.Adam(clf.trainable_params(),
                            learning_rate=1e-4,
                            weight_decay=0.0005)
        # Step 2. Select pseudo-labels.
        test_embeddings = Tensor(test_embeddings, dtype=mstype.float32).to(self.device)
        test_targets = Tensor(test_targets, dtype=mstype.float32).to(self.device)
        unlabel_embeddings = Tensor(unlabel_embeddings, dtype=mstype.float32).to(self.device)
        unlabel_out = clf(unlabel_embeddings)
        nl_pred, nl_conf = get_preds(unlabel_out)

        pseudo_nl_embeddings, nl_label, indexes_nl = self.get_pseudo_labels(clf, unlabel_embeddings, threshold, type='nl',
                                                                       use_uncertainty=False)
        for idx, item in enumerate(indexes_nl):
            nl_pred[item] = nl_label[idx]
        self.train_NL(clf, unlabel_embeddings, nl_pred, 100, optimizer, test_embeddings, test_targets)
        save_checkpoint('best_model_NL.ckpt', clf)
        return clf

    def get_pseudo_labels(self, clf, unlabel_embeddings, uncertainty_thres, type, use_uncertainty=False):
        unlabel_out = clf(unlabel_embeddings)
        if type == 'nl':
            nl_pred, nl_conf = get_preds(unlabel_out)
            nl_indexes, pred = get_indexes(unlabel_out, threshold=self.nlthreshold, reverse=True)
            nl_label = nl_pred[nl_indexes]
            pseudo_nl_embeddings = unlabel_embeddings[nl_indexes]
            return pseudo_nl_embeddings, nl_label, nl_indexes
        else:
            pl_indexes, pred = get_indexes(unlabel_out, threshold=self.plthreshold)
            if use_uncertainty:
                pl_indexes = get_indexes_on_uncertainty(unlabel_out, uncertainty_thres=uncertainty_thres,
                                                        confidence_thres=self.plthreshold, reverse=False)
            pl_label = pred[pl_indexes]
            pseudo_pl_embeddings = unlabel_embeddings[pl_indexes]
            return pseudo_pl_embeddings, pl_label, pl_indexes

    def train_NL(self, clf, pseudo_nl_embeddings, nl_label, epoch, optimizer):
        model = clf
        # best_acc = 0.0
        for epc in range(epoch):
            out = model(pseudo_nl_embeddings)
            loss = w_loss(out, 5, Tensor(nl_label, dtype=mstype.float32).to(self.device))
            optimizer.clear_gradients()
            loss.backward()
            optimizer.step()
            save_checkpoint('best_model_NL.ckpt', model)

        init_learning_rate(optimizer)