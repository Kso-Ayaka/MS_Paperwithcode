import random
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

def NL_loss(f, K, labels):
    Q_1 = 1 - ops.Softmax()(f, 1)
    Q = ops.Softmax()(Q_1, 1)
    weight = 1 - Q
    out = weight * ops.Log()(Q)
    return ops.NLLLoss()(out, labels)  # Equation(14) in paper

def w_loss(f, K, labels):
    loss_class = non_k_softmax_loss(f=f, K=K, labels=labels)
    loss_w = w_loss_p(f=f, K=K, labels=labels)
    entro_loss = entropy_loss(f)
    final_loss = loss_class + loss_w  # Equation(11) in paper
    return final_loss

def w_loss_p(f, K, labels):
    Q_1 = 1 - ops.Softmax()(f, 1)
    Q = ops.Softmax()(Q_1, 1)
    q = ops.Reciprocal()(ops.ReduceSum()(Q_1, dim=1))
    q = q.unsqueeze(1).repeat(1, K)
    w = ops.Mul()(Q_1, q)  # weight
    w_1 = ops.Mul()(w, ops.Log()(Q))
    return ops.NLLLoss()(w_1, labels)  # Equation(14) in paper

def non_k_softmax_loss(f, K, labels):
    Q_1 = 1 - ops.Softmax()(f, 1)
    Q_1 = Q_1 / 1.5
    Q_1 = ops.Softmax()(Q_1, 1)
    return ops.NLLLoss()(ops.Log()(Q_1), labels)  # Equation(8) in paper

def entropy_loss(p):
    p = ops.Softmax()(p, dim=1)
    epsilon = 1e-5
    return -1 * ops.ReduceSum()(ops.Mul()(p, ops.Log()(p + epsilon))) / p.size(0)