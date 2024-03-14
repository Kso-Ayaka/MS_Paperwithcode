import math
import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore.train.callback import LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn import metrics
from copy import deepcopy
from config import config
from datasets import CategoriesSampler, DataSet
from ici_models.ici import ICI
from ici_models.ici_fc import ICI_FC
from utils import get_embedding, mean_confidence_interval, setup_seed
import ilpc.iterative_graph_functions as igf
from sklearn.linear_model import LogisticRegression
from ilpc.fc import FC
from music.music import MUSIC

# Set the environment variable to use GPU
os.environ['DEVICE_ID'] = '0,1'

# Set the context to GPU
context = ms.context('gpu')

# Enable graph mode
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="GPU")

def train_embedding(args):
    setup_seed(42)
    ckpt_root = os.path.join('./ckpt', args.dataset)
    os.makedirs(ckpt_root, exist_ok=True)
    data_root = os.path.join(args.folder, args.dataset)

    from datasets import EmbeddingDataset
    source_set = EmbeddingDataset(data_root, args.img_size, 'train')
    source_loader = ds.GeneratorDataset(source_set, column_names=['image', 'label'], shuffle=True)
    source_loader = source_loader.batch(128, drop_remainder=True)

    test_set = EmbeddingDataset(data_root, args.img_size, 'val')
    test_loader = ds.GeneratorDataset(test_set, column_names=['image', 'label'], shuffle=False)
    test_loader = test_loader.batch(64)

    if args.dataset == 'CUB':
        num_classes = 100
    elif args.dataset == 'tieredimagenet':
        num_classes = 351
    elif args.dataset == 'CIFAR-FS':
        num_classes = 80
    else:
        num_classes = 80

    from ici_models.resnet12 import resnet12
    model = resnet12(num_classes)
    model = nn.DataParallel(model)
    optimizer = nn.SGD(params=model.trainable_params(),
                       learning_rate=args.lr, momentum=0.9, weight_decay=1e-4)

    from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
    config_ck = CheckpointConfig(save_checkpoint_steps=source_loader.get_dataset_size(),
                                 keep_checkpoint_max=120)
    ckpoint_cb = ModelCheckpoint(prefix="res12_epoch", directory=ckpt_root, config=config_ck)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    best_acc = 0.0

    for epoch in range(120):
        loss_list = []
        train_acc_list = []
        for batch in source_loader.create_dict_iterator():
            images = Tensor(batch['image'], ms.float32)
            labels = Tensor(batch['label'], ms.int32)
            preds = model(images)
            loss_value = loss(preds, labels)
            loss_list.append(loss_value.asnumpy())
            train_acc_list.append(metrics.accuracy_score(labels.asnumpy(), np.argmax(preds.asnumpy(), axis=1)))

            grads = loss_value.backward()
            optimizer(grads)

        acc = []
        model.set_train(False)
        for batch in test_loader.create_dict_iterator():
            images = Tensor(batch['image'], ms.float32)
            labels = Tensor(batch['label'], ms.int32)
            preds = model(images)
            acc += (np.argmax(preds.asnumpy(), axis=1) == labels.asnumpy()).tolist()
        acc = np.mean(acc)

        print('Epoch:{} Train-loss:{} Train-acc:{} Valid-acc:{}'.format(epoch, str(np.mean(loss_list))[:6], str(
            np.mean(train_acc_list))[:6], str(acc)[:6]))

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(ckpt_root, "res12_epoch{}.ckpt".format(epoch))
            model.save_checkpoint(save_path)
            model.save_checkpoint(os.path.join(ckpt_root, 'tiered_448_res12_best.ckpt'))

def test(args):
    setup_seed(42)
    import warnings
    warnings.filterwarnings('ignore')

    if args.dataset == 'CUB':
        num_classes = 100
    elif args.dataset == 'tieredimagenet':
        num_classes = 351
    else:
        num_classes = 64

    from ici_models.resnet12 import resnet12
    model = resnet12(num_classes)
    if args.resume is not None:
        model.load_checkpoint(args.resume)
    model.set_train(False)

    data_root = os.path.join(args.folder, args.dataset)
    dataset = DataSet(data_root, 'test', args.img_size)
    sampler = CategoriesSampler(dataset.label, args.num_batches,
                                args.num_test_ways, (args.num_shots, 15, args.unlabel))
    testloader = ds.GeneratorDataset(dataset, column_names=['image', 'label'], sampler=sampler, shuffle=False)
    testloader = testloader.batch(args.num_test_ways * (args.num_shots + 15 + args.unlabel))

    loader = tqdm(testloader, ncols=0)
    iterations = math.ceil(args.unlabel / args.step) + 2 if args.unlabel != 0 else math.ceil(15 / args.step) + 2
    acc = []

    for batch in loader.create_dict_iterator():
        images = Tensor(batch['image'], ms.float32)
        indicator = Tensor(batch['label'], ms.int32)
        if args.algorithm == 'ilpc':
            query_ys, query_ys_pred = ilpc_test_ss(args, model, images, indicator)
        elif args.algorithm == 'ici':
            ici = ICI_FC(classifier=args.classifier, num_class=args.num_test_ways, step=args.step,
                         reduce=args.embed, d=args.dim)
            query_ys, query_ys_pred = ici_test_ss(args, model, images, indicator, ici)
        else:
            print("This algorithm is not available in this experiment")
            break
        acc.append(metrics.accuracy_score(query_ys.asnumpy(), query_ys_pred.asnumpy()))

    test_acc, test_std = mean_confidence_interval(acc)
    print(test_acc * 100)
    print(test_std * 100)


def ilpc_test_ss(args, model, data, indicator):
    k = args.num_shots * args.num_test_ways
    targets = np.arange(args.num_test_ways).repeat(args.num_shots + 15 + args.unlabel)[
        indicator[:args.num_test_ways * (args.num_shots + 15 + args.unlabel)] != 0]
    data = data[indicator != 0].to(args.device)
    train_inputs = data[:k]
    train_targets = targets[:k]
    test_inputs = data[k:k + 15 * args.num_test_ways]
    test_targets = targets[k:k + 15 * args.num_test_ways]
    train_embeddings = get_embedding(model, train_inputs, args.device)
    test_embeddings = get_embedding(model, test_inputs, args.device)
    if args.unlabel != 0:
        unlabel_inputs = data[k + 15 * args.num_test_ways:]
        unlabel_targets = targets[k + 15 * args.num_test_ways:]
        unlabel_embeddings = get_embedding(model, unlabel_inputs, args.device)
    else:
        unlabel_embeddings = test_embeddings
        unlabel_targets = test_targets
    args.no_samples = np.array(np.repeat(float(unlabel_targets.shape[0] / args.num_test_ways), args.num_test_ways))

    support_features = train_embeddings
    query_features = test_embeddings
    unlabelled_features = unlabel_embeddings

    support_ys = train_targets
    query_ys = test_targets
    unlabelled_ys = unlabel_targets

    if args.model == 'resnet12':
        query_features = np.concatenate((query_features, unlabelled_features), axis=0)
        support_features, query_features = igf.dim_reduce(args, support_features, query_features)
        query_features, unlabelled_features = query_features[:args.n_queries * args.num_test_ways], query_features[
                                                                                                    args.n_queries * args.num_test_ways:]

    neg_train = None
    if args.neg_train == 'music':
        neg_train = MUSIC()
    if args.classifier == 'lr':
        clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
        clf.fit(support_features, support_ys)
    elif args.classifier == 'fc':
        clf = FC(num_class=args.num_test_ways)
        clf.fit(support_features, support_ys)
        if neg_train is not None:
            threshold = 0.2
            clf.classifier = neg_train.Negtive_Learning(clf.classifier, support_features, query_features, query_ys,
                                                        threshold)

    labelled_samples = support_ys.shape[0]
    support_ys, support_features = igf.iter_balanced(args, support_features, support_ys, unlabelled_features,
                                                     unlabelled_ys, query_features, query_ys, labelled_samples,
                                                     model=clf, neg_train=neg_train)

    clf.fit(support_features, support_ys)
    if neg_train is not None:
        threshold = 0.2
        clf.classifier = neg_train.Negtive_Learning(clf.classifier, support_features, query_features, query_ys,
                                                    threshold)
    query_ys_pred = clf.predict(query_features)

    return query_ys, query_ys_pred

def ici_test_ss(args, model, data, indicator, ici):
    k = args.num_shots * args.num_test_ways
    targets = torch.arange(args.num_test_ways).repeat(args.num_shots + 15 + args.unlabel).long()[
        indicator[:args.num_test_ways * (args.num_shots + 15 + args.unlabel)] != 0]
    data = data[indicator != 0].to(args.device)
    train_inputs = data[:k]
    train_targets = targets[:k].cpu().numpy()
    test_inputs = data[k:k + 15 * args.num_test_ways]
    test_targets = targets[k:k + 15 * args.num_test_ways].cpu().numpy()
    train_embeddings = get_embedding(model, train_inputs, args.device)
    test_embeddings = get_embedding(model, test_inputs, args.device)
    if args.unlabel != 0:
        unlabel_inputs = data[k + 15 * args.num_test_ways:]
        # unlabel_targets = targets[k + 15 * args.num_test_ways:].cpu().numpy()
        unlabel_embeddings = get_embedding(
            model, unlabel_inputs, args.device)
    else:
        # unlabel_targets = None
        unlabel_embeddings = None
    neg_train = None
    if args.neg_train == 'music':
        neg_train = MUSIC()
    ici.fit(train_embeddings, train_targets)
    pred = ici.predict(test_embeddings, unlabel_embeddings, False, test_targets, neg_train)

    return test_targets, pred

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU" if args.device == "cuda" else "CPU")
    print(args)
    if args.mode == 'train':
        train_embedding(args)
    elif args.mode == 'test':
        my_test(args)
    else:
        raise NameError


if __name__ == '__main__':
    args = config()
    main(args)