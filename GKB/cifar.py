import random
import numpy as np
from itertools import chain
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
from msadapter.pytorch.utils.data import DataLoader
from msadapter.torchvision import datasets, transforms
from msadapter.torchvision.transforms.functional import InterpolationMode
 
import mindspore as ms

from msadapter.pytorch.utils.data import Dataset
from PIL import Image


class Cifar10Attr(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, transform=None):
        super(Cifar10Attr, self).__init__(root, train, transform)
        self.num_classes = 10
        data = [[] for _ in range(10)]
        for img, target in zip(self.data, self.targets):
            data[target].append(img)
        num_per_class = 5000 if train else 1000

        data_3x3 = [[] for _ in range(10)]
        for i in range(10):
            for j in range(num_per_class):
                img = []
                for k in range(10):
                    if i == k: continue
                    img.append(data[k][j])
                random.shuffle(img)
                img = np.stack(img, 0).reshape((3, 3, 32, 32, 3)).transpose((0, 2, 1, 3, 4)).reshape((96, 96, 3))
                data_3x3[i].append(img)

        data = list(chain(*data_3x3))
        targets = list(chain(*[[i] * num_per_class for i in range(10)]))
        ind = list(range(len(data)))
        random.shuffle(ind)
        self.data = [data[i] for i in ind]
        self.targets = [targets[i] for i in ind]
        self._id = list(range(len(data)))


class Cifar10AttrId(Cifar10Attr):
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        _id = self._id[index]
        return data, target, _id


class Cifar10AttrL(Cifar10AttrId):
    def __init__(self, root, train, transform=None, label_ratio=0.1):
        super(Cifar10AttrL, self).__init__(root, train, transform)
        num_per_class = int(5000 * label_ratio)
        data = []
        targets = []
        ans = [0] * 10
        _id = []
        for i in range(len(self.targets)):
            target = self.targets[i]
            if ans[target] >= num_per_class:
                if sum(ans) == num_per_class * 10: break
                continue
            data.append(self.data[i])
            targets.append(self.targets[i])
            _id.append(self._id[i])
            ans[target] += 1
        self.data = data
        self.targets = targets
        self._id = _id


class Cifar10AttrU(Cifar10AttrId):
    def __init__(self, root, train, transform=None, label_ratio=0.1):
        super(Cifar10AttrU, self).__init__(root, train, transform)
        num_per_class = int(5000 * (1 - label_ratio))
        data = []
        targets = []
        ans = [0] * 10
        _id = []
        for i in range(len(self.targets) - 1, -1, -1):
            target = self.targets[i]
            if ans[target] >= num_per_class:
                if sum(ans) == num_per_class * 10: break
                continue
            data.append(self.data[i])
            targets.append(self.targets[i])
            _id.append(self._id[i])
            ans[target] += 1
        self.data = data
        self.targets = targets
        self._id = _id


def get_gkb(dataset, d):
    label = []
    feat = []
    centers = []
    for _ in range(dataset.num_classes):
        centers.append([])
    for i, (j, k) in enumerate(zip(dataset._id, dataset.targets)):
        label.append(dataset.targets[i])
        f = torch.from_numpy(d[j])
        feat.append(f)
        centers[k].append(f)
    feat = torch.stack(feat, dim=0)
    label = torch.tensor(label)
    for i in range(dataset.num_classes):
        centers[i] = torch.stack(centers[i], dim=0).mean(dim=0)
    centers = torch.stack(centers, dim=0)
    return feat, label, centers