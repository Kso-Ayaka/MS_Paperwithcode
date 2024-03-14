import os
import pickle as pkl
import numpy as np
from PIL import Image
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
from msadapter.pytorch.utils.data import DataLoader
from msadapter.torchvision import datasets, transforms
from msadapter.torchvision.transforms.functional import InterpolationMode
 
import mindspore as ms

from msadapter.pytorch.utils.data import Dataset


class CUB200(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, label_ratio=1.0, has_label=True):
        self._transform = transform
        self.train = train
        self.num_classes = 200

        part = '1' if train else '0'
        if train and label_ratio < 1:
            s = pkl.load(open(f'{label_ratio}/cub_{label_ratio}_id.pkl', 'rb'))
        self.data = []
        self.target = []
        self._id = []
        for line_split, line_images, line_labels in \
            zip(open(os.path.join(root, 'train_test_split.txt'), 'r'), 
                open(os.path.join(root, 'images.txt'), 'r'),
                open(os.path.join(root, 'image_class_labels.txt'), 'r')
            ):
            _id, split = line_split.strip().split()
            if split == part:
                if train and label_ratio < 1:
                    if int(_id) in s and not has_label:continue
                    if int(_id) not in s and has_label:continue
                image_path = line_images.strip().split()[1]
                label = int(line_labels.strip().split()[1]) - 1
                self.data.append(os.path.join(root, 'images', image_path))
                self.target.append(label)
                self._id.append(int(_id))


    def __getitem__(self, index):
        image, target = self.data[index], self.target[index]
        image = Image.open(image).convert('RGB')

        image = self._transform(image)
        return image, target

    def __len__(self):
        return len(self.data)


class CUB200Id(CUB200):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        _id = self._id[index]
        return image, target, _id


class CUB200Feat(CUB200Id):
    def __init__(self, root, train=True, transform=None, label_ratio=1.0, has_label=True, d=None):
        super().__init__(root, train, transform, label_ratio, has_label)
        self.d = d
    
    def __getitem__(self, index):
        image, target, _id = super().__getitem__(index)
        feat = self.d[_id]
        return image, target, feat


class StanfordDog(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, label_ratio=1.0, has_label=True):
        self._transform = transform
        self.train = train
        self.num_classes = 120

        part = 'train.txt' if train else 'test.txt'
        if train and label_ratio < 1:
            s = pkl.load(open(f'{label_ratio}/sdog_{label_ratio}_id.pkl', 'rb'))
        self.data = []
        self.target = []
        self._id = []
        for i, line in enumerate(open(os.path.join(root, part), 'r')):
            _id, label = line.strip().split()
            if train and label_ratio < 1:
                if _id in s and not has_label:continue
                if _id not in s and has_label:continue
            image_path = _id
            label = int(label) - 1
            self.data.append(os.path.join(root, 'images', image_path))
            self.target.append(label)
            self._id.append(i)


    def __getitem__(self, index):
        image, target = self.data[index], self.target[index]
        image = Image.open(image).convert('RGB')

        image = self._transform(image)
        return image, target

    def __len__(self):
        return len(self.data)


class StanfordDogId(StanfordDog):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        _id = self._id[index]
        return image, target, _id


class StanfordDogFeat(StanfordDogId):
    def __init__(self, root, train=True, transform=None, label_ratio=1.0, has_label=True, d=None):
        super().__init__(root, train, transform, label_ratio, has_label)
        self.d = d
    
    def __getitem__(self, index):
        image, target, _id = super().__getitem__(index)
        feat = self.d[_id]
        return image, target, feat


class Aircraft(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, label_ratio=1.0, has_label=True):
        self._transform = transform
        self.train = train
        self.num_classes = 100

        part = 'images_variant_train.txt' if train else 'images_variant_test.txt'
        if train and label_ratio < 1:
            s = pkl.load(open(f'{label_ratio}/air_{label_ratio}_id.pkl', 'rb'))
        self.data = []
        self.target = []
        self._id = []
        name2label = {}
        for i, line in enumerate(open(os.path.join(root, 'fgvc-aircraft-2013b/data/', part), 'r')):
            _id, name = line[:7], line.strip()[8:]
            if name not in name2label:
                name2label[name] = len(name2label)
            label = name2label[name]
            if train and label_ratio < 1:
                if _id in s and not has_label:continue
                if _id not in s and has_label:continue
            image_path = _id+'.jpg'
            self.data.append(os.path.join(root, 'fgvc-aircraft-2013b/data/images', image_path))
            self.target.append(label)
            self._id.append(i)


    def __getitem__(self, index):
        image, target = self.data[index], self.target[index]
        image = Image.open(image).convert('RGB')

        image = self._transform(image)
        return image, target

    def __len__(self):
        return len(self.data)


class AircraftId(Aircraft):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        _id = self._id[index]
        return image, target, _id


class AircraftFeat(AircraftId):
    def __init__(self, root, train=True, transform=None, label_ratio=1.0, has_label=True, d=None):
        super().__init__(root, train, transform, label_ratio, has_label)
        self.d = d
    
    def __getitem__(self, index):
        image, target, _id = super().__getitem__(index)
        feat = self.d[_id]
        return image, target, feat


class Food101(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, label_ratio=1.0, has_label=True):
        self._transform = transform
        self.train = train
        self.num_classes = 101

        part = 'train.txt' if train else 'test.txt'
        if train and label_ratio < 1:
            s = pkl.load(open(f'{label_ratio}/food_{label_ratio}_id.pkl', 'rb'))
        self.data = []
        self.target = []
        self._id = []
        name2label = {}
        for i, line in enumerate(open(os.path.join(root, 'meta', part), 'r')):
            line = line.strip()
            _id, name = line, line.split('/')[0]
            if name not in name2label:
                name2label[name] = len(name2label)
            label = name2label[name]
            if train and label_ratio < 1:
                if _id in s and not has_label:continue
                if _id not in s and has_label:continue
            image_path = _id+'.jpg'
            self.data.append(os.path.join(root, 'images', image_path))
            self.target.append(label)
            self._id.append(i)


    def __getitem__(self, index):
        image, target = self.data[index], self.target[index]
        image = Image.open(image).convert('RGB')

        image = self._transform(image)
        return image, target

    def __len__(self):
        return len(self.data)


class Food101Id(Food101):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        _id = self._id[index]
        return image, target, _id


class Food101Feat(Food101Id):
    def __init__(self, root, train=True, transform=None, label_ratio=1.0, has_label=True, d=None):
        super().__init__(root, train, transform, label_ratio, has_label)
        self.d = d
    
    def __getitem__(self, index):
        image, target, _id = super().__getitem__(index)
        feat = self.d[_id]
        return image, target, feat


class Vegfru(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, label_ratio=1.0, has_label=True):
        self._transform = transform
        self.train = train
        self.num_classes = 292

        part = 'vegfru_train.txt' if train else 'vegfru_test.txt'
        if train and label_ratio < 1:
            s = pkl.load(open(f'{label_ratio}/veg_{label_ratio}_id.pkl', 'rb'))
        self.data = []
        self.target = []
        self._id = []
        for i, line in enumerate(open(os.path.join(root, part), 'r')):
            _id, label = line.strip().split(' ')
            label = int(label)
            if train and label_ratio < 1:
                if _id in s and not has_label:continue
                if _id not in s and has_label:continue
            image_path = _id
            self.data.append(os.path.join(root, image_path))
            self.target.append(label)
            self._id.append(i)


    def __getitem__(self, index):
        image, target = self.data[index], self.target[index]
        image = Image.open(image).convert('RGB')

        image = self._transform(image)
        return image, target

    def __len__(self):
        return len(self.data)


class VegfruId(Vegfru):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        _id = self._id[index]
        return image, target, _id


class VegfruFeat(VegfruId):
    def __init__(self, root, train=True, transform=None, label_ratio=1.0, has_label=True, d=None):
        super().__init__(root, train, transform, label_ratio, has_label)
        self.d = d
    
    def __getitem__(self, index):
        image, target, _id = super().__getitem__(index)
        feat = self.d[_id]
        return image, target, feat


def get_gkb(dataset):
    label = []
    feat = []
    centers = []
    for _ in range(dataset.num_classes):
        centers.append([])
    for i, (j, k) in enumerate(zip(dataset._id, dataset.target)):
        label.append(dataset.target[i])
        f = torch.from_numpy(dataset.d[j])
        feat.append(f)
        centers[k].append(f)
    feat = torch.stack(feat, dim=0)
    label = torch.tensor(label)
    for i in range(dataset.num_classes):
        centers[i] = torch.stack(centers[i], dim=0).mean(dim=0)
    centers = torch.stack(centers, dim=0)
    return feat, label, centers
