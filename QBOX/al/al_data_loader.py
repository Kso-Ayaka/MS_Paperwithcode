from utils.datasets import *
from al.train_test_func import hyp
import copy


class QueryRepo:
    """
    Class to store queried examples.
    """

    def __init__(self, partial_label: 'bool' = False):
        self.pl = partial_label
        self.database = dict()
        self.metadata = dict()
        self.tcost = 0
        self.skipped_image = 0
        self.empty_image_list = []
        self.fs_database = []

    def update(self, img_path: 'str', im_info: 'int', gt_boxes: 'torch.Tensor',
               num_pos: 'int', num_neg: 'int', num_outlier: 'int', cost: 'float',
               domain_label: 'int', iteration: 'int', method_name: 'str' = None,
               score=0):
        """Store the queried box.
        IMPORTANT: The label should not be mapped to target domain.
        Just remain the original index in the source domain.

        :param img_path: str, path to the image
        :param im_info:  int, 1: fully supervised; 0: partially supervised.
        :param gt_boxes:  torch.Tensor, gt boxes. class = -1 means background.
            Format: 0, class, xywh relative coord
        :param num_pos: int, number of positive (foreground) instances.
        :param num_neg: int, number of negative (background) instances.
        :param num_outlier: int, number of instances whose class is in the outlier classes.
        :param cost: float, cost of quering this example.
        :param domain_label: int, 1: source domain, 0 target domain.
        :param iteration: int, the number of the AL iteration.
        :return:
        """
        if img_path not in self.database.keys():
            # fully supervised or 1st time partially labeling
            assert len(gt_boxes) > 0
            self.database[img_path] = gt_boxes.clone()
            info = dict()
            info['sup_type'] = im_info
            info['num_pos'] = num_pos
            info['num_neg'] = num_neg
            info['num_outlier'] = num_outlier
            info['cost'] = cost
            info['domain_label'] = domain_label
            info['iteration'] = iteration
            self.metadata[img_path] = info
        else:
            # partially labeling
            queried_boxes = self.database[img_path]
            total_boxes = torch.cat([queried_boxes, gt_boxes], dim=0)
            self.database[img_path] = total_boxes.clone()
            # update metadata
            info = self.metadata[img_path]
            info['num_pos'] += num_pos
            info['num_neg'] += num_neg
            info['num_outlier'] += num_outlier
            info['cost'] += cost
            self.metadata[img_path] = info
        self.tcost += cost

    def __contains__(self, item):
        return True if item in self.database.keys() else False

    def __getitem__(self, item):
        return self.database[item].clone(), copy.copy(self.metadata[item])

    def __len__(self):
        return len(self.database)

    def get_latest_set(self):
        """Return a QueryRepo with the latest updated examples."""
        return list(self.database.keys())

    def keys(self):
        return list(self.database.keys())
