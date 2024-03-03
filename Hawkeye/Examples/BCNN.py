import os
import sys
import mindspore.experimental.optim as optim
from msadapter.torchvision import transforms

sys.path.append(os.path.abspath('.'))
from train import Trainer


class BCNNTrainer(Trainer):
    def __init__(self):
        super(BCNNTrainer, self).__init__()

    def get_optimizer(self, config):
        model = self.get_model_module()
        # Stage 1, freeze backbone parameters.
        if self.config.model.stage == 1:
            params = model.classifier.parameters()
        # Stage 2, train all parameters.
        elif self.config.model.stage == 2:
            params = model.parameters()
        return optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)


if __name__ == '__main__':
    trainer = BCNNTrainer()
    trainer.train()
