import os
import sys
import mindspore.experimental.optim as optim
import msadapter.pytorch as ms
from msadapter.pytorch.utils.data.dataloader import DataLoader

sys.path.append(os.path.abspath('.'))
from dataset.sampler import BalancedBatchSampler
from model.loss.CIN_loss import CINLoss
from train import Trainer
from utils import accuracy


class CINTrainer(Trainer):
    def __init__(self):
        super(CINTrainer, self).__init__()

    def get_dataloader(self, config):
        # CIN use `BalancedBatchSampler` to sample a fixed number of categories
        # and a fixed number of samples in each category.
        train_sampler = BalancedBatchSampler(self.datasets['train'], config.n_classes, config.n_samples)
        dataloaders = {
            'train': DataLoader(
                self.datasets['train'], num_workers=config.num_workers, batch_sampler=train_sampler
            ),
            'val': DataLoader(
                self.datasets['val'], batch_size=config.batch_size, num_workers=config.num_workers
            )
        }
        return dataloaders

    def get_criterion(self, config):
        return self.to_device(CINLoss(config))

    def get_optimizer(self, config):
        return optim.Adam(
            [
                {'params': self.model.parameters(), 'lr': config.lr},
                # CIN use a linear layer when it computes loss.
                {'params': self.criterion.parameters(), 'lr': config.lr},
            ], weight_decay=config.weight_decay)

    def forward_fn(self, images, labels):
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        logits, _ = outputs
        return loss, logits


if __name__ == '__main__':
    trainer = CINTrainer()
    trainer.train()
