import os
import sys
from mindspore import value_and_grad
from mindspore.nn import cosine_decay_lr
import msadapter.pytorch as ms
from mindspore.experimental import optim 
from msadapter.pytorch.utils.data.dataloader import DataLoader
from mindspore.experimental.optim.lr_scheduler import LinearLR, SequentialLR

sys.path.append(os.path.abspath('.'))
from dataset.sampler import BalancedBatchSampler
from model.loss.APINet_loss import APINetLoss
from train import Trainer
from utils import accuracy


class APINetTrainer(Trainer):
    def __init__(self):
        super(APINetTrainer, self).__init__()

    def get_dataloader(self, config):
        # APINet use `BalancedBatchSampler` to sample a fixed number of categories
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
        return APINetLoss(config)

    def get_optimizer(self, config):
        model = self.get_model_module()
        backbone_param_ids = list(map(id, model.backbone.parameters()))
        fc_params = list(filter(lambda p: id(p) not in backbone_param_ids, model.parameters()))
        return optim.Adam(
            [
                {'params': model.backbone.parameters(), 'lr': config.lr},
                {'params': fc_params, 'lr': config.lr}
            ], weight_decay=config.weight_decay
        )

    def forward_fn(self, images, labels):
        outputs = self.model(images, labels, flag='train')
        loss = self.criterion(outputs, labels)
        return loss, outputs

    def batch_training(self, data):
        images, labels = data['img'], data['label']

        outputs = self.model(images, labels, flag='train')
        loss, outputs = self.forward_fn(images, labels)

        (loss, _), grads = self.grad_fn()(images, labels)
        self.optimizer(grads)

        self_logits, other_logits, labels1, labels2 = outputs
        logits = ms.cat([self_logits, other_logits], dim=0)
        targets = ms.cat([labels1, labels2, labels1, labels2], dim=0)

        # record accuracy and loss
        acc = accuracy(logits, targets, 1)
        batch_size = self_logits.shape[0] // 2
        self.average_meters['acc'].update(acc, 4 * batch_size)
        self.average_meters['loss'].update(loss.item(), 2 * batch_size)

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        logits = self.model(images, flag='val')
        acc = accuracy(logits, labels, 1)
        self.average_meters['acc'].update(acc, logits.shape(0))

    def on_start_epoch(self, config):
        if self.epoch == 0:
            self.optimizer.param_groups[0]['lr'] = 0
            self.logger.info('Freeze conv')
        elif self.epoch == 8:
            self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']
            self.logger.info('Unfreeze conv')

        Trainer.on_start_epoch(self, config)


if __name__ == '__main__':
    trainer = APINetTrainer()
    trainer.train()
