import msadapter.pytorch as ms
import msadapter.pytorch.nn as nn


class APINetLoss(nn.Module):
    def __init__(self, config):
        super(APINetLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.rank_loss = nn.MarginRankingLoss(margin=0.05)
        self.softmax_layer = nn.Softmax(dim=1)

    def __call__(self, output, target):
        self_logits, other_logits, labels1, labels2 = output
        labels1, labels2 = labels1.to(ms.int64), labels2.to(ms.int64)

        batch_size = self_logits.shape[0] // 2

        # compute loss
        logits = ms.cat([self_logits, other_logits], dim=0)
        targets = ms.cat([labels1, labels2, labels1, labels2], dim=0).to(ms.int64)
        softmax_loss = self.ce_loss(logits, targets)

        self_scores = self.softmax_layer(self_logits)[ms.arange(2 * batch_size).to(ms.int64),
                                                      ms.cat([labels1, labels2], dim=0)]
        other_scores = self.softmax_layer(other_logits)[ms.arange(2 * batch_size).to(ms.int64),
                                                        ms.cat([labels1, labels2], dim=0)]
        flag = ms.ones((2 * batch_size, ))
        rank_loss = self.rank_loss(self_scores, other_scores, flag)
        loss = softmax_loss + rank_loss
        return loss
