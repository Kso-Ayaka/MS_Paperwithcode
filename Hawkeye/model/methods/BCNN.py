import msadapter.pytorch as ms
import msadapter.pytorch.nn as nn
from model.backbone import vgg16
from model.utils import initialize_weights
from model.registry import MODEL


class BilinearPooling(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        batch_size = x.shape[0]
        channel_size = x.shape[1]
        feature_size = x.shape[2] * x.shape[3]
        x = x.view(batch_size, channel_size, feature_size)
        x = ms.bmm(x, ms.transpose(x, 1, 2)) / feature_size

        x = x.view(batch_size, -1)
        x = ms.sqrt(x + 1e-5)
        x = nn.functional.normalize(x)
        return x


@MODEL.register
class BCNN(nn.Module):

    def __init__(self, config):
        super(BCNN, self).__init__()
        # Training stage for BCNN. Stage 1 freeze backbone parameters.
        self.stage = config.stage if 'stage' in config else 2

        self.backbone = vgg16(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2][0])

        self.bilinear_pooling = BilinearPooling()
        self.classifier = nn.Linear(512 ** 2, config.num_classes)
        self.classifier.apply(initialize_weights)

        if self.stage == 1:
            for params in self.backbone.parameters():
                params.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        if self.stage == 1:
            x = x.detach()
        x = self.bilinear_pooling(x)
        x = self.classifier(x)
        return x
