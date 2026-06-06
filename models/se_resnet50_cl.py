import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH_NAME = "se_resnet50_cl"
MODEL_CONFIGS = {"se_resnet50_cl": {"input_size": 224, "params_M": 28.09}}


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced = max(channels // reduction_ratio, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = self.squeeze(x).view(b, c)
        e = self.excitation(s).view(b, c, 1, 1)
        return x * e.expand_as(x)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample=None, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)

        self.se         = SEBlock(out_channels * self.expansion, reduction)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class SEResNet50(nn.Module):
    def __init__(self, num_classes: int = 5, feature_dim: int = 2048,
                 reduction: int = 16):
        super().__init__()
        self.in_channels = 64

        self.conv1   = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  3, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(128, 4, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(256, 6, stride=2, reduction=reduction)
        self.layer4 = self._make_layer(512, 3, stride=2, reduction=reduction)

        self.avgpool     = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = feature_dim

        self.feature_layer = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

        self._init_weights()

    def _make_layer(self, out_channels: int, num_blocks: int,
                    stride: int, reduction: int) -> nn.Sequential:
        downsample = None
        expanded   = out_channels * SEBottleneck.expansion

        if stride != 1 or self.in_channels != expanded:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, expanded, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(expanded),
            )

        layers = [SEBottleneck(self.in_channels, out_channels,
                               stride, downsample, reduction)]
        self.in_channels = expanded

        for _ in range(1, num_blocks):
            layers.append(SEBottleneck(self.in_channels, out_channels,
                                       reduction=reduction))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        features = self.feature_layer(x)
        logits   = self.classifier(features)
        return logits, features


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int,
                 lambda_c: float = 0.003):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c    = lambda_c
        self.centers = nn.Parameter(
            torch.randn(num_classes, feature_dim), requires_grad=True
        )

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        batch_size    = features.size(0)
        centers_batch = self.centers[labels]
        diff          = features - centers_batch
        loss          = 0.5 * (diff ** 2).sum() / batch_size
        return self.lambda_c * loss


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int,
                 lambda_c: float = 0.003):
        super().__init__()
        self.ce_loss     = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss(num_classes, feature_dim, lambda_c)

    def forward(self, logits: torch.Tensor, features: torch.Tensor,
                labels: torch.Tensor):
        loss_ce     = self.ce_loss(logits, labels)
        loss_center = self.center_loss(features, labels)
        total       = loss_ce + loss_center
        return total, loss_ce, loss_center


def build_model(name="se_resnet50_cl", num_classes=5, in_channels=3, **kwargs):
    feature_dim = kwargs.get("feature_dim", 2048)
    lambda_c    = kwargs.get("lambda_c", 0.003)
    reduction   = kwargs.get("reduction", 16)

    model = SEResNet50(num_classes=num_classes,
                       feature_dim=feature_dim,
                       reduction=reduction)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, 7,
                                stride=2, padding=3, bias=False)

    model.num_classes  = num_classes
    model.criterion    = CombinedLoss(num_classes, feature_dim, lambda_c)

    return model
