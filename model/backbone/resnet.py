import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from abc import ABCMeta, abstractmethod

from utils import ShapeSpec

_resnet_mapper = {18: resnet.resnet18, 50: resnet.resnet50, 101: resnet.resnet101}


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class ResnetBackbone(Backbone):
    def __init__(self, cfg, input_shape=None, pretrained=True):
        super().__init__()
        depth = cfg.BACKBONE_DEPTH
        backbone = _resnet_mapper[depth](pretrained=pretrained)
        self.stage0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

        self._output_shape = cfg.BACKBONE_OUTPUT_SHAPE

    def forward(self, x):
        c1 = self.stage0(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return {'res2': c2, 'res3': c3, 'res4': c4, 'res5': c5}

    def output_shape(self):
        return self._output_shape


def build_torch_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=3)

    backbone = ResnetBackbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone
