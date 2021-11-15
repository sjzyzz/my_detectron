from abc import ABCMeta, abstractmethod
import torch.nn as nn

from my_detectron.layers.shape_spec import ShapeSpec


class Backbone(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @property
    def size_divisibility(self) -> int:
        return 0

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }
