from collections import namedtuple


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    包含一个张量的基本形状的简单结构。它经常被用作模型的辅助输入/输出来弥补PyTorch模型对形状的推理能力。

    属性：
        channels:
        height:
        width:
        stride:
    """

    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)
