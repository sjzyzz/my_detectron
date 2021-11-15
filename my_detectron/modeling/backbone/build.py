from detectron2.utils.registry import Registry

from my_detectron.layers.shape_spec import ShapeSpec

from detectron2.modeling.backbone import Backbone

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


def build_backbone(cfg, input_shpae=None):
    if input_shpae is None:
        input_shpae = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shpae)
    assert isinstance(backbone, Backbone)
    return backbone
