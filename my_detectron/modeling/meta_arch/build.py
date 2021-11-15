# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch
from detectron2.utils.registry import Registry

# from retinanet.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")


def build_model(cfg):
    """
    构建由``cfg.MODEL.META_ARCHITECTURE``定义的整个模型架构。
    注意它不会从``cfg``载入任何权重。
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
