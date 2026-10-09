from typing import List

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class ModuleFreezeHook(Hook):
    """Keep selected modules frozen during fine-tuning.

    This is mainly used for camera-lidar residual fine-tuning where the
    pretrained lidar branch should stay fixed, including BN running stats.
    """

    def __init__(self, module_names: List[str]) -> None:
        self.module_names = module_names

    def before_train(self, runner) -> None:
        self._freeze_modules(runner.model)

    def before_train_epoch(self, runner) -> None:
        self._freeze_modules(runner.model)

    def _freeze_modules(self, model) -> None:
        if is_model_wrapper(model):
            model = model.module

        for module_name in self.module_names:
            module = getattr(model, module_name, None)
            if module is None:
                raise AttributeError(f'Module `{module_name}` not found in model.')

            module.eval()
            for param in module.parameters():
                param.requires_grad = False

            for submodule in module.modules():
                if isinstance(submodule, nn.modules.batchnorm._BatchNorm):
                    submodule.eval()
