import os
import os.path as osp
import importlib
from mynn.utils.logger import log_print
from mynn.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# Import modules automatically.
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in os.listdir(model_folder) if v.endswith('_model.py')]
_model_modules = [importlib.import_module(f'mynn.models.{file_name}') for file_name in model_filenames]


def build_model(opt=None, name=None, **kwargs):
    if opt is not None:
        model = MODEL_REGISTRY.get(opt['model'])()
        log_print(f'Model: {model.__class__.__name__}')
    else:
        model = MODEL_REGISTRY.get(name)(**kwargs)
        log_print(f'Model: {model.__class__.__name__}')
    return model