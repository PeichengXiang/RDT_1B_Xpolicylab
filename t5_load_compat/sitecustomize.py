"""Compatibility shim for loading the official Google T5 PyTorch checkpoint.

Transformers 4.37+ forces torch.load(weights_only=True, mmap=True), which fails
on google/t5-v1_1-xxl's official pickle. This shim is used only by offline
language-embedding jobs and only relaxes calls that already request
weights_only=True. The checkpoint was downloaded from the official repository.
"""

import torch


_original_torch_load = torch.load


def _load_official_t5_checkpoint(*args, **kwargs):
    if kwargs.get("weights_only") is True:
        kwargs["weights_only"] = False
        kwargs.pop("mmap", None)
    return _original_torch_load(*args, **kwargs)


torch.load = _load_official_t5_checkpoint
