"""Explicit, recorded performance options; old configs retain their defaults."""

import torch


def configure_runtime(config, device):
    options = config.get("training", {})
    fast = options.get("fast_cuda", False) and torch.device(device).type == "cuda"
    torch.backends.cudnn.deterministic = not fast
    torch.backends.cudnn.benchmark = fast
    # Explicitly reset TF32 as multiple experiments may share one process.
    torch.backends.cuda.matmul.allow_tf32 = fast
    torch.backends.cudnn.allow_tf32 = fast if "fast_cuda" in options else True
    return {
        "amp": options.get("amp", False) and torch.device(device).type == "cuda",
        "channels_last": options.get("channels_last", False) and torch.device(device).type == "cuda",
        "fast_cuda": fast,
    }
