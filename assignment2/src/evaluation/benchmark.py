'''
Model-only inference cost on synthetic inputs (no image decoding or transfer).

Each setting runs warm-up passes, then times a fixed number of forward passes
with CUDA synchronization. AMP FP16 is measured on CUDA only.
'''

from contextlib import nullcontext
from time import perf_counter

import torch


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_inference(model, device, image_size=224, batch_sizes=(1, 64),
                        channels_last=False, warmup=10, iterations=50):
    device = torch.device(device)
    model = model.to(device).eval()
    precisions = ["fp32"] + (["amp_fp16"] if device.type == "cuda" else [])
    results = []

    for batch_size in batch_sizes:
        images = torch.randn(batch_size, 3, image_size, image_size, device=device)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        for precision in precisions:
            autocast = (
                torch.autocast(device.type, dtype=torch.float16)
                if precision == "amp_fp16" else nullcontext()
            )
            with torch.no_grad(), autocast:
                for _ in range(warmup):
                    model(images)
                _synchronize(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                start = perf_counter()
                for _ in range(iterations):
                    model(images)
                _synchronize(device)
                seconds = perf_counter() - start

            results.append({
                "batch_size": batch_size,
                "precision": precision,
                "iterations": iterations,
                "ms_per_batch": 1000 * seconds / iterations,
                "ms_per_image": 1000 * seconds / (iterations * batch_size),
                "images_per_second": iterations * batch_size / seconds,
                "peak_allocated_mb": (
                    torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda" else None
                ),
            })

    return results
