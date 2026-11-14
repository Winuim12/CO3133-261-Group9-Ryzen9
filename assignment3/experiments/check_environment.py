"""Inspect the environment and optionally exercise PyTorch without training."""

import argparse
import importlib.metadata
import json
import platform
import sys

from src.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/image_only.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_model": config["model"].get("name", "not selected"),
        "labels": config["labels"],
        "requested_device": args.device,
    }
    try:
        import torch
        import torchvision
    except ImportError as error:
        parser.exit(1, f"Missing/broken PyTorch installation: {error}\nInstall requirements-cpu.txt or requirements-gpu.txt.\n")
    cuda_available = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_available:
        parser.exit(1, "CUDA was explicitly requested but is unavailable. Check the GPU build, driver, and Docker GPU access.\n")
    device = "cuda" if args.device != "cpu" and cuda_available else "cpu"
    report.update({
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "selected_device": device,
        "pillow": importlib.metadata.version("Pillow"),
    })
    if cuda_available:
        report["gpu_name"] = torch.cuda.get_device_name(0)
    if args.smoke_test:
        # Real tensor/autograd and TorchVision operator checks; no data/weights download.
        x = torch.ones((4, 4), device=device, requires_grad=True)
        loss = (x @ x).sum()
        loss.backward()
        if loss.item() != 64.0 or x.grad is None or not torch.isfinite(x.grad).all().item():
            raise RuntimeError("Tensor/autograd smoke check failed.")
        boxes = torch.tensor([[0, 0, 2, 2], [3, 3, 5, 5]], dtype=torch.float32, device=device)
        scores = torch.tensor([0.9, 0.8], device=device)
        if torchvision.ops.nms(boxes, scores, 0.5).tolist() != [0, 1]:
            raise RuntimeError("TorchVision compiled-operator smoke check failed.")
        report["smoke_test"] = "passed"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
