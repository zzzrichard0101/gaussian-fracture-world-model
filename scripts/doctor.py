"""Record the actual runtime; never equate driver CUDA with the active toolkit."""
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def main():
    import torch
    import warp as wp

    wp.init()
    packages = {}
    for name in ("torch", "torchvision", "numpy", "warp-lang", "taichi", "open3d", "scipy"):
        packages[name] = importlib.metadata.version(name)
    source = Path(os.environ["WM_SOURCE_ROOT"])
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_toolkit": command([str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"), "--version"]),
        "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"]),
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "warp_cuda_available": wp.is_cuda_available(),
        "warp_devices": [str(device) for device in wp.get_devices()],
        "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "upstream_commit": command(["git", "-C", str(source), "rev-parse", "HEAD"]),
        "submodules": command(["git", "-C", str(source), "submodule", "status", "--recursive"]),
        "runtime_root": os.environ["WM_RESEARCH_ROOT"],
    }
    output = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts/environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["torch_cuda_available"] or not report["warp_cuda_available"]:
        raise RuntimeError("Both PyTorch and Warp must have CUDA access")


if __name__ == "__main__":
    main()
