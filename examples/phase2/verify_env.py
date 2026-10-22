"""Verify Phase 2 / UniST ML environment (CUDA torch + IO deps).

Usage (from repo root):
  .venv-ml\\Scripts\\python.exe examples/phase2/verify_env.py
"""

from __future__ import annotations

import sys


def main() -> int:
    print(f"python: {sys.version}")
    print(f"executable: {sys.executable}")

    try:
        import torch
    except ImportError as exc:
        print(f"FAIL: torch not installed ({exc})")
        return 1

    print(f"torch: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"cuda_available: {cuda_ok}")
    if "+cpu" in torch.__version__.lower() or (not cuda_ok and "cu" not in torch.__version__.lower()):
        print("FAIL: CPU torch wheel detected. Reinstall cu124 torch (see examples/phase2/README.md step 5).")
        return 1
    if cuda_ok:
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")
        x = torch.randn(2, 3, device="cuda")
        y = (x @ x.T).sum().item()
        print(f"cuda_matmul_ok: True (checksum={y:.4f})")
    else:
        print("FAIL: CUDA build present but torch.cuda.is_available() is False (driver/runtime?).")
        return 1

    missing = []
    for name in ("timm", "xarray", "netCDF4", "tensorboard", "sklearn", "joblib"):
        try:
            __import__(name if name != "sklearn" else "sklearn")
            print(f"import {name}: ok")
        except ImportError:
            missing.append(name)
            print(f"import {name}: MISSING")

    try:
        import eda_toolkit  # noqa: F401

        print("import eda_toolkit: ok")
    except ImportError:
        print("import eda_toolkit: MISSING (run: uv pip install -e \".[spatial,kriging,ml]\")")
        missing.append("eda_toolkit")

    if missing:
        print(f"FAIL: missing packages: {', '.join(missing)}")
        return 1

    print("OK: phase2 ML environment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
