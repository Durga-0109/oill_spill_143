import importlib.util
import sys
from pathlib import Path
from typing import Any


class ModelRuntimeError(RuntimeError):
    """Raised when a supplied model handoff cannot provide the required output."""


def load_handoff_module(module_name: str, handoff_dir: Path) -> Any:
    module_path = handoff_dir / "inference.py"
    if not module_path.exists():
        raise ModelRuntimeError(f"Inference module missing: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ModelRuntimeError(f"Could not load inference module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(handoff_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def load_model_runtimes(binary_dir: Path, segmentation_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    binary_module = load_handoff_module("oilguard_binary_inference", binary_dir)
    segmentation_module = load_handoff_module("oilguard_segmentation_inference", segmentation_dir)

    binary_ready = callable(getattr(binary_module, "predict", None))
    binary_message = None
    if binary_ready:
        checkpoint_candidates = [
            binary_dir / "checkpoints" / "best_model.pth",
            binary_dir / "checkpoints" / "model.pth",
            binary_dir / "best_model.pth",
        ]
        if not any(candidate.exists() for candidate in checkpoint_candidates):
            binary_ready = False
            binary_message = f"Binary checkpoint missing under {binary_dir / 'checkpoints'}"

    segmentation_predict = getattr(segmentation_module, "predict", None)
    segmentation_ready = callable(segmentation_predict)
    segmentation_message = None
    if segmentation_ready:
        checkpoint_candidates = [
            segmentation_dir / "checkpoints" / "best_model.pth",
            segmentation_dir / "checkpoints" / "model.pth",
            segmentation_dir / "best_model.pth",
        ]
        if not any(candidate.exists() for candidate in checkpoint_candidates):
            segmentation_ready = False
            segmentation_message = f"Segmentation checkpoint missing under {segmentation_dir / 'checkpoints'}"
    elif not segmentation_message:
        segmentation_message = (
            "The supplied segmentation handoff does not expose a callable predict() implementation. "
            "A real segmentation checkpoint/inference contract is required."
        )

    status = {
        "binary": "ready" if binary_ready else "error",
        "segmentation": "ready" if segmentation_ready else "error",
        "binary_message": binary_message or ("Binary model ready." if binary_ready else "Binary model could not be initialized."),
        "segmentation_message": segmentation_message or "Segmentation model ready.",
    }
    return binary_module, segmentation_module, status
