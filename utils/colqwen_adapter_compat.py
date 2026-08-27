"""Compatibility helpers for loading ColQwen2 adapters with newer Transformers."""

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_OLD_LANGUAGE_LAYER_PATH = "model.layers."
_NEW_LANGUAGE_LAYER_PATH = "model.language_model.layers."
_ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
_ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def _uses_nested_language_model() -> bool:
    """Return whether the installed Qwen2-VL uses ``model.language_model``."""
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

    return "language_model" in Qwen2VLModel.__init__.__code__.co_names


def _rename_colqwen_adapter_key(
    key: str,
    source_layer_path: str,
    target_layer_path: str,
) -> str:
    """Rename one Qwen2-VL language-layer path without changing its tensor."""
    if target_layer_path in key:
        return key
    return key.replace(
        source_layer_path,
        target_layer_path,
        1,
    )


def _update_target_modules(
    adapter_config_path: Path,
    nested_language_model: bool,
) -> None:
    """Restrict LoRA injection to the renamed language-model hierarchy."""
    with adapter_config_path.open("r", encoding="utf-8") as config_file:
        adapter_config = json.load(config_file)

    target_modules = adapter_config.get("target_modules")
    if isinstance(target_modules, str):
        if nested_language_model and "(language_model)" not in target_modules:
            adapter_config["target_modules"] = target_modules.replace(
                "(model)",
                "(language_model)",
                1,
            )
        elif not nested_language_model and "(language_model)" in target_modules:
            adapter_config["target_modules"] = target_modules.replace(
                "(language_model)",
                "(model)",
                1,
            )

    with adapter_config_path.open("w", encoding="utf-8") as config_file:
        json.dump(adapter_config, config_file, ensure_ascii=False, indent=2)
        config_file.write("\n")


@contextmanager
def transformers_compatible_colqwen_adapter(adapter_path: str) -> Iterator[str]:
    """Yield an adapter path compatible with the installed Transformers hierarchy.

    ColQwen2 v1.0 was trained against Transformers 4.45, where Qwen2-VL language
    layers lived below ``model.layers``. Newer Transformers versions expose the
    same layers below ``model.language_model.layers``. If the supplied local
    adapter uses the old names, this function creates a temporary copy and only
    renames its state-dict keys; tensor values and the original checkpoint are
    left unchanged.
    """
    source_dir = Path(adapter_path).expanduser()
    if not source_dir.is_dir():
        # Preserve the existing behavior for Hub model IDs and let Transformers
        # resolve them. Local first-stage checkpoints take the conversion path.
        yield adapter_path
        return

    weights_path = source_dir / _ADAPTER_WEIGHTS_FILENAME
    config_path = source_dir / _ADAPTER_CONFIG_FILENAME
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "ColQwen adapter conversion requires adapter_model.safetensors and "
            f"adapter_config.json in {source_dir}"
        )

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required to convert the ColQwen adapter parameter names"
        ) from exc

    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())

    nested_language_model = _uses_nested_language_model()
    if nested_language_model:
        source_layer_path = _OLD_LANGUAGE_LAYER_PATH
        target_layer_path = _NEW_LANGUAGE_LAYER_PATH
    else:
        source_layer_path = _NEW_LANGUAGE_LAYER_PATH
        target_layer_path = _OLD_LANGUAGE_LAYER_PATH

    keys_to_rename = [
        key
        for key in keys
        if source_layer_path in key and target_layer_path not in key
    ]
    if not keys_to_rename:
        yield adapter_path
        return

    with tempfile.TemporaryDirectory(prefix="colqwen2_adapter_compat_") as temp_dir:
        compatible_dir = Path(temp_dir) / source_dir.name
        shutil.copytree(
            source_dir,
            compatible_dir,
            ignore=shutil.ignore_patterns(_ADAPTER_WEIGHTS_FILENAME),
        )

        converted_state_dict = {}
        with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata()
            for key in checkpoint.keys():
                converted_key = _rename_colqwen_adapter_key(
                    key,
                    source_layer_path,
                    target_layer_path,
                )
                if converted_key in converted_state_dict:
                    raise ValueError(
                        "ColQwen adapter key conversion produced a duplicate key: "
                        f"{converted_key}"
                    )
                converted_state_dict[converted_key] = checkpoint.get_tensor(key)

        save_file(
            converted_state_dict,
            str(compatible_dir / _ADAPTER_WEIGHTS_FILENAME),
            metadata=metadata,
        )
        _update_target_modules(
            compatible_dir / _ADAPTER_CONFIG_FILENAME,
            nested_language_model,
        )

        print(
            "Converted ColQwen adapter parameter names for this Transformers "
            f"version ({len(keys_to_rename)} tensors): "
            f"{source_layer_path} -> {target_layer_path}"
        )
        yield str(compatible_dir)
