import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from utils.colqwen_adapter_compat import transformers_compatible_colqwen_adapter


class ColQwenAdapterCompatTest(unittest.TestCase):
    def test_old_language_layer_keys_are_renamed_without_changing_values(self):
        old_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        new_key = (
            "base_model.model.model.language_model.layers.0."
            "self_attn.q_proj.lora_A.weight"
        )
        projection_key = "base_model.model.custom_text_proj.lora_A.weight"
        old_value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        projection_value = torch.ones(2, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "colqwen-adapter"
            source_dir.mkdir()
            save_file(
                {
                    old_key: old_value,
                    projection_key: projection_value,
                },
                str(source_dir / "adapter_model.safetensors"),
            )
            with (source_dir / "adapter_config.json").open(
                "w",
                encoding="utf-8",
            ) as config_file:
                json.dump(
                    {
                        "target_modules": (
                            ".*(model).*(q_proj).*$|.*(custom_text_proj).*$"
                        )
                    },
                    config_file,
                )

            with transformers_compatible_colqwen_adapter(str(source_dir)) as result:
                compatible_dir = Path(result)
                self.assertNotEqual(compatible_dir, source_dir)
                self.assertTrue(compatible_dir.is_dir())

                with safe_open(
                    str(compatible_dir / "adapter_model.safetensors"),
                    framework="pt",
                    device="cpu",
                ) as checkpoint:
                    self.assertNotIn(old_key, checkpoint.keys())
                    self.assertIn(new_key, checkpoint.keys())
                    self.assertIn(projection_key, checkpoint.keys())
                    self.assertTrue(torch.equal(checkpoint.get_tensor(new_key), old_value))
                    self.assertTrue(
                        torch.equal(
                            checkpoint.get_tensor(projection_key),
                            projection_value,
                        )
                    )

                with (compatible_dir / "adapter_config.json").open(
                    "r",
                    encoding="utf-8",
                ) as config_file:
                    compatible_config = json.load(config_file)
                self.assertIn(
                    "(language_model)",
                    compatible_config["target_modules"],
                )

            self.assertFalse(compatible_dir.exists())

            with safe_open(
                str(source_dir / "adapter_model.safetensors"),
                framework="pt",
                device="cpu",
            ) as original_checkpoint:
                self.assertIn(old_key, original_checkpoint.keys())


if __name__ == "__main__":
    unittest.main()
