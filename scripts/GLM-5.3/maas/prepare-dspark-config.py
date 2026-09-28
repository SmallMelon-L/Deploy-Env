#!/usr/bin/env python3
"""Adapt the Red Hat GLM-5.3 Speculators checkpoint for SGLang."""

import json
from pathlib import Path


SOURCE = Path("/mnt/models2/GLM-5.3-speculator.dspark")
OUTPUT = Path("/mnt/models2/GLM-5.3-speculator.dspark-sglang")
TARGET = Path("/mnt/models2/GLM-5.3")


def main() -> None:
    source_config = json.loads((SOURCE / "config.json").read_text())
    target_config = json.loads((TARGET / "config.json").read_text())

    if source_config.get("architectures") != ["DSparkDraftModel"]:
        raise ValueError("Unexpected draft architecture")
    if source_config.get("speculators_model_type") != "dspark":
        raise ValueError("Source checkpoint is not a DSpark speculator")

    backbone = source_config["transformer_layer_config"]
    layer_ids = source_config["aux_hidden_state_layer_ids"]
    target_layers = target_config["num_hidden_layers"]
    if any(not 0 <= layer_id < target_layers for layer_id in layer_ids):
        raise ValueError("Draft target layer IDs are outside the target model")

    config = {
        **backbone,
        **{
            key: value
            for key, value in source_config.items()
            if key not in {"auto_map", "transformer_layer_config"}
        },
        "model_type": "qwen3",
        "architectures": ["DSparkDraftModel"],
        "target_layer_ids": layer_ids,
        "num_target_layers": target_layers,
    }
    if config["vocab_size"] != target_config["vocab_size"]:
        raise ValueError("Draft and target vocabularies differ")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    weights = OUTPUT / "model.safetensors"
    if weights.is_symlink():
        if weights.resolve() != (SOURCE / "model.safetensors").resolve():
            raise ValueError(f"Unexpected weight symlink: {weights}")
    elif weights.exists():
        raise ValueError(f"Refusing to replace existing weight file: {weights}")
    else:
        weights.symlink_to(SOURCE / "model.safetensors")

    (OUTPUT / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
