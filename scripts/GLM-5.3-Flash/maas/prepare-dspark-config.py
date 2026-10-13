#!/usr/bin/env python3
"""Adapt the Red Hat GLM-5.3-Flash Speculators checkpoint for SGLang."""

import json
import os
from pathlib import Path


SOURCE = Path(os.environ.get("DRAFT_SOURCE_PATH", "/mnt/models2/GLM-5.3-Flash-speculator.dspark"))
OUTPUT = Path(os.environ.get("DRAFT_MODEL_PATH", "/mnt/models2/GLM-5.3-Flash-speculator.dspark-sglang"))
TARGET = Path(os.environ.get("MODEL_PATH", "/mnt/models2/GLM-5.3-Flash"))


def main() -> None:
    source = json.loads((SOURCE / "config.json").read_text())
    target = json.loads((TARGET / "config.json").read_text())
    if target.get("architectures") != ["Glm5NextForConditionalGeneration"]:
        raise ValueError("Target checkpoint is not GLM-5.3-Flash")
    target = {**target, **target.get("text_config", {})}
    if source.get("architectures") != ["DSparkDraftModel"] or source.get("speculators_model_type") != "dspark":
        raise ValueError("Source checkpoint is not a DSpark speculator")
    verifier = source["speculators_config"]["verifier"]["architectures"]
    if verifier != ["Glm5NextForConditionalGeneration"]:
        raise ValueError("Draft checkpoint is not for GLM-5.3-Flash")
    # Speculators stores layer-input IDs; SGLang's capture hook adds one internally.
    layers = [layer - 1 for layer in source["aux_hidden_state_layer_ids"]]
    if any(not 0 <= layer < target["num_hidden_layers"] for layer in layers):
        raise ValueError("Draft target layer IDs are outside the target model")
    config = {
        **source["transformer_layer_config"],
        **{key: value for key, value in source.items() if key not in {"auto_map", "transformer_layer_config"}},
        "model_type": "qwen3",
        "target_layer_ids": layers,
        "num_target_layers": target["num_hidden_layers"],
    }
    if config["vocab_size"] != target["vocab_size"]:
        raise ValueError("Draft and target vocabularies differ")
    weights = SOURCE / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if SOURCE.resolve() == OUTPUT.resolve():
        raise ValueError("Output must differ from the original checkpoint directory")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    link = OUTPUT / weights.name
    if link.is_symlink():
        if link.resolve() != weights.resolve():
            raise ValueError(f"Unexpected weight symlink: {link}")
    elif link.exists():
        raise ValueError(f"Refusing to replace existing weight file: {link}")
    else:
        link.symlink_to(weights.resolve())
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
