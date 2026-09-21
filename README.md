# Install

```bash
uv venv --python 3.12
uv pip install "torch==2.13.0" setuptools wheel ninja

CARGO_HOME=/tmp/deploy-env-cargo \
CARGO_TARGET_DIR=/tmp/deploy-env-target \
uv sync
```