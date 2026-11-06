"""Compute ESM-2 embeddings: embed.py <dir> with a JSON list of sequences on stdin."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from nesso.data.esm import DEFAULT_ESM2_MODEL, extract_esm_embedding, setup_esm_model  # noqa: E402
from nesso.main import get_cache_path  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1])
    sequences = json.loads(sys.stdin.read())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = setup_esm_model(
        DEFAULT_ESM2_MODEL, device, cache_dir=get_cache_path() / "huggingface")
    for sequence in sequences:
        embedding = extract_esm_embedding(sequence, model, tokenizer)
        target = out / f"{hashlib.md5(sequence.encode('utf-8')).hexdigest()}.safetensors"
        staged = target.with_suffix(".partial")
        save_file({"embeddings": embedding}, staged, metadata={"sequence": sequence})
        staged.replace(target)


if __name__ == "__main__":
    main()
