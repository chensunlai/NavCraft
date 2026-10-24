"""CPU-only, fixed student tokenizer for checking visible summary length."""

import json
from pathlib import Path
import sys


def count(text):
    from tokenizers import Tokenizer
    path = Path(__file__).resolve().parents[2] / "model/nas/Qwen3.5-9B/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


if __name__ == "__main__":
    print(json.dumps({"count": count(sys.stdin.read())}))
