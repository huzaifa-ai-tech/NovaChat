import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "training_data.json"


def format_example(example: dict) -> str:
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['response']}\n"
    )


def validate_dataset(path=DEFAULT_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    required = {"instruction", "response"}
    for i, item in enumerate(data):
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Item {i} is missing keys: {missing}")
        if not item["instruction"].strip() or not item["response"].strip():
            raise ValueError(f"Item {i} has empty text")

    print(f"Dataset OK: {len(data)} items at {path}")
    return data


def build_training_file(path=DEFAULT_PATH, output=None):
    data = validate_dataset(path)
    if output is None:
        output = path.parent / "training_text.txt"
    with open(output, "w", encoding="utf-8") as f:
        for item in data:
            f.write(format_example(item) + "\n")
    print(f"Training text written to {output}")


if __name__ == "__main__":
    build_training_file()
