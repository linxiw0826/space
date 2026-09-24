#!/usr/bin/env python3
"""Convert the official ScanQA_v1.0_val.json (a JSON array) into a jsonl file,
one record per line, for lmms-eval's `json` dataset loader."""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/data2/wlx/data/ScanQA/ScanQA_v1.0_val.json")
    parser.add_argument("--output", default="/data2/wlx/data/ScanQA/scanqa_val.jsonl")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    with open(args.output, "w") as f:
        for row in data:
            f.write(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "scene_id": row["scene_id"],
                        "question": row["question"],
                        "answers": row["answers"],
                        "object_ids": row.get("object_ids", []),
                        "object_names": row.get("object_names", []),
                    }
                )
                + "\n"
            )

    print(f"wrote {len(data)} rows to {args.output}")


if __name__ == "__main__":
    main()
