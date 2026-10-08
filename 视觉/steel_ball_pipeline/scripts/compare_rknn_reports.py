from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the FP16 versus INT8 recall acceptance rule.")
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fp16 = json.loads(args.fp16.read_text(encoding="utf-8"))
    int8 = json.loads(args.int8.read_text(encoding="utf-8"))
    fp16_recall = float(fp16["subsets"]["all_ball_frames"]["recall"])
    int8_recall = float(int8["subsets"]["all_ball_frames"]["recall"])
    drop_points = (fp16_recall - int8_recall) * 100
    report = {
        "fp16_recall": fp16_recall,
        "int8_recall": int8_recall,
        "int8_drop_percentage_points": drop_points,
        "maximum_allowed_drop_percentage_points": 0.5,
        "selected_deployment": "int8" if drop_points <= 0.5 else "fp16",
        "passes_int8_rule": drop_points <= 0.5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
