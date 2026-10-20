"""Run a saved tabular conversation policy and emit turn traces as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rl import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, help="JSON artifact written by psychgym.conversation.rl")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    artifact = json.loads(args.policy.read_text(encoding="utf-8"))
    report = evaluate(artifact["q_table"], max_turns=artifact["metadata"]["max_turns"])
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
