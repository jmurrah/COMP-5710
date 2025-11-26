"""
COMP-5710 Workshop 12: Pairwise Testing
Author: Jacob Murrah
Date: 11/20/2025
"""

from collections import OrderedDict
from allpairspy import AllPairs

from pathlib import Path
import subprocess
import json

BANDIT_FLAGS = OrderedDict(
    {
        "a": ["file", "vuln"],
        "l": ["-l", "-ll", "-lll"],
        "i": ["-i", "-ii", "-iii"],
        "f": ["csv", "custom", "html", "json", "screen", "txt", "xml", "yaml"],
    }
)
RESULTS_DIR = Path("bandit-results")


def run_bandit_case(id, pair) -> dict[str, str]:
    command = [
        "bandit",
        "-r",
        "w12",
        "-a",
        pair.a,
        pair.l,
        pair.i,
        "-f",
        pair.f,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "id": id,
        "flags": pair,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"pair_{id}.txt").write_text(
        "\n".join(
            [
                f"CASE {id}",
                f"Command : {' '.join(command)}",
                f"Return  : {completed.returncode}",
                "--- STDOUT ---",
                completed.stdout.strip(),
                "--- STDERR ---",
                completed.stderr.strip(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    return record


if __name__ == "__main__":
    print("PAIRWISE:")
    for i, pairs in enumerate(AllPairs(BANDIT_FLAGS)):
        print("{:2d}: {}".format(i, pairs))

    all_records = []
    for i, pair in enumerate(AllPairs(BANDIT_FLAGS)):
        record = run_bandit_case(i, pair)
        all_records.append(record)

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"\nRecorded {len(all_records)} executions in '{RESULTS_DIR}'.")
    print(f"Summary JSON written to '{summary_path}'.")
