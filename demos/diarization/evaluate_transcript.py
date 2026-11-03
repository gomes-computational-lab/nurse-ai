"""Score speaker attribution against a two-speaker transcript, without timings.

Exact words are aligned with edit distance and speaker IDs are mapped globally.
Missing or misrecognized words are reported as reduced lexical coverage. This
is a transcript diagnostic, not time-based diarization error rate (DER).
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import permutations
import json
from pathlib import Path
import re


def words(text: str) -> list[str]:
    return [word.replace("'", "") for word in
            re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower().replace("\u2019", "'"))]


def read_text(path: Path) -> str:
    data = path.read_bytes()
    return data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def evaluate(reference: str, prediction: str) -> dict:
    expected = []
    speaker = None
    for line in reference.splitlines():
        if not line.strip():
            continue
        match = re.match(r"^(Nurse|Patient):\s*(.*)$", line, re.IGNORECASE)
        if match:
            speaker, text = match[1].upper(), match[2]
        elif speaker is not None:
            text = line
        else:
            raise ValueError("Reference must start with Nurse: or Patient:.")
        expected.extend((word, speaker) for word in words(text))
    if {speaker for _, speaker in expected} != {"NURSE", "PATIENT"}:
        raise ValueError("Reference must contain both Nurse: and Patient: turns.")

    observed = []
    for line in prediction.splitlines():
        match = re.match(r"^\[[^\]]+\]\s+(SPEAKER_\w+)(?:\s+\[[^\]]+\])?:\s*(.*)$", line)
        if match:
            observed.extend((word, match[1]) for word in words(match[2]))
    if not observed:
        raise ValueError("Prediction contains no timestamped speaker-text lines.")
    labels = sorted({speaker for _, speaker in observed} - {"SPEAKER_UNKNOWN"})
    if len(labels) > 2:
        raise ValueError("This evaluator supports at most two known speaker IDs.")

    # A monotonic minimum-edit alignment avoids counting ASR substitutions as
    # speaker errors. Omitted speech is visible in the per-speaker coverage.
    costs = [list(range(len(observed) + 1))]
    for i, (word, _) in enumerate(expected, 1):
        row = [i]
        for j, (other, _) in enumerate(observed, 1):
            row.append(min(costs[i - 1][j - 1] + (word != other),
                           costs[i - 1][j] + 1, row[j - 1] + 1))
        costs.append(row)
    aligned = []
    i, j = len(expected), len(observed)
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (expected[i - 1][0] != observed[j - 1][0]):
            if expected[i - 1][0] == observed[j - 1][0]:
                aligned.append((expected[i - 1][1], observed[j - 1][1]))
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1

    mappings = [dict(zip(labels, order)) for order in permutations(("NURSE", "PATIENT"), len(labels))]
    mapping = max(mappings, key=lambda candidate: sum(candidate.get(actual) == target
                                                     for target, actual in aligned))
    totals = Counter(speaker for _, speaker in expected)
    by_speaker = {}
    for speaker in ("NURSE", "PATIENT"):
        matches = [(target, actual) for target, actual in aligned if target == speaker]
        correct = sum(mapping.get(actual) == speaker for _, actual in matches)
        unknown = sum(actual == "SPEAKER_UNKNOWN" for _, actual in matches)
        by_speaker[speaker] = {
            "reference_words": totals[speaker],
            "matched_words": len(matches),
            "lexical_coverage": len(matches) / totals[speaker],
            "correct_speaker_words": correct,
            "wrong_speaker_words": len(matches) - correct - unknown,
            "unknown_speaker_words": unknown,
            "matched_word_speaker_accuracy": correct / len(matches) if matches else None,
        }
    correct = sum(mapping.get(actual) == target for target, actual in aligned)
    return {
        "metric": "Speaker attribution on lexically matched words; not time-based DER.",
        "speaker_mapping": mapping,
        "reference_words": len(expected),
        "prediction_words": len(observed),
        "matched_words": len(aligned),
        "lexical_coverage": len(aligned) / len(expected),
        "matched_word_speaker_accuracy": correct / len(aligned) if aligned else None,
        "by_speaker": by_speaker,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate(read_text(args.reference), read_text(args.prediction))
    except (OSError, ValueError, UnicodeError) as exc:
        parser.error(str(exc))
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
