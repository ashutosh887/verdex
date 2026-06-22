from __future__ import annotations

import csv
from typing import List, Union

from .schema import INPUT_COLUMNS, OUTPUT_COLUMNS, InputRow, OutputRow


class OutputValidationError(Exception):
    pass


def read_input_rows(path: str) -> List[InputRow]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != INPUT_COLUMNS:
            raise OutputValidationError(
                f"input header mismatch: got {reader.fieldnames}, expected {INPUT_COLUMNS}"
            )
        return [InputRow(**row) for row in reader]


def read_labeled_rows(path: str) -> List[OutputRow]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != OUTPUT_COLUMNS:
            raise OutputValidationError(
                f"labeled header mismatch: got {reader.fieldnames}, expected {OUTPUT_COLUMNS}"
            )
        return [OutputRow(**row) for row in reader]


def write_output_csv(path: str, rows: List[OutputRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_csv_dict())


def _uid(row: Union[OutputRow, dict]) -> str:
    return row.user_id if isinstance(row, OutputRow) else row.get("user_id", "?")


def validate_output(output_rows: List[OutputRow], input_rows: List[InputRow]) -> None:
    errors: List[str] = []

    if len(output_rows) != len(input_rows):
        errors.append(f"row count: {len(output_rows)} output vs {len(input_rows)} input")

    for i, row in enumerate(output_rows):
        try:
            payload = row.model_dump() if isinstance(row, OutputRow) else row
            OutputRow(**payload)
        except Exception as e:
            errors.append(f"row {i} ({_uid(row)}): schema invalid: {e}")

    for i, (out, inp) in enumerate(zip(output_rows, input_rows)):
        d = out.to_csv_dict() if isinstance(out, OutputRow) else out
        for col in INPUT_COLUMNS:
            ov = d.get(col)
            iv = getattr(inp, col)
            iv = iv.value if hasattr(iv, "value") else iv
            if ov != iv:
                errors.append(f"row {i}: {col} passthrough mismatch: {ov!r} != {iv!r}")

    if errors:
        raise OutputValidationError(
            f"output validation failed with {len(errors)} error(s):\n  - " + "\n  - ".join(errors)
        )
