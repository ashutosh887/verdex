from __future__ import annotations

import argparse
import os
import sys
import time

_CODE = os.path.dirname(os.path.abspath(__file__))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from pipeline.fusion import load_user_history, parse_history_flags  # noqa: E402
from pipeline.pregates import PHashStore  # noqa: E402
from pipeline.run import load_requirements_for, process_claim  # noqa: E402
from pipeline.validate import read_input_rows, validate_output, write_output_csv  # noqa: E402
from pipeline.vlm import make_client  # noqa: E402

_REPO = os.path.dirname(_CODE)


def run(dataset: str, input_csv: str, output_csv: str, limit, offline: bool) -> None:
    rows = read_input_rows(input_csv)
    if limit:
        rows = rows[:limit]
    requirements = load_requirements_for(dataset)
    history = load_user_history(os.path.join(dataset, "user_history.csv"))
    phash_store = PHashStore(os.path.join(_CODE, ".cache", "phash_store.json"))

    client = make_client()
    client.offline = offline

    outputs = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        hflags = parse_history_flags(history.get(row.user_id))
        pc = process_claim(
            client, row, requirements, dataset, hflags, phash_store=phash_store
        )
        outputs.append(pc.output)
        print(f"  [{i}/{len(rows)}] {row.user_id}: {pc.output.claim_status.value}", flush=True)
    phash_store.save()
    elapsed = time.time() - t0

    # Fail loud BEFORE writing: never emit a malformed output.csv.
    validate_output(outputs, rows)
    write_output_csv(output_csv, outputs)

    cost = client.cost_estimate()
    print(
        f"\nWrote {len(outputs)} rows -> {output_csv}\n"
        f"calls={cost['calls']} (billed={cost['billed_calls']}, cached={cost['cached_calls']})  "
        f"tokens={cost['input_tokens']:,}in/{cost['output_tokens']:,}out  "
        f"est_cost=${cost['est_cost_usd']}  wall={elapsed:.0f}s"
    )


def main():
    ap = argparse.ArgumentParser(description="Run the evidence-review pipeline and write output.csv.")
    ap.add_argument("--dataset", default=os.path.join(_REPO, "dataset"))
    ap.add_argument("--input", default=None, help="input CSV (default: <dataset>/claims.csv)")
    ap.add_argument("--output", default=os.path.join(_REPO, "output.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offline", action="store_true",
                    help="cache-only: a cache miss raises instead of billing the API")
    args = ap.parse_args()
    input_csv = args.input or os.path.join(args.dataset, "claims.csv")
    run(args.dataset, input_csv, args.output, args.limit, args.offline)


if __name__ == "__main__":
    main()
