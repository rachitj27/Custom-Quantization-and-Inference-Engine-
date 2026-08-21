"""Benchmark every runtime and emit a markdown table.

Getting trustworthy latencies on a laptop took three corrections, all of which
are baked into how this script runs:

1. **One process per runtime.** PyTorch, ONNX Runtime and OpenVINO each build a
   thread pool sized to the core count. Loaded together they oversubscribe the
   CPU: ONNX FP32 measured 360 ms sharing a process, versus 34 ms alone.
2. **A settle gap between runtimes.** Even in separate processes, launching them
   back to back leaves the previous runtime's threads winding down. ONNX FP32
   measured 341 ms immediately after a PyTorch run, versus 34 ms after a pause.
3. **Report the minimum.** Interference only ever makes a run slower, so the
   fastest observed time is the best estimate of the runtime's real cost. The
   median is shown alongside it to expose the spread.

Accuracy is unaffected by any of this and is scored over the full test set.

    python benchmarks/run_all_benchmarks.py
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RUNTIMES = ["pytorch", "onnx_fp32", "onnx_int8", "ov_fp32", "ov_int8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--settle", type=float, default=8.0,
                    help="seconds to idle between runtimes so the previous "
                         "runtime's threads fully wind down")
    ap.add_argument("--engine-json", default=os.path.join(HERE, "engine_results.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results.md"))
    args = ap.parse_args()

    rows = []
    for i, key in enumerate(RUNTIMES):
        if i > 0:
            print(f"  (settling {args.settle:g}s)")
            time.sleep(args.settle)

        print(f"Benchmarking {key}...")
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "bench_one.py"), key,
             "--rounds", str(args.rounds), "--images", str(args.images)],
            capture_output=True, text=True, cwd=HERE)

        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("RESULT ")), None)
        if line is None:
            print(f"  failed:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
            continue

        result = json.loads(line[len("RESULT "):])
        if result.get("skipped"):
            print(f"  skipped ({key}: model file missing)")
            continue
        print(f"  {result['label']:<28} min {result['min_ms']:7.1f} ms   "
              f"mAP {result['map50']:.4f}")
        rows.append(result)

    # Fold in the custom C++ engine, measured separately by cpp_engine/run_both.sh.
    if args.engine_json and os.path.exists(args.engine_json):
        with open(args.engine_json) as f:
            rows.extend(json.load(f))

    lines = [
        "| Runtime | Precision | Best latency | Median | mAP@0.5 |",
        "|---------|-----------|--------------|--------|---------|",
    ]
    for r in rows:
        label = r["label"]
        precision = "INT8" if "INT8" in label else "FP32"
        name = label.replace(" INT8", "").replace(" FP32", "").strip()
        median = f"{r['median_ms']:.1f} ms" if r.get("median_ms") else "--"
        lines.append(f"| {name} | {precision} | {r['min_ms']:.1f} ms | "
                     f"{median} | {r['map50']:.4f} |")

    table = "\n".join(lines)
    print("\n" + table)

    with open(args.out, "w") as f:
        f.write("# Benchmark results\n\n")
        f.write(f"{args.rounds} rounds x {args.images} images per runtime, "
                f"each in its own process with a {args.settle:g}s settle gap; "
                f"mAP scored over the full test set.\n\n")
        f.write(table + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
