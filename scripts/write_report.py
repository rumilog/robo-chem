"""
Write the experiment report. Runs never write one by themselves: this is the
only way a report gets written, and only when you run it.

With no arguments it finds everything itself:
  * the experiment: the newest run record in experiments/ that moved the
    robot (dry runs are skipped), chained back through every --resume to
    the session that started it, so the report covers the whole experiment;
  * the console log: any experiments/*.log written since the experiment
    started (what `... | tee experiments/<name>.log` kept). For each failed
    step the skill's own lines are quoted, which the record keeps only as a
    one-line error;
  * the run's own photos: run_experiment.py photographs the bench after
    every pour (experiments/photos_<time>/, paths in the step records), so
    the model sees each cup's colour before it was poured out;
  * the cameras: the model gets overview photos of the bench taken now, and
    while it writes it can ask for more -- look_at("clear cup a") for a
    close-up of one object (the colour of the liquid in it), or fresh views
    of the bench. Read-only; the robot is never commanded. The photos are
    the only source for the colours, so run this before the bench is
    cleared, or pass --no-photos.

The model (the strong tier, gpt-4.1) writes the prose, always in English; the
step table, the times and the measured values come straight from the records.

Output: experiments/report_<time>/report.md, every photo the model looked at,
facts.json (with the camera calls it made).

Usage:
    source scripts/env.sh
    python scripts/write_report.py                       # newest experiment, cameras on
    python scripts/write_report.py --no-photos           # bench already cleared
    python scripts/write_report.py --note "sub-task 6 was a false contact reading"
    python scripts/write_report.py experiments/agent_run_20261015_174505.json
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochem.agents import config
from robochem.agents.reporter import (BenchCamera, find_console_logs, latest_record,
                                      load_run_chain, write_report)


def open_cameras(cameras, bench_name):
    """The report writer's cameras (and object locating), or None if they cannot be set up."""
    try:
        from robochem.vision import VisionSystem
        from robochem.vision.bench_manifest import BENCHES
        vision = VisionSystem(cameras={}, bench=BENCHES.get(bench_name),
                              grounding_url=os.environ.get("GROUNDING_URL",
                                                           "http://127.0.0.1:5005"))
        return BenchCamera(vision, cameras=cameras, color_order="bgr")
    except Exception as exc:
        print(f"No cameras ({type(exc).__name__}: {exc}); writing from the records only.")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", nargs="?", default=None,
                        help="A run record (default: the newest one that moved the robot)")
    parser.add_argument("--no-photos", action="store_true",
                        help="No cameras: write from the records only (e.g. the bench "
                             "has been cleared)")
    parser.add_argument("--no-console", action="store_true",
                        help="Do not quote the console log")
    parser.add_argument("--cameras", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--note", action="append", default=[], metavar="TEXT",
                        help="An operator remark for the report, e.g. a failure's real "
                             "cause; repeat for several")
    parser.add_argument("--experiments", default=str(ROOT / "experiments"),
                        help="Where the run records and console logs are")
    parser.add_argument("--bench", default="magic_beaker")
    args = parser.parse_args()

    config.ensure_loaded()
    record = Path(args.record) if args.record else latest_record(args.experiments)
    if record is None:
        print(f"No run record in {args.experiments} that moved the robot.")
        return 1
    records, paths = load_run_chain(record)
    last = records[-1]
    print(f"Experiment: {len(records)} session(s), started {records[0].get('started')}, "
          f"outcome {last.get('outcome')}")
    for p in paths:
        print(f"  record {p}")

    logs = [] if args.no_console else find_console_logs(record.parent, records)
    for p in logs:
        print(f"  console log {p}")

    camera = None
    if not args.no_photos:
        try:
            ended = (time.mktime(time.strptime(last.get("started"), "%Y-%m-%d %H:%M:%S"))
                     + float(last.get("seconds") or 0))
            minutes = (time.time() - ended) / 60.0
            if minutes > 30:
                print(f"Note: the experiment ended {minutes:.0f} min ago. The photos show the "
                      f"bench as it is now; if it was changed since, rerun with --no-photos.")
        except (TypeError, ValueError):
            pass
        camera = open_cameras(args.cameras, args.bench)

    print("Writing the report (the model may look at the bench as it goes)...")
    path = write_report(records, camera=camera, record_paths=paths, out_dir=args.experiments,
                        notes=args.note, console_logs=logs)
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
