from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


class ImportFailed(RuntimeError):
    def __init__(self, job_path: Path, attempts: int):
        super().__init__(f"import failed after {attempts} attempts: {job_path}")
        self.job_path = job_path
        self.attempts = attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and import completed job directories with retry."
    )
    parser.add_argument(
        "patterns",
        nargs="+",
        help="Job directory glob(s), for example 'jobs/kumo*__batch1__phase2'.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum import attempts per job. Default 0 means retry until success.",
    )
    return parser.parse_args()


def discover_job_paths(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()

    for pattern in patterns:
        matched = sorted(Path().glob(pattern))
        for path in matched:
            if path.is_dir():
                paths.add(path)

    return sorted(paths)


def run_tbx(command: str, job_path: Path) -> int:
    args = ["uv", "run", "tbx", command, "--job-path", str(job_path)]
    print(f"$ {' '.join(args)}", flush=True)
    return subprocess.run(args).returncode


def import_with_retry(job_path: Path, max_attempts: int) -> int:
    attempt = 1

    while True:
        status = run_tbx("import", job_path)
        if status == 0:
            print(f"import succeeded: {job_path}", flush=True)
            return attempt

        if max_attempts > 0 and attempt >= max_attempts:
            raise ImportFailed(job_path, attempt)

        sleep_sec = min(10.0 * 2 ** (attempt - 1), 300.0)
        print(
            f"import failed: {job_path} attempt {attempt}; "
            f"retrying in {sleep_sec:.1f}s",
            flush=True,
        )
        time.sleep(sleep_sec)
        attempt += 1


def main() -> None:
    args = parse_args()
    job_paths = discover_job_paths(args.patterns)

    if not job_paths:
        raise SystemExit(f"no job directories matched: {', '.join(args.patterns)}")

    print(f"found {len(job_paths)} job(s)", flush=True)

    imported: list[tuple[Path, int]] = []
    validate_failures: list[Path] = []
    import_failures: list[tuple[Path, int]] = []
    for index, job_path in enumerate(job_paths, start=1):
        print(f"\n[{index}/{len(job_paths)}] validating {job_path}", flush=True)
        validate_status = run_tbx("validate", job_path)
        if validate_status != 0:
            print(f"validate failed, skipping import: {job_path}", flush=True)
            validate_failures.append(job_path)
            continue

        print(f"[{index}/{len(job_paths)}] importing {job_path}", flush=True)
        try:
            attempts = import_with_retry(job_path=job_path, max_attempts=args.max_attempts)
            imported.append((job_path, attempts))
        except ImportFailed as exc:
            print(str(exc), flush=True)
            import_failures.append((exc.job_path, exc.attempts))

    print("\nsummary:", flush=True)
    print(f"  imported: {len(imported)}", flush=True)
    print(f"  validate failures: {len(validate_failures)}", flush=True)
    print(f"  import failures: {len(import_failures)}", flush=True)

    retried = [(path, attempts) for path, attempts in imported if attempts > 1]
    if retried:
        print("\nimports that needed retry:", flush=True)
        for path, attempts in retried:
            print(f"  {path} ({attempts} attempts)", flush=True)

    if validate_failures:
        print("\nvalidate failures:", flush=True)
        for path in validate_failures:
            print(f"  {path}", flush=True)

    if import_failures:
        print("\nimport failures:", flush=True)
        for path, attempts in import_failures:
            print(f"  {path} ({attempts} attempts)", flush=True)

    if validate_failures or import_failures:
        raise SystemExit(1)

    print("\nall jobs imported", flush=True)


if __name__ == "__main__":
    main()
