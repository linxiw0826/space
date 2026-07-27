#!/usr/bin/env python3
"""Download source-native ADT files for the exact VSI-590K sequences.

The Dataset Explorer JSON contains temporary URLs. This tool never prints or
copies those URLs into its report. Downloads are resumable and SHA1-verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


DEFAULT_GROUPS = ("main_groundtruth",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-json", required=True, type=Path)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument(
        "--group", action="append", dest="groups", default=[]
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--sequence-limit", type=int)
    return parser.parse_args()


def sha1sum(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jobs(args: argparse.Namespace) -> list[dict]:
    manifest = json.loads(args.download_json.read_text(encoding="utf-8"))
    requested = [
        line.strip()
        for line in args.sequences.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.sequence_limit is not None:
        requested = requested[: args.sequence_limit]
    groups = tuple(args.groups or DEFAULT_GROUPS)
    absent = sorted(set(requested) - set(manifest["sequences"]))
    if absent:
        raise ValueError(f"Sequences absent from download JSON: {absent}")
    jobs = []
    for sequence in requested:
        available = manifest["sequences"][sequence]
        missing_groups = sorted(set(groups) - set(available))
        if missing_groups:
            raise ValueError(
                f"{sequence} is missing data groups {missing_groups}"
            )
        for group in groups:
            item = available[group]
            jobs.append(
                {
                    "sequence": sequence,
                    "group": group,
                    "filename": item["filename"],
                    "download_url": item["download_url"],
                    "file_size_bytes": int(item["file_size_bytes"]),
                    "sha1sum": item["sha1sum"],
                }
            )
    return jobs


def download(job: dict, args: argparse.Namespace) -> dict:
    destination = (
        args.output_dir / job["sequence"] / job["filename"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = job["file_size_bytes"]
    expected_sha1 = job["sha1sum"]
    if (
        destination.is_file()
        and destination.stat().st_size == expected_size
        and sha1sum(destination) == expected_sha1
    ):
        return {
            key: job[key]
            for key in (
                "sequence",
                "group",
                "filename",
                "file_size_bytes",
                "sha1sum",
            )
        } | {"status": "already_valid"}

    partial = destination.with_suffix(destination.suffix + ".part")
    error = None
    for attempt in range(args.max_retries):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(
                job["download_url"],
                headers=headers,
                stream=True,
                timeout=args.timeout,
            ) as response:
                if offset and response.status_code == 200:
                    offset = 0
                    partial.unlink(missing_ok=True)
                response.raise_for_status()
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size != expected_size:
                raise ValueError(
                    f"size={partial.stat().st_size}, expected={expected_size}"
                )
            actual_sha1 = sha1sum(partial)
            if actual_sha1 != expected_sha1:
                raise ValueError(
                    f"sha1={actual_sha1}, expected={expected_sha1}"
                )
            partial.replace(destination)
            return {
                key: job[key]
                for key in (
                    "sequence",
                    "group",
                    "filename",
                    "file_size_bytes",
                    "sha1sum",
                )
            } | {"status": "downloaded"}
        except Exception as current:
            error = current
            if attempt + 1 < args.max_retries:
                time.sleep(min(2**attempt, 30))
    return {
        "sequence": job["sequence"],
        "group": job["group"],
        "filename": job["filename"],
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
    }


def write_report(path: Path, jobs: list[dict], results: list[dict]) -> None:
    counts = Counter(result["status"] for result in results)
    report = {
        "schema_version": "vsi_adt_gold_download_v1",
        "requested_files": len(jobs),
        "requested_sequences": len({job["sequence"] for job in jobs}),
        "requested_bytes": sum(job["file_size_bytes"] for job in jobs),
        "completed_results": len(results),
        "status": dict(counts),
        "results": sorted(
            results, key=lambda item: (item["sequence"], item["group"])
        ),
        "security_note": "Temporary download URLs are intentionally omitted.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    jobs = load_jobs(args)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download, job, args): job for job in jobs
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            write_report(args.report_output, jobs, results)
            print(
                f"[{index}/{len(jobs)}] {result['sequence']} "
                f"{result['group']} {result['status']}",
                flush=True,
            )
    write_report(args.report_output, jobs, results)
    failures = [result for result in results if result["status"] == "error"]
    if failures:
        raise SystemExit(f"{len(failures)} downloads failed; see report")


if __name__ == "__main__":
    main()
