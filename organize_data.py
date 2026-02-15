import argparse
import csv
import hashlib
import itertools
import shutil
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".m4a", ".3gp", ".caf"}


def normalize_label(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


CLASS_ALIASES = {
    "bellypain": "belly_pain",
    "burping": "burping",
    "coldhot": "cold_hot",
    "discomfort": "discomfort",
    "hungry": "hungry",
    "lonely": "lonely",
    "scared": "scared",
    "tired": "tired",
    "unknown": "unknown",
    # Added datasets include these non-cry folders.
    "laugh": "unknown",
    "noise": "unknown",
    "silence": "unknown",
}

CANONICAL_CLASSES = [
    "belly_pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "lonely",
    "scared",
    "tired",
    "unknown",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def infer_label(path: Path, data_root: Path) -> str | None:
    current = path.parent
    while True:
        key = normalize_label(current.name)
        if key in CLASS_ALIASES:
            return CLASS_ALIASES[key]
        if current == data_root or current.parent == current:
            return None
        current = current.parent


def in_output_tree(path: Path, output_dir: Path) -> bool:
    try:
        path.relative_to(output_dir)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and deduplicate audio datasets under Data/")
    parser.add_argument("--data-root", default="Data")
    parser.add_argument("--output-dir", default="Data/v3_organized")
    parser.add_argument(
        "--conflict-policy",
        choices=["skip", "first"],
        default="skip",
        help="How to handle identical audio hashes that appear with different labels.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output directory before writing merged results.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cls in CANONICAL_CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    files = []
    skipped_unmapped = 0
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if in_output_tree(path.resolve(), output_dir):
            continue
        label = infer_label(path.resolve(), data_root)
        if label is None:
            skipped_unmapped += 1
            continue
        files.append((path.resolve(), label))

    files.sort(key=lambda item: str(item[0]).lower())

    grouped = {}
    summary = {
        "total_candidates": 0,
        "kept": 0,
        "duplicates_same_label": 0,
        "duplicates_label_conflict": 0,
        "conflicts_skipped": 0,
        "skipped_unmapped": skipped_unmapped,
    }
    kept_per_class = {cls: 0 for cls in CANONICAL_CLASSES}
    hashed_rows = []

    for src, label in files:
        summary["total_candidates"] += 1
        file_hash = sha256_file(src)
        rel_src = src.relative_to(data_root)
        source_bucket = rel_src.parts[0] if rel_src.parts else ""
        record = {
            "src": src,
            "label": label,
            "sha256": file_hash,
            "source_path": str(rel_src),
            "source_bucket": source_bucket,
        }
        grouped.setdefault(file_hash, []).append(record)
        hashed_rows.append(record)

    status_by_row_key = {}
    dest_by_hash = {}

    for file_hash, records in grouped.items():
        labels = {r["label"] for r in records}
        records = sorted(records, key=lambda r: r["source_path"].lower())

        if len(labels) > 1:
            summary["duplicates_label_conflict"] += len(records)
            if args.conflict_policy == "skip":
                summary["conflicts_skipped"] += len(records)
                for rec in records:
                    status_by_row_key[(rec["source_path"], rec["sha256"])] = ("conflict_skipped", "")
                continue

        keep = records[0]
        label = keep["label"]
        ext = keep["src"].suffix.lower()
        dest_name = f"{label}_{file_hash[:16]}{ext}"
        dest = output_dir / label / dest_name
        shutil.copy2(keep["src"], dest)
        dest_by_hash[file_hash] = str(dest.relative_to(output_dir))
        summary["kept"] += 1
        kept_per_class[label] += 1
        status_by_row_key[(keep["source_path"], keep["sha256"])] = ("kept", "")

        for rec in records[1:]:
            if rec["label"] == label:
                summary["duplicates_same_label"] += 1
                status = "duplicate_same_label"
            else:
                status = "duplicate_label_conflict"
            status_by_row_key[(rec["source_path"], rec["sha256"])] = (status, dest_by_hash[file_hash])

    rows = []
    for rec in sorted(hashed_rows, key=lambda r: (r["source_path"].lower(), r["sha256"])):
        status, duplicate_of = status_by_row_key.get((rec["source_path"], rec["sha256"]), ("unknown", ""))
        rows.append(
            {
                "source_path": rec["source_path"],
                "source_bucket": rec["source_bucket"],
                "class_label": rec["label"],
                "sha256": rec["sha256"],
                "status": status,
                "duplicate_of": duplicate_of,
            }
        )

    report_csv = output_dir / "dedup_report.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "source_bucket",
                "class_label",
                "sha256",
                "status",
                "duplicate_of",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_txt = output_dir / "dedup_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as handle:
        handle.write("Deduplication Summary\n")
        handle.write("=====================\n")
        handle.write(f"conflict_policy: {args.conflict_policy}\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")
        handle.write("\nKept per class\n")
        for cls in CANONICAL_CLASSES:
            handle.write(f"{cls}: {kept_per_class[cls]}\n")

    print("Deduplication complete.")
    print(f"Output directory: {output_dir}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Kept per class:")
    for cls in CANONICAL_CLASSES:
        print(f"  {cls}: {kept_per_class[cls]}")
    print(f"Report: {report_csv}")
    print(f"Summary: {summary_txt}")


if __name__ == "__main__":
    main()
