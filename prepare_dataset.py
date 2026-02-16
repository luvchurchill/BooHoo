import argparse
from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf


CLASS_MAPPING = {
    "belly_pain": "bp",
    "burping": "bu",
    "cold_hot": "ch",
    "discomfort": "dc",
    "hungry": "hu",
    "lonely": "lo",
    "scared": "sc",
    "tired": "ti",
    "unknown": "un",
}
SUPPORTED_EXTS = (".wav", ".mp3", ".caf", ".3gp", ".ogg", ".m4a")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def prepare_dataset(
    source_dir="Data/v3_organized",
    output_dir="processed_dataset",
    test_size=0.2,
    seed=42,
    unknown_strategy="cap",
    unknown_cap_ratio=0.35,
    stratify=True,
    min_per_class=5,
):
    """
    Prepare the dataset by organizing and splitting audio files.
    
    Args:
        source_dir (str): Source directory containing class folders
        output_dir (str): Output directory for processed dataset
        test_size (float): Proportion of data to use for testing
    """
    # Create output directories
    output_path = Path(output_dir)
    train_path = output_path / "train"
    test_path = output_path / "test"
    
    for path in [train_path, test_path]:
        path.mkdir(parents=True, exist_ok=True)
        
    # Create metadata list
    metadata = []
    rng = random.Random(seed)
    
    # Collect all files per class first so we can optionally rebalance unknown.
    source_path = Path(source_dir)
    class_files = {}
    for class_folder in source_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            audio_files = [
                p for p in class_folder.glob("*.*") if p.suffix.lower() in SUPPORTED_EXTS
            ]
            class_files[class_name] = audio_files

    if unknown_strategy == "drop":
        class_files["unknown"] = []
        print("Unknown handling: dropped all unknown samples.")
    elif unknown_strategy == "cap":
        unknown_files = class_files.get("unknown", [])
        known_total = sum(len(v) for k, v in class_files.items() if k != "unknown")
        max_unknown = int(round(known_total * unknown_cap_ratio))
        if len(unknown_files) > max_unknown:
            class_files["unknown"] = rng.sample(unknown_files, k=max_unknown)
            print(
                f"Unknown handling: capped unknown from {len(unknown_files)} "
                f"to {max_unknown} samples (ratio={unknown_cap_ratio:.2f} of known total)."
            )
        else:
            print(
                f"Unknown handling: cap requested but no trimming needed "
                f"({len(unknown_files)} <= {max_unknown})."
            )
    else:
        print("Unknown handling: keeping all unknown samples.")

    # Validate class counts and flatten all files for global split.
    all_rows = []
    for class_name, audio_files in class_files.items():
        if class_name not in CLASS_MAPPING:
            continue
        if class_name == "unknown" and unknown_strategy == "drop" and len(audio_files) == 0:
            continue
        if len(audio_files) < min_per_class:
            raise ValueError(
                f"Class '{class_name}' has {len(audio_files)} samples, below --min-per-class={min_per_class}."
            )
        class_code = CLASS_MAPPING[class_name]
        for audio_file in audio_files:
            all_rows.append((audio_file, class_name, class_code))

    if not all_rows:
        raise ValueError("No audio files found after filtering.")

    all_classes = [row[1] for row in all_rows]
    indices = list(range(len(all_rows)))
    stratify_labels = all_classes if stratify else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
    )

    # Remove stale generated WAVs so repeated runs stay consistent.
    for split_path in (train_path, test_path):
        for old_file in split_path.glob("*.wav"):
            old_file.unlink()

    split_items = [
        (train_idx, train_path, "train"),
        (test_idx, test_path, "test"),
    ]

    # Process and copy files
    for split_idx, split_path, split_name in split_items:
        for idx in split_idx:
            audio_file, class_name, class_code = all_rows[idx]
            # Load and resample audio to 16kHz
            try:
                y, sr = librosa.load(audio_file, sr=16000)

                # Generate new filename
                new_filename = f"{audio_file.stem}-{class_code}.wav"
                output_file = split_path / new_filename

                # Save processed audio
                sf.write(output_file, y, sr, subtype="PCM_16")

                # Add to metadata
                metadata.append(
                    {
                        "filename": new_filename,
                        "class": class_name,
                        "class_code": class_code,
                        "split": split_name,
                    }
                )

            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_path / 'metadata.csv', index=False)
    
    print(f"Dataset prepared successfully in {output_dir}")
    print("\nClass distribution:")
    print(metadata_df.groupby(['split', 'class']).size().unstack(fill_value=0))

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DeepInfant dataset splits")
    parser.add_argument("--source-dir", default="Data/v3_organized", help="Source class folder root")
    parser.add_argument("--output-dir", default="processed_dataset", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument(
        "--unknown-strategy",
        choices=["keep", "cap", "drop"],
        default="cap",
        help="How to handle the unknown class before splitting.",
    )
    parser.add_argument(
        "--unknown-cap-ratio",
        type=float,
        default=0.35,
        help="When using --unknown-strategy cap, keep at most ratio*known_total unknown samples.",
    )
    parser.add_argument(
        "--stratify",
        type=str2bool,
        default=True,
        help="Whether to use stratified global train/test split.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=5,
        help="Fail if any class has fewer than this many samples after filtering.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        unknown_strategy=args.unknown_strategy,
        unknown_cap_ratio=args.unknown_cap_ratio,
        stratify=args.stratify,
        min_per_class=args.min_per_class,
    )
