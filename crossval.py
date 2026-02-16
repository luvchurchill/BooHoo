import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from train import (
    DeepInfantDataset,
    DeepInfantModel,
    LABEL_MAP,
    build_criterion,
    configure_warning_filters,
    compute_class_weights,
    set_seed,
    str2bool,
    train_model,
)


CLASS_NAME_TO_CODE = {
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
SUPPORTED_EXTS = {".wav", ".mp3", ".caf", ".3gp", ".ogg", ".m4a"}


def collect_samples(source_dir, unknown_strategy, unknown_cap_ratio, min_per_class, seed):
    source_path = Path(source_dir)
    rng = random.Random(seed)

    class_files = {}
    for class_name, class_code in CLASS_NAME_TO_CODE.items():
        class_dir = source_path / class_name
        if not class_dir.exists() or not class_dir.is_dir():
            continue
        files = [p for p in class_dir.glob("*.*") if p.suffix.lower() in SUPPORTED_EXTS]
        class_files[class_name] = files

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
                f"to {max_unknown} samples (ratio={unknown_cap_ratio:.2f})."
            )
        else:
            print("Unknown handling: cap requested but no trimming needed.")
    else:
        print("Unknown handling: keeping all unknown samples.")

    samples = []
    labels = []
    for class_name, files in class_files.items():
        class_code = CLASS_NAME_TO_CODE[class_name]
        if class_name == "unknown" and unknown_strategy == "drop" and len(files) == 0:
            continue
        if len(files) < min_per_class:
            raise ValueError(
                f"Class '{class_name}' has {len(files)} samples, below --min-per-class={min_per_class}."
            )
        class_idx = LABEL_MAP[class_code]
        for file_path in files:
            samples.append(str(file_path))
            labels.append(class_idx)

    if not samples:
        raise ValueError("No samples found for cross-validation.")
    return samples, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-validation for DeepInfant training")
    parser.add_argument("--source-dir", default="Data/v3_organized")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    parser.add_argument("--checkpoint-prefix", default="deepinfant_cv")
    parser.add_argument("--output-csv", default="cv_results.csv")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_true")

    parser.add_argument(
        "--imbalance-mode",
        choices=["none", "sampler", "loss", "both"],
        default="loss",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "sqrt_inv", "inv"],
        default="sqrt_inv",
    )
    parser.add_argument("--max-class-weight", type=float, default=3.0)
    parser.add_argument(
        "--scheduler",
        choices=["none", "plateau", "cosine"],
        default="cosine",
    )
    parser.add_argument(
        "--monitor",
        choices=["val_macro_f1", "val_loss"],
        default="val_macro_f1",
    )
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument(
        "--show-warnings",
        type=str2bool,
        default=False,
        help="Show all Python warnings. By default, known noisy audio warnings are suppressed.",
    )
    parser.add_argument(
        "--warning-log-path",
        default="",
        help="Optional file path to append suppressed warning lines.",
    )
    parser.add_argument(
        "--no-progress",
        type=str2bool,
        default=True,
        help="Disable tqdm progress bars. Default is true for cleaner cross-validation logs.",
    )

    parser.add_argument(
        "--unknown-strategy",
        choices=["keep", "cap", "drop"],
        default="cap",
    )
    parser.add_argument("--unknown-cap-ratio", type=float, default=0.35)
    parser.add_argument("--min-per-class", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_warning_filters(
        show_warnings=args.show_warnings,
        warning_log_path=args.warning_log_path,
    )
    set_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples, labels = collect_samples(
        source_dir=args.source_dir,
        unknown_strategy=args.unknown_strategy,
        unknown_cap_ratio=args.unknown_cap_ratio,
        min_per_class=args.min_per_class,
        seed=args.seed,
    )
    labels_np = np.asarray(labels, dtype=np.int64)
    print(f"Total samples for CV: {len(samples)}")

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fold_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(samples)), labels_np), start=1):
        print(f"\n=== Fold {fold_idx}/{args.cv_folds} ===")
        fold_seed = args.seed + fold_idx
        set_seed(fold_seed)

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        train_labels = labels_np[train_idx].tolist()
        val_labels = labels_np[val_idx].tolist()

        train_dataset = DeepInfantDataset(samples=train_samples, labels=train_labels, transform=(not args.no_augment))
        val_dataset = DeepInfantDataset(samples=val_samples, labels=val_labels, transform=False)

        class_counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=len(LABEL_MAP))
        print("Fold train class counts:", class_counts.tolist())
        if args.class_weighting != "none":
            class_weights = compute_class_weights(
                class_counts,
                scheme=args.class_weighting,
                max_weight=args.max_class_weight,
            )
            print("Fold class weights:", [round(x, 4) for x in class_weights.tolist()])
        else:
            class_weights = torch.ones(len(LABEL_MAP), dtype=torch.float32)
            print("Fold class weights: uniform")

        sample_weights = class_weights[torch.tensor(train_labels, dtype=torch.long)].double()
        use_sampler = args.imbalance_mode in ("sampler", "both")
        sampler = None
        if use_sampler:
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            print("Imbalance handling: using WeightedRandomSampler")
        elif args.imbalance_mode == "none":
            print("Imbalance handling: disabled")
        else:
            print("Imbalance handling: sampler not enabled")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = DeepInfantModel()
        criterion = build_criterion(args, class_weights, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = None
        if args.scheduler == "plateau":
            mode = "max" if args.monitor == "val_macro_f1" else "min"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            )
        elif args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs),
                eta_min=1e-6,
            )

        checkpoint_path = f"{args.checkpoint_prefix}_fold{fold_idx}.pth"
        metrics = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=args.epochs,
            device=device,
            scheduler=scheduler,
            grad_clip=args.grad_clip,
            checkpoint_path=checkpoint_path,
            monitor=args.monitor,
            early_stop_patience=args.early_stop_patience,
            mixup_alpha=args.mixup_alpha,
            show_progress=(not args.no_progress),
        )
        fold_rows.append(
            {
                "fold": fold_idx,
                "seed": fold_seed,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "best_epoch": metrics["best_epoch"],
                "best_val_loss": metrics["best_val_loss"],
                "best_val_acc": metrics["best_val_acc"],
                "best_val_macro_f1": metrics["best_val_macro_f1"],
                "best_val_balanced_acc": metrics["best_val_balanced_acc"],
                "best_monitor_value": metrics["best_monitor_value"],
                "checkpoint_path": checkpoint_path,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [
        "best_epoch",
        "best_val_loss",
        "best_val_acc",
        "best_val_macro_f1",
        "best_val_balanced_acc",
        "best_monitor_value",
    ]
    mean_row = {"fold": "mean", "seed": "", "n_train": "", "n_val": "", "checkpoint_path": ""}
    std_row = {"fold": "std", "seed": "", "n_train": "", "n_val": "", "checkpoint_path": ""}
    for col in metric_cols:
        mean_row[col] = fold_df[col].mean()
        std_row[col] = fold_df[col].std(ddof=0)

    out_df = pd.concat([fold_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved CV report to {args.output_csv}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
