import argparse
from pathlib import Path
import warnings

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


LABEL_MAP = {
    "bp": 0,  # belly pain
    "bu": 1,  # burping
    "ch": 2,  # cold/hot
    "dc": 3,  # discomfort
    "hu": 4,  # hungry
    "lo": 5,  # lonely
    "sc": 6,  # scared
    "ti": 7,  # tired
    "un": 8,  # unknown
}

SUPPORTED_AUDIO_EXTS = {".wav", ".caf", ".3gp", ".mp3", ".ogg", ".m4a"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def _should_quiet_warning(message, category, filename):
    text = str(message)
    if category is UserWarning and "PySoundFile failed. Trying audioread instead" in text:
        return True
    if category is FutureWarning and "__audioread_load" in text and "librosa" in str(filename):
        return True
    return False


def configure_warning_filters(show_warnings=False, warning_log_path=""):
    if show_warnings:
        return

    original_showwarning = warnings.showwarning
    log_path = warning_log_path.strip()

    def custom_showwarning(message, category, filename, lineno, file=None, line=None):
        if _should_quiet_warning(message, category, filename):
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{category.__name__}: {message} "
                        f"({filename}:{lineno})\n"
                    )
            return
        original_showwarning(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = custom_showwarning


class DeepInfantDataset(Dataset):
    def __init__(self, data_dir=None, transform=None, samples=None, labels=None):
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.transform = transform
        # Backward-compatible behavior: callers currently pass bool for augmentation.
        self.augment = bool(transform) if isinstance(transform, bool) else False
        self.samples = []
        self.labels = []
        self.label_map = LABEL_MAP.copy()

        if samples is not None:
            if labels is None:
                raise ValueError("labels must be provided when samples are passed explicitly")
            if len(samples) != len(labels):
                raise ValueError("samples and labels must have the same length")
            self.samples = [str(s) for s in samples]
            self.labels = [int(y) for y in labels]
            return

        if self.data_dir is None:
            raise ValueError("data_dir is required when samples are not provided")

        # Load metadata if available
        metadata_file = self.data_dir.parent / "metadata.csv"
        if metadata_file.exists():
            self._load_from_metadata(metadata_file)
        else:
            self._load_dataset()

    def _load_from_metadata(self, metadata_file):
        df = pd.read_csv(metadata_file)
        for _, row in df.iterrows():
            if row["split"] != self.data_dir.name:  # train/test
                continue
            class_code = row["class_code"]
            if class_code not in self.label_map:
                continue
            audio_path = self.data_dir / row["filename"]
            if audio_path.exists():
                self.samples.append(str(audio_path))
                self.labels.append(self.label_map[class_code])

    def _load_dataset(self):
        for audio_file in self.data_dir.glob("*.*"):
            if audio_file.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
                continue
            label = audio_file.stem.split("-")[-1][:2]  # reason code suffix
            if label in self.label_map:
                self.samples.append(str(audio_file))
                self.labels.append(self.label_map[label])

    def _process_audio(self, audio_path):
        waveform, sample_rate = librosa.load(audio_path, sr=16000)

        # Augmentation is only enabled for train datasets.
        if self.augment:
            # Random time shift (-100ms to 100ms)
            shift = np.random.randint(-1600, 1600)
            if shift > 0:
                waveform = np.pad(waveform, (shift, 0))[: len(waveform)]
            else:
                waveform = np.pad(waveform, (0, -shift))[(-shift):]

            # Random noise injection
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 0.005, len(waveform))
                waveform = waveform + noise

        # Ensure consistent length (7 seconds)
        target_length = 7 * 16000
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))

        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            fmin=20,
            fmax=8000,
        )

        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-6)

        return torch.FloatTensor(mel_spec)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel_spec = self._process_audio(self.samples[idx])
        label = self.labels[idx]
        if callable(self.transform):
            mel_spec = self.transform(mel_spec)
        return mel_spec, label


class DeepInfantModel(nn.Module):
    def __init__(self, num_classes=9):
        super(DeepInfantModel, self).__init__()

        # With n_mels=80 and 3x MaxPool2d(2), frequency bins become 80 -> 40 -> 20 -> 10.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(
            input_size=256 * 10,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, -1, 256 * 10)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.classifier(x)


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def compute_class_weights(class_counts, scheme="sqrt_inv", max_weight=3.0):
    counts = np.asarray(class_counts, dtype=np.float64)
    weights = np.ones_like(counts, dtype=np.float64)
    nonzero = counts > 0
    if not nonzero.any() or scheme == "none":
        return torch.tensor(weights, dtype=torch.float32)

    if scheme == "inv":
        weights[nonzero] = 1.0 / counts[nonzero]
    elif scheme == "sqrt_inv":
        weights[nonzero] = 1.0 / np.sqrt(counts[nonzero])
    else:
        raise ValueError(f"Unsupported class weighting scheme: {scheme}")

    weights = weights / weights[nonzero].mean()
    weights = np.clip(weights, 1.0 / max_weight, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _mixup_batch(inputs, labels, alpha):
    if alpha <= 0:
        return inputs, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed_inputs, labels, labels[index], lam


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=60,
    device="cuda",
    scheduler=None,
    grad_clip=1.0,
    checkpoint_path="deepinfant.pth",
    monitor="val_macro_f1",
    early_stop_patience=10,
    mixup_alpha=0.0,
    show_progress=True,
):
    model = model.to(device)
    monitor_mode = "min" if monitor == "val_loss" else "max"
    if monitor_mode == "max":
        best_monitor_value = float("-inf")
    else:
        best_monitor_value = float("inf")
    epochs_without_improve = 0
    best_metrics = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=(not show_progress)
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            inputs, labels_a, labels_b, lam = _mixup_batch(inputs, labels, mixup_alpha)

            optimizer.zero_grad()
            outputs = model(inputs)
            if mixup_alpha > 0:
                loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss_avg = train_loss / max(1, len(train_loader))
        train_acc = 100.0 * train_correct / max(1, train_total)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_preds.extend(predicted.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_loss_avg = val_loss / max(1, len(val_loader))
        val_acc = 100.0 * val_correct / max(1, val_total)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_balanced_acc = recall_score(val_labels, val_preds, average="macro", zero_division=0)
        monitor_value = val_macro_f1 if monitor == "val_macro_f1" else val_loss_avg

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_value)
            else:
                scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Val Macro-F1: {val_macro_f1:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}")
        print(f"Monitor ({monitor}): {monitor_value:.6f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if monitor_mode == "max":
            is_better = monitor_value > best_monitor_value
        else:
            is_better = monitor_value < best_monitor_value

        if is_better:
            best_monitor_value = monitor_value
            epochs_without_improve = 0
            best_metrics = {
                "best_epoch": epoch + 1,
                "best_train_loss": train_loss_avg,
                "best_train_acc": train_acc,
                "best_val_loss": val_loss_avg,
                "best_val_acc": val_acc,
                "best_val_macro_f1": val_macro_f1,
                "best_val_balanced_acc": val_balanced_acc,
                "best_monitor_value": monitor_value,
            }
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Saved improved checkpoint to {checkpoint_path} "
                f"({monitor}={monitor_value:.4f}, val_macro_f1={val_macro_f1:.4f}, val_acc={val_acc:.2f}%)"
            )
        else:
            epochs_without_improve += 1

        if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1} "
                f"(no {monitor} improvement for {early_stop_patience} epochs)."
            )
            break

    if best_metrics is None:
        best_metrics = {
            "best_epoch": 0,
            "best_train_loss": float("nan"),
            "best_train_acc": float("nan"),
            "best_val_loss": float("nan"),
            "best_val_acc": float("nan"),
            "best_val_macro_f1": float("nan"),
            "best_val_balanced_acc": float("nan"),
            "best_monitor_value": float("nan"),
        }
    return best_metrics


def parse_seed_list(seed, seed_list_text):
    if seed_list_text is None or seed_list_text.strip() == "":
        return [seed]
    return [int(x.strip()) for x in seed_list_text.split(",") if x.strip()]


def build_criterion(args, class_weights, device):
    use_loss_weights = args.imbalance_mode in ("loss", "both")
    weight = None
    if use_loss_weights and class_weights is not None and args.class_weighting != "none":
        weight = class_weights.to(device)
        print("Imbalance handling: using class-weighted loss")
    elif args.imbalance_mode in ("loss", "both"):
        print("Imbalance handling: class-weighted loss unavailable, using unweighted loss")

    if args.focal_gamma > 0:
        print(f"Loss: focal cross-entropy (gamma={args.focal_gamma}, label_smoothing={args.label_smoothing})")
        return FocalCrossEntropyLoss(
            gamma=args.focal_gamma,
            weight=weight,
            label_smoothing=args.label_smoothing,
        )

    print(f"Loss: cross-entropy (label_smoothing={args.label_smoothing})")
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepInfant model")
    parser.add_argument("--train-dir", default="processed_dataset/train")
    parser.add_argument("--val-dir", default="processed_dataset/test")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-list", default="", help="Comma-separated seed list, e.g. 42,43")
    parser.add_argument("--checkpoint-path", default="deepinfant.pth")
    parser.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    parser.add_argument(
        "--imbalance-mode",
        choices=["none", "sampler", "loss", "both"],
        default="loss",
        help="How to handle class imbalance in training set.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "sqrt_inv", "inv"],
        default="sqrt_inv",
        help="Class weighting function used by imbalance modes.",
    )
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=3.0,
        help="Clip absolute class weights to avoid unstable extremes.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "plateau", "cosine"],
        default="cosine",
        help="LR scheduler strategy.",
    )
    parser.add_argument(
        "--monitor",
        choices=["val_macro_f1", "val_loss"],
        default="val_macro_f1",
        help="Validation metric to monitor for checkpointing and early stopping.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop after this many epochs without monitor improvement (0 disables).",
    )
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
        default=False,
        help="Disable tqdm progress bars.",
    )
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentations.")
    return parser.parse_args()


def run_single_seed(args, seed, checkpoint_path, device):
    set_seed(seed)

    train_dataset = DeepInfantDataset(args.train_dir, transform=(not args.no_augment))
    val_dataset = DeepInfantDataset(args.val_dir, transform=False)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_labels = np.array(train_dataset.labels, dtype=np.int64)
    num_classes = len(train_dataset.label_map)
    class_counts = np.bincount(train_labels, minlength=num_classes)
    print("Train class counts:", class_counts.tolist())

    if args.class_weighting != "none":
        class_weights = compute_class_weights(
            class_counts,
            scheme=args.class_weighting,
            max_weight=args.max_class_weight,
        )
        print("Class weights:", [round(x, 4) for x in class_weights.tolist()])
    else:
        class_weights = torch.ones(num_classes, dtype=torch.float32)
        print("Class weights: uniform")

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
    return metrics


def main():
    args = parse_args()
    configure_warning_filters(
        show_warnings=args.show_warnings,
        warning_log_path=args.warning_log_path,
    )
    seeds = parse_seed_list(args.seed, args.seed_list)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training seeds: {seeds}")

    all_results = []
    ckpt_path = Path(args.checkpoint_path)
    for seed in seeds:
        if len(seeds) == 1:
            run_ckpt = str(ckpt_path)
        else:
            run_ckpt = str(ckpt_path.with_name(f"{ckpt_path.stem}_seed{seed}{ckpt_path.suffix}"))
        print(f"\n=== Seed {seed} | checkpoint: {run_ckpt} ===")
        metrics = run_single_seed(args, seed, run_ckpt, device)
        metrics["seed"] = seed
        metrics["checkpoint_path"] = run_ckpt
        all_results.append(metrics)

    if len(all_results) > 1:
        macro_f1_values = np.array([r["best_val_macro_f1"] for r in all_results], dtype=np.float64)
        balanced_values = np.array([r["best_val_balanced_acc"] for r in all_results], dtype=np.float64)
        print("\n=== Multi-seed summary ===")
        print(f"Best Val Macro-F1: mean={macro_f1_values.mean():.4f}, std={macro_f1_values.std(ddof=0):.4f}")
        print(
            "Best Val Balanced Acc: "
            f"mean={balanced_values.mean():.4f}, std={balanced_values.std(ddof=0):.4f}"
        )


if __name__ == "__main__":
    main()
