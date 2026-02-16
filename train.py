import os
import argparse
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm

class DeepInfantDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        # Backward-compatible behavior: callers currently pass bool for augmentation.
        self.augment = bool(transform) if isinstance(transform, bool) else False
        self.samples = []
        self.labels = []
        
        # Updated label mapping based on new classes
        self.label_map = {
            'bp': 0,  # belly pain
            'bu': 1,  # burping
            'ch': 2,  # cold/hot
            'dc': 3,  # discomfort
            'hu': 4,  # hungry
            'lo': 5,  # lonely
            'sc': 6,  # scared
            'ti': 7,  # tired
            'un': 8,  # unknown
        }
        
        # Load metadata if available
        metadata_file = Path(data_dir).parent / 'metadata.csv'
        if metadata_file.exists():
            self._load_from_metadata(metadata_file)
        else:
            self._load_dataset()
    
    def _load_from_metadata(self, metadata_file):
        df = pd.read_csv(metadata_file)
        for _, row in df.iterrows():
            if row['split'] == self.data_dir.name:  # 'train' or 'test'
                audio_path = self.data_dir / row['filename']
                if audio_path.exists():
                    self.samples.append(str(audio_path))
                    self.labels.append(self.label_map[row['class_code']])
    
    def _load_dataset(self):
        for audio_file in self.data_dir.glob('*.*'):
            if audio_file.suffix in ['.wav', '.caf', '.3gp']:
                # Parse filename for label
                label = audio_file.stem.split('-')[-1][:2]  # Get reason code
                if label in self.label_map:
                    self.samples.append(str(audio_file))
                    self.labels.append(self.label_map[label])
    
    def _process_audio(self, audio_path):
        # Load audio with 16kHz sample rate
        waveform, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Add basic audio augmentation (during training)
        if self.augment:
            # Random time shift (-100ms to 100ms)
            shift = np.random.randint(-1600, 1600)
            if shift > 0:
                waveform = np.pad(waveform, (shift, 0))[:len(waveform)]
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
        
        # Generate mel spectrogram with adjusted parameters
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,  # Reduced from 2048 for better temporal resolution
            hop_length=256,  # Reduced from 512
            n_mels=80,  # Standard for speech/audio
            fmin=20,  # Minimum frequency
            fmax=8000  # Maximum frequency, suitable for infant cries
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # Per-sample normalization for stabler optimization.
        mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-6)
        
        return torch.FloatTensor(mel_spec)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        label = self.labels[idx]
        
        # Process audio to mel spectrogram
        mel_spec = self._process_audio(audio_path)
        
        if callable(self.transform):
            mel_spec = self.transform(mel_spec)
        
        return mel_spec, label

class DeepInfantModel(nn.Module):
    def __init__(self, num_classes=9):
        super(DeepInfantModel, self).__init__()
        
        # CNN feature extractor: preserve temporal/frequency structure for LSTM.
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
            nn.MaxPool2d(2)
        )
        
        # Bi-directional LSTM for better temporal modeling
        self.lstm = nn.LSTM(
            input_size=256 * 10,  # Adjusted based on new mel spec parameters
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),  # 1024 due to bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Input shape: (batch, 1, freq_bins, time_steps)
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.conv_layers(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, -1, 256 * 10)  # (batch, time, features)
        
        # LSTM processing
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last time step
        
        # Classification
        x = self.classifier(x)
        return x

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=50,
    device='cuda',
    scheduler=None,
    grad_clip=1.0,
    checkpoint_path='deepinfant.pth',
):
    model = model.to(device)
    best_val_f1 = -1.0
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
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
        
        val_acc = 100. * val_correct / val_total
        val_macro_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_balanced_acc = recall_score(val_labels, val_preds, average='macro', zero_division=0)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss / len(val_loader))
            else:
                scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Macro-F1: {val_macro_f1:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}')
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model by macro-F1, then by accuracy for tie-break.
        is_better = (
            val_macro_f1 > best_val_f1
            or (abs(val_macro_f1 - best_val_f1) < 1e-8 and val_acc > best_val_acc)
        )
        if is_better:
            best_val_f1 = val_macro_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Saved improved checkpoint to {checkpoint_path} "
                f"(val_macro_f1={val_macro_f1:.4f}, val_acc={val_acc:.2f}%)"
            )


def compute_class_weights(class_counts, scheme='sqrt_inv', max_weight=4.0):
    counts = np.asarray(class_counts, dtype=np.float64)
    weights = np.ones_like(counts, dtype=np.float64)
    nonzero = counts > 0
    if not nonzero.any() or scheme == 'none':
        return torch.tensor(weights, dtype=torch.float32)

    if scheme == 'inv':
        weights[nonzero] = 1.0 / counts[nonzero]
    elif scheme == 'sqrt_inv':
        weights[nonzero] = 1.0 / np.sqrt(counts[nonzero])
    else:
        raise ValueError(f"Unsupported class weighting scheme: {scheme}")

    # Normalize around 1.0 and clip extremes to reduce instability.
    weights = weights / weights[nonzero].mean()
    weights = np.clip(weights, 1.0 / max_weight, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepInfant model")
    parser.add_argument("--train-dir", default="processed_dataset/train")
    parser.add_argument("--val-dir", default="processed_dataset/test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", default="deepinfant.pth")
    parser.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    parser.add_argument(
        "--imbalance-mode",
        choices=["none", "sampler", "loss", "both"],
        default="both",
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
        default=4.0,
        help="Clip absolute class weights to avoid unstable extremes.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "plateau"],
        default="plateau",
        help="LR scheduler strategy.",
    )
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentations.")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets using processed data
    train_dataset = DeepInfantDataset(args.train_dir, transform=(not args.no_augment))
    val_dataset = DeepInfantDataset(args.val_dir, transform=False)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Compute class counts from training labels.
    train_labels = np.array(train_dataset.labels, dtype=np.int64)
    num_classes = len(train_dataset.label_map)
    class_counts = np.bincount(train_labels, minlength=num_classes)
    print("Train class counts:", class_counts.tolist())

    class_weights = None
    sample_weights = None
    if args.class_weighting != "none":
        class_weights = compute_class_weights(
            class_counts, scheme=args.class_weighting, max_weight=args.max_class_weight
        )
        print("Class weights:", [round(x, 4) for x in class_weights.tolist()])
    else:
        class_weights = torch.ones(num_classes, dtype=torch.float32)
        print("Class weights: uniform")

    if class_weights is not None:
        sample_weights = class_weights[torch.tensor(train_labels, dtype=torch.long)].double()

    # Create data loaders
    use_sampler = args.imbalance_mode in ("sampler", "both")
    sampler = None
    if use_sampler and sample_weights is not None:
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
    
    # Initialize model, loss function, and optimizer
    model = DeepInfantModel()
    use_loss_weights = args.imbalance_mode in ("loss", "both")
    if use_loss_weights and class_weights is not None and args.class_weighting != "none":
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Imbalance handling: using class-weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        if args.imbalance_mode in ("loss", "both"):
            print("Imbalance handling: class-weighted loss unavailable, using standard loss")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )
    
    # Train the model
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=args.epochs,
        device=device,
        scheduler=scheduler,
        grad_clip=args.grad_clip,
        checkpoint_path=args.checkpoint_path,
    )

if __name__ == '__main__':
    main() 
