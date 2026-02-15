import torch
import librosa
import numpy as np
from pathlib import Path
from train import DeepInfantModel  # Import the model architecture

class InfantCryPredictor:
    def __init__(self, model_path='deepinfant.pth', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        state_dict = self._load_state_dict(model_path)
        num_classes = self._infer_num_classes(state_dict)

        # Initialize model with inferred class count.
        self.model = DeepInfantModel(num_classes=num_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Label mapping based on model output classes.
        label_maps = {
            5: {
                0: 'hungry',
                1: 'burping',
                2: 'belly_pain',
                3: 'discomfort',
                4: 'tired',
            },
            9: {
                0: 'belly_pain',
                1: 'burping',
                2: 'cold_hot',
                3: 'discomfort',
                4: 'hungry',
                5: 'lonely',
                6: 'scared',
                7: 'tired',
                8: 'unknown',
            },
        }
        if num_classes not in label_maps:
            raise ValueError(
                f"Unsupported model class count: {num_classes}. "
                "Expected 5 or 9 output classes."
            )
        self.label_map = label_maps[num_classes]

    def _load_state_dict(self, model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{model_path}'. "
                "Place a .pth checkpoint in the repository root or pass model_path explicitly."
            )

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
                return checkpoint['state_dict']
            if 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
                return checkpoint['model_state_dict']
            if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                return checkpoint
        raise ValueError(
            f"Unsupported checkpoint format for '{model_path}'. "
            "Expected a PyTorch state_dict or a checkpoint with state_dict/model_state_dict."
        )

    def _infer_num_classes(self, state_dict):
        classifier_weight_keys = ['classifier.3.weight', 'classifier.1.weight']
        for key in classifier_weight_keys:
            if key in state_dict:
                return int(state_dict[key].shape[0])
        raise ValueError(
            "Unable to infer output classes from checkpoint. "
            "Expected classifier final layer weights in state_dict."
        )
    
    def _process_audio(self, audio_path):
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: '{audio_path}'")

        # Load audio with 16kHz sample rate.
        try:
            waveform, sample_rate = librosa.load(audio_path, sr=16000)
        except Exception as exc:
            raise ValueError(f"Failed to read audio file '{audio_path}': {exc}") from exc
        
        # Ensure consistent length (7 seconds)
        target_length = 7 * 16000
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            fmin=20,
            fmax=8000
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return torch.FloatTensor(mel_spec)
    
    def predict(self, audio_path):
        """
        Predict the class of a single audio file
        Returns tuple of (predicted_label, confidence)
        """
        # Process audio
        mel_spec = self._process_audio(audio_path)
        mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        mel_spec = mel_spec.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(mel_spec)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][pred_class].item()
        
        return self.label_map[pred_class], confidence
    
    def predict_batch(self, audio_dir, file_extensions=('.wav', '.caf', '.3gp')):
        """
        Predict classes for all audio files in a directory
        Returns list of tuples (filename, predicted_label, confidence)
        """
        results = []
        audio_dir = Path(audio_dir)
        
        for audio_file in audio_dir.glob('*.*'):
            if audio_file.suffix.lower() in file_extensions:
                label, confidence = self.predict(str(audio_file))
                results.append((audio_file.name, label, confidence))
        
        return results

def main():
    # Example usage
    predictor = InfantCryPredictor()
    
    # Single file prediction
    audio_path = "path/to/your/audio.wav"
    label, confidence = predictor.predict(audio_path)
    print(f"\nPrediction for {audio_path}:")
    print(f"Label: {label}")
    print(f"Confidence: {confidence:.2%}")
    
    # Batch prediction
    audio_dir = "path/to/audio/directory"
    results = predictor.predict_batch(audio_dir)
    print("\nBatch Predictions:")
    for filename, label, confidence in results:
        print(f"{filename}: {label} ({confidence:.2%})")

if __name__ == "__main__":
    main() 
