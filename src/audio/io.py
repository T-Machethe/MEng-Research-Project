import torchaudio
import torch
from typing import Tuple

def load_audio(filepath: str) -> Tuple[torch.Tensor, int]:
    """
    Load a WAV file as a mono float32 tensor.

    torchaudio.load returns (waveform [C, T], sample_rate).
    We convert to mono by averaging channels so that recordings from
    different microphone setups (mono / stereo) are treated uniformly.
    """
    waveform, sr = torchaudio.load(filepath)
    
    # Average across channels → shape [1, T]
    waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr
