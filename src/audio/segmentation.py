import torch
from src.config import WINDOW_SAMPLES, HOP_SAMPLES, FINETUNE_MIN_S, FINETUNE_MAX_S, TARGET_SR
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Amplitude Normalization
# ─────────────────────────────────────────────────────────────────────────────

def global_amplitude_normalize(waveform: torch.Tensor,
                                eps: float = 1e-8) -> torch.Tensor:
    """
    Global peak normalization: divide by the maximum absolute amplitude.

    This maps every recording to the range [-1, +1] regardless of the
    microphone gain, speaker distance, or recording device used across
    the 107-speaker clinical dataset.  Without this step, amplitude
    differences between sessions would act as a spurious feature,
    allowing the model to learn recording-session identity rather than
    voice pathology.
    """
    peak = waveform.abs().max()
    return waveform / (peak + eps)


def instance_normalize(segment: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """
    Per-segment zero-mean, unit-variance normalization.

    Applied immediately before the wav2vec 2.0 feature encoder.
    The feature encoder's first layer (a Conv1d) is sensitive to the
    mean and variance of its input.  Instance normalization ensures
    every 1.024-second window enters the encoder with identical
    first- and second-order statistics, preventing any single segment
    from dominating gradient updates during training.
    """
    mean = segment.mean()
    std  = segment.std() + eps
    
    # Guard: if segment is silent/near-zero, return zeros rather than NaN
    if std < 1e-6:
        return torch.zeros_like(segment)
      
    return (segment - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Temporal Segmentation — Sliding Window
# ─────────────────────────────────────────────────────────────────────────────

def sliding_window_segments(waveform: torch.Tensor,
                             window: int = WINDOW_SAMPLES,
                             hop: int = HOP_SAMPLES) -> List[torch.Tensor]:
    """
    Split the VAD-filtered waveform into overlapping windows.

    Window size : 8 192 samples = 1.024 s @ 16 kHz.
    Overlap     : 50 % (hop = 4 096 samples).

    Rationale
    ---------
    • The small corpus (~107 speakers) would be severely data-starved
      at the utterance level.  Windowing increases the number of training
      examples by roughly (duration / hop) per speaker.
    • 50 % overlap ensures transitional phoneme contexts near window
      boundaries appear in both adjacent windows, reducing boundary
      artefacts.
    • 8 192 samples is a power-of-2 length compatible with the strided
      CNN stack in wav2vec 2.0's feature encoder, which down-samples by
      a total factor of 320 (giving 25.6 latent frames/second).
    """
    audio  = waveform.squeeze(0)
    length = audio.shape[0]
    segs   = []

    start = 0
    while start + window <= length:
        segs.append(audio[start: start + window].unsqueeze(0))
        start += hop

    return segs


def build_finetune_chunks(waveform: torch.Tensor,
                          sr: int = TARGET_SR,
                          min_s: float = FINETUNE_MIN_S,
                          max_s: float = FINETUNE_MAX_S) -> List[torch.Tensor]:
    """
    For fine-tuning: produce longer chunks (15–20 s) to feed the
    Transformer context network.

    wav2vec 2.0's Transformer stack captures long-range temporal
    dependencies.  Feeding only 1-second windows during fine-tuning
    would prevent the model from learning multi-second prosodic patterns
    (e.g., sustained nasal airflow characteristic of sinusitis).
    Chunks of 15–20 s allow ≈ 384–512 latent frames per forward pass,
    giving the Transformer sufficient context.
    """
    min_samples = int(min_s * sr)
    max_samples = int(max_s * sr)
    audio  = waveform.squeeze(0)
    length = audio.shape[0]

    chunks = []
    start  = 0
    while start < length:
        end = min(start + max_samples, length)
        chunk = audio[start:end]
        if len(chunk) >= min_samples:
            chunks.append(chunk.unsqueeze(0))
        start = end

    return chunks