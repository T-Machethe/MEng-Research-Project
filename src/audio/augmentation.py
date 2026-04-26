import torch
import numpy as np
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from src.config import PITCH_SEMITONES, TIME_STRETCH_RATES, TARGET_SR,NOISE_SNR_DB
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(waveform: torch.Tensor,
                       snr_db: Tuple[float, float] = NOISE_SNR_DB) -> torch.Tensor:
    """
    Additive white Gaussian noise at a random SNR in [snr_db[0], snr_db[1]] dB.

    Clinic recordings exhibit variable HVAC hum, distant speech, and
    equipment beeps.  Noisy augmentation teaches wav2vec 2.0 to extract
    vocal features robustly regardless of environmental noise levels —
    critical for deployment across different hospital sites.
    """
    snr = np.random.uniform(*snr_db)
    signal_power = waveform.pow(2).mean()
    noise_power  = signal_power / (10 ** (snr / 10))
    noise        = torch.randn_like(waveform) * noise_power.sqrt()
    return (waveform + noise).clamp(-1.0, 1.0)


def pitch_shift(waveform: torch.Tensor, sr: int = TARGET_SR,
                semitone_range: Tuple[float, float] = PITCH_SEMITONES) -> torch.Tensor:
    """
    Random pitch shift in [-2, +2] semitones without changing duration.

    Sinusitis affects resonance chamber geometry, which shifts formant
    frequencies.  Pitch augmentation simulates natural inter-speaker
    variation in fundamental frequency (F0), preventing the model from
    over-fitting to the specific F0 distribution of 107 speakers.

    Uses librosa's phase-vocoder implementation, which operates in the
    time-domain short-time sense but does NOT produce a spectrogram input
    to the model — only an augmented waveform.
    """
    n_steps = np.random.uniform(*semitone_range)
    audio   = waveform.squeeze(0).numpy()
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    return torch.from_numpy(shifted).unsqueeze(0)


def time_stretch(waveform: torch.Tensor,
                 rate_range: Tuple[float, float] = TIME_STRETCH_RATES) -> torch.Tensor:
    """
    Random time-stretch (rate ∈ [0.9, 1.1]) without changing pitch.

    Speaking rate varies between healthy and sinusitis-affected subjects
    (congestion reduces airflow, slowing speech).  Time-stretch augmentation
    exposes the model to a wider range of speaking rates, improving
    generalisation to this clinically relevant variability.

    NOTE: Time-stretching changes the waveform length; segments are
    re-windowed afterwards to maintain consistent 8 192-sample inputs.
    """
    rate  = np.random.uniform(*rate_range)
    audio = waveform.squeeze(0).numpy()
    # librosa.effects.time_stretch uses a phase-vocoder internally but
    # the OUTPUT is a raw waveform — no spectrogram is fed to the model.
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return torch.from_numpy(stretched.astype(np.float32)).unsqueeze(0)


def augment_waveform(waveform: torch.Tensor, sr: int = TARGET_SR,
                     p_noise: float = 0.6,
                     p_pitch: float = 0.5,
                     p_stretch: float = 0.4) -> torch.Tensor:
    """
    Stochastically apply physical augmentations to a single waveform.

    Each augmentation is applied independently with its own probability,
    allowing the pipeline to generate a diverse combination of conditions
    (noisy + pitch-shifted, stretched + noisy, etc.) without always
    applying all three — which would make the training distribution
    unrealistically uniform.
    """
    if np.random.rand() < p_noise:
        waveform = add_gaussian_noise(waveform)
    if np.random.rand() < p_pitch:
        waveform = pitch_shift(waveform, sr)
    if np.random.rand() < p_stretch:
        waveform = time_stretch(waveform)
    return waveform


def spec_augment_latent(latent: torch.Tensor,
                        time_mask_param: int = 50,
                        freq_mask_param: int = 20,
                        n_time_masks: int = 2,
                        n_freq_masks: int = 2) -> torch.Tensor:
    """
    SpecAugment applied in the *latent* (feature-encoder output) space.

    wav2vec 2.0's original masking strategy (used during self-supervised
    pre-training) masks consecutive latent time-steps before the
    Transformer.  During fine-tuning we can optionally also mask latent
    channels (analogous to frequency masking in SpecAugment) to prevent
    the Transformer from relying on any single feature dimension.

    Args:
        latent : Tensor [B, T, C]  — output of the feature encoder.
        time_mask_param  : max consecutive time-steps to mask.
        freq_mask_param  : max consecutive channels (dimensions) to mask.
        n_time_masks     : number of independent time masks.
        n_freq_masks     : number of independent channel masks.

    Returns:
        Masked latent tensor of the same shape.

    NOTE: This function is called *after* the feature encoder, not on the
    raw waveform.  Insert it between `model.feature_extractor()` and
    `model.encoder()` in your fine-tuning training loop.
    """
    B, T, C = latent.shape
    out = latent.clone()

    # Time masking — zero out random consecutive time-steps
    for _ in range(n_time_masks):
        mask_len   = np.random.randint(0, time_mask_param + 1)
        mask_start = np.random.randint(0, max(1, T - mask_len))
        out[:, mask_start: mask_start + mask_len, :] = 0.0

    # Channel (latent-frequency) masking
    for _ in range(n_freq_masks):
        mask_len   = np.random.randint(0, freq_mask_param + 1)
        mask_start = np.random.randint(0, max(1, C - mask_len))
        out[:, :, mask_start: mask_start + mask_len] = 0.0

    return out
