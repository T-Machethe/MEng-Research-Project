import numpy as np
import torch
import torchaudio.transforms as T
from scipy.signal import butter, sosfilt
from src.config import (
    LEADING_TRIM_S, VAD_THRESHOLD, TARGET_SR, MIN_DURATION_S
)
from typing import Tuple


def remove_leading_silence(waveform: torch.Tensor,
                            sr: int,
                            trim_seconds: float = LEADING_TRIM_S) -> torch.Tensor:
    """
    Strip leading non-informative audio.

    Reduced from 0.15s to 0.05s because isolated vowel recordings are
    often under 1 second — a 0.15s trim was removing up to 26% of those
    files before VAD even ran.
    """
    trim_samples = int(trim_seconds * sr)
    trimmed = waveform[:, trim_samples:]

    # Safety: if trim leaves less than MIN_DURATION_S, return original
    # This protects extremely short files from being gutted by the trim
    if trimmed.shape[1] / sr < MIN_DURATION_S:
        return waveform

    return trimmed


def highpass_filter(waveform: torch.Tensor, sr: int,
                    cutoff_hz: float = 80.0) -> torch.Tensor:
    sos = butter(N=4, Wn=cutoff_hz / (sr / 2), btype='high', output='sos')
    filtered = sosfilt(sos, waveform.numpy())
    return torch.from_numpy(filtered.astype(np.float32))


def voice_activity_detection(waveform: torch.Tensor, sr: int,
                              threshold: float = VAD_THRESHOLD,
                              frame_ms: int = 20) -> torch.Tensor:
    """
    Energy-based VAD with two improvements over the original:

    1. Adaptive floor: threshold is set to max(fixed_threshold, 5% of
       the file's own peak RMS). This prevents the fixed threshold from
       being too harsh on quiet clinical recordings (sinusitis patients
       have reduced vocal intensity) while still removing true silence.

    2. Minimum chunk protection: after VAD, if less than MIN_DURATION_S
       of audio remains, return the original pre-VAD audio rather than
       an almost-empty tensor. This is the right behaviour for short
       sustained vowel recordings where nearly all frames are voiced.
    """
    frame_samples = int(sr * frame_ms / 1000)
    audio = waveform.squeeze(0).numpy()
    n_frames = len(audio) // frame_samples

    # Adaptive threshold: 5% of file peak RMS, floored at fixed threshold
    # This scales with recording volume rather than using a global constant
    frame_rms_values = []
    for i in range(n_frames):
        chunk = audio[i * frame_samples: (i + 1) * frame_samples]
        frame_rms_values.append(np.sqrt(np.mean(chunk ** 2)))

    if frame_rms_values:
        peak_rms        = np.percentile(frame_rms_values, 95)
        adaptive_thresh = max(threshold, 0.05 * peak_rms)
    else:
        adaptive_thresh = threshold

    voiced_chunks = []
    for i, rms in enumerate(frame_rms_values):
        if rms > adaptive_thresh:
            chunk = audio[i * frame_samples: (i + 1) * frame_samples]
            voiced_chunks.append(chunk)

    if not voiced_chunks:
        return waveform

    voiced_audio = np.concatenate(voiced_chunks)
    voiced_tensor = torch.from_numpy(voiced_audio).unsqueeze(0)

    # Protection: if VAD removed too much, return pre-VAD audio
    if voiced_tensor.shape[1] / sr < MIN_DURATION_S:
        return waveform

    return voiced_tensor


def resample(waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
    if orig_sr == TARGET_SR:
        return waveform
    resampler = T.Resample(orig_freq=orig_sr, new_freq=TARGET_SR)
    return resampler(waveform)