import os
import pandas as pd
import torch
from pathlib import Path
import torchaudio.transforms as T


from src.audio.io import load_audio
from src.audio.cleaning import (
    remove_leading_silence,
    highpass_filter,
    voice_activity_detection,
)
from src.audio.segmentation import (
    sliding_window_segments,
    build_finetune_chunks,
    global_amplitude_normalize,
    instance_normalize,
)
from src.audio.augmentation import augment_waveform
from src.utils.paths import resolve_path
from src.config import TARGET_SR, MIN_DURATION_S


def preprocess_file(filepath, mode="scratch", augment=True):
    waveform, sr = load_audio(filepath)
    waveform = remove_leading_silence(waveform, sr)

    if sr != TARGET_SR:
        waveform = T.Resample(sr, TARGET_SR)(waveform)
        sr = TARGET_SR

    waveform = highpass_filter(waveform, sr)
    waveform = voice_activity_detection(waveform, sr)

     
    if waveform.shape[1] / sr < MIN_DURATION_S:
        return []
    
    waveform = global_amplitude_normalize(waveform)

    if augment:
        waveform = augment_waveform(waveform, sr)

    if mode == "finetune":
        segments = build_finetune_chunks(waveform)
    else:
        segments = sliding_window_segments(waveform)   # ← removed spurious ()

    return [instance_normalize(s) for s in segments]


def process_from_csv(csv_path, project_root, output_dir,
                     mode="scratch", augment=True):

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    audio_cols = df.columns[df.columns.get_loc("TIME") + 1: -1]

    total_segments = 0
    total_missing = 0
    total_skipped_empty = 0

    print(f"\nLoaded CSV: {df.shape}")
    print(f"Audio columns: {list(audio_cols)}\n")

    for _, row in df.iterrows():
        subject_id = row["ID"]
        session = row["session"]

        for col in audio_cols:
            rel_path = row[col]

            if pd.isna(rel_path) or str(rel_path).strip() == "":
                total_skipped_empty += 1
                continue

            abs_path = resolve_path(rel_path, project_root, col=col)

            if not Path(abs_path).exists():
                print(f"[MISSING] ID={subject_id} SES={session} COL={col}")
                print(f"          -> {abs_path}")
                total_missing += 1
                continue

            try:
                segments = preprocess_file(str(abs_path), mode=mode, augment=augment)
            except Exception as e:
                print(f"[ERROR] Processing failed: {abs_path}")
                print(f"        {repr(e)}")
                continue

            for i, seg in enumerate(segments):
                fname = f"ID{subject_id}_ses{session}_{col}_seg{i:04d}.pt"
                out_path = os.path.join(output_dir, fname)
                torch.save(seg, out_path)
                total_segments += 1

    print(f"Done. Total segments: {total_segments}")   # ← fixed variable name
    print(f"Missing files: {total_missing}")
    print(f"Skipped empty cells: {total_skipped_empty}")