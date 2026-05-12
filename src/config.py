# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SR      = 16000
LEADING_TRIM_S = 0.05       # reduced from 0.15 → only strip 50ms
                             # 0.15s was designed for long recordings;
                             # it disproportionately harms short vowels (<1s)

VAD_THRESHOLD  = 0.01       # reduced from 0.02 → less aggressive
                             # sinusitis patients have lower vocal intensity;
                             # 0.02 was discarding genuine voiced frames

MIN_DURATION_S = 0.1        # skip files shorter than 300ms entirely —
                             # they are too short to produce any 8192-sample
                             # segment and are likely recording errors

WINDOW_SAMPLES     = 16000 # 1.0s @ 16kHz — matches natural vowel duration, which are the shortest in the files
HOP_SAMPLES        = 8000  # 0.5s overlap
FINETUNE_MIN_S     = 15.0
FINETUNE_MAX_S     = 20.0
NOISE_SNR_DB       = (10, 30)
PITCH_SEMITONES    = (-2, 2)
TIME_STRETCH_RATES = (0.9, 1.1)
