from pathlib import Path
from typing import Union
import re

COLUMN_TO_SUBFOLDER = {
    # Isolated vowels
    "a":          ("Vowels", "A"),
    "e":          ("Vowels", "E"),
    "i":          ("Vowels", "I"),
    "o":          ("Vowels", "O"),
    "u":          ("Vowels", "U"),
    # Sustained vowels
    "a1":         ("Sustained vowels", "A1"),
    "a2":         ("Sustained vowels", "A2"),
    "a3":         ("Sustained vowels", "A3"),
    # TDU words
    "agua":       ("TDU", "Agua"),
    "brasero":    ("TDU", "Brasero"),
    "dia":        ("TDU", "Dia"),
    "mesa":       ("TDU", "Mesa"),
    # Speech
    "speech":     ("Speech",),
    # Raw concatenated read
    "un":         ("Raw", "concatenateread"),
}

# Maps every observed CSV group-directory variant to the canonical Drive folder.
GROUP_DIR_MAP = {
    "FESS":     "Fess",
    "fess":     "Fess",
    "Fess":     "Fess",
    "Contr":    "Contr",
    "contr":    "Contr",
    "Contra":   "Contr",
    "contra":   "Contr",
    "Contract": "Contr",
    "contract": "Contr",
    "Control":  "Contr",
    "control":  "Contr",
    "Sept":     "Sept",
    "sept":     "Sept",
    "Tonsill":  "Tonsill",
    "tonsill":  "Tonsill",
}

CANONICAL_DATA_FOLDER = "data_final"

SUSTAINED_CAPITAL_V_GROUPS = {"fess"}


def _find_file_in_dir(parent: Path, filename: str, col_key: str) -> Path:
    """
    Three-level fallback search inside `parent` for a file that cannot be
    found by exact match.

    Level 1 — case-insensitive exact name match.
        Handles .WAV / .Wav / .wav extension differences.

    Level 2 — ID-anchored fuzzy match.
        Extracts the 4-digit zero-padded patient ID from the CSV filename
        (e.g. "0012" from "contr_se1_agua_0012.wav"), then finds any file
        in the directory whose name contains that ID and the column name.
        This handles prefix variants: contr_se1_agua_0012 vs
        Contr_ses1_agua_0012.

    Level 3 — ID-only match (last resort).
        If no file matches both ID and column, fall back to ID alone.
        Rare but covers files where the column is abbreviated differently.

    Returns the candidate path if found, or the original expected path so
    the caller receives a meaningful FileNotFoundError rather than None.
    """
    if not parent.exists():
        return parent / filename

    name_lower = filename.lower()

    # Level 1: case-insensitive exact name
    for candidate in parent.iterdir():
        if candidate.name.lower() == name_lower:
            return candidate

    # Extract 4-digit ID from the CSV filename (always at the end before ext)
    id_match = re.search(r"_(\d{4})\.\w+$", filename)
    if not id_match:
        return parent / filename  # can't do better
    id_str = id_match.group(1)   # e.g. "0012"

    # Level 2: file contains both the patient ID and the audio column name
    for candidate in parent.iterdir():
        cname = candidate.name.lower()
        if id_str in cname and col_key in cname:
            return candidate

    # Level 3: file contains the patient ID alone (last resort)
    for candidate in parent.iterdir():
        if id_str in candidate.name:
            return candidate

    # Nothing found — return the expected path for a clear error
    return parent / filename


def resolve_path(rel_path: str,
                 project_root: Union[str, Path],
                 col: str = "") -> Path:
    """
    Resolve a relative CSV audio path to an absolute Drive path.

    Handles all observed CSV path inconsistencies:
      - data_final / Date_final / date_final  -> always data_final
      - FESS / fess                           -> Fess  (actual folder)
      - Contr / contr / contra / contract     -> Contr (actual folder)
      - .WAV / .Wav                           -> case-insensitive fallback
      - contr_se1_agua_ / contract_ses1_dia_  -> ID+column fuzzy fallback
    """
    project_root = Path(project_root)

    rel_path = rel_path.strip().replace("\\", "/").lstrip("./").lstrip("/")
    parts = list(Path(rel_path).parts)

    if len(parts) < 5:
        raise ValueError(f"Unexpected CSV path format: {rel_path!r}")

    group    = GROUP_DIR_MAP.get(parts[3], parts[3])
    session  = parts[-2]
    filename = parts[-1]

    col_key = col.strip().lower()
    if col_key not in COLUMN_TO_SUBFOLDER:
        raise KeyError(
            f"Column '{col}' not in COLUMN_TO_SUBFOLDER. "
            f"Add it to src/utils/paths.py."
        )

    category_parts = list(COLUMN_TO_SUBFOLDER[col_key])

    # FESS uses 'Sustained Vowels' (capital V); all others use 'Sustained vowels'
    if col_key in ("a1", "a2", "a3") and parts[3].lower() in SUSTAINED_CAPITAL_V_GROUPS:
        category_parts[0] = "Sustained Vowels"

    path = project_root.joinpath(
        "Data",
        CANONICAL_DATA_FOLDER,
        "Audios",
        group,
        *category_parts,
        session,
        filename,
    )

    if path.exists():
        return path

    # Multi-level fallback: case differences, filename prefix variants, etc.
    return _find_file_in_dir(path.parent, filename, col_key)


# Maps the canonical group folder name to the prefix used in audio filenames.
# e.g. group folder "Fess" → filename prefix "FESS" in FESS_ses1_a_0017.wav
FILENAME_PREFIX_MAP = {
    "Fess":    "FESS",
    "Contr":   "Contr",
    "Sept":    "Sept",
    "Tonsill": "Tonsill",
}


def find_audio_file(group: str,
                    session: int,
                    patient_id,
                    col: str,
                    project_root: Union[str, Path]) -> Path:
    """
    Locate an audio file purely from row metadata, without a CSV path.

    Used when the CSV cell for a given audio column is empty or NaN.
    Constructs the expected canonical path and falls back to the same
    ID+column fuzzy search as resolve_path.

    Parameters
    ----------
    group       : raw GROUP value from the CSV row (e.g. "FESS", "Contr")
    session     : session number (1, 2, or 3)
    patient_id  : patient ID (numeric or string)
    col         : audio column name (e.g. "a", "speech", "a3")
    project_root: Drive root (resolve_path convention)

    Returns
    -------
    Path to the audio file if found, or the expected canonical path
    (which will not exist) so the caller can log it as MISSING.
    """
    project_root = Path(project_root)
    col_key      = col.strip().lower()

    if col_key not in COLUMN_TO_SUBFOLDER:
        raise KeyError(f"Column '{col}' not in COLUMN_TO_SUBFOLDER.")

    # Resolve canonical group directory name
    group_dir = GROUP_DIR_MAP.get(str(group).strip(), str(group).strip())

    # Construct subfolder path
    category_parts = list(COLUMN_TO_SUBFOLDER[col_key])
    if col_key in ("a1", "a2", "a3") and group_dir.lower() in SUSTAINED_CAPITAL_V_GROUPS:
        category_parts[0] = "Sustained Vowels"

    parent = project_root.joinpath(
        "Data", CANONICAL_DATA_FOLDER, "Audios",
        group_dir, *category_parts, str(session),
    )

    # Build expected filename using the canonical naming convention:
    # {PREFIX}_ses{N}_{col}_{id:04d}.wav
    prefix   = FILENAME_PREFIX_MAP.get(group_dir, group_dir)
    id_str   = f"{int(patient_id):04d}"
    expected = parent / f"{prefix}_ses{session}_{col_key}_{id_str}.wav"

    if expected.exists():
        return expected

    # Fuzzy fallback — same three-level search as resolve_path
    return _find_file_in_dir(parent, expected.name, col_key)