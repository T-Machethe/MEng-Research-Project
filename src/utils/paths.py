from pathlib import Path
from typing import Union

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

# Maps every observed CSV group name variant to the canonical Drive folder name.
# The CSV was created on Windows/Mac (case-insensitive) so paths were recorded
# with inconsistent capitalisation. On Linux (Colab) these are case-sensitive.
GROUP_DIR_MAP = {
    # FESS — CSV says FESS, actual Drive folder is Fess
    "FESS":     "Fess",
    "fess":     "Fess",
    "Fess":     "Fess",
    # Control — CSV has four variants, actual Drive folder is Contr
    "Contr":    "Contr",
    "contr":    "Contr",
    "Contra":   "Contr",
    "contra":   "Contr",
    "Contract": "Contr",
    "contract": "Contr",
    "Control":  "Contr",
    "control":  "Contr",
    # Septoplasty
    "Sept":     "Sept",
    "sept":     "Sept",
    # Tonsillitis
    "Tonsill":  "Tonsill",
    "tonsill":  "Tonsill",
}

# The data subfolder is always data_final regardless of what the CSV recorded.
# Some CSV paths have Date_final or date_final (typos/case errors on Windows).
CANONICAL_DATA_FOLDER = "data_final"

SUSTAINED_CAPITAL_V_GROUPS = {"fess"}


def resolve_path(rel_path: str,
                 project_root: Union[str, Path],
                 col: str = "") -> Path:
    """
    Resolve a relative CSV audio path to an absolute Drive path.

    Handles all observed CSV path inconsistencies:
      - data_final / Date_final / date_final   → always data_final
      - FESS / fess                             → Fess
      - Contr / contr / contra / contract       → Contr
      - .WAV / .Wav                             → case-insensitive glob fallback
    """
    project_root = Path(project_root)

    rel_path = rel_path.strip().replace("\\", "/").lstrip("./").lstrip("/")
    parts = list(Path(rel_path).parts)

    if len(parts) < 5:
        raise ValueError(f"Unexpected CSV path format: {rel_path!r}")

    # parts[1] carries data_final (sometimes Date_final / date_final) — normalise
    group      = GROUP_DIR_MAP.get(parts[3], parts[3])
    session    = parts[-2]
    filename   = parts[-1]

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

    # ── Case-insensitive fallback ─────────────────────────────────────────────
    # Handles .WAV / .Wav extension variants and any remaining capitalisation
    # mismatches in the filename itself. Only searches the parent directory.
    parent = path.parent
    if parent.exists():
        name_lower = filename.lower()
        for candidate in parent.iterdir():
            if candidate.name.lower() == name_lower:
                return candidate

    # File genuinely missing — return the constructed path so the caller
    # receives a meaningful error rather than a silent skip.
    return path