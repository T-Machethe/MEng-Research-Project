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

SUSTAINED_CAPITAL_V_GROUPS = {"fess"}


def resolve_path(rel_path: str,
                 project_root: Union[str, Path],
                 col: str = "") -> Path:
    project_root = Path(project_root)

    rel_path = rel_path.strip().replace("\\", "/").lstrip("./").lstrip("/")
    parts = list(Path(rel_path).parts)

    if len(parts) < 5:
        raise ValueError(f"Unexpected CSV path format: {rel_path!r}")

    data_final = parts[1]
    group      = parts[3]
    session    = parts[-2]
    filename   = parts[-1]

    col_key = col.strip().lower()
    if col_key not in COLUMN_TO_SUBFOLDER:
        raise KeyError(
            f"Column '{col}' not in COLUMN_TO_SUBFOLDER. "
            f"Add it to src/utils/paths.py."
        )

    category_parts = list(COLUMN_TO_SUBFOLDER[col_key])

    # Fess uses 'Sustained Vowels' (capital V), all others use 'Sustained vowels'
    if col_key in ("a1", "a2", "a3") and group.lower() in SUSTAINED_CAPITAL_V_GROUPS:
        category_parts[0] = "Sustained Vowels"

    return project_root.joinpath(
        "Data",
        data_final,
        "Audios",
        group,
        *category_parts,
        session,
        filename,
    )