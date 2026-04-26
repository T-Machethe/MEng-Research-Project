
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipeline.preprocess import process_from_csv
from pathlib import Path

# parents[1] = Project Folder  (parents[0] would be the scripts/ subfolder)
#PROJECT_ROOT = Path(__file__).resolve().parents[1]

csv_path = PROJECT_ROOT / "Data" / "data_final" / "Clinical" / "clinical_all_sessions.csv"
output_dir = PROJECT_ROOT / "Data" / "data_final" / "clean_audio"

process_from_csv(
    csv_path=str(csv_path),
    project_root=PROJECT_ROOT,
    output_dir=str(output_dir),
    mode="scratch",
    augment=True,
)