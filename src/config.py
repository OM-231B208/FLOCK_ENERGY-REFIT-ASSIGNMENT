import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"

RAW_DIR = Path(
    os.environ.get("REFIT_RAW_DIR", DOWNLOADS / "REFIT_RAW_081116")
).expanduser()
CLEAN_DIR = Path(
    os.environ.get("REFIT_CLEAN_DIR", DOWNLOADS / "CLEAN_REFIT_081116")
).expanduser()

PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"

SEED = 42


def raw_file(house, part):
    name = f"RAW_House_{house}" if house >= 13 else f"RAW_House{house}"
    return RAW_DIR / f"{name}_Part{part}.csv"


def clean_file(house):
    return CLEAN_DIR / f"CLEAN_House{house}.csv"
