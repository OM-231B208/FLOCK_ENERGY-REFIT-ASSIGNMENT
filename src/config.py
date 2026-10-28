from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Where the extracted datasets live on this computer
RAW_DIR = Path("C:/Users/om885/Downloads/REFIT_RAW_081116")
CLEAN_DIR = Path("C:/Users/om885/Downloads/CLEAN_REFIT_081116")

PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
PROC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def raw_file(house, part):
    name = f"RAW_House_{house}" if house >= 13 else f"RAW_House{house}"
    return RAW_DIR / f"{name}_Part{part}.csv"


def clean_file(house):
    return CLEAN_DIR / f"CLEAN_House{house}.csv"
