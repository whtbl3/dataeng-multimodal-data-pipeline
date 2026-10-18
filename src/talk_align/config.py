"""Paths and default parameters. Everything that changes output goes into provenance.params."""

from dataclasses import dataclass
from pathlib import Path

# Project root directory (automatically determines the location of the config.py file)
BASE_DIR = Path(__file__).resolve().parents[2]

# Working directories
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
WORK_DIR = DATA_DIR / "work"
DB_PATH = DATA_DIR / "lab1.duckdb"
OUT_DIR = DATA_DIR / "out"

# Automatically create the directory if it does not yet exist
WORK_DIR.mkdir(parents=True, exist_ok=True)

QUARANTINE_THRESHOLD = 0.3


@dataclass(frozen=True)
class AudioParams:
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    language: str = "en"
    temperature: float = 0.0

SILENCE_THRESHOLD = 0.6
TIME_TOLERANCE_S = 1.0