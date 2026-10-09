"""Wspólna konfiguracja Zadania 1 (Część I – anotacja PolEmo 2.0)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Korpus -----------------------------------------------------------------
DATA_URL = "hf://datasets/clarin-pl/polemo2-official/data/all.sentence.train.txt"
LABEL_SEP = " __label__"  # podwójne podkreślniki
GOLD_MAP = {"z_plus_m": "POZ", "z_minus_m": "NEG", "z_zero": "NEU", "z_amb": "AMB"}

# --- Schemat anotacji -------------------------------------------------------
SENT_LABELS = ["POZ", "NEG", "NEU", "AMB"]
TOKEN_LABELS = ["WYR_POZ", "WYR_NEG", "WZM", "OSL", "ODW"]
NO_LABEL = "O"  # puste pole w arkuszu
ANNOTATORS = ["A1", "A2", "A3"]  # A3 = superanotator (procedura 2+1)

# --- Losowanie --------------------------------------------------------------
N_SAMPLE = 100  # zdań na iterację, po N_SAMPLE / 4 z każdej klasy gold
MIN_TOKENS, MAX_TOKENS = 5, 40
SEEDS = {1: 20261, 2: 20262}  # osobny seed dla każdej iteracji

# --- Ścieżki ----------------------------------------------------------------
PROCESSED = ROOT / "data" / "processed" / "polemo_sentences.parquet"
CORPUS_STATS = ROOT / "results" / "corpus_stats.json"
ANNOT_DIR = ROOT / "annotation"
GUIDELINES_DIR = ANNOT_DIR / "guidelines"
EXAMPLE_IDS = GUIDELINES_DIR / "example_ids.txt"  # zdania z wytycznych – wykluczone z próbek


def iter_dir(iteration: int) -> Path:
    return ANNOT_DIR / f"iter{iteration}"
