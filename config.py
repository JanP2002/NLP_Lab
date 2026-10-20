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
RESULTS_DIR = ROOT / "results"

# --- Zgodność ---------------------------------------------------------------
BOOTSTRAP = {"n": 2000, "seed": 2026}  # przedziały ufności: losowanie zdań ze zwracaniem

# --- Osadzenia słów (Część II) ------------------------------------------------
MODELS_DIR = ROOT / "models"  # duże pliki modeli – poza repozytorium (.gitignore)
W2V = {  # word2vec IPI PAN; ścieżkę można nadpisać zmienną środowiskową W2V_PATH
    "file": "nkjp+wiki-forms-all-100-skipg-ns.txt.gz",
    "url": "http://dsmodels.nlp.ipipan.waw.pl/dsmodels/nkjp+wiki-forms-all-100-skipg-ns.txt.gz",
    "limit": 300_000,  # najczęstsze formy (zalecenie z uwag technicznych)
}
W2V_POLEMO = {  # własny word2vec na zdaniach PolEmo (gensim), parametry jak w modelu IPI PAN
    "vector_size": 100, "sg": 1, "negative": 5, "window": 5, "min_count": 3,
    "epochs": 15, "seed": 2026, "workers": 1,  # workers=1 → powtarzalny wynik
}
FASTTEXT_FILE = "kgr10.plain.skipgram.dim100.neg10.bin"  # zalecenie z uwag technicznych
WORD_EMB_DIR = RESULTS_DIR / "word_embeddings"


def iter_dir(iteration: int) -> Path:
    return ANNOT_DIR / f"iter{iteration}"


def filled_sheets_dir(iteration: int) -> Path:
    """Wypełnione arkusze pobrane z Google Sheets (konwencja zespołu: iterN_artefacts/)."""
    return ANNOT_DIR / f"iter{iteration}_artefacts"


def export_dir(iteration: int) -> Path:
    """Eksport oetykietowanych danych (src.export)."""
    return iter_dir(iteration) / "export"


def results_dir(iteration: int) -> Path:
    return RESULTS_DIR / f"iter{iteration}"
