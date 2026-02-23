# config for the project

from pathlib import Path

# paths
ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# make sure dirs exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# stocks we're tracking
STOCK_SYMBOLS = [
    "AAPL",  # Apple
    "GOOGL", # Google
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "TSLA",  # Tesla
]

# time series stuff
SEQUENCE_LENGTH = 60  # days of history to use
TRAIN_TEST_SPLIT = 0.8
FORECAST_DAYS = 5

# model params
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# sentiment model
FINBERT_MODEL = "ProsusAI/finbert"

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Streamlit
STREAMLIT_PORT = 8501