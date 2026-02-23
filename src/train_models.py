import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

sys.path.append(str(Path(__file__).parent))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    STOCK_SYMBOLS, SEQUENCE_LENGTH, TRAIN_TEST_SPLIT,
    BATCH_SIZE, EPOCHS, LEARNING_RATE
)
from data_processing.feature_engineering import StockFeatureEngineer
from models.lstm_forecaster import StockForecaster


def train_lstm_model(symbol):
    print(f"\n{'='*60}")
    print(f"Training LSTM for {symbol}")
    print(f"{'='*60}\n")
    
    # load data
    filepath = RAW_DATA_DIR / f"{symbol}_historical.csv"
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        return
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records")
    
    # add features
    engineer = StockFeatureEngineer()
    df = engineer.add_technical_indicators(df)
    
    # prepare data
    feature_cols = ['Close', 'Volume', 'MA_7', 'MA_21', 'RSI', 'MACD', 'Volatility', 'Daily_Return']
    features, target = engineer.prepare_for_lstm(df, target_col='Close', feature_cols=feature_cols)
    
    # normalize
    features_normalized, feature_scaler = engineer.normalize_data(features.values)
    target_normalized, target_scaler = engineer.normalize_data(target.values.reshape(-1, 1))
    target_normalized = target_normalized.flatten()
    
    # create sequences
    forecaster = StockForecaster(input_size=len(feature_cols), sequence_length=SEQUENCE_LENGTH)
    X, y = forecaster.create_sequences(features_normalized, target_normalized)
    
    print(f"Created {len(X)} sequences")
    
    # split train/test
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # train
    history = forecaster.train(
        X_train, y_train,
        X_test, y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # save model
    model_path = MODELS_DIR / f"{symbol}_lstm.pth"
    forecaster.save(model_path)
    
    # save scalers
    scaler_path = MODELS_DIR / f"{symbol}_scalers.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'feature_cols': feature_cols
        }, f)
    print(f"Saved scalers to {scaler_path}")
    
    # evaluate
    predictions = forecaster.predict(X_test)
    predictions = target_scaler.inverse_transform(predictions)
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mse = np.mean((predictions - y_test_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test_actual))
    
    print(f"\nResults for {symbol}:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    
    return forecaster, history


def main():
    print("Starting training...")
    
    for symbol in STOCK_SYMBOLS:
        try:
            train_lstm_model(symbol)
        except Exception as e:
            print(f"Error training {symbol}: {e}")
            continue
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
