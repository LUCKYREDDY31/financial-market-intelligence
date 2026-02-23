import pandas as pd
import numpy as np


class StockFeatureEngineer:
    
    def add_technical_indicators(self, df):
        # add common technical indicators
        df = df.copy()
        
        # moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # bollinger bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # momentum and volatility
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        return df
    
    def prepare_for_lstm(self, df, target_col='Close', feature_cols=None):
        df = df.copy()
        df = df.dropna()  # remove NaN from rolling windows
        
        if feature_cols is None:
            # use all numeric columns except date/symbol
            feature_cols = [col for col in df.columns 
                           if col not in ['Date', 'Symbol', 'Dividends', 'Stock Splits']]
        
        features = df[feature_cols]
        target = df[target_col]
        
        return features, target
    
    def normalize_data(self, data, scaler=None):
        from sklearn.preprocessing import MinMaxScaler
        
        if scaler is None:
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(data)
        else:
            normalized = scaler.transform(data)
        
        return normalized, scaler
