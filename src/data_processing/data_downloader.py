import yfinance as yf
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import STOCK_SYMBOLS, RAW_DATA_DIR


class StockDataDownloader:
    def __init__(self, symbols=None, period="5y"):
        self.symbols = symbols if symbols else STOCK_SYMBOLS
        self.period = period
        
    def download_stock_data(self, symbol):
        # download data for one stock
        print(f"Downloading {symbol}...")
        
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=self.period)
            
            if df.empty:
                print(f"Warning: No data for {symbol}")
                return None
            
            df.reset_index(inplace=True)
            df['Symbol'] = symbol
            
            print(f"Got {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error with {symbol}: {e}")
            return None
    
    def download_all(self, save=True):
        data_dict = {}
        
        for symbol in self.symbols:
            df = self.download_stock_data(symbol)
            if df is not None:
                data_dict[symbol] = df
                
                if save:
                    filepath = RAW_DATA_DIR / f"{symbol}_historical.csv"
                    df.to_csv(filepath, index=False)
                    print(f"Saved to {filepath}")
        
        print(f"\nDone! Downloaded {len(data_dict)} stocks")
        return data_dict


if __name__ == "__main__":
    downloader = StockDataDownloader()
    downloader.download_all(save=True)
