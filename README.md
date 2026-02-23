# Financial Market Intelligence Platform

## Overview

A tool for analyzing financial markets using deep learning. This platform predicts stock price movements with an LSTM model and analyzes market sentiment from news articles using FinBERT. It's all wrapped up in an interactive Streamlit dashboard and a FastAPI backend.

## Features

- **Stock Price Prediction**: LSTM model to forecast future stock prices.
- **Sentiment Analysis**: FinBERT model to analyze financial news sentiment.
- **Interactive Dashboard**: Streamlit app for visualizing data and predictions.
- **REST API**: FastAPI backend to serve model predictions.
- **Containerized**: Docker support for easy setup.

## Tech Stack

| Category      | Technology                                      |
|---------------|-------------------------------------------------|
| **Backend**     | FastAPI, Uvicorn                                |
| **Frontend**    | Streamlit                                       |
| **Deep Learning** | PyTorch, Hugging Face Transformers              |
| **Data**        | Pandas, NumPy, Scikit-learn, yfinance           |
| **Deployment**  | Docker                                          |

## Getting Started

### Installation

1.  **Clone the repo**
    ```bash
    git clone https://github.com/LUCKYREDDY31/financial-market-intelligence.git
    cd financial-market-intelligence
    ```

2.  **Run the setup script**
    ```bash
    ./setup.sh
    ```
    This will create a virtual environment, install dependencies, download data, and train the models.

3.  **Activate the environment**
    ```bash
    source venv/bin/activate
    ```

### Usage

-   **Run the dashboard**
    ```bash
    streamlit run app/dashboard.py
    ```
    The app will be available at `http://localhost:8501`.

-   **Run the API**
    ```bash
    python src/api/main.py
    ```
    The API will be available at `http://localhost:8000` with docs at `/docs`.

## Docker

1.  **Build the image**
    ```bash
    docker build -t market-intelligence .
    ```

2.  **Run the container**
    ```bash
    docker run -p 8501:8501 market-intelligence
    ```
    The app will be available at `http://localhost:8501`.

## Project Structure

```
financial-market-intelligence/
├── app/                    # Streamlit dashboard
├── src/
│   ├── api/               # FastAPI backend
│   ├── data_processing/   # Data download & features
│   ├── models/            # LSTM & FinBERT
│   └── train_models.py    # Training script
├── data/                  # Data storage
└── Dockerfile             # For deployment
```

## How It Works

1.  **Data Ingestion**: Historical stock data is downloaded from Yahoo Finance using `yfinance`.
2.  **Feature Engineering**: Technical indicators (RSI, MACD, etc.) are calculated to create features for the LSTM model.
3.  **Model Training**: The LSTM model is trained on the historical data to predict future prices. The FinBERT model is used for sentiment analysis on news headlines.
4.  **API & Dashboard**: The trained models are served via a FastAPI backend and visualized in a Streamlit dashboard.

## Disclaimer

This project is for educational purposes only and is not financial advice. Do not use it for making real investment decisions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.