#!/bin/bash

echo "Setting up Financial Market Intelligence Platform..."

# create venv
echo "Creating virtual environment..."
python3 -m venv venv

# activate
echo "Activating venv..."
source venv/bin/activate

# install
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# download data
echo "Downloading stock data..."
python3 src/data_processing/data_downloader.py

# train models
echo "Training models (this might take a while)..."
python3 src/train_models.py

echo ""
echo "Setup complete!"
echo ""
echo "Run the dashboard:"
echo "  streamlit run app/dashboard.py"
echo ""
echo "Run the API:"
echo "  python3 src/api/main.py"
echo ""
