import time
import requests
from dotenv import load_dotenv
import os
from scraper import get_fear_greed_index  # Import the scraping function

# Load environment variables from .env file
load_dotenv()

# Configuration settings (use your own API key and secret)
API_KEY = os.getenv('GEMINI_API_KEY')
API_SECRET = os.getenv('GEMINI_API_SECRET')
TRADING_PAIR = "solusd"
BASE_URL = "https://api.gemini.com"
TRADE_AMOUNT = 1  # Amount of SOL to trade

# Define your trading strategy based on the Fear and Greed Index
def trading_strategy(fear_greed_index):
    if fear_greed_index < 25:
        print(f"Index {fear_greed_index}: Extreme Fear. Buying 1 SOL.")
        # Add your buy order logic here
        # place_order("buy", TRADE_AMOUNT, current_price, TRADING_PAIR)
    elif fear_greed_index > 75:
        print(f"Index {fear_greed_index}: Extreme Greed. Shorting 1 SOL.")
        # Add your short/sell order logic here
        # place_order("sell", TRADE_AMOUNT, current_price, TRADING_PAIR)
    else:
        print(f"Index {fear_greed_index}: Neutral. Holding position.")

# Main loop for the bot
while True:
    # Get the current Fear and Greed Index
    index = get_fear_greed_index()
    if index is not None:
        # Execute the trading strategy based on the index
        trading_strategy(index)
    
    # Sleep for an hour (3600 seconds)
    time.sleep(3600)
