import requests
import time
import pandas as pd
import ta
import base64
import hashlib
import hmac
import json
import ntplib
from datetime import datetime, timedelta
from scraper import get_cfgi  # Import the CFGI scraping function
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configuration settings (use your own API key and secret)
API_KEY = os.getenv('GEMINI_API_KEY')
API_SECRET = os.getenv('GEMINI_API_SECRET').encode()  # Ensure it's encoded in bytes
TRADING_PAIR = "solusd"
BASE_URL = "https://api.gemini.com"
TRADE_AMOUNT = 1  # Amount of SOL to trade

# Set global parameters for trading
STOP_LOSS_MARGIN = 1.5  # Multiplier for ATR-based stop-loss (1.5x ATR)
TAKE_PROFIT_MARGIN = 2.0  # Multiplier for ATR-based take-profit (2.0x ATR)

# Global variable to hold the time difference between local and NTP server time
time_difference = 0

# Function to sync local time with NTP server
def sync_with_ntp_server():
    global time_difference
    ntp_client = ntplib.NTPClient()
    try:
        response = ntp_client.request('pool.ntp.org', version=3)
        ntp_time = response.tx_time  # NTP time in seconds
        local_time = time.time()  # Local time in seconds
        time_difference = int(ntp_time - local_time)  # Time difference in seconds
        print(f"NTP time: {ntp_time}, Local time: {local_time}, Time difference: {time_difference} seconds")
    except Exception as e:
        print(f"Failed to sync with NTP server: {e}")
        time_difference = 0  # Reset time difference on failure

# Function to generate a time-based nonce using adjusted time in seconds
def generate_nonce():
    global time_difference
    adjusted_time = time.time() + time_difference  # Adjust local time with NTP time difference, in seconds
    return str(int(adjusted_time))  # Return nonce as a string in seconds

# Function to generate the Gemini API signature
def generate_signature(payload, secret_key):
    # Encode the payload as JSON and then base64 encode it
    encoded_payload = json.dumps(payload).encode()
    b64 = base64.b64encode(encoded_payload)
    # Create HMAC SHA384 signature using the encoded payload and the API secret
    signature = hmac.new(secret_key, b64, hashlib.sha384).hexdigest()
    return b64, signature

# Function to get current SOL balance from Gemini
def get_sol_balance():
    try:
        url = f"{BASE_URL}/v1/balances"
        nonce = generate_nonce()  # Generate a time-based nonce using adjusted time
        payload = {
            "request": "/v1/balances",
            "nonce": nonce
        }

        # Generate signature and headers
        b64, signature = generate_signature(payload, API_SECRET)
        headers = {
            "Content-Type": "text/plain",
            "Content-Length": "0",  # Gemini API expects this to be 0
            "X-GEMINI-APIKEY": API_KEY,
            "X-GEMINI-PAYLOAD": b64.decode(),  # Convert encoded payload to string
            "X-GEMINI-SIGNATURE": signature,
            "Cache-Control": "no-cache"
        }

        # Make the API request
        response = requests.post(url, headers=headers)
        
        # Print response for debugging
        print(f"Response from Gemini API: {response.text}")

        # Check for common error reasons
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return 0.0

        # Parse the JSON response
        try:
            balances = response.json()
        except json.JSONDecodeError:
            print("Error: Failed to parse JSON response from Gemini API.")
            return 0.0

        # Check if balances is a list and return the SOL balance
        if isinstance(balances, list):
            for balance in balances:
                print(f"Balance Record: {balance}")  # Print each balance record for debugging
                if balance['currency'] == 'SOL':
                    return float(balance['available'])
            print("Error: SOL balance not found in the response.")
        else:
            print("Unexpected response format. Expected a list of balances.")
            print(balances)

        return 0.0
    except Exception as e:
        print(f"Error fetching SOL balance: {e}")
        return 0.0

# Function to get historical data
def get_historical_data(trading_pair, interval="1day", limit=200):
    url = f"{BASE_URL}/v2/candles/{trading_pair}/{interval}?limit={limit}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df[['open', 'high', 'low', 'close', 'volume']]
            else:
                print(f"No data returned from API for {trading_pair}.")
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        else:
            print(f"Failed to fetch data from API. Status code: {response.status_code}")
            print(response.text)  # Print response for debugging
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

# Function to calculate indicators
def calculate_indicators(df):
    if len(df) >= 21:  # Ensure enough data points for the longest indicator window
        # Calculate EMAs
        df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        
        # Calculate Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        
        # Calculate ATR for volatility-based stops
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    else:
        print("Not enough data to calculate indicators. Need at least 21 rows.")
    return df

# Function to place an order on Gemini
def place_order(side, amount, symbol, price=None):
    try:
        url = f"{BASE_URL}/v1/order/new"
        nonce = generate_nonce()  # Generate a time-based nonce using adjusted time
        payload = {
            "request": "/v1/order/new",
            "nonce": nonce,
            "symbol": symbol,
            "amount": str(amount),
            "price": str(price),
            "side": side,
            "type": "exchange limit",
            "options": ["immediate-or-cancel"]  # Optional: Set order options
        }

        # Generate signature and headers
        b64, signature = generate_signature(payload, API_SECRET)
        headers = {
            "Content-Type": "text/plain",
            "Content-Length": "0",  # Gemini API expects this to be 0
            "X-GEMINI-APIKEY": API_KEY,
            "X-GEMINI-PAYLOAD": b64.decode(),  # Convert encoded payload to string
            "X-GEMINI-SIGNATURE": signature,
            "Cache-Control": "no-cache"
        }

        # Make the API request
        response = requests.post(url, headers=headers)
        
        # Print response for debugging
        print(f"Response from Gemini API: {response.text}")

        # Check for common error reasons
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return None

        # Parse the JSON response
        try:
            order_result = response.json()
            return order_result
        except json.JSONDecodeError:
            print("Error: Failed to parse JSON response from Gemini API.")
            return None
    except Exception as e:
        print(f"Error placing {side} order: {e}")
        return None

# Updated trading strategy based on new indicators
def trading_strategy(df, cfg_index):
    # Get the latest data point
    latest_data = df.iloc[-1]
    current_price = latest_data['close']
    ema9 = latest_data['ema9']
    ema21 = latest_data['ema21']
    stoch_k = latest_data['stoch_k']
    stoch_d = latest_data['stoch_d']
    bb_upper = latest_data['bb_upper']
    bb_lower = latest_data['bb_lower']
    atr = latest_data['atr']

    # Calculate stop-loss and take-profit levels based on ATR
    stop_loss = current_price - (STOP_LOSS_MARGIN * atr)
    take_profit = current_price + (TAKE_PROFIT_MARGIN * atr)

    # Check current SOL balance
    sol_balance = get_sol_balance()

    # Buy condition: EMA crossover, Stoch below 15, price near lower Bollinger Band, and no SOL in account
    if ema9 > ema21 and stoch_k < 15 and stoch_d < 15 and current_price <= bb_lower and sol_balance == 0:
        print(f"CFGI: {cfg_index}, EMA Crossover, Stochastic Oversold, Near Lower BB - Buying 1 SOL at {current_price}")
        place_order("buy", TRADE_AMOUNT, TRADING_PAIR, current_price)
        print(f"Set Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")
    # Sell condition: EMA crossover, Stoch above 85, price near upper Bollinger Band, and SOL in account
    elif ema9 < ema21 and stoch_k > 85 and stoch_d > 85 and current_price >= bb_upper and sol_balance >= 1:
        print(f"CFGI: {cfg_index}, EMA Crossover, Stochastic Overbought, Near Upper BB - Selling 1 SOL at {current_price}")
        place_order("sell", TRADE_AMOUNT, TRADING_PAIR, current_price)
        print(f"Set Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")
    else:
        print(f"CFGI: {cfg_index}, EMA: {ema9} vs {ema21}, Stoch: {stoch_k}/{stoch_d}, Price: {current_price} - Holding position.")

# Main loop for the bot
sync_with_ntp_server()  # Sync local time with NTP server time at the start

while True:
    # Get the current Fear and Greed Index
    cfg_index = get_cfgi()
    if cfg_index is not None:
        # Get historical data for the last 7 days
        historical_data = get_historical_data(TRADING_PAIR, interval="1day", limit=200)  # Increase limit for more data
        
        # Print number of rows and check for NaNs
        print(f"Number of rows in historical data: {len(historical_data)}")
        print(historical_data.head())  # Print first few rows
        print(historical_data.isnull().sum())  # Check for missing values

        if not historical_data.empty:  # Check if the DataFrame is not empty
            # Calculate technical indicators if there's enough data
            historical_data = calculate_indicators(historical_data)
            # Execute the trading strategy based on indicators and CFGI
            trading_strategy(historical_data, cfg_index)

    # Sleep for an hour (3600 seconds)
    time.sleep(3600)

