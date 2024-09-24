

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
STOP_LOSS_MARGIN = 0.01  # Stop-loss margin (1%)
PROFIT_MARGIN = 0.02     # Take-profit margin (2%)

# Global variable to hold the time difference between local and NTP server time
time_difference = 0

# Function to sync local time with NTP server time
def sync_with_ntp_server():
    global time_difference
    ntp_client = ntplib.NTPClient()
    try:
        response = ntp_client.request('pool.ntp.org', version=3)
        ntp_time = response.tx_time  # NTP time in seconds (no need to convert to milliseconds)
        local_time = time.time()  # Local time in seconds
        time_difference = int(ntp_time - local_time)  # Time difference in seconds
        print(f"NTP time: {ntp_time}, Local time: {local_time}, Time difference: {time_difference} seconds")
    except Exception as e:
        print(f"Failed to sync with NTP server: {e}")
        time_difference = 0  # Reset time difference on failure

# Function to generate a time-based nonce using adjusted time
def generate_nonce():
    global time_difference
    adjusted_time = time.time() + time_difference  # Adjust local time with NTP time difference, in seconds
    return str(int(adjusted_time))  # Return nonce as a string in milliseconds

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
def get_historical_data(trading_pair, interval="1hr", limit=200):
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
    if len(df) >= 14:  # Check if there are enough rows to calculate indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=5).average_true_range()
    else:
        print("Not enough data to calculate indicators. Need at least 14 rows.")
    return df

# Function to calculate dynamic indicator thresholds based on recent data
def calculate_dynamic_thresholds(df):
    # Calculate recent highs and lows for indicators over the past 'n' periods
    recent_period = 50  # Number of periods to calculate recent high/low, adjust as needed
    
    # Ensure we have enough data
    if len(df) < recent_period:
        print("Not enough data to calculate dynamic thresholds.")
        return None, None, None, None
    
    # Calculate RSI high and low over recent period
    recent_rsi_high = df['rsi'].tail(recent_period).max()
    recent_rsi_low = df['rsi'].tail(recent_period).min()
    
    # Calculate MACD high and low over recent period
    recent_macd_high = df['macd'].tail(recent_period).max()
    recent_macd_low = df['macd'].tail(recent_period).min()
    
    # Print calculated values for debugging
    print(f"Recent RSI High: {recent_rsi_high}, Recent RSI Low: {recent_rsi_low}")
    print(f"Recent MACD High: {recent_macd_high}, Recent MACD Low: {recent_macd_low}")
    
    return recent_rsi_high, recent_rsi_low, recent_macd_high, recent_macd_low

# Function to place an order on Gemini
def place_order(side, amount, symbol, price=None):
    try:
        url = f"{BASE_URL}/v1/order/new"
        nonce = generate_nonce()  # Generate a time-based nonce using adjusted time
        payload = {
            "request": "/v1/order/new",
            "nonce": nonce,
            "symbol": symbol,
            "amount": str(amount),  # Amount must be a string
            "price": str(price) if price else "1.00",  # Set a default price if not provided (for market orders)
            "side": side,  # "buy" or "sell"
            "type": "exchange limit",  # You can use "exchange market" for market orders
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

# Updated trading strategy based on dynamic thresholds
def trading_strategy(df, cfg_index):
    # Calculate dynamic thresholds
    recent_rsi_high, recent_rsi_low, recent_macd_high, recent_macd_low = calculate_dynamic_thresholds(df)
    
    # If any of the thresholds are None, do not proceed
    if None in [recent_rsi_high, recent_rsi_low, recent_macd_high, recent_macd_low]:
        print("Not enough data to calculate dynamic thresholds. Holding position.")
        return
    
    # Adjust these margins if needed
    rsi_buy_margin = 0.1  # Margin below the recent low to trigger a buy
    rsi_sell_margin = 0.1  # Margin above the recent high to trigger a sell
    
    # Calculate buy/sell RSI thresholds
    rsi_buy_threshold = recent_rsi_low + (recent_rsi_high - recent_rsi_low) * rsi_buy_margin
    rsi_sell_threshold = recent_rsi_high - (recent_rsi_high - recent_rsi_low) * rsi_sell_margin
    
    # MACD dynamic thresholds
    macd_buy_threshold = recent_macd_low * 0.8  # 20% below recent MACD low to trigger a buy
    macd_sell_threshold = recent_macd_high * 1.2  # 20% above recent MACD high to trigger a sell
    
    # Get the latest data point
    latest_data = df.iloc[-1]
    current_price = latest_data['close']
    rsi = latest_data['rsi']
    macd = latest_data['macd']
    atr = latest_data['atr']
    
    # Calculate stop-loss and take-profit levels based on ATR
    stop_loss = current_price - (STOP_LOSS_MARGIN * atr)
    take_profit = current_price + (PROFIT_MARGIN * atr)

    # Check current SOL balance
    sol_balance = get_sol_balance()

    # Buy condition: CFGI < 30, RSI near recent low, MACD near recent low, and no SOL in account
    if cfg_index < 30 and rsi < rsi_buy_threshold and macd < macd_buy_threshold and sol_balance == 0:
        print(f"CFGI: {cfg_index}, RSI: {rsi} - Bullish signal. Buying 1 SOL at {current_price}")
        place_order("buy", TRADE_AMOUNT, TRADING_PAIR, current_price)
        print(f"Set Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")
    # Sell condition: CFGI > 70, RSI near recent high, MACD near recent high, and SOL in account
    elif cfg_index > 70 and rsi > rsi_sell_threshold and macd > macd_sell_threshold and sol_balance >= 1:
        print(f"CFGI: {cfg_index}, RSI: {rsi} - Bearish signal. Shorting 1 SOL at {current_price}")
        place_order("sell", TRADE_AMOUNT, TRADING_PAIR, current_price)
        print(f"Set Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")
    else:
        print(f"CFGI: {cfg_index}, RSI: {rsi} - Holding position.")

# Main loop for the bot
sync_with_ntp_server()  # Sync local time with NTP server time at the start

while True:
    # Get the current Fear and Greed Index
    cfg_index = get_cfgi()
    if cfg_index is not None:
        # Get historical data for the last 7 days
        historical_data = get_historical_data(TRADING_PAIR, interval="1hr", limit=500)  # Increase limit for more data
        
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
