import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# Parameters
symbol = "ETHUSDT"  # Revert to ETH-USDT
interval = "1m"
lookback_days = 30
output_file = f"data/binance_ETH-USDT_1m.csv"

# Binance endpoint
url = "https://api.binance.com/api/v3/klines"

# Calculate start and end times
end_time = int(datetime.now().timestamp() * 1000)
start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)

# Download in chunks (Binance max 1000 per request)
all_rows = []
fetch_time = start_time
print(f"Downloading {symbol} {interval} candles from Binance...")
while fetch_time < end_time:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": fetch_time,
        "endTime": end_time,
        "limit": 1000
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    if not data:
        break
    all_rows.extend(data)
    fetch_time = data[-1][0] + 60_000  # next minute
    print(f"Fetched up to {datetime.fromtimestamp(data[-1][0]/1000)} ({len(all_rows)} rows)")
    if len(data) < 1000:
        break

# Convert to DataFrame
cols = ["timestamp","open","high","low","close","volume","close_time","quote_asset_vol","num_trades","taker_buy_base_vol","taker_buy_quote_vol","ignore"]
df = pd.DataFrame(all_rows, columns=cols)
df = df[["timestamp","open","high","low","close","volume"]]
df["timestamp"] = df["timestamp"].astype(int)
for col in ["open","high","low","close","volume"]:
    df[col] = df[col].astype(float)

# Save to CSV
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False)
print(f"Saved {len(df)} rows to {output_file}") 