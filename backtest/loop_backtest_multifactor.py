import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import csv
from decimal import Decimal
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from hummingbot.core.data_type.order_candidate import OrderCandidate

print("Script started")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)
    logger.info(f"Added {project_root} to Python path")

try:
    from backtest.sim_connector import SimConnector
    from scripts.multifactor import AdvancedMultiFactorPMM as Strategy
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    logger.error("Make sure you're running this script from the project root with the correct Python environment")
    sys.exit(1)

# CONFIG
CSV_FILE = Path("data/binance_ETH-USDT_1m.csv")  # candle file to replay
TRADING_PAIR = "ETH-USDT"
EXCHANGE_NAME = "sim"  # name exposed to the strategy
START_QUOTE_BAL = Decimal("100000")  # 100k USDT paper balance
# TAKER_FEE = Decimal("0.0005")  # 0.05% fee for realistic backtest
TAKER_FEE = Decimal("0")  # 0 fee for testing profitability
MAKER_REBATE = Decimal('0.0002')  # 0.02% maker rebate

assert CSV_FILE.exists(), f"Historical candles not found: {CSV_FILE}"

# Initialize stub connector and strategy
try:
    connector = SimConnector(EXCHANGE_NAME, quote_balance=float(START_QUOTE_BAL), base_balance=0)
    logger.info("Created SimConnector successfully")
    print("SimConnector created")
except Exception as e:
    logger.error(f"Failed to create SimConnector: {str(e)}")
    print(f"Failed to create SimConnector: {str(e)}")
    sys.exit(1)

class BacktestStrategy(Strategy):
    """Subclass that injects stub connector without Cython type issues."""
    def __init__(self, connector):
        super().__init__(connectors={EXCHANGE_NAME: connector})
        self._offline_orders = []
        self.exchange = EXCHANGE_NAME
        self.candle_exchange = EXCHANGE_NAME
        self.trading_pair = TRADING_PAIR
        self.markets = {EXCHANGE_NAME: {TRADING_PAIR}}
        self._price_history = []
        self._volume_history = []
        self.create_timestamp = 0
        self._current_timestamp = 0
        logger.info("Strategy initialized with multifactor logic.")
    def on_tick(self):
        # Use the real multifactor logic
        if self.create_timestamp <= self._current_timestamp:
            self.cancel_all_orders()
            self.update_all_metrics()
            proposal: List[OrderCandidate] = self.create_advanced_proposal()
            logger.info(f"Proposal generated: {proposal}")
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            logger.info(f"Proposal after budget check: {proposal_adjusted}")
            self.place_orders(proposal_adjusted)
            logger.info(f"Orders placed: {proposal_adjusted}")
            self.create_timestamp = self.order_refresh_time + self._current_timestamp
    def buy(self, connector_name, trading_pair, amount, order_type, price, position_action=None):
        oid = f"B-{len(self._offline_orders)}"
        order = type("DummyOrder", (), {
            "is_buy": True,
            "trading_pair": trading_pair,
            "quantity": float(amount),
            "price": float(price),
            "client_order_id": oid,
            "creation_timestamp": self._current_timestamp,
        })()
        self._offline_orders.append(order)
        logger.info(f"Created BUY order: {amount} @ {price}")
        return oid
    def sell(self, connector_name, trading_pair, amount, order_type, price, position_action=None):
        oid = f"S-{len(self._offline_orders)}"
        order = type("DummyOrder", (), {
            "is_buy": False,
            "trading_pair": trading_pair,
            "quantity": float(amount),
            "price": float(price),
            "client_order_id": oid,
            "creation_timestamp": self._current_timestamp,
        })()
        self._offline_orders.append(order)
        logger.info(f"Created SELL order: {amount} @ {price}")
        return oid
    def cancel(self, connector_name, trading_pair, order_id):
        self._offline_orders = [o for o in self._offline_orders if o.client_order_id != order_id]
        logger.info(f"Cancelled order: {order_id}")
    def get_active_orders(self, connector_name):
        return self._offline_orders

try:
    strategy = BacktestStrategy(connector)
    logger.info("Created BacktestStrategy successfully")
    print("BacktestStrategy created")
except Exception as e:
    logger.error(f"Failed to create BacktestStrategy: {str(e)}")
    print(f"Failed to create BacktestStrategy: {str(e)}")
    sys.exit(1)

equity_curve: List[float] = []
price_history: List[float] = []
volume_history: List[float] = []
trades_history: List[dict] = []
trade_pnls: List[float] = []  # Store per-trade PnL
running_pnl: List[float] = []  # Store cumulative PnL
min_price_move = Decimal('0')  # realize PnL immediately
min_holding_period = 0  # bars
fifo_inventory = []  # List of (qty, price, entry_bar)
bar_count = 0

def push_price(price: float, timestamp: float, volume: float = None):
    depth = volume if volume is not None else 10.0
    connector.attach_snapshot(price, timestamp=timestamp, depth=depth)
    price_history.append(price)
    logger.debug(f"Updated price to {price} with depth {depth}")

def should_fill_order(order, current_price: float) -> bool:
    return True

print("Starting backtest replay...")
logger.info("Starting backtest replay...")

with CSV_FILE.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts_raw = int(row["timestamp"])
        ts = ts_raw / 1000 if ts_raw > 1e12 else ts_raw
        price = float(row["close"])
        volume = float(row.get("volume", 10.0))
        push_price(price, ts, volume)
        strategy._current_timestamp = ts
        bar_count += 1
        strategy.on_tick()
        active_orders = list(strategy.get_active_orders(EXCHANGE_NAME))
        for order in active_orders:
            if should_fill_order(order, price):
                qty = Decimal(order.quantity)
                if order.is_buy:
                    cost = qty * Decimal(price) * (1 + TAKER_FEE)
                    if connector.get_available_balance("USDT") >= cost:
                        connector._quote -= cost
                        connector._base += qty
                        # Maker rebate
                        connector._quote += qty * Decimal(price) * MAKER_REBATE
                        fifo_inventory.append((qty, Decimal(price), bar_count))
                        print(f"BUY {qty} @ {price}")
                        trades_history.append({
                            "timestamp": ts,
                            "side": "buy",
                            "price": price,
                            "amount": float(qty),
                            "cost": float(cost),
                            "pnl": 0
                        })
                        trade_pnls.append(0)
                        logger.info(f"Filled BUY order: {qty} @ {price}, cost: {cost}")
                else:
                    if connector.get_available_balance("ETH") >= qty:
                        proceeds = qty * Decimal(price) * (1 - TAKER_FEE)
                        connector._base -= qty
                        connector._quote += proceeds
                        # Maker rebate
                        connector._quote += qty * Decimal(price) * MAKER_REBATE
                        # FIFO matching for PnL
                        qty_to_match = qty
                        realized_pnl = Decimal("0")
                        while qty_to_match > 0 and fifo_inventory:
                            inv_qty, inv_price, entry_bar = fifo_inventory[0]
                            match_qty = min(inv_qty, qty_to_match)
                            # Only realize PnL if price moved at least min_price_move and min holding period
                            if (Decimal(price) >= inv_price * (1 + min_price_move)) and (bar_count - entry_bar >= min_holding_period):
                                pnl = (Decimal(price) - inv_price) * match_qty
                                realized_pnl += pnl
                                fifo_inventory[0] = (inv_qty - match_qty, inv_price, entry_bar)
                                if fifo_inventory[0][0] == 0:
                                    fifo_inventory.pop(0)
                                qty_to_match -= match_qty
                            else:
                                # Not enough price move or not enough holding period, don't realize PnL, break
                                break
                        print(f"SELL {qty} @ {price} | Realized PnL: {realized_pnl}")
                        trades_history.append({
                            "timestamp": ts,
                            "side": "sell",
                            "price": price,
                            "amount": float(qty),
                            "proceeds": float(proceeds),
                            "pnl": float(realized_pnl)
                        })
                        trade_pnls.append(float(realized_pnl))
                        logger.info(f"Filled SELL order: {qty} @ {price}, proceeds: {proceeds}, pnl: {realized_pnl}")
                strategy.cancel(EXCHANGE_NAME, order.trading_pair, order.client_order_id)
        equity = (
            connector.get_balance("ETH") * Decimal(price) + connector.get_balance("USDT")
        )
        equity_curve.append(float(equity))
        volume_history.append(volume)
        # Running PnL: equity - initial balance
        running_pnl.append(float(equity) - float(START_QUOTE_BAL))

# --- Force-close any remaining inventory at last price ---
if fifo_inventory:
    print("\nUnclosed inventory at end of backtest:")
    for qty, entry_price, entry_bar in fifo_inventory:
        print(f"Qty: {qty}, Entry price: {entry_price}, Entry bar: {entry_bar}")
    last_price = Decimal(str(price_history[-1]))
    print(f"Force-closing all remaining inventory at last price: {last_price}")
    for qty, entry_price, entry_bar in fifo_inventory:
        pnl = (last_price - entry_price) * qty
        trades_history.append({
            "timestamp": ts,
            "side": "force_close",
            "price": float(last_price),
            "amount": float(qty),
            "proceeds": float(qty * last_price),
            "pnl": float(pnl)
        })
        trade_pnls.append(float(pnl))
        print(f"Force-closed {qty} @ {last_price} | Realized PnL: {pnl}")
    fifo_inventory.clear()

print("Backtest completed.")
print(f"Final equity: {equity_curve[-1]:.2f} USDT")
print(f"PnL: {equity_curve[-1] - float(START_QUOTE_BAL):.2f} USDT")
print(f"Number of trades: {len(trades_history)}")
logger.info("Back-test completed.")
logger.info(f"Final equity: {equity_curve[-1]:.2f} USDT")
logger.info(f"PnL: {equity_curve[-1] - float(START_QUOTE_BAL):.2f} USDT")
logger.info(f"Number of trades: {len(trades_history)}")

# --- Profit Analytics Pack ---
import numpy as np

def sharpe_ratio(pnls, risk_free_rate=0.0):
    pnls = [float(p) for p in pnls]
    if len(pnls) < 2 or np.std(pnls) == 0:
        return 0.0
    return (np.mean(pnls) - risk_free_rate) / np.std(pnls) * np.sqrt(len(pnls))

# Only consider nonzero PnL trades for stats
nonzero_pnls = [p for p in trade_pnls if p != 0]
num_trades = len(nonzero_pnls)
num_wins = len([p for p in nonzero_pnls if p > 0])
num_losses = len([p for p in nonzero_pnls if p < 0])
win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0
avg_profit = np.mean(nonzero_pnls) if num_trades > 0 else 0
max_profit = np.max(nonzero_pnls) if num_trades > 0 else 0
max_loss = np.min(nonzero_pnls) if num_trades > 0 else 0
sharpe = sharpe_ratio(nonzero_pnls)

print("\n--- Profit Analytics ---")
print(f"Total trades (with PnL): {num_trades}")
print(f"Win rate: {win_rate:.2f}%")
print(f"Average profit per trade: {avg_profit:.2f} USDT")
print(f"Max profit: {max_profit:.2f} USDT")
print(f"Max loss: {max_loss:.2f} USDT")
print(f"Sharpe ratio: {sharpe:.2f}")

# --- Enhanced Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
ax1.plot(price_history, label='Price', color='blue', alpha=0.6)
ax1.set_ylabel('Price (USDT)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1_twin = ax1.twinx()
ax1_twin.plot(equity_curve, label='Equity', color='green')
ax1_twin.set_ylabel('Equity (USDT)', color='green')
ax1_twin.tick_params(axis='y', labelcolor='green')
ax2.bar(range(len(volume_history)), volume_history, alpha=0.6, color='gray')
ax2.set_ylabel('Volume')
ax2.set_xlabel('Bars')
ax3.plot(running_pnl, label='Running PnL', color='red')
ax3.set_ylabel('Running PnL (USDT)', color='red')
ax3.set_xlabel('Bars')
ax3.legend()
plt.title("Backtest Results - AdvancedMultiFactorPMM")
plt.tight_layout()
plt.savefig("backtest_results.png")
logger.info("Saved backtest results to backtest_results.png")

trades_df = pd.DataFrame(trades_history)
trades_df.to_csv("trade_history.csv", index=False)
print("Saved trade history to trade_history.csv") 