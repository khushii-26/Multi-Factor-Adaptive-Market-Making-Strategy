import pandas as pd
from pathlib import Path

from hummingbot.strategy_v2.backtesting import BacktestingEngine as BacktestingRunner
from hummingbot.strategy_v2.backtesting.feeds.csv_candles_feed import CsvCandlesFeed

from backtest.sim_connector import SimConnector
from scripts.multifactor import AdvancedMultiFactorPMM

PAIR = "ETH-USDT"
EXCHANGE = "binance"
DATA_PATH = Path("data/binance_ETH-USDT_1m.csv")

assert DATA_PATH.exists(), (
    f"Historical candle file {DATA_PATH} not found. "
    "Download it first with scripts/utility/download_candles.py"
)

feed = CsvCandlesFeed(
    connector=EXCHANGE,
    trading_pair=PAIR,
    filepath=str(DATA_PATH),
    interval="1m",
)

# stub connector with large balances
sim_connector = SimConnector(EXCHANGE, quote_balance=1_000_000, base_balance=0)


class BacktestMultiFactor(AdvancedMultiFactorPMM):
    """Reuse the live strategy but point candle+exchange names to stub."""

    exchange = EXCHANGE
    candle_exchange = EXCHANGE

    def __init__(self, connectors):
        super().__init__(connectors)


runner = BacktestingRunner(
    strategy_cls=BacktestMultiFactor,
    connectors={EXCHANGE: sim_connector},
    market_feeds=[feed],
    start_time=feed.start_timestamp,
    end_time=feed.end_timestamp,
)


def update_order_book(ctx):
    # Each bar close => update simulated order book
    price = ctx.feed_closes[EXCHANGE][PAIR]
    sim_connector.attach_snapshot(price)


runner.add_hook("on_bar", update_order_book)

stats = runner.run()
print("==== Backtest complete ====")
print(stats.to_frame().T)

try:
    runner.plot_equity_curve()  # saves equity_curve.png
    print("Equity curve saved to equity_curve.png")
except Exception:
    pass 