import traceback
from decimal import Decimal
import logging
from typing import Dict

import pandas as pd  # Import pandas if converting DataFrames to list of tuples

from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.core.data_type.common import PriceType, OrderType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.connector.connector_base import ConnectorBase


class AltLiquidityThresholdMaker(ScriptStrategyBase):
    # User-configurable parameters for the strategy
    exchange = "binance_paper_trade"  # Use Binance paper trading for simulation
    trading_pair = "ETH-USDT"  # Trading pair (ETH/USDT)
    price_source = PriceType.MidPrice  # Mid price as the reference for orders

    order_refresh_time = 15  # Time in seconds between refreshing orders
    order_amount = Decimal("0.01")  # Amount of base asset to buy or sell per order

    # Liquidity thresholds (based on volume in ±1% of mid price)
    low_liquidity_cutoff = Decimal("50")
    medium_liquidity_cutoff = Decimal("200")

    # Spread settings for each liquidity tier
    spread_low_liquidity = Decimal("0.01")  # 1.00% spread for low liquidity
    spread_medium_liquidity = Decimal("0.005")  # 0.50% spread for medium liquidity
    spread_high_liquidity = Decimal("0.002")  # 0.20% spread for high liquidity

    # Inventory adjustment settings
    target_ratio = Decimal("0.5")  # Desired ratio of base asset in portfolio
    inventory_adjustment_factor = Decimal("0.001")  # Adjustment factor for inventory imbalance

    markets = {exchange: {trading_pair}}  # Define the markets to trade in

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        """
        Initializes the strategy. Connectors are passed to manage connections to the exchange.
        """
        super().__init__(connectors)
        self.base_asset, self.quote_asset = self.trading_pair.split("-")  # Split the trading pair into base and quote
        self.create_timestamp = 0  # Initialize timestamp for order refresh
        self.last_mid_price = None  # Initialize the last mid price (used for spread calculation)

    def on_tick(self):
        """
        The on_tick function runs every tick to manage order placement and inventory adjustment.
        """
        now = self.current_timestamp
        if now < self.create_timestamp:
            return  # Wait until the next scheduled refresh

        try:
            # Get the current mid price (average of bid-ask spread)
            mid_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            self.last_mid_price = mid_price  # Store the last mid price

            # Fetch the current order book (bids and asks)
            ob = self.connectors[self.exchange].get_order_book(self.trading_pair)
            if not ob:
                self.logger().warning("Order book is None or empty.")
                return  # If no order book data is available, return

            # Extract bids and asks from the order book snapshot
            bids, asks = ob.snapshot  # Might be DataFrames in paper trade mode

            # Log the order book for debugging purposes
            self.logger().info(f"Bids snapshot: {bids}")
            self.logger().info(f"Asks snapshot: {asks}")

            # Convert DataFrame to list of tuples if the order book is a DataFrame
            if isinstance(bids, pd.DataFrame):
                bids = list(zip(bids["price"], bids["amount"]))
            if isinstance(asks, pd.DataFrame):
                asks = list(zip(asks["price"], asks["amount"]))

            # Cancel all existing orders before placing new ones
            self.cancel_all_orders()

            # Compute the liquidity score based on the order book data
            liquidity_score = self.compute_liquidity(bids, asks, mid_price, Decimal("0.01"))

            # Choose the appropriate spread based on liquidity conditions
            if liquidity_score < self.low_liquidity_cutoff:
                used_spread = self.spread_low_liquidity
                tier_label = "LOW"
            elif liquidity_score < self.medium_liquidity_cutoff:
                used_spread = self.spread_medium_liquidity
                tier_label = "MEDIUM"
            else:
                used_spread = self.spread_high_liquidity
                tier_label = "HIGH"

            # Log the liquidity score, tier, and spread being used
            self.logger().info(f"Liquidity={liquidity_score:.2f}, Tier={tier_label}, Spread={used_spread*100:.2f}%")

            # Adjust the spreads based on inventory ratio
            buy_spread, sell_spread = self.adjust_spreads_for_inventory(used_spread)
            buy_price = mid_price * (Decimal("1") - buy_spread)  # Calculate buy price
            sell_price = mid_price * (Decimal("1") + sell_spread)  # Calculate sell price
            self.logger().info(f"Placing BUY at {buy_price:.4f}, SELL at {sell_price:.4f}")

            # Create buy and sell order candidates
            buy_order = OrderCandidate(
                trading_pair=self.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=self.order_amount,
                price=buy_price
            )
            sell_order = OrderCandidate(
                trading_pair=self.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=self.order_amount,
                price=sell_price
            )
            proposal = [buy_order, sell_order]

            # Adjust the orders according to available budget
            adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal)
            for order in adjusted:
                self.place_order(order)

            # Update the timestamp for the next order refresh
            self.create_timestamp = now + self.order_refresh_time

        except Exception as e:
            # Log any errors that occur during the tick process
            self.logger().error(f"Error in on_tick: {str(e)}")
            tb = "".join(traceback.format_exc())
            self.logger().error(f"Traceback:\n{tb}")

    def compute_liquidity(self, bids_list, asks_list, mid_price: Decimal, threshold_pct: Decimal) -> Decimal:
        """
        Computes the liquidity score based on bid and ask volumes within a ±threshold_pct range of the mid price.
        """
        lower_bound = mid_price * (Decimal("1") - threshold_pct)  # Define the lower bound for liquidity
        upper_bound = mid_price * (Decimal("1") + threshold_pct)  # Define the upper bound for liquidity

        total_base_volume = Decimal("0")  # Initialize total liquidity volume

        # Process bids to calculate liquidity
        for entry in bids_list:
            if len(entry) != 2:
                self.logger().error(f"Unexpected bid entry format: {entry}")
                continue
            price, size = entry
            price = Decimal(str(price))
            size = Decimal(str(size))
            if price >= lower_bound:
                total_base_volume += size
            else:
                break  # Bids are sorted from high to low; break once price is lower than lower_bound

        # Process asks to calculate liquidity
        for entry in asks_list:
            if len(entry) != 2:
                self.logger().error(f"Unexpected ask entry format: {entry}")
                continue
            price, size = entry
            price = Decimal(str(price))
            size = Decimal(str(size))
            if price <= upper_bound:
                total_base_volume += size
            else:
                break  # Asks are sorted from low to high; break once price exceeds upper_bound

        # Log the total liquidity within the price range
        self.logger().info(f"Liquidity ±{threshold_pct*100}% around {mid_price} => {total_base_volume}")
        return total_base_volume

    def adjust_spreads_for_inventory(self, base_spread: Decimal):
        """
        Adjusts the bid/ask spreads based on the inventory imbalance.
        If the base asset balance deviates from the target ratio, the spread is adjusted to reduce imbalance.
        """
        if self.last_mid_price is None:
            return base_spread, base_spread  # If no previous mid price, return the base spread unchanged

        # Get the current base and quote asset balances
        base_balance = self.connectors[self.exchange].get_balance(self.base_asset)
        quote_balance = self.connectors[self.exchange].get_balance(self.quote_asset)
        mid_price = Decimal(str(self.last_mid_price))

        total_val = base_balance * mid_price + quote_balance
        if total_val == 0:
            return base_spread, base_spread  # Avoid division by zero

        current_ratio = (base_balance * mid_price) / total_val  # Current ratio of base to total value
        imbalance = current_ratio - self.target_ratio  # Calculate imbalance
        shift = imbalance * self.inventory_adjustment_factor  # Adjust spread based on imbalance

        # Adjust the spreads accordingly
        buy_spread = base_spread
        sell_spread = base_spread
        if imbalance > 0:
            sell_spread = max(base_spread - shift, Decimal("0.0001"))  # Narrower sell spread if too much base
        else:
            buy_spread = max(base_spread + shift, Decimal("0.0001"))  # Narrower buy spread if too little base

        return buy_spread, sell_spread

    def place_order(self, order: OrderCandidate):
        """
        Places a buy or sell order based on the order side.
        """
        if order.order_side == TradeType.BUY:
            self.buy(
                connector_name=self.exchange,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
        else:
            self.sell(
                connector_name=self.exchange,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )

    def cancel_all_orders(self):
        """
        Cancels all active orders for the trading pair.
        """
        for o in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, o.trading_pair, o.client_order_id)

    def format_status(self) -> str:
        """
        Returns a formatted string summarizing the strategy's status, including balances and active orders.
        """
        lines = []
        bal_base = self.connectors[self.exchange].get_balance(self.base_asset)
        bal_quote = self.connectors[self.exchange].get_balance(self.quote_asset)
        lines.append(f"ALT Maker - {self.trading_pair}")
        lines.append(f"Balances => {self.base_asset}={bal_base}, {self.quote_asset}={bal_quote}")

        active = self.get_active_orders(connector_name=self.exchange)
        if active:
            lines.append("Active Orders:")
            for o in active:
                lines.append(f" - {o.trade_type} {o.amount} @ {o.price}")
        else:
            lines.append("No active orders.")

        return "\n".join(lines)
