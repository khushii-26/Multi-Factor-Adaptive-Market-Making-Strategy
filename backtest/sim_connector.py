from decimal import Decimal
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_book_row import OrderBookRow
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.core.data_type.common import OrderType, TradeType
import pandas as pd
import numpy as np
import random
import time
import logging

logger = logging.getLogger(__name__)


class DummyBudgetChecker:
    def adjust_candidates(self, proposal, all_or_none=True):
        return proposal


class SimConnector(ConnectorBase):
    """A minimal in-memory connector that fulfils only the methods used by
    AdvancedMultiFactorPMM when running in back-testing mode. It does **not**
    talk to any exchange – it merely stores an `OrderBook` you update from
    your back-test driver and exposes balances.
    """

    def __init__(self, name: str, quote_balance: float = 1_000_000, base_balance: float = 0):
        # Initialize name first to avoid circular reference
        self._name = name
        
        # Create minimal client config
        client_config = ClientConfigMap()
        client_config_adapter = ClientConfigAdapter(client_config)
        
        super().__init__(client_config_adapter)
        self._order_book = OrderBook()
        self._quote = Decimal(str(quote_balance))
        self._base = Decimal(str(base_balance))
        self._trading_pairs = ["ETH-USDT"]  # Hardcoded for simplicity
        self._last_price = None
        self._last_timestamp = None
        self._last_volume = None
        self._balances = {"ETH": Decimal("1.0"), "USDT": Decimal("100000.0")}
        self._available_balances = {"ETH": Decimal("1.0"), "USDT": Decimal("100000.0")}
        self._trades = []
        self._orders = {}
        self._next_order_id = 1
        self._last_trade_price = Decimal("2000.0")  # Initial price
        self._last_trade_timestamp = 0
        self._fill_probability = 0.7  # 70% chance of fill for orders near mid price
        self._price_sensitivity = 0.002  # 20 bps price sensitivity
        
        # Initialize order book with some depth
        self._initialize_order_book()
        self._budget_checker = DummyBudgetChecker()
        logger.info("Created SimConnector successfully")

    def _initialize_order_book(self):
        """Initialize order book with multiple price levels using apply_snapshot."""
        try:
            base_price = float(self._last_trade_price)
            bids = []
            asks = []
            update_id = int(time.time() * 1e6)
            for i in range(1, 6):
                bid_price = base_price * (1 - i * 0.001)
                ask_price = base_price * (1 + i * 0.001)
                bid_volume = 1.0 + random.random()
                ask_volume = 1.0 + random.random()
                bids.append(OrderBookRow(price=bid_price, amount=bid_volume, update_id=update_id))
                asks.append(OrderBookRow(price=ask_price, amount=ask_volume, update_id=update_id))
            self._order_book.apply_snapshot(bids, asks, update_id)
            logger.info(f"Initialized order book with 5 levels on each side using apply_snapshot")
        except Exception as e:
            logger.error(f"Error initializing order book: {str(e)}")

    def buy(self, trading_pair: str, amount: Decimal, order_type: OrderType, price: Decimal) -> str:
        """Place a buy order"""
        try:
            order_id = str(self._next_order_id)
            self._next_order_id += 1
            
            # Check if we have enough balance
            required_funds = amount * price
            if self._available_balances["USDT"] < required_funds:
                logger.warning(f"Insufficient USDT balance for buy order. Required: {required_funds}, Available: {self._available_balances['USDT']}")
                return order_id
            
            # Reserve the funds
            self._available_balances["USDT"] -= required_funds
            
            # Store the order
            self._orders[order_id] = {
                "trading_pair": trading_pair,
                "amount": amount,
                "price": price,
                "type": order_type,
                "side": TradeType.BUY,
                "created_timestamp": time.time()
            }
            
            logger.info(f"Created buy order {order_id}: {amount} ETH @ {price} USDT")
            return order_id
        except Exception as e:
            logger.error(f"Error placing buy order: {str(e)}")
            return str(self._next_order_id - 1)

    def sell(self, trading_pair: str, amount: Decimal, order_type: OrderType, price: Decimal) -> str:
        """Place a sell order"""
        try:
            order_id = str(self._next_order_id)
            self._next_order_id += 1
            
            # Check if we have enough balance
            if self._available_balances["ETH"] < amount:
                logger.warning(f"Insufficient ETH balance for sell order. Required: {amount}, Available: {self._available_balances['ETH']}")
                return order_id
            
            # Reserve the funds
            self._available_balances["ETH"] -= amount
            
            # Store the order
            self._orders[order_id] = {
                "trading_pair": trading_pair,
                "amount": amount,
                "price": price,
                "type": order_type,
                "side": TradeType.SELL,
                "created_timestamp": time.time()
            }
            
            logger.info(f"Created sell order {order_id}: {amount} ETH @ {price} USDT")
            return order_id
        except Exception as e:
            logger.error(f"Error placing sell order: {str(e)}")
            return str(self._next_order_id - 1)

    def process_tick(self, timestamp: float, mid_price: Decimal):
        """Process orders on each tick"""
        try:
            self._last_trade_timestamp = timestamp
            self._last_trade_price = mid_price
            
            # Update order book
            self._initialize_order_book()
            
            # Process each order
            filled_orders = []
            for order_id, order in self._orders.items():
                # Skip orders we've already decided to fill
                if order_id in filled_orders:
                    continue
                    
                # Calculate fill probability based on price distance from mid
                price_distance = abs(float(order["price"] - mid_price) / float(mid_price))
                fill_prob = max(0.0, self._fill_probability - price_distance / self._price_sensitivity)
                
                # Decide if order fills
                if random.random() < fill_prob:
                    # Execute the trade
                    if order["side"] == TradeType.BUY:
                        self._balances["ETH"] += order["amount"]
                        self._balances["USDT"] -= order["amount"] * order["price"]
                        self._available_balances["ETH"] += order["amount"]
                    else:
                        self._balances["ETH"] -= order["amount"]
                        self._balances["USDT"] += order["amount"] * order["price"]
                        self._available_balances["USDT"] += order["amount"] * order["price"]
                    
                    # Record the trade
                    self._trades.append({
                        "timestamp": timestamp,
                        "order_id": order_id,
                        "price": order["price"],
                        "amount": order["amount"],
                        "side": order["side"]
                    })
                    
                    filled_orders.append(order_id)
                    logger.info(f"Filled order {order_id}: {'BUY' if order['side'] == TradeType.BUY else 'SELL'} {order['amount']} @ {order['price']}")
            
            # Remove filled orders
            for order_id in filled_orders:
                del self._orders[order_id]
                
            # Log current state
            logger.info(f"Processed tick at {timestamp}")
            logger.info(f"Current balances - ETH: {float(self._balances['ETH']):.4f}, USDT: {float(self._balances['USDT']):.2f}")
            logger.info(f"Active orders: {len(self._orders)}")
            
        except Exception as e:
            logger.error(f"Error processing tick: {str(e)}")

    # ---------------------------------------------------------------------
    # Public helpers for the back-test driver
    # ---------------------------------------------------------------------
    def attach_snapshot(self, price: float, timestamp: float = None, depth: float = 10.0, spread: float = 0.0002):
        """Inject a naïve order-book around *price* with configurable depth and spread.
        """
        self._last_price = price
        self._last_timestamp = timestamp if timestamp is not None else self._last_timestamp
        self._last_volume = depth

        # Create multiple levels with decreasing depth
        num_levels = 5
        bids = []
        asks = []
        
        for i in range(num_levels):
            bid_price = price * (1 - spread * (i + 1))
            ask_price = price * (1 + spread * (i + 1))
            level_depth = depth / (2 ** i)  # Exponentially decrease depth
            
            bids.append(OrderBookRow(float(bid_price), float(level_depth), 0))
            asks.append(OrderBookRow(float(ask_price), float(level_depth), 0))

        self._order_book.apply_snapshot(bids, asks, int(price * 1e6))

    # ------------------------------------------------------------------
    # Methods consumed by the strategy
    # ------------------------------------------------------------------
    def get_order_book(self, trading_pair: str) -> OrderBook:
        return self._order_book

    def get_price(self, trading_pair: str, is_buy: bool) -> Decimal:
        return Decimal(str(self._order_book.get_price(is_buy)))

    def get_price_by_type(self, trading_pair: str, price_type):
        return self.get_price(trading_pair, True)

    def get_balance(self, asset: str) -> Decimal:
        return self._base if asset == "ETH" else self._quote

    def get_available_balance(self, asset: str) -> Decimal:
        return self.get_balance(asset)

    # Required ConnectorBase methods
    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def display_name(self) -> str:
        return self._name  # Use _name directly to avoid potential circular reference

    @property
    def trading_pairs(self):
        return self._trading_pairs

    def get_all_balances(self) -> dict:
        return {"ETH": self._base, "USDT": self._quote}

    # no-op networking methods (strategy never calls these in back-test)
    def _create_web_assistant_factory(self):
        return None

    def start_network(self):
        pass

    def stop_network(self):
        pass

    async def start_network_async(self):
        pass

    async def stop_network_async(self):
        pass

    @property
    def ready(self) -> bool:
        return True

    @property
    def budget_checker(self):
        if not hasattr(self, "_budget_checker"):
            self._budget_checker = DummyBudgetChecker()
        return self._budget_checker

    @budget_checker.setter
    def budget_checker(self, value):
        self._budget_checker = value 