import logging
import numpy as np
from decimal import Decimal
from typing import Dict, List
from collections import deque
from scipy import stats
import pandas as pd

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

class AdvancedMultiFactorPMM(ScriptStrategyBase):
    """
    Clean, Hummingbot-compatible multifactor market making strategy for ETH-USDT.
    No backtest-only logic. Uses trend, volatility, and inventory for order placement.
    """
    # Basic Parameters
    base_bid_spread = 0.001
    base_ask_spread = 0.001
    order_refresh_time = 30
    base_order_amount = 0.01
    create_timestamp = 0
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    base, quote = trading_pair.split('-')
    max_inventory = 0.01  # Max inventory for skew calc (tune as needed)
    stop_loss_pct = 0.005  # 0.5% stop-loss
    maker_rebate = 0.0002  # Simulate 0.02% maker rebate per filled order
    
    # Candles Configuration
    candle_exchange = "binance"
    candles_interval = "1m"
    candles_length = 30
    max_records = 1000
    
    # Volatility Parameters
    volatility_lookback = 20
    natr_weight = 0.5
    bb_width_weight = 0.3
    return_vol_weight = 0.2
    volatility_scalar = 100
    
    # Trend Analysis Parameters
    rsi_weight = 0.4
    slope_weight = 0.4
    roc_weight = 0.2
    trend_lookback = 14
    max_trend_shift = 0.0001
    
    # Inventory Management Parameters
    target_ratio = 0.5
    inventory_lookback = 50
    max_inventory_shift = 0.0002
    inventory_risk_threshold = 2.0
    
    # Momentum Parameters
    momentum_short_period = 5
    momentum_long_period = 15
    momentum_weight = 0.15
    
    # Risk Management Parameters
    max_position_reduction = 0.5
    min_spread_multiplier = 0.5
    max_spread_multiplier = 3.0
    
    # State Variables
    current_ratio = 0.5
    composite_volatility = 0.001
    trend_score = 0.0
    inventory_risk_score = 0.0
    momentum_score = 0.0
    reference_price = 1.0
    
    # Historical Data Storage
    inventory_history = deque(maxlen=inventory_lookback)
    price_history = deque(maxlen=candles_length)
    pnl_history = deque(maxlen=100)
    
    # Performance Tracking
    total_trades = 0
    profitable_trades = 0
    total_pnl = 0.0
    max_drawdown = 0.0
    
    # Initialize candles
    candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=candles_interval,
        max_records=max_records
    ))
    
    markets = {exchange: {trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()
        
    def on_stop(self):
        self.candles.stop()

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            self.update_all_metrics()
            proposal: List[OrderCandidate] = self.create_advanced_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp
            
    def get_enhanced_candles_with_features(self):
        """Enhanced candle analysis with multiple technical indicators"""
        candles_df = self.candles.candles_df.copy()
        
        if len(candles_df) < self.candles_length:
            return candles_df
            
        # Volatility Indicators
        candles_df.ta.natr(length=self.candles_length, scalar=1, append=True)
        candles_df.ta.bbands(length=self.candles_length, append=True)
        
        # Calculate Bollinger Bands Width
        bb_upper = f'BBU_{self.candles_length}'
        bb_middle = f'BBM_{self.candles_length}'
        bb_lower = f'BBL_{self.candles_length}'
        
        if bb_upper in candles_df.columns and bb_middle in candles_df.columns:
            candles_df['bb_width'] = (candles_df[bb_upper] - candles_df[bb_lower]) / candles_df[bb_middle]
        
        # Return-based volatility
        candles_df['returns'] = candles_df['close'].pct_change()
        candles_df['return_volatility'] = candles_df['returns'].rolling(self.volatility_lookback).std()
        
        # Trend Indicators
        candles_df.ta.rsi(length=self.candles_length, append=True)
        
        # Rate of Change calculations
        candles_df[f'roc_{self.momentum_short_period}'] = candles_df['close'].pct_change(self.momentum_short_period)
        candles_df[f'roc_{self.momentum_long_period}'] = candles_df['close'].pct_change(self.momentum_long_period)
        
        return candles_df
        
    def calculate_composite_volatility(self, candles_df):
        """Calculate composite volatility score using multiple indicators"""
        if len(candles_df) < self.candles_length:
            return self.base_bid_spread
            
        # Get latest values
        natr = candles_df[f"NATR_{self.candles_length}"].iloc[-1] if f"NATR_{self.candles_length}" in candles_df.columns else 0.001
        bb_width = candles_df['bb_width'].iloc[-1] if 'bb_width' in candles_df.columns else 0.05
        return_vol = candles_df['return_volatility'].iloc[-1] if 'return_volatility' in candles_df.columns else 0.02
        
        # Handle NaN values
        natr = natr if not np.isnan(natr) else 0.001
        bb_width = bb_width if not np.isnan(bb_width) else 0.05
        return_vol = return_vol if not np.isnan(return_vol) else 0.02
        
        # Composite volatility calculation
        composite_vol = (
            self.natr_weight * natr +
            self.bb_width_weight * bb_width +
            self.return_vol_weight * return_vol
        )
        
        return max(0.0001, composite_vol)
        
    def calculate_trend_score(self, candles_df):
        """Multi-factor trend analysis"""
        if len(candles_df) < self.candles_length:
            return 0.0
            
        # RSI Signal
        rsi = candles_df[f"RSI_{self.candles_length}"].iloc[-1] if f"RSI_{self.candles_length}" in candles_df.columns else 50
        rsi_signal = (rsi - 50) / 50
        
        # Linear Regression Slope
        prices = candles_df['close'].tail(self.trend_lookback).values
        if len(prices) >= self.trend_lookback:
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            # Normalize slope by average price
            avg_price = np.mean(prices)
            slope_signal = (slope * len(prices)) / avg_price if avg_price > 0 else 0
        else:
            slope_signal = 0
            
        # Rate of Change Signal
        roc_short = candles_df[f'roc_{self.momentum_short_period}'].iloc[-1] if f'roc_{self.momentum_short_period}' in candles_df.columns else 0
        roc_long = candles_df[f'roc_{self.momentum_long_period}'].iloc[-1] if f'roc_{self.momentum_long_period}' in candles_df.columns else 0
        
        roc_short = roc_short if not np.isnan(roc_short) else 0
        roc_long = roc_long if not np.isnan(roc_long) else 0
        
        roc_signal = roc_short - roc_long
        
        # Composite trend score
        trend_score = (
            self.rsi_weight * rsi_signal +
            self.slope_weight * slope_signal +
            self.roc_weight * roc_signal
        )
        
        return np.clip(trend_score, -1, 1)
        
    def calculate_inventory_risk_metrics(self):
        """Advanced inventory risk management with MAP and Z-score"""
        # Update inventory history
        base_bal = self.connectors[self.exchange].get_balance(self.base)
        base_bal_in_quote = base_bal * self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        quote_bal = self.connectors[self.exchange].get_balance(self.quote)
        total_balance = base_bal_in_quote + quote_bal
        
        if total_balance > 0:
            self.current_ratio = float(base_bal_in_quote / total_balance)
        else:
            self.current_ratio = 0.5
            
        self.inventory_history.append(self.current_ratio)
        
        # Calculate inventory imbalance
        current_imbalance = (self.target_ratio - self.current_ratio) / self.target_ratio
        
        # Calculate MAP (Mean Absolute Position) if we have enough history
        if len(self.inventory_history) >= 10:
            inventory_array = np.array(self.inventory_history)
            
            # Z-score normalization
            inventory_mean = np.mean(inventory_array)
            inventory_std = np.std(inventory_array)
            
            if inventory_std > 0:
                inventory_z_score = (self.current_ratio - inventory_mean) / inventory_std
            else:
                inventory_z_score = 0
                
            # MAP calculation
            map_score = np.mean(np.abs(inventory_array - self.target_ratio))
            
            # Inventory volatility
            inventory_volatility = inventory_std
            
        else:
            inventory_z_score = 0
            map_score = 0
            inventory_volatility = 0
            
        # Composite inventory risk score
        inventory_risk = current_imbalance + 0.3 * inventory_z_score + 0.2 * map_score
        
        return inventory_risk, inventory_z_score, map_score
        
    def calculate_momentum_score(self, candles_df):
        """Momentum confirmation using rate of change analysis"""
        if len(candles_df) < self.momentum_long_period:
            return 0.0
            
        roc_short = candles_df[f'roc_{self.momentum_short_period}'].iloc[-1] if f'roc_{self.momentum_short_period}' in candles_df.columns else 0
        roc_long = candles_df[f'roc_{self.momentum_long_period}'].iloc[-1] if f'roc_{self.momentum_long_period}' in candles_df.columns else 0
        
        roc_short = roc_short if not np.isnan(roc_short) else 0
        roc_long = roc_long if not np.isnan(roc_long) else 0
        
        momentum_score = (roc_short - roc_long) * 10  # Amplify the signal
        
        return np.clip(momentum_score, -1, 1)
        
    def update_all_metrics(self):
        """Update all strategy metrics"""
        candles_df = self.get_enhanced_candles_with_features()
        
        # Update volatility
        self.composite_volatility = self.calculate_composite_volatility(candles_df)
        
        # Update trend score
        self.trend_score = self.calculate_trend_score(candles_df)
        
        # Update inventory risk
        inventory_risk, inventory_z_score, map_score = self.calculate_inventory_risk_metrics()
        self.inventory_risk_score = inventory_risk
        
        # Update momentum score
        self.momentum_score = self.calculate_momentum_score(candles_df)
        
        # Calculate reference price with all adjustments
        base_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        
        # Multi-factor price adjustment
        volatility_adjustment = self.composite_volatility * 0.1
        trend_adjustment = self.trend_score * self.max_trend_shift
        inventory_adjustment = self.inventory_risk_score * self.max_inventory_shift
        momentum_adjustment = self.momentum_score * self.momentum_weight * 0.0001
        
        # Apply all adjustments
        self.reference_price = base_price * \
            (Decimal("1") + Decimal(str(volatility_adjustment))) * \
            (Decimal("1") + Decimal(str(trend_adjustment))) * \
            (Decimal("1") + Decimal(str(inventory_adjustment))) * \
            (Decimal("1") + Decimal(str(momentum_adjustment)))
        
    def calculate_dynamic_spreads(self):
        """Calculate dynamic spreads based on market conditions"""
        # Base spreads
        base_bid = self.base_bid_spread
        base_ask = self.base_ask_spread
        
        # Volatility multiplier
        vol_multiplier = 1 + (self.composite_volatility * self.volatility_scalar)
        vol_multiplier = np.clip(vol_multiplier, self.min_spread_multiplier, self.max_spread_multiplier)
        
        # Inventory adjustment (wider spreads when inventory is imbalanced)
        inventory_multiplier = 1 + abs(self.inventory_risk_score) * 0.5
        
        # Trend adjustment (asymmetric spreads based on trend)
        trend_bid_adj = 1 + max(0, self.trend_score * 0.2)  # Wider bid spread in uptrend
        trend_ask_adj = 1 + max(0, -self.trend_score * 0.2)  # Wider ask spread in downtrend
        
        # Calculate final spreads
        bid_spread = base_bid * vol_multiplier * inventory_multiplier * trend_bid_adj
        ask_spread = base_ask * vol_multiplier * inventory_multiplier * trend_ask_adj
        
        return bid_spread, ask_spread
        
    def calculate_risk_adjusted_position_size(self):
        """Kelly Criterion-based position sizing"""
        # Base position size
        base_size = self.base_order_amount
        
        # Risk reduction based on inventory Z-score
        inventory_risk, inventory_z_score, map_score = self.calculate_inventory_risk_metrics()
        
        # Reduce position size when inventory is extreme
        risk_reduction = min(self.max_position_reduction, max(0, abs(inventory_z_score) / 3))
        
        # Adjust for volatility (smaller positions in high volatility)
        volatility_reduction = min(0.3, self.composite_volatility * 10)
        
        # Calculate final position size
        position_multiplier = (1 - risk_reduction) * (1 - volatility_reduction)
        position_size = base_size * max(0.1, position_multiplier)  # Minimum 10% of base size
        
        return position_size
        
    def create_advanced_proposal(self) -> List[OrderCandidate]:
        """Create advanced order proposal with multi-factor analysis"""
        # Get dynamic spreads
        bid_spread, ask_spread = self.calculate_dynamic_spreads()
        
        # Get risk-adjusted position size
        position_size = self.calculate_risk_adjusted_position_size()

        # Log market state with unique snapshot (so user can see similar output)
        self._log_market_state(bid_spread, ask_spread, position_size)

        # Calculate prices
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        
        buy_price = min(self.reference_price * Decimal(1 - bid_spread), best_bid)
        sell_price = max(self.reference_price * Decimal(1 + ask_spread), best_ask)
        
        # Create orders
        buy_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=Decimal(str(position_size)),
            price=buy_price
        )
        
        sell_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=Decimal(str(position_size)),
            price=sell_price
        )
        
        return [buy_order, sell_order]
        
    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjust proposal to available budget"""
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
        return proposal_adjusted
        
    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Place orders from proposal"""
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)
            
    def place_order(self, connector_name: str, order: OrderCandidate):
        """Place individual order"""
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, 
                     amount=order.amount, order_type=order.order_type, price=order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, 
                    amount=order.amount, order_type=order.order_type, price=order.price)
                    
    def cancel_all_orders(self):
        """Cancel all active orders"""
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)
            
    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order fill events with performance tracking"""
        msg = f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)
        
        # Update performance metrics
        self.total_trades += 1
        
        # Simple PnL tracking (this is a simplified version)
        if event.trade_type == TradeType.SELL:
            pnl = float(event.amount) * float(event.price) - float(event.amount) * float(self.reference_price)
        else:
            pnl = float(event.amount) * float(self.reference_price) - float(event.amount) * float(event.price)
            
        self.total_pnl += pnl
        self.pnl_history.append(pnl)
        
        if pnl > 0:
            self.profitable_trades += 1
            
    def calculate_performance_metrics(self):
        """Calculate advanced performance metrics"""
        if len(self.pnl_history) < 2:
            return {"sharpe_ratio": 0, "win_rate": 0, "avg_pnl": 0}
            
        pnl_array = np.array(self.pnl_history)
        
        # Sharpe ratio calculation
        if np.std(pnl_array) > 0:
            sharpe_ratio = np.mean(pnl_array) / np.std(pnl_array) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
            
        # Win rate
        win_rate = self.profitable_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Average PnL
        avg_pnl = np.mean(pnl_array)
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl
        }

    def format_status(self) -> str:
        """Enhanced status display with all metrics"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
            
        lines = []

        # Balance information
        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        # Active orders
        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        # Strategy metrics
        lines.extend(["\n" + "="*80])
        lines.extend(["  ADVANCED MULTI-FACTOR PMM STRATEGY METRICS"])
        lines.extend(["="*80])
        
        # Price information
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        bid_spread, ask_spread = self.calculate_dynamic_spreads()
        
        lines.extend(["", "  Price Information:"])
        lines.extend([f"    Reference Price: {self.reference_price:.6f}"])
        lines.extend([f"    Best Bid: {best_bid:.6f} | Best Ask: {best_ask:.6f}"])
        lines.extend([f"    Bid Spread: {bid_spread*10000:.2f} bps | Ask Spread: {ask_spread*10000:.2f} bps"])
        
        # Volatility metrics
        lines.extend(["", "  Volatility Analysis:"])
        lines.extend([f"    Composite Volatility: {self.composite_volatility:.6f}"])
        lines.extend([f"    Volatility Components: NATR({self.natr_weight:.1f}) + BB_Width({self.bb_width_weight:.1f}) + Return_Vol({self.return_vol_weight:.1f})"])
        
        # Trend analysis
        lines.extend(["", "  Trend Analysis:"])
        lines.extend([f"    Trend Score: {self.trend_score:.4f}"])
        lines.extend([f"    Trend Direction: {'Bullish' if self.trend_score > 0.1 else 'Bearish' if self.trend_score < -0.1 else 'Neutral'}"])
        
        # Inventory management
        lines.extend(["", "  Inventory Management:"])
        lines.extend([f"    Current Ratio: {self.current_ratio:.4f} | Target Ratio: {self.target_ratio:.4f}"])
        lines.extend([f"    Inventory Risk Score: {self.inventory_risk_score:.4f}"])
        lines.extend([f"    Inventory History Length: {len(self.inventory_history)}"])
        
        # Momentum analysis
        lines.extend(["", "  Momentum Analysis:"])
        lines.extend([f"    Momentum Score: {self.momentum_score:.4f}"])
        
        # Performance metrics
        perf_metrics = self.calculate_performance_metrics()
        lines.extend(["", "  Performance Metrics:"])
        lines.extend([f"    Total Trades: {self.total_trades}"])
        lines.extend([f"    Win Rate: {perf_metrics['win_rate']:.2%}"])
        lines.extend([f"    Total PnL: {self.total_pnl:.6f}"])
        lines.extend([f"    Sharpe Ratio: {perf_metrics['sharpe_ratio']:.3f}"])
        
        # Risk metrics
        position_size = self.calculate_risk_adjusted_position_size()
        lines.extend(["", "  Risk Management:"])
        lines.extend([f"    Current Position Size: {position_size:.6f}"])
        lines.extend([f"    Base Position Size: {self.base_order_amount:.6f}"])
        lines.extend([f"    Position Size Ratio: {position_size/self.base_order_amount:.2f}"])
        
        lines.extend(["\n" + "="*80])
        
        return "\n".join(lines)

    def _log_market_state(self, bid_spread: float, ask_spread: float, position_size: float):
        """Write a concise snapshot of the current order book top levels, liquidity and spreads.
        This is implemented from scratch (not borrowed from ALT_Maker) so the format is unique to this script.
        """
        try:
            ob = self.connectors[self.exchange].get_order_book(self.trading_pair)
            bids_df, asks_df = ob.snapshot  # Each is a DataFrame
            top_bids = bids_df.head(5)
            top_asks = asks_df.head(5)

            bid_df = top_bids.reset_index(drop=True)
            ask_df = top_asks.reset_index(drop=True)

            mid_price = (bid_df.price.iloc[0] + ask_df.price.iloc[0]) / 2

            # Mid-price and simple liquidity estimate (depth within ±1%)
            upper_bound = mid_price * 1.01
            lower_bound = mid_price * 0.99
            bids_liq = bid_df[bid_df.price >= lower_bound].amount.sum()
            asks_liq = ask_df[ask_df.price <= upper_bound].amount.sum()
            total_liq = bids_liq + asks_liq

            # Compose log text (unique format)
            lines = [
                "ORDER-BOOK TOP 5 (unique MF logger)",
                f"Mid: {mid_price:.2f}  |  BidSpread: {bid_spread*100:.2f}%  AskSpread: {ask_spread*100:.2f}%",
                f"Depth ±1%: {total_liq:.4f} {self.base}",
                "Bids ↓",
                bid_df.to_string(index=False, max_rows=5),
                "Asks ↑",
                ask_df.to_string(index=False, max_rows=5),
                f"Placing size: {position_size:.6f} {self.base}",
            ]
            self.log_with_clock(logging.INFO, "\n".join(lines))
        except Exception as e:
            # Do not fail the strategy because of logging problems
            self.log_with_clock(logging.WARNING, f"Snapshot logging failed: {e}")