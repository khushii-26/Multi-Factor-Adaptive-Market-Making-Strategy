# Multi-Factor Adaptive Market Making Strategy

Welcome to the **Multi-Factor Adaptive Market Making Strategy** — a professional-grade, algorithmic trading bot built for [Hummingbot](https://hummingbot.org). This strategy dynamically manages risk and adapts to evolving market conditions using advanced quantitative techniques.

---

## Overview

This strategy acts as an intelligent market maker: it continuously places buy and sell orders around the current market price to profit from the bid–ask spread.  
What sets it apart is its ability to analyze multiple market factors—such as volatility, trend, inventory, and momentum—to adjust pricing, order size, and risk exposure in real time.

Goal: Resilient, data-driven trading that delivers consistent profitability and strong risk management across all market conditions.

---

## Key Features

- Uses volatility, trend, inventory balance, and momentum to inform every trading decision.
- Widens or narrows spreads based on real-time market volatility and liquidity.
- Shifts pricing in the direction of prevailing trends and confirms with momentum signals.
- Monitors and rebalances holdings to avoid risky imbalances.
- Uses Kelly Criterion principles to size trades based on current risk.
- Monitors Sharpe ratio, win rate, drawdown, and more for ongoing evaluation.

---

## How It Works

- Continuously fetches and analyzes market data using indicators like NATR, Bollinger Bands, RSI, linear regression, and rate of change.
- Computes composite scores for volatility, trend, inventory risk, and momentum.
- Adjusts reference price and spread width based on these factor scores.
- Places and refreshes buy/sell orders at optimal price and size.
- Automatically manages inventory and reduces position sizing in high-risk periods.
- Tracks key metrics to ensure the strategy remains profitable and risk-aware.

---

## Why Use This Strategy?

- Adaptive: Performs well in both volatile and stable markets  
- Risk-Managed: Incorporates several layers of risk control  
- Data-Driven: Makes decisions based on quantitative signals  
- Extensible: Modular design for customization and enhancements  

---

## Quick Start

### Requirements

- [Hummingbot](https://hummingbot.org) installed and configured  
- Python 3.8+  
- Required packages:
  - `pandas`
  - `numpy`
  - `pandas_ta`
  - `scipy`

---

## Conclusion

This strategy is designed for traders who seek to go beyond static market-making. By integrating multiple dynamic market signals, it adapts intelligently to changing conditions and prioritizes both profitability and risk control. Whether you're exploring algorithmic trading or building robust, production-ready bots, this strategy offers a strong foundation.

Give it a try, customize it to your needs, and contribute to improving it further!

---

**Khushi Singh** 