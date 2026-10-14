# Advanced ML Trading Backtester

A robust, object-oriented, and vectorized Python backtesting engine for quantitative trading strategies. This repository features a complete Machine Learning pipeline using Walk-Forward validation, alongside a realistic execution engine simulating dynamic market frictions.

## Features

- **Machine Learning Pipeline:** Integrates feature engineering (momentum, rolling volatility, moving average distances), standard scaling, Principal Component Analysis (PCA) for dimensionality reduction, and Ridge Regression (L2 regularization) for predictive signal generation.
- **Walk-Forward Validation:** Prevents look-ahead bias and overfitting by using a rolling window approach. The model is periodically recalibrated on past data to predict out-of-sample forward returns.
- **Advanced Market Frictions:**
  - *Dynamic Slippage:* Transaction costs are scaled by local market volatility acting as a proxy for bid-ask spread widening.
  - *Asymmetrical Shorting Costs:* Incorporates annualized borrow rates (fees) that are dynamically deducted when holding short positions.
- **Vectorized Engine:** Fast execution using `pandas` and `numpy`, avoiding slow iterative loops while strictly enforcing T+1 execution to prevent look-ahead bias.
- **Performance Analytics:** Computes key quantitative metrics including Annualized Return, Annualized Volatility, Sharpe Ratio, and Maximum Drawdown.

## Architecture

- `BaseStrategy`: Abstract base class enforcing the `generate_signals` interface for modularity. Any new strategy must inherit from this class.
- `MLRidgePCAStrategy`: The concrete machine learning strategy implementing the PCA + Ridge regression pipeline with rolling recalibration.
- `AdvancedVectorizedBacktester`: The core engine handling data ingestion, transaction logic, dynamic costs, and equity curve generation.

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Quick Start

Clone the repository and execute the main script. The code includes a synthetic Geometric Brownian Motion (GBM) data generator for immediate testing out-of-the-box. To test on real markets, replace the synthetic generator with real historical data (e.g., via `yfinance`).

```bash
git clone [https://github.com/your-username/Advanced-ML-Trading-Backtester.git](https://github.com/your-username/Advanced-ML-Trading-Backtester.git)
cd Advanced-ML-Trading-Backtester
python backtester.py
