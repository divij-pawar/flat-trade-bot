# Flat-trade-bot

## Usage

1. First, build the C extension:
```bash
python setup.py build_ext --inplace
```
2. Then run your trading bot as before:
```bash
python run.py
```
## Features
### Core functionality:
<ul>
<li> Market data fetching
<li> Technical indicator calculation (SMA, RSI, MACD)
<li> Signal generation based on indicator values
<li> Trade execution
<li> Risk management
</ul>

Trading strategies:
<ul>
<li> Moving average crossover (SMA20 crosses SMA50)
<li> RSI overbought/oversold conditions
</ul>

Risk management features:
<ul>
<li> Capital allocation limits
<li> Volatility checks
<li> Position sizing rules
</ul>

Performance tracking:
<ul>
<li> Trade history
<li> Profit/loss calculation
<li> Win rate metrics
</ul>