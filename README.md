## Portfolio Optimization

### Description
Portfolio Optimization project, conducted at Stanford University together with Alexander E. Karolin and under supervision of Prof. Stephen P. Boyd and research members:
- Nikhil Devanathan
- Alexandros Tzikas
- Daniel Cederberg
- Maximilian Schaller

The goal of this project is to identify and formulate simple methods for portfolio constructions that yields long-term decent and consistent returns. The project employs methods of: volatility control through Markowitz and volatility scaling, alpha generation through return prediction and asset ranking, strategies and meta strategies. 

### Data
All data is publicly available through APIs provided by Yahoo Finance and Federal Reserve Economic Data (FRED). Our asset universe consists of a semi-broad basket of selected ETFs, long-only, and all assets are well known and highly liquid. 

### Results
Results found from this study do not gurantee future financial sucess, however provide soft empirical proof of what is likely to continue being solid long term investment strategies. We benchmark results against the classic 60/40 portfolio (consisting of 60% SPY and 40% AGG) and measure portfolio performance in terms of sharpe primarily. Other important metrics include: annualized return, annualized volatility, total return, and maximum drawdown. These results are derived from a 20 years long backtest between 2004-01-03 and 2024-12-31. Each year is assumed to have 252 trading days. 

### Environment Setup
```zsh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
``` 
Create a new `scripts/env.sh` file where you export your FRED API key to the environment. You can order a FRED API key from [here](https://fred.stlouisfed.org/docs/api/api_key.html).
```zsh
export FRED_API_KEY="your_key_here"
``` 
Then download data and process risk with the setup command.
```zsh
source setup.sh  # this requires your FRED API key to be set, see 'env'
```
