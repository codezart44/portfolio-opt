FIRST_DATE     = "1954-01-03"  # first date
FINAL_DATE     = "2026-03-25"  # final date
FIRST_DATE_BTC = "2017-09-07"  # according to Kasper's paper
AUTO_ADJUST = True

D0 = "2005-01-03"
D1 = "2024-12-31"

indx = ["SPY", "QQQ", "VTV", "VUG", "MDY", "IWM", "SCHD", "USMV", "QUAL"]  # market
bond = ["AGG", "TLT", "IEF", "TIP", "LQD", "LQDH", "HYG", "MUB", "BNDX", "EMB", "IAGG", "VWOB"]  # bonds
sect = ["XLK", "XLV", "XLF", "XLY", "XLI", "XLP", "XLE", "XLU", "XLB", "IBB", "IYR"]  # sectors
intn = ["EWJ", "EWG", "EWU", "EWA", "EWH", "EWS", "EWZ", "EWT", "EWY", "EWP", "EWW", "EWI", "EWD", "EWL", "EWC"]  # international "EEM"
comd = ["GLD", "SLV", "CPER", "USO", "UGA", "CORN", "WEAT", "SOYB", "CANE"]  # commodities COTN.L, "DBC", "DBA", 
# metl = ["GLD", "SLV", "CPER"]  # precious metals  "DBB"
# crpt = ["BTC-USD"]  # crypto
universe = sorted([*indx, *sect, *intn, *bond, *comd])

# Time periods
_1W   = 5
_2W   = 10
_4W   = _1M  = 21
_12W  = _3M  = _1Q  = 63
_26W  = _6M  = _2Q  = 126
_52W  = _12M = _4Q  = _1Y = 252
_104W = _24M = _8Q  = _2Y = 504
_156W = _36M = _12Q = _3Y = 756

# Plan: 
# 1. Run models on categories of assets
# 2. For equities, residualize on SPY and hold only sectors/international (replaces SPY)
# 3. For metals, residualize against GLD (still keep GLD!)
# 4. For bonds, do NOT residualize against AGG. Raw returns!

# Apply transforms for the output and target : rank, squared + sign, sigmoid

# Targets: MOM 1M, MOM Vol Scaled (VS) 1M

# Macro Features
# Include for bonds : CPI, Yield Curve Slope 3M - 10Y, (Unemployment, GDP)
# Include for equities : GDP, Inflation, Unemployment (Yield curve slope 3M - 10Y)
# Include for commodities : PCE, inflation, unemployment, 

# Return features
# Include for bonds : MOM VS 3Y +- 1Y, <1M momentum
# 

# Short term momentum for sectors (<1M) - outperforms in burts, otherwise SPY is better
# Consumer staples performs well under recessions, get in and out quickly
# For commodities - short term momentum (2W) can be good, short term draw down is good signal to exit

# Equities: Short term momentum (sectors and international)
# 
