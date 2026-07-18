import MetaTrader5 as mt5
import sys

if not mt5.initialize():
    print("MT5 failed")
    sys.exit()

mt5.symbol_select("XAUUSDc", True)
rates = mt5.copy_rates_from_pos("XAUUSDc", mt5.TIMEFRAME_M5, 0, 100000)
if rates is None:
    print(f"Error: {mt5.last_error()}")
else:
    print(f"Got {len(rates)} bars for M5")
mt5.shutdown()
