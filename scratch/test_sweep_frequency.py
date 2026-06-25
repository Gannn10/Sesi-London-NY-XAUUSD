import sys
import os
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import pickle
import xgboost as xgb
import polars as pl

# Tambahkan path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ln-ny up'))
from feature_eng import FeatureEngineer
from smc_polars import SMCAnalyzer
import config_london_ny as cfg

def main():
    if not mt5.initialize():
        print("MT5 init failed")
        return

    # Ambil data 5 hari ke belakang
    sekarang = datetime.now(pytz.timezone(cfg.TIMEZONE))
    utc_end = sekarang.astimezone(pytz.utc)
    utc_start = utc_end - timedelta(days=14)
    
    rates = mt5.copy_rates_range(cfg.SYMBOL, mt5.TIMEFRAME_M5, utc_start, utc_end)
    if rates is None or len(rates) == 0:
        print("Gagal ambil data")
        return
        
    df_m5_pd = pd.DataFrame(rates)
    df_m5_pd['time'] = pd.to_datetime(df_m5_pd['time'], unit='s')
    
    rates_h1 = mt5.copy_rates_range(cfg.SYMBOL, mt5.TIMEFRAME_H1, utc_start - timedelta(days=2), utc_end)
    df_h1_pd = pd.DataFrame(rates_h1)
    df_h1_pd['time'] = pd.to_datetime(df_h1_pd['time'], unit='s')
    
    # Feature extraction (like live, but vectorized for backtest speed)
    df_m5 = pl.from_pandas(df_m5_pd)
    df_h1 = pl.from_pandas(df_h1_pd)
    
    fe = FeatureEngineer()
    smc = SMCAnalyzer()
    
    df = fe.calculate_all(df_m5, include_ml_features=True)
    df = smc.calculate_all(df)
    
    df_h1 = fe.calculate_all(df_h1, include_ml_features=False)
    df_h1 = smc.calculate_all(df_h1)
    
    h1_feature_cols = ["time", "close", "rsi", "atr", "bb_upper", "bb_lower", "macd", "macd_signal", "ema_20", "ema_50", "ob", "fvg", "market_structure"]
    h1_cols_present = [c for c in h1_feature_cols if c in df_h1.columns]
    h1_subset = df_h1.select(h1_cols_present)
    h1_subset = h1_subset.rename({c: f"h1_{c}" for c in h1_cols_present if c != "time"})
    h1_subset = h1_subset.with_columns((pl.col("time") + pl.duration(hours=1)).alias("time"))
    
    df = df.join_asof(h1_subset, on="time", strategy="backward")
    if "h1_close" in df.columns and "h1_ema_20" in df.columns:
        df = df.with_columns([((pl.col("h1_close") - pl.col("h1_ema_20")) / pl.col("h1_ema_20")).alias("h1_ema20_distance")])
        
    df = df.fill_null(strategy="forward").fill_null(strategy="zero")
    df_pd = df.to_pandas()
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backtests', 'ml_v3', 'xgboost_model_v3.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    xgb_model = model_data['xgb_model']
    feature_cols = model_data['feature_names']
    
    # Predict on ALL rows (these represent the [-2] state when shifted by 1)
    dmatrix = xgb.DMatrix(df_pd[feature_cols])
    preds = xgb_model.predict(dmatrix)
    df_pd['prob_buy'] = preds
    df_pd['prob_sell'] = 1.0 - preds
    
    # Shift predictions to simulate [-2] logic.
    # When we are at row i (live candle [-1]), the closed candle [-2] is row i-1.
    df_pd['closed_prob_buy'] = df_pd['prob_buy'].shift(1)
    df_pd['closed_prob_sell'] = df_pd['prob_sell'].shift(1)
    
    # The last_swing limits are also from [-2] or [-1]?
    # In smc_polars, last_swing_low is updated. If we use [-1]'s last_swing_low, it's fine.
    df_pd['last_swing_low'] = df_pd['last_swing_low'].fillna(method='ffill')
    df_pd['last_swing_high'] = df_pd['last_swing_high'].fillna(method='ffill')
    
    # Calculate sweep on live candle [-1]
    # We define a sweep as: low < last_swing_low AND close > last_swing_low
    # Since this is a M5 candle, a sweep inside the candle means the final 'low' was < swing_low,
    # and the 'close' (or high/any price during the candle) recovered.
    # For a backtest approximation: low < swing_low AND close > swing_low
    
    df_pd['sweep_buy'] = (df_pd['low'] < df_pd['last_swing_low']) & (df_pd['close'] > df_pd['last_swing_low'])
    df_pd['sweep_sell'] = (df_pd['high'] > df_pd['last_swing_high']) & (df_pd['close'] < df_pd['last_swing_high'])
    
    total_candles = len(df_pd)
    
    print(f"Total M5 Candles (14 days): {total_candles}")
    
    for threshold in [0.60, 0.55, 0.52, 0.50]:
        xgb_buy = df_pd['closed_prob_buy'] >= threshold
        xgb_sell = df_pd['closed_prob_sell'] >= threshold
        
        xgb_only = (xgb_buy | xgb_sell).sum()
        sweep_only = (df_pd['sweep_buy'] | df_pd['sweep_sell']).sum()
        
        combined_buy = xgb_buy & df_pd['sweep_buy']
        combined_sell = xgb_sell & df_pd['sweep_sell']
        combined = (combined_buy | combined_sell).sum()
        
        print(f"--- Threshold {threshold*100:.1f}% ---")
        print(f"XGBoost Only entries: {xgb_only}")
        print(f"Sweep Only entries: {sweep_only}")
        print(f"Combined (Hybrid) entries: {combined}")
        print()

if __name__ == '__main__':
    main()
