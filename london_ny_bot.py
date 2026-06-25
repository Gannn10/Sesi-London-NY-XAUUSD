import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import requests
import json
import os
import pickle
import xgboost as xgb
import polars as pl
from regime_detector import detect_market_regime

# Import config
import config_london_ny as cfg

# Tambahkan path ln-ny up agar bisa import FeatureEngineer
sys.path.append(os.path.join(os.path.dirname(__file__), 'ln-ny up'))
from feature_eng import FeatureEngineer
from smc_polars import SMCAnalyzer

# Set up logging
logger = logging.getLogger('london_ny_bot')
logger.setLevel(cfg.LOG_LEVEL)

fh = TimedRotatingFileHandler(cfg.LOG_FILE, when='midnight', interval=1, backupCount=getattr(cfg, 'LOG_BACKUP_COUNT', 1), encoding='utf-8')
fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(ch)

wib = pytz.timezone(cfg.TIMEZONE)

def init_mt5():
    if not mt5.initialize():
        logger.error(f"Inisialisasi MT5 gagal, error code: {mt5.last_error()}")
        return False
    logger.info("Koneksi MT5 berhasil")
    return True

def get_mode_sesi():
    sekarang = datetime.now(wib)
    jam = sekarang.hour
    
    if cfg.JAM_LONDON_BUKA <= jam < cfg.JAM_LONDON_TUTUP:
        return "LONDON"
    elif cfg.JAM_NY_BUKA <= jam < cfg.JAM_NY_TUTUP:
        return "NEW_YORK"
    else:
        return "INACTIVE"

def hitung_menit_ke_london():
    sekarang = datetime.now(wib)
    target = sekarang.replace(hour=cfg.JAM_LONDON_BUKA, minute=0, second=0, microsecond=0)
    if sekarang >= target:
        target += timedelta(days=1)
    return (target - sekarang).total_seconds() / 60.0

def hitung_asia_range():
    sekarang = datetime.now(wib)
    asia_open = wib.localize(datetime(sekarang.year, sekarang.month, sekarang.day, cfg.ASIA_RANGE_START, 0))
    asia_close = wib.localize(datetime(sekarang.year, sekarang.month, sekarang.day, cfg.ASIA_RANGE_END, 0))
    
    utc_open = asia_open.astimezone(pytz.utc)
    utc_close = asia_close.astimezone(pytz.utc)
    
    rates = mt5.copy_rates_range(cfg.SYMBOL, cfg.TIMEFRAME, utc_open, utc_close)
    if rates is None or len(rates) == 0:
        return None, None
    
    df_asia = pd.DataFrame(rates)
    asia_high = df_asia['high'].max()
    asia_low = df_asia['low'].min()
    return asia_high, asia_low

def get_tp_sl_multiplier(mode_sesi):
    if mode_sesi == "LONDON":
        return {'sl': cfg.LONDON_SL_MULTIPLIER, 'tp': cfg.LONDON_TP_MULTIPLIER}
    elif mode_sesi == "NEW_YORK":
        return {'sl': cfg.NY_SL_MULTIPLIER, 'tp': cfg.NY_TP_MULTIPLIER}
    return {'sl': 1.0, 'tp': 1.0}

def get_data(symbol, timeframe, n):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def cek_posisi_terbuka():
    positions = mt5.positions_get(symbol=cfg.SYMBOL, magic=cfg.MAGIC_NUMBER)
    return len(positions) > 0 if positions else False

def terapkan_smart_exits(regime, rsi, harga_close):
    positions = mt5.positions_get(symbol=cfg.SYMBOL)
    if positions is None or len(positions) == 0:
        return

    for pos in positions:
        if pos.magic == cfg.MAGIC_NUMBER:
            tick = mt5.symbol_info_tick(cfg.SYMBOL)
            if tick is None: continue
            
            # --- 1. RSI EXIT ---
            # Jika Buy, harga di atas entry (profit), dan RSI Overbought (> 75) -> Close
            if pos.type == mt5.ORDER_TYPE_BUY and tick.bid > pos.price_open and rsi >= 75:
                logger.info(f"[SMART EXIT] RSI Overbought ({rsi:.1f}). Menutup posisi BUY #{pos.ticket} untuk amankan profit.")
                close_position(pos, tick.bid)
                continue
                
            # Jika Sell, harga di bawah entry (profit), dan RSI Oversold (< 25) -> Close
            elif pos.type == mt5.ORDER_TYPE_SELL and tick.ask < pos.price_open and rsi <= 25:
                logger.info(f"[SMART EXIT] RSI Oversold ({rsi:.1f}). Menutup posisi SELL #{pos.ticket} untuk amankan profit.")
                close_position(pos, tick.ask)
                continue
                
            # --- 2. REGIME EXIT ---
            # Jika market berubah jadi Choppy dan posisi sedang profit -> Close
            profit_usd = 0.0
            if pos.type == mt5.ORDER_TYPE_BUY:
                profit_usd = tick.bid - pos.price_open
            else:
                profit_usd = pos.price_open - tick.ask
                
            if regime == "CHOPPY" and profit_usd >= 1.0: # Profit minimal $1.00 (10 pips XAUUSD)
                logger.info(f"[SMART EXIT] Market menjadi CHOPPY. Menutup posisi #{pos.ticket} untuk amankan profit (${profit_usd:.2f}).")
                close_position(pos, tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask)
                continue

            # --- 3. DYNAMIC TRAILING STOP (Step Trailing) ---
            # Jika profit >= $3.00 (30 pips), trailing stop = harga_sekarang - $1.50
            # Jika profit >= $1.50 (15 pips), SL digeser ke Breakeven
            if pos.type == mt5.ORDER_TYPE_BUY:
                if profit_usd >= 3.0: # 30 pips
                    new_sl = tick.bid - 1.50
                    if pos.sl == 0.0 or new_sl > pos.sl:
                        modify_sl(pos, new_sl)
                        logger.info(f"[TRAILING STOP] SL BUY #{pos.ticket} digeser ke {new_sl:.2f}")
                elif profit_usd >= 1.50: # 15 pips
                    if pos.sl == 0.0 or pos.sl < pos.price_open:
                        modify_sl(pos, pos.price_open)
                        logger.info(f"[SMART BE] Posisi BUY #{pos.ticket} diamankan di BEP.")
            elif pos.type == mt5.ORDER_TYPE_SELL:
                if profit_usd >= 3.0: # 30 pips
                    new_sl = tick.ask + 1.50
                    if pos.sl == 0.0 or new_sl < pos.sl:
                        modify_sl(pos, new_sl)
                        logger.info(f"[TRAILING STOP] SL SELL #{pos.ticket} digeser ke {new_sl:.2f}")
                elif profit_usd >= 1.50: # 15 pips
                    if pos.sl == 0.0 or pos.sl > pos.price_open:
                        modify_sl(pos, pos.price_open)
                        logger.info(f"[SMART BE] Posisi SELL #{pos.ticket} diamankan di BEP.")

def close_position(pos, price):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": cfg.SYMBOL,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": pos.ticket,
        "price": price,
        "deviation": cfg.DEVIATION,
        "magic": cfg.MAGIC_NUMBER,
        "comment": "Smart Exit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    mt5.order_send(request)

def modify_sl(pos, new_sl):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": pos.ticket,
        "sl": new_sl,
        "tp": pos.tp
    }
    mt5.order_send(request)

def kirim_notion(tanggal):
    token = getattr(cfg, 'NOTION_TOKEN', '')
    db_id = getattr(cfg, 'NOTION_DATABASE_ID', '')
    
    if not token or not db_id:
        return
        
    try:
        logger.info(f"Mengirim laporan harian ke Notion untuk {tanggal}...")
        start_date = datetime.combine(tanggal, datetime.min.time()) - timedelta(hours=6)
        end_date = datetime.combine(tanggal, datetime.max.time()) + timedelta(hours=6)
        
        deals = mt5.history_deals_get(start_date, end_date)
        
        total_profit = 0.0
        win_count = 0
        loss_count = 0
        trade_count = 0
        
        if deals:
            for deal in deals:
                if deal.magic == cfg.MAGIC_NUMBER and deal.entry == mt5.DEAL_ENTRY_OUT:
                    trade_count += 1
                    net_profit = deal.profit + deal.swap + deal.commission
                    total_profit += net_profit
                    if net_profit > 0:
                        win_count += 1
                    else:
                        loss_count += 1
        
        if trade_count == 0:
            logger.info("Tidak ada trade kemarin, Notion di-skip.")
            return

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        
        judul = f"Laporan Bot London/NY - {tanggal.strftime('%Y-%m-%d')}"
        
        data = {
            "parent": {"database_id": db_id},
            "properties": {
                "Name": {"title": [{"text": {"content": judul}}]},
            },
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": f"🎯 Total Trade : {trade_count} posisi\n"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": f"⚖️ Win / Loss  : {win_count} Win / {loss_count} Loss\n"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": f"💰 Net Profit  : ${total_profit:.2f}\n"}}]
                    }
                }
            ]
        }

        # URL Notion fix
        response = requests.post("https://api.notion.com/v1/pages", headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            logger.info("✅ Laporan harian berhasil masuk ke Notion!")
        else:
            logger.error(f"❌ Gagal mengirim ke Notion: {response.text}")
            
    except Exception as e:
        logger.error(f"Error saat mengirim ke Notion: {e}")

def extract_live_features(df_m5_pd, df_h1_pd):
    df_m5 = pl.from_pandas(df_m5_pd)
    df_h1 = pl.from_pandas(df_h1_pd)
    
    fe = FeatureEngineer()
    smc = SMCAnalyzer()
    
    # Calculate M5 features
    df = fe.calculate_all(df_m5, include_ml_features=True)
    df = smc.calculate_all(df)
    
    # Calculate H1 features
    df_h1 = fe.calculate_all(df_h1, include_ml_features=False)
    df_h1 = smc.calculate_all(df_h1)
    
    # Join H1 features
    h1_feature_cols = [
        "time", "close", "rsi", "atr", "bb_upper", "bb_lower",
        "macd", "macd_signal", "ema_20", "ema_50",
        "ob", "fvg", "market_structure", "last_swing_high", "last_swing_low"
    ]
    h1_feature_cols = [c for c in h1_feature_cols if c in df_h1.columns]
    
    h1_subset = df_h1.select(h1_feature_cols)
    h1_subset = h1_subset.rename({c: f"h1_{c}" for c in h1_feature_cols if c != "time"})
    
    # FIX: Shift H1 time forward by 1 hour to prevent look-ahead bias (Data Leakage)
    h1_subset = h1_subset.with_columns(
        (pl.col("time") + pl.duration(hours=1)).alias("time")
    )
    
    df = df.join_asof(
        h1_subset,
        on="time",
        strategy="backward"
    )
    
    # Calculate H1 derived features (must match train_ml_v3.py)
    if "h1_close" in df.columns and "h1_ema_20" in df.columns:
        df = df.with_columns([
            ((pl.col("h1_close") - pl.col("h1_ema_20")) / pl.col("h1_ema_20")).alias("h1_ema20_distance")
        ])
    
    df = df.fill_null(strategy="forward").fill_null(strategy="zero")
    return df.to_pandas()


def main():
    if not init_mt5():
        return
        
    symbol_info = mt5.symbol_info(cfg.SYMBOL)
    if not symbol_info:
        logger.error(f"Symbol {cfg.SYMBOL} tidak ditemukan")
        mt5.shutdown()
        return

    # LOAD MODEL AI
    model_path = os.path.join(os.path.dirname(__file__), 'backtests', 'ml_v3', 'xgboost_model_v3.pkl')
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        xgb_model = model_data['xgb_model']
        feature_cols = model_data['feature_names']
        confidence_threshold = getattr(cfg, 'AI_CONFIDENCE_THRESHOLD', model_data.get('confidence_threshold', 0.60))
        logger.info(f"✅ AI Model V3 Loaded! Threshold: {confidence_threshold*100:.1f}%")
        logger.info(f" Fitur yang digunakan: {len(feature_cols)} kolom")
    except Exception as e:
        logger.error(f"❌ Gagal meload model AI: {e}. Pastikan sudah menjalankan Fase 1.")
        return

    logger.info("Bot London & NY (AI V3) mulai berjalan...")

    tick = mt5.symbol_info_tick(cfg.SYMBOL)
    main.bot_start_time_server = tick.time if tick else 0

    trade_count = 0
    loss_beruntun = 0
    tanggal_terakhir = datetime.now(wib).date()
    last_log_time = None
    last_active_ticket = None
    last_signal_time = None  # Menyimpan waktu candle terakhir yang memicu OP

    while True:
        sekarang_wib = datetime.now(wib)
        if sekarang_wib.date() != tanggal_terakhir:
            kirim_notion(tanggal_terakhir)
            trade_count = 0
            loss_beruntun = 0
            tanggal_terakhir = sekarang_wib.date()
            logger.info("Reset target dan hitungan trade untuk hari baru.")

        mode = get_mode_sesi()
        
        if mode == "INACTIVE":
            menit_tunggu = hitung_menit_ke_london()
            logger.info(f"Di luar sesi London/NY. Bot hibernasi selama {menit_tunggu:.1f} menit...")
            time.sleep(menit_tunggu * 60)
            continue
            
        if trade_count >= cfg.MAX_TRADE_PER_SESI:
            logger.info("Maksimal trade per sesi tercapai. Menunggu sesi berikutnya...")
            time.sleep(60 * 15)
            continue
            
        # if loss_beruntun >= cfg.MAX_LOSS_BERUNTUN:
        #     logger.info(f"Loss beruntun {loss_beruntun}x. Pause {cfg.PAUSE_SETELAH_LOSS} menit.")
        #     time.sleep(cfg.PAUSE_SETELAH_LOSS * 60)
        #     loss_beruntun = 0
        #     continue

        # Ambil data untuk mengevaluasi harga saat ini (live)
        # Kita butuh minimal 200 candle untuk EMA_200 dsb
        df_m5_pd = get_data(cfg.SYMBOL, mt5.TIMEFRAME_M5, 300)
        df_h1_pd = get_data(cfg.SYMBOL, mt5.TIMEFRAME_H1, 100)
        
        if df_m5_pd is None or df_h1_pd is None or len(df_m5_pd) < 200 or len(df_h1_pd) < 50:
            logger.warning("Gagal mengambil data M5/H1 yang cukup, coba lagi...")
            time.sleep(5)
            continue
            
        # HMM Regime Detector (menggunakan candle yang sudah close saja / index[:-1])
        regime = detect_market_regime(df_m5_pd.iloc[:-1])
            
        try:
            # 1. Ekstrak fitur menggunakan Polars
            df_features = extract_live_features(df_m5_pd, df_h1_pd)
            
            # 2. Arsitektur Hybrid (XGBoost di candle closed, Eksekusi di candle live)
            closed_features = df_features.iloc[[-2]]
            live_features = df_features.iloc[[-1]]
            
            waktu_candle = closed_features['time'].iloc[0]  # Waktu candle tertutup
            harga_close = float(live_features['close'].iloc[0]) # Harga eksekusi/live
            rsi = float(live_features['rsi'].iloc[0]) if 'rsi' in live_features.columns else 50.0
            
            # --- TAHAP 1: Liquidity Sweep Filter (Lookback 10 Candle) ---
            recent_features = df_features.tail(10)
            
            last_swing_low = float(live_features['last_swing_low'].iloc[0]) if 'last_swing_low' in live_features.columns else None
            last_swing_high = float(live_features['last_swing_high'].iloc[0]) if 'last_swing_high' in live_features.columns else None
            
            # Cek opsi di config, apakah filter ini mau dipakai atau dibypass (murni AI)
            use_sweep_filter = getattr(cfg, 'USE_SWEEP_FILTER', True)
            
            is_buy_allowed = not use_sweep_filter  # Jika False, default ke True (bypass)
            is_sell_allowed = not use_sweep_filter
            
            if use_sweep_filter:
                if last_swing_low is not None and not pd.isna(last_swing_low):
                    # Cari apakah ada candle di 10 candle terakhir yang low-nya tembus swing low, tapi close-nya mantul naik
                    sweep_buy_df = recent_features[(recent_features['low'] < recent_features['last_swing_low']) & (recent_features['close'] > recent_features['last_swing_low'])]
                    if len(sweep_buy_df) > 0:
                        is_buy_allowed = True
                        
                if last_swing_high is not None and not pd.isna(last_swing_high):
                    # Cari apakah ada candle di 10 candle terakhir yang high-nya tembus swing high, tapi close-nya mantul turun
                    sweep_sell_df = recent_features[(recent_features['high'] > recent_features['last_swing_high']) & (recent_features['close'] < recent_features['last_swing_high'])]
                    if len(sweep_sell_df) > 0:
                        is_sell_allowed = True
            
            # 3. Prediksi dengan XGBoost (menggunakan fitur closed candle)
            X_closed = closed_features[feature_cols]
            dmatrix_closed = xgb.DMatrix(X_closed)
            
            prob_buy = float(xgb_model.predict(dmatrix_closed)[0])
            prob_sell = 1.0 - prob_buy
            
            signal_buy = prob_buy >= confidence_threshold
            signal_sell = prob_sell >= confidence_threshold
            
            atr = float(live_features['atr'].iloc[0]) if 'atr' in live_features.columns else 3.0
            multiplier = get_tp_sl_multiplier(mode)

        except Exception as e:
            logger.error(f"Error saat komputasi AI/Feature: {e}")
            time.sleep(10)
            continue
            
        # Eksekusi Smart Exits (termasuk BEP, Trailing Stop, RSI Exit, Regime Exit)
        terapkan_smart_exits(regime, rsi, harga_close)
        
        if regime == "CHOPPY":
            # Jangan spam log setiap loop, cukup tunggu
            time.sleep(60) # Wait 1 minute before checking again
            continue
            
        # Logging
        sekarang_wib = datetime.now(wib)
        posisi_terbuka = mt5.positions_get(symbol=cfg.SYMBOL, magic=cfg.MAGIC_NUMBER)
        ada_posisi = len(posisi_terbuka) > 0 if posisi_terbuka else False

        # --- TRACK CLOSED TRADES FOR LOSS_BERUNTUN ---
        if ada_posisi:
            last_active_ticket = posisi_terbuka[0].ticket
        else:
            if last_active_ticket is not None:
                time.sleep(1)
                from_date = datetime.now() - timedelta(days=1)
                to_date = datetime.now() + timedelta(days=1)
                deals = mt5.history_deals_get(from_date, to_date)
                
                if deals:
                    net_profit = 0.0
                    closed = False
                    for deal in deals:
                        if getattr(deal, 'position_id', None) == last_active_ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                            net_profit += deal.profit + deal.swap + deal.commission
                            closed = True
                    
                    if closed:
                        if net_profit < 0:
                            loss_beruntun += 1
                            logger.warning(f"❌ Trade #{last_active_ticket} tertutup LOSS (${net_profit:.2f}). Loss beruntun: {loss_beruntun}")
                        else:
                            loss_beruntun = 0
                            logger.info(f"✅ Trade #{last_active_ticket} tertutup PROFIT (${net_profit:.2f}). Loss beruntun direset.")
                
                last_active_ticket = None
        # ----------------------------------------------

        if last_log_time is None or (sekarang_wib - last_log_time).total_seconds() >= 60:
            logger.info("==========================================")
            logger.info(f"Waktu Live: {sekarang_wib.strftime('%Y-%m-%d %H:%M:%S')} (Harga: {harga_close:.2f})")
            
            if ada_posisi:
                pos = posisi_terbuka[0]
                tipe_order = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                logger.info("--- POSISI AKTIF ---")
                logger.info(f"Tipe   : {tipe_order} | Tiket: {pos.ticket}")
                logger.info(f"Open   : {pos.price_open:.2f} | Profit: ${pos.profit:.2f}")
                logger.info(f"SL     : {pos.sl:.2f} | TP : {pos.tp:.2f}")
                logger.info("[INFO] Radar AI DIJEDA sementara sampai posisi ini selesai.")
            else:
                logger.info("--- AI PREDICTION V3 ---")
                logger.info(f"Probabilitas BUY : {prob_buy*100:.1f}%")
                logger.info(f"Probabilitas SELL: {prob_sell*100:.1f}%")
                logger.info(f"Threshold OP     : {confidence_threshold*100:.1f}%")
            logger.info("==========================================")
            last_log_time = sekarang_wib

        # Eksekusi Order
        if not ada_posisi:
            # --- FUNNEL LOGGING (Catat Sinyal AI yg Diblokir Filter) ---
            if (signal_buy and not is_buy_allowed) and (waktu_candle != last_signal_time):
                logger.info(f"[BLOCKED] AI prediksi BUY ({prob_buy*100:.1f}%), tapi DIBLOKIR oleh Liquidity Sweep Filter (tidak ada sweep di 10 candle terakhir).")
                last_signal_time = waktu_candle # Supaya tidak spam
                
            if (signal_sell and not is_sell_allowed) and (waktu_candle != last_signal_time):
                logger.info(f"[BLOCKED] AI prediksi SELL ({prob_sell*100:.1f}%), tapi DIBLOKIR oleh Liquidity Sweep Filter (tidak ada sweep di 10 candle terakhir).")
                last_signal_time = waktu_candle # Supaya tidak spam
                
            # Menggabungkan Signal ML dan Izin dari Sweep Filter
            if (signal_buy and is_buy_allowed) or (signal_sell and is_sell_allowed):
                if waktu_candle != last_signal_time:
                    strategy_name = "AI_BUY+SWEEP" if signal_buy else "AI_SELL+SWEEP"
                    logger.info(f"[🔥] SINYAL {strategy_name} TERDETEKSI! Confidence: {max(prob_buy, prob_sell)*100:.1f}%")
                
                    if signal_buy:
                        ask = mt5.symbol_info_tick(cfg.SYMBOL).ask
                        order_type = mt5.ORDER_TYPE_BUY
                        price = ask
                        prob = prob_buy
                        
                        # --- Dynamic SL ---
                        sl_statik = ask - (atr * multiplier['sl'])
                        sl_dynamic = (last_swing_low - (0.3 * atr)) if last_swing_low and not pd.isna(last_swing_low) else sl_statik
                        sl = min(sl_statik, sl_dynamic)  # Pilih yang lebih protektif (rendah)
                        
                        # --- Dynamic TP ---
                        tp_statik = ask + (atr * multiplier['tp'])
                        h1_swing_high = float(live_features['h1_last_swing_high'].iloc[0]) if 'h1_last_swing_high' in live_features.columns else None
                        if h1_swing_high and not pd.isna(h1_swing_high) and h1_swing_high > ask:
                            tp = h1_swing_high
                        else:
                            tp = tp_statik
                    else:
                        bid = mt5.symbol_info_tick(cfg.SYMBOL).bid
                        order_type = mt5.ORDER_TYPE_SELL
                        price = bid
                        prob = prob_sell
                        
                        # --- Dynamic SL ---
                        sl_statik = bid + (atr * multiplier['sl'])
                        sl_dynamic = (last_swing_high + (0.3 * atr)) if last_swing_high and not pd.isna(last_swing_high) else sl_statik
                        sl = max(sl_statik, sl_dynamic)  # Pilih yang lebih protektif (tinggi)
                        
                        # --- Dynamic TP ---
                        tp_statik = bid - (atr * multiplier['tp'])
                        h1_swing_low = float(live_features['h1_last_swing_low'].iloc[0]) if 'h1_last_swing_low' in live_features.columns else None
                        if h1_swing_low and not pd.isna(h1_swing_low) and h1_swing_low < bid:
                            tp = h1_swing_low
                        else:
                            tp = tp_statik
                        
                    # Kelly Position Scaler
                    lot_size = cfg.LOT_SIZE
                    if prob > 0.85:
                        lot_size = cfg.LOT_SIZE * 3
                        logger.info(f"💎 High Conviction! Prob {prob*100:.1f}% > 85%, Lot x3 -> {lot_size}")
                    elif prob > 0.70:
                        lot_size = cfg.LOT_SIZE * 2
                        logger.info(f"🚀 Medium Conviction! Prob {prob*100:.1f}% > 70%, Lot x2 -> {lot_size}")
                        
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": cfg.SYMBOL,
                        "volume": lot_size,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": cfg.DEVIATION,
                        "magic": cfg.MAGIC_NUMBER,
                        "comment": f"{strategy_name} {mode}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        logger.error(f"Order {strategy_name} gagal: {result.retcode}")
                    else:
                        logger.info(f"ORDER BERHASIL! Tiket #{result.order} | TP: {tp:.2f} | SL: {sl:.2f}")
                        trade_count += 1
                        last_signal_time = waktu_candle  # Catat waktu candle agar tidak OP lagi di candle yang sama
                        
        time.sleep(2)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot dihentikan oleh user. Mengumpulkan laporan trading...")
        try:
            from_date = datetime.now() - timedelta(days=2)
            to_date = datetime.now() + timedelta(days=2)
            deals = mt5.history_deals_get(from_date, to_date)
            
            if deals:
                total_profit = 0.0
                win_count = 0
                loss_count = 0
                trade_count = 0
                
                for deal in deals:
                    if deal.magic == cfg.MAGIC_NUMBER and deal.entry == mt5.DEAL_ENTRY_OUT and deal.time >= getattr(main, 'bot_start_time_server', 0):
                        trade_count += 1
                        net_profit = deal.profit + deal.swap + deal.commission
                        total_profit += net_profit
                        if net_profit > 0:
                            win_count += 1
                        else:
                            loss_count += 1
                            
                if trade_count > 0:
                    logger.info("========== LAPORAN SESI INI ==========")
                    logger.info(f"Total Trade Tertutup : {trade_count} posisi")
                    logger.info(f"Win / Loss           : {win_count} Win / {loss_count} Loss")
                    logger.info(f"Net Profit/Loss      : ${total_profit:.2f}")
                    logger.info("======================================")
                else:
                    logger.info("Tidak ada posisi yang selesai dieksekusi selama sesi ini berjalan.")
        except Exception as e:
            logger.error(f"Gagal memuat laporan: {e}")
            
        mt5.shutdown()
