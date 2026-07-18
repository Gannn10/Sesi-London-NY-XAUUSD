import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
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
    positions = mt5.positions_get(symbol=cfg.SYMBOL)
    if not positions: return False
    return any(p.magic == cfg.MAGIC_NUMBER for p in positions)

def terapkan_smart_exits(regime, rsi, harga_close):
    positions = mt5.positions_get(symbol=cfg.SYMBOL)
    if positions is None or len(positions) == 0:
        return

    for pos in positions:
        if pos.magic == cfg.MAGIC_NUMBER:
            tick = mt5.symbol_info_tick(cfg.SYMBOL)
            if tick is None: continue
            
            # Hitung Profit
            profit_usd = 0.0
            if pos.type == mt5.ORDER_TYPE_BUY:
                profit_usd = tick.bid - pos.price_open
            else:
                profit_usd = pos.price_open - tick.ask
                
            # --- 1. RSI EXIT ---
            # Jika Buy, harga profit > $2.00, dan RSI Overbought (> 75) -> Close
            if pos.type == mt5.ORDER_TYPE_BUY and profit_usd >= 2.0 and rsi >= 75:
                logger.info(f"[SMART EXIT] RSI Overbought ({rsi:.1f}). Menutup posisi BUY #{pos.ticket} untuk amankan profit (${profit_usd:.2f}).")
                close_position(pos, tick.bid)
                continue
                
            # Jika Sell, harga profit > $2.00, dan RSI Oversold (< 25) -> Close
            elif pos.type == mt5.ORDER_TYPE_SELL and profit_usd >= 2.0 and rsi <= 25:
                logger.info(f"[SMART EXIT] RSI Oversold ({rsi:.1f}). Menutup posisi SELL #{pos.ticket} untuk amankan profit (${profit_usd:.2f}).")
                close_position(pos, tick.ask)
                continue
                
            # --- 2. REGIME EXIT ---
            # Jika market berubah jadi Choppy dan posisi sedang profit -> Close
            if regime == "CHOPPY" and profit_usd >= 3.0: # Profit minimal $3.00 (30 pips XAUUSD)
                logger.info(f"[SMART EXIT] Market menjadi CHOPPY. Menutup posisi #{pos.ticket} untuk amankan profit (${profit_usd:.2f}).")
                close_position(pos, tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask)
                continue

            # --- 3. DYNAMIC TRAILING STOP (Step Trailing) ---
            # Jika profit >= $6.00 (60 pips), trailing stop = harga_sekarang - $3.00
            # Jika profit >= $4.00 (40 pips), SL digeser ke Breakeven
            if pos.type == mt5.ORDER_TYPE_BUY:
                if profit_usd >= 6.0: # 60 pips
                    new_sl = tick.bid - 3.00
                    if pos.sl == 0.0 or new_sl > pos.sl:
                        modify_sl(pos, new_sl)
                        logger.info(f"[TRAILING STOP] SL BUY #{pos.ticket} digeser ke {new_sl:.2f}")
                elif profit_usd >= 4.0: # 40 pips
                    if pos.sl == 0.0 or pos.sl < pos.price_open:
                        modify_sl(pos, pos.price_open)
                        logger.info(f"[SMART BE] Posisi BUY #{pos.ticket} diamankan di BEP.")
            elif pos.type == mt5.ORDER_TYPE_SELL:
                if profit_usd >= 6.0: # 60 pips
                    new_sl = tick.ask + 3.00
                    if pos.sl == 0.0 or new_sl < pos.sl:
                        modify_sl(pos, new_sl)
                        logger.info(f"[TRAILING STOP] SL SELL #{pos.ticket} digeser ke {new_sl:.2f}")
                elif profit_usd >= 4.0: # 40 pips
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
        
        # Details for table
        trade_details = []
        
        if deals:
            for deal in deals:
                if deal.magic == cfg.MAGIC_NUMBER and deal.entry == mt5.DEAL_ENTRY_OUT:
                    trade_count += 1
                    net_profit = deal.profit + deal.swap + deal.commission
                    total_profit += net_profit
                    is_win = net_profit > 0
                    if is_win:
                        win_count += 1
                    else:
                        loss_count += 1
                        
                    # Deteksi strategi dari comment entry
                    pos_id = getattr(deal, 'position_id', None)
                    strat_name = "Manual / Unknown"
                    entry_time_str = "-"
                    entry_price = 0.0
                    order_type_str = "BUY" if deal.type == mt5.DEAL_TYPE_SELL else "SELL" # Reverse logic because out deal
                    
                    if pos_id:
                        entry_deals = [d for d in deals if getattr(d, 'position_id', None) == pos_id and d.entry == mt5.DEAL_ENTRY_IN]
                        if entry_deals:
                            ed = entry_deals[0]
                            strat_name = ed.comment
                            entry_time_str = datetime.fromtimestamp(ed.time).strftime('%H:%M:%S')
                            entry_price = ed.price
                            order_type_str = "BUY" if ed.type == mt5.DEAL_TYPE_BUY else "SELL"
                            
                    # Formatting weapon name for UI
                    if "AI_XGBOOST" in strat_name:
                        weapon = "🤖 AI XGBoost"
                    elif "SNIPER" in strat_name or "REVERSAL_SNIPER" in strat_name:
                        weapon = "🎯 Reversal Sniper"
                    elif "SBR" in strat_name or "RBS" in strat_name:
                        weapon = "🧱 SBR/RBS Bounce"
                    else:
                        weapon = strat_name
                        
                    pips = abs(deal.price - entry_price) * 10 # rough pips calc for XAUUSD (assumes 2 decimal point)
                    pips_formatted = f"+{pips:.1f}" if is_win else f"-{pips:.1f}"
                    
                    trade_details.append({
                        "time": entry_time_str,
                        "weapon": weapon,
                        "type": order_type_str,
                        "price": f"{entry_price:.2f}",
                        "status": "Win" if is_win else "Loss",
                        "pips": pips_formatted,
                        "profit": f"${net_profit:.2f}"
                    })
        
        # Win rate
        win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
        net_profit_pips = total_profit / cfg.LOT_SIZE / 10 # Rough estimate: 1 lot = $10 per pip
        
        # Account info
        account = mt5.account_info()
        balance = account.balance if account else 0.0
        equity = account.equity if account else 0.0
        
        if trade_count == 0:
            logger.info("Tidak ada trade kemarin, Notion di-skip.")
            return

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        
        judul = f"📊 LAPORAN HARIAN BOT TRADING XAU/USD ({tanggal.strftime('%Y-%m-%d')})"
        
        # Build children blocks
        children = []
        
        # Header quote
        children.append({
            "object": "block",
            "type": "quote",
            "quote": {
                "rich_text": [{"type": "text", "text": {"content": f"Hari/Tanggal: {tanggal.strftime('%Y-%m-%d')} | Timeframe: M5 | Versi Bot: v2.1 (Hybrid Multi-Strategy Engine)"}}]
            }
        })
        
        children.append({"object": "block", "type": "divider", "divider": {}})
        
        # 1. Ringkasan Performa
        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "📈 1. RINGKASAN PERFORMA"}}]}
        })
        
        children.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": f"Net Profit/Loss: {net_profit_pips:.1f} pips / ${total_profit:.2f}"}}]}
        })
        children.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": f"Total Trade: {trade_count} (🟢 {win_count} Win / 🔴 {loss_count} Loss)"}}]}
        })
        children.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": f"Win Rate: {win_rate:.1f}%"}}]}
        })
        children.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": f"Status Akun: Balance: ${balance:.2f} | Equity: ${equity:.2f}"}}]}
        })
        
        children.append({"object": "block", "type": "divider", "divider": {}})
        
        # 2. Rincian Eksekusi
        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "⚔️ 2. RINCIAN EKSEKUSI STRATEGI (INDEPENDEN)"}}]}
        })
        
        # Create Table block
        table_rows = []
        # Table Header
        table_rows.append({
            "object": "block",
            "type": "table_row",
            "table_row": {
                "cells": [
                    [{"type": "text", "text": {"content": "Waktu (WIB)"}}],
                    [{"type": "text", "text": {"content": "Senjata"}}],
                    [{"type": "text", "text": {"content": "Posisi"}}],
                    [{"type": "text", "text": {"content": "Entry"}}],
                    [{"type": "text", "text": {"content": "Status"}}],
                    [{"type": "text", "text": {"content": "Pips"}}],
                    [{"type": "text", "text": {"content": "Profit"}}]
                ]
            }
        })
        
        for t in trade_details:
            table_rows.append({
                "object": "block",
                "type": "table_row",
                "table_row": {
                    "cells": [
                        [{"type": "text", "text": {"content": t['time']}}],
                        [{"type": "text", "text": {"content": t['weapon']}}],
                        [{"type": "text", "text": {"content": t['type']}}],
                        [{"type": "text", "text": {"content": t['price']}}],
                        [{"type": "text", "text": {"content": t['status']}}],
                        [{"type": "text", "text": {"content": t['pips']}}],
                        [{"type": "text", "text": {"content": t['profit']}}]
                    ]
                }
            })
            
        children.append({
            "object": "block",
            "type": "table",
            "table": {
                "table_width": 7,
                "has_column_header": True,
                "has_row_header": False,
                "children": table_rows
            }
        })
        
        children.append({"object": "block", "type": "divider", "divider": {}})
        
        # 3. Evaluasi
        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "📝 3. EVALUASI OTOMATIS BOT"}}]}
        })
        
        eval_notes = "Market sedang sulit ditebak, mode bertahan aktif untuk melindungi equity."
        if win_rate >= 60 and total_profit > 0:
            eval_notes = "Hari yang sangat baik. Filter berjalan sempurna menghindari noise market."
        elif win_rate >= 50 and total_profit > 0:
            eval_notes = "Performa standar. Bot berhasil mencetak net profit positif dari scalping."
            
        children.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": f"Catatan Hari Ini: {eval_notes}"}}]}
        })
        
        data = {
            "parent": {"page_id": db_id},
            "properties": {
                "title": {"title": [{"text": {"content": judul}}]},
            },
            "children": children
        }

        response = requests.post("https://api.notion.com/v1/pages", headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            logger.info("✅ Laporan harian berhasil masuk ke Notion!")
        else:
            logger.error(f"❌ Gagal mengirim ke Notion: {response.text}")
            
    except Exception as e:
        logger.error(f"Error saat mengirim ke Notion: {e}")

def extract_live_features_v5(df_m5_pd, df_m15_pd):
    df_m5 = pl.from_pandas(df_m5_pd)
    df_m15 = pl.from_pandas(df_m15_pd)
    
    fe = FeatureEngineer()
    smc = SMCAnalyzer()
    
    # Calculate M5 features
    df = fe.calculate_all(df_m5, include_ml_features=True)
    df = smc.calculate_all(df)
    
    # Calculate M15 features
    df_m15 = fe.calculate_all(df_m15, include_ml_features=False)
    df_m15 = smc.calculate_all(df_m15)
    
    # Join M15 features
    m15_feature_cols = [
        "time", "close", "rsi", "atr", "bb_upper", "bb_lower",
        "macd", "macd_signal", "ema_20", "ema_50",
        "ob", "fvg", "market_structure", "last_swing_high", "last_swing_low"
    ]
    m15_feature_cols = [c for c in m15_feature_cols if c in df_m15.columns]
    
    m15_subset = df_m15.select(m15_feature_cols)
    m15_subset = m15_subset.rename({c: f"m15_{c}" for c in m15_feature_cols if c != "time"})
    
    # FIX: Shift M15 time forward by 15 minutes to prevent look-ahead bias
    m15_subset = m15_subset.with_columns(
        (pl.col("time") + pl.duration(minutes=15)).alias("time")
    )
    
    df = df.join_asof(
        m15_subset,
        on="time",
        strategy="backward"
    )
    
    # Calculate M15 derived features
    if "m15_close" in df.columns and "m15_ema_20" in df.columns:
        df = df.with_columns([
            ((pl.col("m15_close") - pl.col("m15_ema_20")) / pl.col("m15_ema_20")).alias("m15_ema20_distance")
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
    model_path = os.path.join(os.path.dirname(__file__), 'backtests', 'ml_v5', 'xgboost_model_v5_scalper.pkl')
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        xgb_model = model_data['xgb_model']
        feature_cols = model_data['feature_names']
        confidence_threshold = getattr(cfg, 'AI_CONFIDENCE_THRESHOLD', model_data.get('confidence_threshold', 0.60))
        logger.info(f"✅ AI Model V5 (Enhanced Scalper) Loaded! Threshold: {confidence_threshold*100:.1f}%")
        logger.info(f" Fitur yang digunakan: {len(feature_cols)} kolom")
    except Exception as e:
        logger.error(f"❌ Gagal meload model AI: {e}. Pastikan sudah menjalankan Fase 1.")
        return

    logger.info("Bot London & NY (AI V5 Scalper) mulai berjalan...")

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
        df_m15_pd = get_data(cfg.SYMBOL, mt5.TIMEFRAME_M15, 100)
        
        if df_m5_pd is None or df_m15_pd is None or len(df_m5_pd) < 200 or len(df_m15_pd) < 50:
            logger.warning("Gagal mengambil data M5/M15 yang cukup, coba lagi...")
            time.sleep(5)
            continue
            
        # HMM Regime Detector (menggunakan candle yang sudah close saja / index[:-1])
        regime = detect_market_regime(df_m5_pd.iloc[:-1])
            
        try:
            # 1. Ekstrak fitur menggunakan Polars
            df_features = extract_live_features_v5(df_m5_pd, df_m15_pd)
            
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
            use_sweep_filter = getattr(cfg, 'USE_SWEEP_FILTER', False)
            use_momentum_filter = getattr(cfg, 'USE_MOMENTUM_FILTER', True)
            use_trend_filter = getattr(cfg, 'USE_TREND_FILTER', False)
            
            is_buy_allowed = True
            is_sell_allowed = True
            
            # --- 1. SWEEP FILTER ---
            if use_sweep_filter:
                is_buy_allowed = False
                is_sell_allowed = False
                
                if last_swing_low is not None and not pd.isna(last_swing_low):
                    sweep_buy_df = recent_features[(recent_features['low'] < recent_features['last_swing_low']) & (recent_features['close'] > recent_features['last_swing_low'])]
                    if len(sweep_buy_df) > 0:
                        is_buy_allowed = True
                        
                if last_swing_high is not None and not pd.isna(last_swing_high):
                    sweep_sell_df = recent_features[(recent_features['high'] > recent_features['last_swing_high']) & (recent_features['close'] < recent_features['last_swing_high'])]
                    if len(sweep_sell_df) > 0:
                        is_sell_allowed = True

            # --- 2. MOMENTUM FILTER (Mencegah tangkap pisau jatuh) ---
            atr = float(live_features['atr'].iloc[0]) if 'atr' in live_features.columns else 3.0
            live_open = float(live_features['open'].iloc[0])
            live_body = abs(live_open - harga_close)
            
            blocked_by_momentum = False
            blocked_by_trend = False
            
            if use_momentum_filter:
                # Jika candle saat ini sedang turun tajam (body merah > 0.8 ATR)
                if harga_close < live_open and live_body > (0.8 * atr):
                    is_buy_allowed = False
                    blocked_by_momentum = True
                    
                # Jika candle saat ini sedang naik tajam (body hijau > 0.8 ATR)
                if harga_close > live_open and live_body > (0.8 * atr):
                    is_sell_allowed = False
                    blocked_by_momentum = True
                    
            # --- 3. TREND FILTER (EMA 200) ---
            if use_trend_filter and 'ema_200' in live_features.columns:
                ema_200 = float(live_features['ema_200'].iloc[0])
                if harga_close < ema_200:
                    is_buy_allowed = False # Trend turun, jangan buy
                    blocked_by_trend = True
                if harga_close > ema_200:
                    is_sell_allowed = False # Trend naik, jangan sell
                    blocked_by_trend = True
            
            # --- 4. FALLING KNIFE / ANTI-CRASH GUARD ---
            # Hitung pergerakan harga dalam 3 candle terakhir (closed)
            recent_3 = df_features.tail(4).iloc[:-1]  # 3 candle terakhir (exclude live)
            if len(recent_3) >= 3:
                price_drop_3c = recent_3['close'].iloc[0] - recent_3['close'].iloc[-1]  # Positif = harga turun
                price_rise_3c = recent_3['close'].iloc[-1] - recent_3['close'].iloc[0]  # Positif = harga naik
            else:
                price_drop_3c = 0
                price_rise_3c = 0
            
            # EMA 50 untuk Trend Guard Sniper
            ema_50 = float(live_features['ema_50'].iloc[0]) if 'ema_50' in live_features.columns else harga_close
            
            # Flags untuk logging
            falling_knife_blocked = False
            sniper_trend_blocked = False
            sniper_crash_blocked = False
            
            # 3. Prediksi dengan XGBoost (menggunakan fitur closed candle)
            X_closed = closed_features[feature_cols]
            dmatrix_closed = xgb.DMatrix(X_closed)
            
            prob_buy = float(xgb_model.predict(dmatrix_closed)[0])
            prob_sell = 1.0 - prob_buy
            
            ai_signal_buy = prob_buy >= confidence_threshold
            ai_signal_sell = prob_sell >= confidence_threshold
            
            # FALLING KNIFE GUARD untuk AI (1x ATR)
            if ai_signal_buy and price_drop_3c > (1.0 * atr):
                ai_signal_buy = False
                falling_knife_blocked = True
                logger.info(f"🛡️ FALLING KNIFE GUARD: AI BUY diblokir! Harga jatuh {price_drop_3c:.2f} > ATR {atr:.2f} dalam 3 candle")
            if ai_signal_sell and price_rise_3c > (1.0 * atr):
                ai_signal_sell = False
                falling_knife_blocked = True
                logger.info(f"🛡️ FALLING KNIFE GUARD: AI SELL diblokir! Harga naik {price_rise_3c:.2f} > ATR {atr:.2f} dalam 3 candle")
            
            # --- SNIPER LOGIC EVALUATION ---
            live_open_s = float(live_features['open'].iloc[0])
            live_high_s = float(live_features['high'].iloc[0])
            live_low_s = float(live_features['low'].iloc[0])
            
            bb_upper = float(live_features['bb_upper'].iloc[0]) if 'bb_upper' in live_features.columns else harga_close
            bb_mid = float(live_features['bb_middle'].iloc[0]) if 'bb_middle' in live_features.columns else harga_close
            std_value = (bb_upper - bb_mid) / getattr(cfg, 'BB_STD_DEV', 2.0)
            
            bb_std_ex = getattr(cfg, 'BB_STD_DEV_EXTREME', 2.5)
            upper_bb_extreme = bb_mid + (bb_std_ex * std_value)
            lower_bb_extreme = bb_mid - (bb_std_ex * std_value)
            
            candle_range = live_high_s - live_low_s
            if candle_range == 0: candle_range = 0.0001
            upper_wick = live_high_s - max(live_open_s, harga_close)
            lower_wick = min(live_open_s, harga_close) - live_low_s
            
            is_rejection_top = (upper_wick > (0.4 * candle_range)) or (harga_close < live_open_s)
            is_rejection_bottom = (lower_wick > (0.4 * candle_range)) or (harga_close > live_open_s)
            
            rsi_ob_ex = getattr(cfg, 'RSI_OVERBOUGHT_EXTREME', 70)
            rsi_os_ex = getattr(cfg, 'RSI_OVERSOLD_EXTREME', 30)
            
            sniper_signal_sell = (harga_close >= upper_bb_extreme or live_high_s >= upper_bb_extreme) and (rsi >= rsi_ob_ex) and is_rejection_top
            sniper_signal_buy = (harga_close <= lower_bb_extreme or live_low_s <= lower_bb_extreme) and (rsi <= rsi_os_ex) and is_rejection_bottom
            
            # SNIPER ANTI-CRASH FILTER (2x ATR dalam 3 candle — hanya blokir saat crash parah)
            if sniper_signal_buy and price_drop_3c > (2.0 * atr):
                sniper_signal_buy = False
                sniper_crash_blocked = True
                logger.info(f"🛡️ ANTI-CRASH: Sniper BUY diblokir! Crash {price_drop_3c:.2f} > 2×ATR {2.0*atr:.2f}")
            if sniper_signal_sell and price_rise_3c > (2.0 * atr):
                sniper_signal_sell = False
                sniper_crash_blocked = True
                logger.info(f"🛡️ ANTI-CRASH: Sniper SELL diblokir! Rally {price_rise_3c:.2f} > 2×ATR {2.0*atr:.2f}")
            
            # --- SBR/RBS LOGIC EVALUATION ---
            sbr_signal_sell = False
            rbs_signal_buy = False
            
            if last_swing_high is not None and not pd.isna(last_swing_high) and last_swing_low is not None and not pd.isna(last_swing_low):
                # 1. Breakout Valid & Freshness (Lookback 5 candles)
                recent_5 = df_features.tail(6).iloc[:-1] # Exclude live candle
                
                # Check RBS (Buy)
                breakout_rbs_df = recent_5[recent_5['close'] > last_swing_high]
                if not breakout_rbs_df.empty:
                    buffer = 0.20 # 2 pips
                    if live_low_s <= last_swing_high + buffer and live_open_s > last_swing_high:
                        if harga_close > last_swing_high:
                            rbs_signal_buy = True
                            
                # Check SBR (Sell)
                breakout_sbr_df = recent_5[recent_5['close'] < last_swing_low]
                if not breakout_sbr_df.empty:
                    buffer = 0.20 # 2 pips
                    if live_high_s >= last_swing_low - buffer and live_open_s < last_swing_low:
                        if harga_close < last_swing_low:
                            sbr_signal_sell = True
                            
            # --- STRATEGY OVERRIDE ---
            strategy_mode = getattr(cfg, 'STRATEGY_MODE', 'HYBRID')
            strategy_name_buy = "🤖 AI_XGBOOST"
            strategy_name_sell = "🤖 AI_XGBOOST"
            
            if strategy_mode == 'REVERSAL_SNIPER':
                signal_buy = sniper_signal_buy
                signal_sell = sniper_signal_sell
                if signal_buy: 
                    prob_buy = 1.0; prob_sell = 0.0; is_buy_allowed = True
                    strategy_name_buy = "🎯 REVERSAL_SNIPER"
                if signal_sell: 
                    prob_sell = 1.0; prob_buy = 0.0; is_sell_allowed = True
                    strategy_name_sell = "🎯 REVERSAL_SNIPER"
            elif strategy_mode == 'HYBRID':
                signal_buy = ai_signal_buy or sniper_signal_buy or rbs_signal_buy
                signal_sell = ai_signal_sell or sniper_signal_sell or sbr_signal_sell
                
                if rbs_signal_buy:
                    prob_buy = 1.0; prob_sell = 0.0; is_buy_allowed = True
                    strategy_name_buy = "🧱 RBS_BUY"
                elif sniper_signal_buy:
                    prob_buy = 1.0; prob_sell = 0.0; is_buy_allowed = True
                    strategy_name_buy = "🎯 REVERSAL_SNIPER"
                elif ai_signal_buy:
                    strategy_name_buy = "🤖 AI_XGBOOST"
                    
                if sbr_signal_sell:
                    prob_sell = 1.0; prob_buy = 0.0; is_sell_allowed = True
                    strategy_name_sell = "🧱 SBR_SELL"
                elif sniper_signal_sell:
                    prob_sell = 1.0; prob_buy = 0.0; is_sell_allowed = True
                    strategy_name_sell = "🎯 REVERSAL_SNIPER"
                elif ai_signal_sell:
                    strategy_name_sell = "🤖 AI_XGBOOST"
            else: # AI_SCALPER
                signal_buy = ai_signal_buy
                signal_sell = ai_signal_sell
            
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
        semua_posisi = mt5.positions_get(symbol=cfg.SYMBOL)
        posisi_terbuka = [p for p in semua_posisi if p.magic == cfg.MAGIC_NUMBER] if semua_posisi else []
        ada_posisi = len(posisi_terbuka) > 0

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
                logger.info(f"--- POSISI AKTIF ({len(posisi_terbuka)}/{getattr(cfg, 'MAX_POSITIONS', 3)}) ---")
                for p in posisi_terbuka:
                    t_order = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
                    logger.info(f"Tiket: {p.ticket} | {t_order} | Open: {p.price_open:.2f} | Profit: ${p.profit:.2f}")
            
            if len(posisi_terbuka) >= getattr(cfg, 'MAX_POSITIONS', 3):
                logger.info("[INFO] Radar AI DIJEDA sementara karena sudah mencapai batas maksimal posisi aktif.")
            else:
                strategy_mode = getattr(cfg, 'STRATEGY_MODE', 'HYBRID')
                bb_upper_log = float(live_features['bb_upper'].iloc[0]) if 'bb_upper' in live_features.columns else harga_close
                bb_mid_log = float(live_features['bb_middle'].iloc[0]) if 'bb_middle' in live_features.columns else harga_close
                std_log = (bb_upper_log - bb_mid_log) / getattr(cfg, 'BB_STD_DEV', 2.0)
                bb_ex = getattr(cfg, 'BB_STD_DEV_EXTREME', 2.5)
                up_ex = bb_mid_log + (bb_ex * std_log)
                dn_ex = bb_mid_log - (bb_ex * std_log)
                
                if strategy_mode == 'REVERSAL_SNIPER':
                    logger.info("--- 🎯 MODE SNIPER PUCUK/LEMBAH ---")
                    logger.info(f"Target Pucuk (SELL) : > {up_ex:.2f}")
                    logger.info(f"Target Lembah (BUY) : < {dn_ex:.2f}")
                    logger.info(f"RSI Saat Ini        : {rsi:.2f}")
                elif strategy_mode == 'HYBRID':
                    logger.info("--- 🤖 AI SCALPER & 🎯 SNIPER & 🧱 SBR/RBS ---")
                    logger.info(f"AI Prob BUY : {prob_buy*100:.1f}% | SELL : {prob_sell*100:.1f}% (Threshold: {confidence_threshold*100:.1f}%)")
                    logger.info(f"Sniper Pucuk: > {up_ex:.2f} | Lembah: < {dn_ex:.2f} | RSI: {rsi:.2f}")
                    
                    jarak_rbs = harga_close - last_swing_high if last_swing_high and not pd.isna(last_swing_high) else 0
                    jarak_sbr = last_swing_low - harga_close if last_swing_low and not pd.isna(last_swing_low) else 0
                    logger.info(f"SnR Flip -> RBS (Buy): {last_swing_high if last_swing_high else 0:.2f} (Jarak: {jarak_rbs:.2f}) | SBR (Sell): {last_swing_low if last_swing_low else 0:.2f} (Jarak: {jarak_sbr:.2f})")
                    
                    # Guard status
                    guards = []
                    if falling_knife_blocked: guards.append("🛡️ FallingKnife(AI)")
                    if sniper_trend_blocked: guards.append("🛡️ TrendGuard(Sniper)")
                    if sniper_crash_blocked: guards.append("🛡️ AntiCrash(Sniper)")
                    if guards:
                        logger.info(f"GUARDS AKTIF: {' | '.join(guards)}")
                    else:
                        logger.info(f"Guards: ✅ Semua senjata siap tempur | Drop3C: {price_drop_3c:.2f} | Rise3C: {price_rise_3c:.2f} | EMA50: {ema_50:.2f}")
                else:
                    logger.info("--- AI PREDICTION V3 ---")
                    logger.info(f"Probabilitas BUY : {prob_buy*100:.1f}%")
                    logger.info(f"Probabilitas SELL: {prob_sell*100:.1f}%")
                    logger.info(f"Threshold OP     : {confidence_threshold*100:.1f}%")
            logger.info("==========================================")
            last_log_time = sekarang_wib

        # Eksekusi Order
        if len(posisi_terbuka) < getattr(cfg, 'MAX_POSITIONS', 3):
            # --- FUNNEL LOGGING (Catat Sinyal AI yg Diblokir Filter) ---
            if (signal_buy and not is_buy_allowed) and (waktu_candle != last_signal_time):
                if blocked_by_momentum:
                    logger.info(f"[BLOCKED] AI BUY ({prob_buy*100:.1f}%), tapi DIBLOKIR: Momentum sedang terjun tajam (Pisau Jatuh)!")
                elif blocked_by_trend:
                    logger.info(f"[BLOCKED] AI BUY ({prob_buy*100:.1f}%), tapi DIBLOKIR: Melawan Trend (Harga di bawah EMA 200).")
                else:
                    logger.info(f"[BLOCKED] AI BUY ({prob_buy*100:.1f}%), tapi DIBLOKIR oleh Filter (Sweep).")
                last_signal_time = waktu_candle # Supaya tidak spam
                
            if (signal_sell and not is_sell_allowed) and (waktu_candle != last_signal_time):
                if blocked_by_momentum:
                    logger.info(f"[BLOCKED] AI SELL ({prob_sell*100:.1f}%), tapi DIBLOKIR: Momentum sedang terbang kencang!")
                elif blocked_by_trend:
                    logger.info(f"[BLOCKED] AI SELL ({prob_sell*100:.1f}%), tapi DIBLOKIR: Melawan Trend (Harga di atas EMA 200).")
                else:
                    logger.info(f"[BLOCKED] AI SELL ({prob_sell*100:.1f}%), tapi DIBLOKIR oleh Filter (Sweep).")
                last_signal_time = waktu_candle # Supaya tidak spam
                
            # Menggabungkan Signal ML dan Izin dari Sweep Filter
            if (signal_buy and is_buy_allowed) or (signal_sell and is_sell_allowed):
                if waktu_candle != last_signal_time:
                    strategy_name = strategy_name_buy if signal_buy else strategy_name_sell
                    logger.info(f"[🔥] SINYAL {strategy_name} TERDETEKSI! Confidence/Strength: {max(prob_buy, prob_sell)*100:.1f}%")
                
                    if signal_buy:
                        ask = mt5.symbol_info_tick(cfg.SYMBOL).ask
                        order_type = mt5.ORDER_TYPE_BUY
                        price = ask
                        prob = prob_buy
                        
                        # --- Dynamic SL ---
                        sl_statik = ask - (atr * multiplier['sl'])
                        sl_dynamic = (last_swing_low - (0.3 * atr)) if last_swing_low and not pd.isna(last_swing_low) else sl_statik
                        
                        # Pastikan sl_dynamic masuk akal (harus di bawah ask)
                        if sl_dynamic >= ask:
                            sl_dynamic = sl_statik
                            
                        sl = max(sl_statik, sl_dynamic)  # Pilih harga yang lebih tinggi (lebih dekat ke entry)
                        
                        # Pengaman terakhir:
                        if sl >= ask:
                            sl = ask - 2.0  # Default SL 20 pips jika perhitungan gagal
                        
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
                        
                        # Pastikan sl_dynamic masuk akal (harus di atas bid)
                        if sl_dynamic <= bid:
                            sl_dynamic = sl_statik
                            
                        sl = min(sl_statik, sl_dynamic)  # Pilih harga yang lebih rendah (lebih dekat ke entry)
                        
                        # Pengaman terakhir:
                        if sl <= bid:
                            sl = bid + 2.0  # Default SL 20 pips jika perhitungan gagal
                        
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
                        logger.error(f"Order {strategy_name} gagal: {result.retcode}. Price: {price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}")
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
