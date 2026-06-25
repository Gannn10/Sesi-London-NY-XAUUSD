import os
from dotenv import load_dotenv
import MetaTrader5 as mt5

load_dotenv()

# ── CORE ──────────────────────────────────────────────────
SYMBOL          = "XAUUSDm"
TIMEFRAME       = mt5.TIMEFRAME_M5
LOT_SIZE        = 0.01
MAGIC_NUMBER    = 999222             # BEDA dari Asia bot (888111)!
DEVIATION       = 20

# ── EMA ────────────────────────────────────────────────────
EMA_FAST        = 50
EMA_SLOW        = 200
NUM_CANDLES     = 250                # Harus > EMA_SLOW

# ── BOLLINGER BANDS ────────────────────────────────────────
BB_PERIOD       = 20
BB_STD_DEV      = 2.0
BB_WIDTH_MIN    = 1.0                # MINIMUM (kebalikan dari Asia bot)
                                     # Naikkan → lebih selektif
                                     # Turunkan → lebih banyak sinyal
TOLERANSI_BB    = 0.5                # ±$0.5 dari Upper/Lower untuk zona breakout

# ── RSI ────────────────────────────────────────────────────
RSI_PERIOD      = 14
RSI_BULL_MIN    = 52                 # RSI harus > ini untuk BUY Breakout
RSI_BEAR_MAX    = 48                 # RSI harus < ini untuk SELL Breakout
RSI_OVERBOUGHT  = 60                 # RSI harus > ini untuk SELL Reversal (Pucuk)
RSI_OVERSOLD    = 40                 # RSI harus < ini untuk BUY Reversal (Lembah)

# ── ASIA RANGE ─────────────────────────────────────────────
ASIA_RANGE_START     = 7            # Jam WIB mulai hitung range Asia
ASIA_RANGE_END       = 15           # Jam WIB selesai range Asia
BREAKOUT_BUFFER_ATR  = 0.1          # Buffer breakout = ATR × ini

# ── ATR ────────────────────────────────────────────────────
ATR_PERIOD           = 14

# London mode
LONDON_SL_MULTIPLIER = 0.8
LONDON_TP_MULTIPLIER = 1.0

# NY mode (lebih agresif)
NY_SL_MULTIPLIER     = 1.0
NY_TP_MULTIPLIER     = 1.2

# ── SESI WAKTU ─────────────────────────────────────────────
JAM_LONDON_BUKA      = 15           # WIB
JAM_LONDON_TUTUP     = 20           # WIB
JAM_NY_BUKA          = 20           # WIB
JAM_NY_TUTUP         = 24           # WIB (00:00)
JAM_LONDON_TUTUP_AMAN = 19          # 19:45 WIB (Opsional, tapi disini dipasang aman 19:00)
JAM_NY_TUTUP_AMAN    = 23           # 23:45 WIB (Opsional, tapi disini dipasang aman 23:00)
TIMEZONE             = "Asia/Jakarta"

# ── RISK CONTROL ───────────────────────────────────────────
MAX_TRADE_PER_SESI   = 999          # Dibikin sangat besar agar tidak ada batas entry
MAX_LOSS_BERUNTUN    = 2            # Pause 1 jam setelah loss beruntun
PAUSE_SETELAH_LOSS   = 60           # Menit
MAX_LOSS_HARIAN      = 50.0

# ── LOGGING ────────────────────────────────────────────────
LOG_FILE        = "trading_london_ny.log"
LOG_LEVEL       = "INFO"
LOG_BACKUP_COUNT = 1   # Berapa hari log lama yang ingin disimpan (1 = simpan log kemarin, hapus sisanya)

# ── NOTION INTEGRATION ─────────────────────────────────────
NOTION_TOKEN         = os.getenv("NOTION_TOKEN", "")  # Diambil dari file .env
NOTION_DATABASE_ID   = os.getenv("NOTION_DATABASE_ID", "") # Diambil dari file .env

# ── AI THRESHOLD & FILTER ──────────────────────────────────
AI_CONFIDENCE_THRESHOLD = 0.53      # XGBoost V3 probability maxes out ~54-55%. Turunkan dari 0.60 agar bisa entry.
USE_SWEEP_FILTER        = False     # False = Agresif (Murni sinyal AI), True = Sabar (Wajib ada Liquidity Sweep)
