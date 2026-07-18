import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from london_ny_bot import kirim_notion
from dotenv import load_dotenv

# Load credentials
load_dotenv()

import MetaTrader5 as mt5

print("Mencoba mengirim laporan fiktif ke Notion...")
if not mt5.initialize():
    print("Gagal inisialisasi MT5!")
else:
    kirim_notion(datetime.now().date())
    print("Selesai. Silakan cek Notion kamu.")
    mt5.shutdown()
