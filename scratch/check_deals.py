import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import MetaTrader5 as mt5
import config_london_ny as cfg
from datetime import datetime, timedelta

if not mt5.initialize():
    print("Gagal init MT5")
    sys.exit(1)

tanggal = datetime(2026, 7, 17)
start_date = tanggal - timedelta(hours=6)
end_date = tanggal + timedelta(hours=30)

deals = mt5.history_deals_get(start_date, end_date)

if not deals:
    print("Tidak ada deals ditemukan.")
    mt5.shutdown()
    sys.exit(0)

print(f"{'No':>3} | {'Waktu (WIB)':>19} | {'Tipe':>6} | {'Entry':>5} | {'Harga':>10} | {'Profit':>10} | {'Comment'}")
print("-" * 95)

no = 0
total_profit = 0
win = 0
loss = 0

for d in deals:
    if d.magic == cfg.MAGIC_NUMBER:
        no += 1
        t = datetime.fromtimestamp(d.time).strftime('%Y-%m-%d %H:%M:%S')
        tipe = "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL"
        entry = "IN" if d.entry == mt5.DEAL_ENTRY_IN else "OUT"
        net = d.profit + d.swap + d.commission
        comment = d.comment if d.comment else ""
        
        if d.entry == mt5.DEAL_ENTRY_OUT:
            total_profit += net
            if net > 0:
                win += 1
            else:
                loss += 1
        
        print(f"{no:>3} | {t} | {tipe:>6} | {entry:>5} | {d.price:>10.2f} | {net:>+10.2f} | {comment}")

print("-" * 95)
print(f"Total Closed: {win + loss} | Win: {win} | Loss: {loss} | Net Profit: ${total_profit:+.2f}")
mt5.shutdown()
