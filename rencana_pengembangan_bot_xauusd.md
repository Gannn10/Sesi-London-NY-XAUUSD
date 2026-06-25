# Rencana Pengembangan Bot Scalping XAUUSD (London/NY)
**Berdasarkan insight: Marco Acetony, Fabio Valentini, Okala**

---

## Status Saat Ini (Baseline)

- Model: XGBoost V3 Binary Classification
- Test Accuracy: **57.7%** | Train Accuracy: **57.5%** (gap sehat, tidak overfit)
- Fitur: 72 kolom (M5 + H1 + SMC + session sin/cos)
- HMM Regime Detector: 4 fitur (log_return, range, ADX, autocorrelation)
- Risk Control: Smart Exit (RSI exit, regime exit, step trailing stop)

**Catatan penting:** Baseline ini sudah stabil. Semua tahap di bawah harus diuji secara **terpisah (modular)** sebelum digabung, supaya kalau ada penurunan performa, jelas tahap mana penyebabnya.

---

## ⚠️ Peringatan Umum Sebelum Mulai

1. **Jangan gabung 4 tahap sekaligus.** Tahap 1, 3, dan 4 semuanya bersifat *filter* yang mengurangi jumlah sinyal. Jika digabung tanpa diukur satu-satu, bot bisa berhenti total entry tanpa kamu tahu filter mana yang terlalu ketat.
2. **Wajib bikin funnel logging.** Setiap filter harus mencatat: berapa sinyal masuk → berapa lolos. Tanpa ini, evaluasi jadi buta.
3. **Swing point harus konsisten.** Gunakan swing point yang sudah dikonfirmasi (delayed, non-lookahead) dari `smc_polars.py` yang sudah ada — jangan buat ulang versi real-time yang belum terkonfirmasi, karena itu sumber data leakage baru.
4. **Urutan implementasi disarankan:** Tahap 3 → Tahap 1 → Tahap 4 → Tahap 2 (paling akhir, karena butuh data terbanyak).

---

## Tahap 1: Filter Liquidity Sweep (Marco Acetony)

### Ide Dasar
Ritel sering masuk *terlalu cepat* di area support/resistance. Institusi menunggu level itu disapu (stop loss ritel tersapu) dulu sebelum masuk ke arah sebaliknya.

### Logika Implementasi
```
is_buy_allowed  = False (default)
is_sell_allowed = False (default)

IF current_close menembus last_swing_low (lalu kembali naik):
    is_buy_allowed = True

IF current_close menembus last_swing_high (lalu kembali turun):
    is_sell_allowed = True
```

### Cara Implementasi (Teknis)
- **Manfaatkan kolom yang sudah ada**: `last_swing_low`, `last_swing_high` dari `smc_polars.py` — jangan bikin sistem swing baru.
- Tambahkan kolom baru: `liquidity_swept_buy` (bool), `liquidity_swept_sell` (bool).
- Filter ini diterapkan **sebelum** prediksi XGBoost dipanggil (gerbang awal), atau sebagai filter tambahan **setelah** prediksi (mengonfirmasi sinyal AI).

### ⚠️ Yang Perlu Diwaspadai
- Window swing point default di `smc_polars.py` adalah `2*swing_length+1` (≈11 candle) — ini cukup micro/sensitif. Jika ingin sapuan yang lebih makro (seperti contoh 20-50 candle), pertimbangkan swing length lebih besar **khusus untuk filter ini**, terpisah dari swing point yang dipakai SMC signal lain.
- Risiko data leakage jika swing point yang dipakai belum dikonfirmasi forward bar.
- Filter ini akan **mengurangi jumlah sinyal** secara signifikan — wajib diukur dengan funnel logging.

### Validasi Sebelum Live
- [ ] Backtest filter ini sendirian (tanpa tahap lain) dibandingkan baseline
- [ ] Hitung % sinyal XGBoost yang lolos filter ini
- [ ] Bandingkan win rate: dengan filter vs tanpa filter

---

## Tahap 2: Routing Dua Model Berdasarkan Regime (Fabio Valentini)

### Ide Dasar
Market punya dua karakter: *Trending* (NY, momentum kencang) dan *Ranging* (London, mean reversion). Satu model tidak ideal untuk menebak keduanya sekaligus.

### Logika Implementasi
```
regime = detect_market_regime(df)  # sudah ada, 3 state

IF regime == HIGH_VOLATILITY (trending):
    gunakan trend_model.pkl

IF regime == LOW_VOLATILITY / CHOPPY (ranging):
    gunakan reversion_model.pkl
```

### ⚠️ Yang PALING Perlu Diwaspadai (Prioritas Tertinggi)
- **Ini bukan upgrade ringan — ini downgrade risiko tinggi jika dipaksa sekarang.**
- Data training sekarang ~47k sample. Jika dipecah ke 2 model, masing-masing model kebagian ~20-23k sample — **berisiko overfitting baru** karena data per model jadi terlalu sedikit.
- HMM regime detector yang sudah ada (4 fitur) jauh lebih nuanced dibanding sekadar threshold ATR tinggi/rendah. Jangan downgrade logika ini untuk keperluan routing.
- Maintenance jadi 2x lipat: 2 model, 2 set hyperparameter, 2x validasi, 2x risiko bug.

### Rekomendasi
- **Jangan dikerjakan dulu.** Gunakan regime detector yang sudah ada sebagai **filter pelengkap** (skip trade saat CHOPPY), bukan untuk routing ke model berbeda.
- Baru pertimbangkan Tahap 2 setelah data historis tersedia 100k+ bar **per regime** (butuh total 200k+ bar mentah), supaya masing-masing model punya sample yang cukup.

### Validasi Sebelum Live (jika tetap dilanjutkan nanti)
- [ ] Pastikan minimal 50k sample per model setelah split by regime
- [ ] Bandingkan test accuracy gabungan (2 model) vs model tunggal sekarang
- [ ] Jangan lanjut jika kedua model baru test accuracy-nya lebih rendah dari 57.7%

---

## Tahap 3: Full Candle Closure + Validasi Bentuk Candle (Okala & Fabio)

### Ide Dasar
Prediksi AI bisa benar arah tapi entry terlalu cepat saat candle belum close, menyebabkan floating minus akibat noise.

### Logika Implementasi
```
# A. Full Candle Closure (WAJIB)
current_features = df_features.iloc[[-2]]   # bukan [-1] yang masih live

# B. Validasi Bentuk Candle (OPSIONAL, perlu testing terpisah)
IF candle_closed == True AND body_to_wick_ratio memenuhi kriteria:
    FIRE_ORDER = True
```

### ⚠️ Yang Perlu Diwaspadai
- **Bagian A (candle closure)**: low-risk, sangat disarankan. Tinggal ganti index `[-1]` → `[-2]` di `london_ny_bot.py`.
- **Bagian B (validasi bentuk candle / body-wick ratio)**: berisiko over-filtering. Filter ini didesain Okala untuk Nasdaq di timeframe 200 detik dengan psikologi level 80/20 — karakteristiknya beda dengan XAUUSD M5. Bisa jadi sinyal valid malah ke-skip karena bentuk candle "tidak sempurna" menurut kriteria yang belum tervalidasi di instrumen ini.

### Validasi Sebelum Live
- [ ] Implementasi bagian A dulu, backtest, bandingkan dengan baseline
- [ ] Bagian B: backtest TERPISAH dari bagian A, ukur apakah benar menambah win rate atau cuma mengurangi jumlah trade

---

## Tahap 4: Dynamic SL/TP Berbasis Struktur Market (Marco & Fabio)

### Ide Dasar
SL statis (`ATR × multiplier`) tidak peduli struktur pasar. SL seharusnya diletakkan di titik yang paling tidak masuk akal untuk disentuh ulang oleh harga (di luar swing point), dan TP di area likuiditas berikutnya (swing high/low timeframe lebih besar).

### Logika Implementasi
```
# Saat BUY (setelah sweep swing low di Tahap 1):
SL = last_swing_low - buffer        # buffer, JANGAN persis di swing low
TP = swing_high_H1                  # likuiditas eksternal, timeframe lebih besar

# Saat SELL (setelah sweep swing high):
SL = last_swing_high + buffer
TP = swing_low_H1
```

### ⚠️ Yang Perlu Diwaspadai
- **SL persis di swing point M5 sangat rentan whipsaw.** Jarak swing low/high di M5 sering hanya beberapa pip dari entry — RR keliatan bagus di atas kertas, tapi win rate riil bisa jatuh karena gampang tersentuh noise sebelum momentum sungguhan terjadi.
- **Wajib pakai buffer**, seperti yang sudah ada di `smc_polars.py` function `generate_signal()` (pakai `min_sl_distance = 1.5 * ATR` sebagai pembanding, ambil yang lebih jauh/protektif).
- TP di swing H1 bisa membuat jarak TP sangat jauh dibanding SL M5 — RR bisa jadi ekstrem (misal 1:8) yang terdengar bagus tapi probabilitas tercapainya rendah. Perlu dicek realistis atau tidak lewat backtest.

### Validasi Sebelum Live
- [ ] Tentukan buffer optimal (mulai dari kombinasi swing point + 0.3-0.5×ATR seperti pola yang sudah ada)
- [ ] Backtest distribusi RR yang dihasilkan — apakah realistis atau terlalu ekstrem
- [ ] Bandingkan win rate dan expectancy vs sistem SL/TP statis saat ini

---

## Urutan Implementasi yang Disarankan

| Urutan | Tahap | Risiko | Alasan |
|--------|-------|--------|--------|
| 1 | Tahap 3A (candle closure) | Sangat rendah | Fix simple, langsung kurangi noise |
| 2 | Tahap 1 (liquidity filter) | Rendah-Menengah | Manfaatkan kolom yang sudah ada |
| 3 | Tahap 4 (dynamic SL/TP) | Menengah | Perlu buffer, perlu backtest RR |
| 4 | Tahap 3B (validasi bentuk candle) | Menengah | Perlu testing terpisah, riskan over-filter |
| 5 | Tahap 2 (routing 2 model) | **Tinggi** | Butuh data jauh lebih banyak, jangan dipaksa |

---

## Checklist Sebelum Live Trading (Semua Tahap)

- [ ] Setiap tahap diuji backtest **terpisah** sebelum digabung
- [ ] Funnel logging aktif (jumlah sinyal masuk vs lolos di setiap filter)
- [ ] Tidak ada penurunan test accuracy dibanding baseline (57.7%)
- [ ] Distribusi RR hasil Tahap 4 sudah realistis (bukan ekstrem)
- [ ] Risk control (`MAX_LOSS_BERUNTUN`) sudah aktif kembali (saat ini di-comment di `london_ny_bot.py`)
- [ ] Forward test di demo account minimal 2-4 minggu sebelum live
